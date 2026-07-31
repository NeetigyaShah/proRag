# QDMS-AI (document search + RAG backend) — Implementation Notes

> Source root: `D:\QDMS\MariApps.ocean.ai.QDMS`. Package: `qdms_core` (+ optional `qdms_client` plugin package). Python `>3.10,<3.11`, Poetry, FastAPI/uvicorn. These notes are written to let a developer reproduce the design and reasoning end-to-end. Every file path is absolute.

---

## 1. What it is (one tight paragraph)

QDMS-AI is a maritime-domain **RAG backend** that powers an AI document-search + chat experience over a Quality Document Management System (Articles, internal docs, and uploaded files). It is a FastAPI app fronted by a C# wiki/proxy, mounted under a dynamic root path (`/qdms` for core, `/qdms-{client}` for a branded client). A request flows through **three specialist Azure OpenAI agents** built as import-time singletons in `qdms_core/config/deps.py`: a **FilterAgent** (`gpt-4.1-mini`) that turns natural language into MongoDB filters + a refined search phrase + follow-up/greeting decisions, a **SearchAgent** (`gpt-4.1`) that runs **dual retrieval** (vector MMR over Azure Cosmos DB vCore + `$text` keyword search over the source MongoDB) fused with normalized-score fusion and optionally reranked by a **cross-encoder ONNX model loaded at startup**, and a **ChatAgent** (`gpt-4.1-mini`) that synthesizes a cited, word-capped answer (non-streaming JSON or SSE token stream). It is heavily tuned for throughput under concurrency (separate httpx pools, per-endpoint semaphores, dual sync+async Mongo clients, raw-ASGI middleware) and fully instrumented with OpenTelemetry (traces → Tempo, metrics → Prometheus `/metrics`, logs → Alloy→Loki via OTLP).

---

## 2. Architecture at a glance

```
                       C# QDMS Wiki / reverse proxy   (shared service-account JWT)
                                     │
                                     ▼   root_path = /qdms  or  /qdms-{client}
┌──────────────────────────────────────────────────────────────────────────────┐
│ FastAPI app (qdms_core/main.py)                                                │
│   middleware (PURE ASGI, outer→inner):                                         │
│     CORS → GZipCompressMiddleware(skips text/event-stream) → RequestTiming     │
│   exception handlers: 404 / HTTPException / catch-all 500                       │
│   lifespan: create tables, raise anyio thread limiter→200, start APScheduler,  │
│             create Cosmos indexes, warm_connections(), graceful OTel shutdown  │
│                                                                                │
│   Routers:  auth (/token)  users  search  chat  scripts  vessel                │
│             + client routers (registered FIRST → override on dup paths)        │
│   /metrics (Prometheus)   /  (version dict)                                     │
└──────────────────────────────────────────────────────────────────────────────┘
        │ Depends: get_current_user (JWT+user cache), require_endpoint_access
        ▼
SmartChat / SmartSearch pipeline (qdms_core/ai/agents.py)
  is_greeting()? ── yes ──► skip filter+search ──► ChatAgent direct answer
        │ no
        ▼
  ┌── QDMSFilterAgent (gpt-4.1-mini, JSON parser) ──┐  (runs in PARALLEL with ACL fetch
  │   is_search_required / is_follow_up             │   on cache MISS via asyncio.gather)
  │   remaining_query + alternative_query           │
  │   filters[] ($eq/$ne/$lt/.../$in/$nin)          │
  └────────────────────┬────────────────────────────┘
                       │  MongoDynamicFilter → per-collection Mongo filters + ACL + vessel/role
                       ▼
   ┌─────────────────────────── parallel asyncio.gather ───────────────────────────┐
   │  $text content search (source MongoDB, Motor)   │  vector MMR search (Cosmos)  │
   │  Articles / docs / files                        │  Articles / docs / files     │
   │  +AND term forcing, textScore floor, 5s timeout │  simsimd cosine, HNSW efSearch│
   └────────────────────┬───────────────────────────┴──────────────┬───────────────┘
                        ▼  Normalized Score Fusion (text 0.4 / vector 0.6)
                  optional Cross-Encoder rerank (ONNX, accuracy mode)
                        ▼
       get_cropped_context() adaptive selection (dedup + revision-aware + dynamic floor)
                        ▼
   QDMSChatAgent (gpt-4.1-mini)  ── ainvoke (JSON) or astream (SSE tokens) ──►
       inline markdown citations → mistune → HTML → process_citations() links
                        ▼
   persist ChatHistory (SQL)        return cards/sources/chat_response

Data stores:
  SQL (SQLAlchemy): SQLite(local) / pymssql(develop,qa) / pyodbc ADIntegrated(uat,prod)
     tables: users, roles, chathistory, costtracking, user_document_access
  MongoDB SOURCE DB (oceanai_qdms_data_*): Articles, docs, files, fs.chunks, categories
     — sync pymongo + async Motor, $text indexes
  Cosmos DB vCore VECTOR DB (oceanai_qdms_mol_*): per-collection chunk+vector store
     — sync pymongo (LangChain) + async Motor (direct queries), HNSW vector index
Observability: OTel SDK → OTLP/http → Alloy → {Tempo, Prometheus(+/metrics), Loki}
```

---

## 3. Feature-by-feature implementation

### 3.1 Three specialist agents as import-time singletons (with filter injected into search)

**How.** `qdms_core/config/deps.py` constructs `qdms_filter_agent = QDMSFilterAgent(...)`, then `qdms_search_agent = QDMSSearchAgent(filter_agent=qdms_filter_agent)`, then `qdms_chat_agent = QDMSChatAgent(...)` — all at module import time. Every router imports these names (`from qdms_core.config.deps import qdms_chat_agent, qdms_search_agent`). The search agent stores the filter agent as `self.qdms_filter_agent` and calls it inside `QDMSSearchAgent.invoke()`. Each agent wraps its own `AzureChatOpenAI` (`langchain_openai`) with a distinct `azure_deployment` (`settings.qdms_search_agent_model` = `gpt-4.1`, chat/filter = `gpt-4.1-mini`).

**What.** LangChain `AzureChatOpenAI` + LCEL pipes (`prompt | llm | parser`). Module-level singletons (not FastAPI `Depends`).

**Why.** The module docstring says it plainly: *"avoids duplicate LLM clients, embedding clients, and vector DB connections."* Building agents once at import keeps a single shared httpx pool, single Cosmos connection, and preloads prompt templates. Using a *cheaper* model (`mini`) for the high-frequency filter + chat steps and the *stronger* `gpt-4.1` only for the search agent is a deliberate cost/quality split.

### 3.2 QDMSFilterAgent — filter extraction, follow-up detection, greeting short-circuit, prompt caching

**How.** `QDMSFilterAgent` (`qdms_core/ai/agents.py` ~line 1759). Prompt is split into `DYNAMIC_FILTER_SYSTEM` (static rules) + `DYNAMIC_FILTER_HUMAN` (per-request `current_date`, `chat_history`, `user_query`) in `qdms_core/ai/prompts.py`. Output is parsed with `JsonOutputParser(pydantic_object=DynamicFilter)` where `DynamicFilter` (`qdms_core/ai/output_parsers.py`) declares `is_search_required`, `is_follow_up`, `remaining_query`, `alternative_query`, `n`, `filters[]`. `ainvoke()` is split into three timed stages so only the LLM call holds the `_filter_semaphore`: (1) `dynamic_filter_prompt.ainvoke()` renders, (2) `self.llm.ainvoke(rendered)` under semaphore, (3) `output_parser.ainvoke()` parses. It then reads `cached_tokens` from `response_metadata["token_usage"]["prompt_tokens_details"]` and logs it.
- **Follow-up detection** is prompt-engineered (rule 3 in `DYNAMIC_FILTER_SYSTEM`): numerical refs ("step 2"), ordinals ("first/second/next/previous"), demonstratives ("that/this/it/those"), sequential words ("explain/elaborate"), process refs ("next phase/following step"). On a follow-up it extracts the referenced topic into `remaining_query`; if the answer is already in chat history it sets `is_search_required=false, n=0` to avoid a needless retrieval.
- **Greeting short-circuit** is *separate* and faster: `qdms_core/helpers/greeting.py::is_greeting()` is a pure keyword/token-gate function (≤4 tokens, normalized match against `_GREETING_WORDS`) called by the routers *before* the filter agent. When true, the whole filter+search stage is skipped and the chat agent answers directly.
- **remaining_query / alternative_query** rules force official maritime/IMO terminology, expand acronyms ("ism" → "ISM code safety management system"), and require the alternative query to use *genuinely different vocabulary* (used later for dual-query RRF).

**What.** Azure OpenAI `gpt-4.1-mini`, LangChain JSON output parser, Pydantic v2 model, `asyncio.Semaphore`.

**Why.** A small LLM is cheap enough to run on every turn and turns messy NL into structured filters + a clean retrieval phrase that works for *both* `$text` keyword search and vector embedding. The static/dynamic prompt split exists for **Azure prompt caching**: the comment at the top of `prompts.py` states Azure caches identical SystemMessage prefixes ≥1024 tokens, so the large rulebook lives in `_SYSTEM` (cacheable) and per-request data in `_HUMAN`. Splitting the semaphore-held region to only the LLM call lets prompt rendering and JSON parsing run without consuming Azure concurrency budget.

### 3.3 DynamicFilter operators + date-range expansion + ACL/vessel/role merge

**How.** `qdms_core/ai/filters.py`. `MongoDynamicFilter._combine_filters()` folds the LLM `filters[]` into `{field: {op: value}}`. Any field whose name contains "date" is parsed with `datetime.strptime(val, "%Y-%m-%d")`; a `$eq` on a date is **expanded to a full-day range** `{"$gte": dt, "$lt": dt + timedelta(days=1)}`. Supported operators (declared in the `Filter` Pydantic model and the prompt) are `$eq, $ne, $lt, $lte, $gt, $gte, $in, $nin` (`$regex` explicitly forbidden for dates). `dynamic_filter(filter_type)` then (1) maps LLM field names → collection-specific Mongo paths via `DEFAULT_FIELD_MAPPING` (e.g. `date` → `LastUpdatedTime` for files but `Date` for docs/articles), (2) ANDs `metadata.document_id: {$in: allowed_ids}` (ACL), (3) ANDs a vessel filter and (4) a role filter. Vessel/role use `_universal_or_match()` which builds an `$or` of `{field: {$exists:False}} / None / "" / [] / {$in:["all",...]} / {$in:[value variants]}` — note the explicit comment that this uses `$in` exact matches *instead of `$regex`* to be 30–50% faster during HNSW traversal.

**What.** MongoDB query operators, BSON datetime, hand-built `$or` ACL clauses.

**Why.** Date `$eq` on a timestamp field would never match (it's a full datetime); expanding to a day range fixes that. Mapping per-collection field names lets one LLM filter target three differently-shaped collections. The `$in`-not-`$regex` choice is a measured perf optimization because the filter is a `pre_filter` evaluated per-candidate inside the vector index scan.

### 3.4 Dual retrieval: vector MMR (simsimd) + `$text` content search, fused 0.4/0.6

**How.** `QDMSSearchAgent.invoke()` runs both arms in one `asyncio.gather`:
- **Vector arm** `_mmr_search()`: embeds the query (per-collection embedder), then issues a native async Motor `$search.cosmosSearch` aggregate against the Cosmos vector collection (`get_async_vector_collection`) with `path="vectorContent"`, `k=effective_fetch_k`, `efSearch=settings.hnsw_ef_search` (40), filtering by `min_score_threshold` (0.5). In **accuracy mode** it re-fetches the stored chunk embeddings and runs `_fast_mmr()` — a hand-written MMR that pre-computes the full similarity matrix once using **`simsimd.cdist(metric="cosine")`** (falls back to numpy if `simsimd` missing), replacing LangChain's `maximal_marginal_relevance` which recomputes similarities every loop iteration. In **latency mode** MMR is skipped (Cosmos scores already ranked) saving ~1.3 MB transfer + numpy work.
- **Content arm** `_content_search()`: a Mongo `$text` query on the *source* collection. Significant words are prefixed with `+` to force **AND semantics** (comment: without it "ISM Code" matches "Dress code" via OR on "Code"); a stopword set is excluded from the `+` forcing. Results are floored by `textScore >= max(len(significant)*1.5, 2.0)`. Uses `find()` (not `aggregate()`) deliberately so the optimizer pushes sort+limit into the text index scan (2–3× faster under load), with `max_time_ms` + `asyncio.wait_for(timeout=5.0)` as a hard server-side + client-side deadline.
- **Fusion** `_score_fuse()` / `_fuse()`: min-max normalize each list to [0,1] then combine with `fusion_text_weight=0.4` + `fusion_vector_weight=0.6` (settings), keyed by a stable business id (`ArticleNumber`/`DocumentNumber`/`FileID`) so matches survive reimports. Degenerate all-equal lists fall back to rank-based scoring. Documents found by both arms get contributions from both. There is also `_dedup_by_key()` (keep best-scoring chunk per business key) and, when an `alternative_query` exists, `_merge_vector_results()` fuses primary+alt vector results with **Reciprocal Rank Fusion (rrf_k=60)**.

> Note: the README claims fusion is "0.3/0.7"; the live defaults in `settings.py` are **0.4 text / 0.6 vector** (`fusion_text_weight=0.4`, `fusion_vector_weight=0.6`). Trust the code.

**What.** Azure Cosmos DB vCore vector search (Mongo API), `simsimd` SIMD cosine, MongoDB `$text` inverted index, custom NSF + RRF fusion.

**Why.** Vector search captures semantic similarity; `$text` captures exact keyword/title hits (acronyms, document numbers) that embeddings miss. Fusing both with vector-favored weights gives recall without keyword noise dominating. `simsimd` + precomputed matrix removes the dominant CPU cost of MMR.

### 3.5 Cross-encoder ONNX reranker, loaded at startup in a threadpool

**How.** `QDMSSearchAgent.load_reranker()` loads `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder` with `backend="onnx"`, `file_name="onnx/model_quint8_avx2.onnx"` (a quantized AVX2 ONNX), resolving the model to a local cache path via `huggingface_hub.snapshot_download(local_files_only=True)` so it never hits the HF API. It is **preloaded at startup**: `deps.warm_connections()` calls `await run_in_threadpool(qdms_search_agent.load_reranker)` (only in `SEARCH_MODE=accuracy`). Inference runs in a **dedicated `ThreadPoolExecutor`** (`self._reranker_executor`, `thread_name_prefix="reranker"`, sized `cpu_count//2`) with `OMP_NUM_THREADS=1`, isolated from the shared anyio threadpool. `_rerank_batch()` flattens all collections into one `predict()` call (prepending each chunk's title to its content) and filters by `reranker_threshold` (0.0).

**What.** `sentence-transformers` CrossEncoder, ONNX Runtime quantized model, `huggingface_hub` offline snapshot, dedicated `ThreadPoolExecutor`.

**Why.** The comment is explicit: the model-load cost is 10–15 s, so paying it once at startup keeps the first user query fast. `OMP_NUM_THREADS=1` + a private executor prevents "thread-in-thread" contention that caused 0.8–5.6 s latency variance against the anyio pool used by vector searches. `HF_HUB_OFFLINE=1` (forced in `main.py` before any HF import) avoids 429 rate limits on model metadata checks.

### 3.6 Per-collection embedders + differing chunk sizes (1536 vs 6144)

**How.** `QDMSSearchAgent.__init__` builds three `QDMSEmbedder` (`qdms_core/ai/embedder.py`): `Articles`, `docs` (chunk 1536), and `files` with **`chunk_size=6144`**. Each `QDMSEmbedder` owns its own `AzureOpenAIEmbeddings(model=text-embedding-3-small)` on the dedicated embed httpx client and its own `AzureCosmosDBVectorSearch` store. The actual chunking happens in `QDMSMongoLoader` (`qdms_core/ai/document_loader.py`) via `RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=..., chunk_overlap=100)`.

**What.** LangChain `AzureOpenAIEmbeddings` + `AzureCosmosDBVectorSearch`, tiktoken-aware splitter.

**Why.** Articles/docs are short structured HTML, so smaller 1536-token chunks give precise retrieval; uploaded **files** (PDFs/manuals) are long and benefit from larger 6144-token chunks to preserve context. Separate embedders keep the per-collection vector store + custom `_document_from_point` loader cleanly bound.

### 3.7 Adaptive context assembly (quality-not-count, revision-aware dedup)

**How.** `qdms_core/helpers/chat.py::get_cropped_context()` pools all reranked results, sorts by `similarity_score`, computes a **dynamic score floor** = `max(top_score - _SCORE_GAP, _SCORE_FLOOR)` whose constants auto-calibrate by mode (`accuracy`: CE-logit gap 8.0, floor 0.0; `latency`: NSF gap 0.8, floor 0.1). It enforces `_MIN_CONTEXT=3` and `_MAX_CONTEXT=max_context_docs(15)`, drops chunks shorter than `MIN_CONTENT_CHARS(200)`, dedups by business key, and then **revision-aware dedups** by `_normalize_title()` (a regex in `agents.py` that strips version/rev numbers, dates, month names, possessives, parenthesized numbers so "BS FMD Reporting Procedures _May-2021" collapses with "BS FMD Reporting Procedures").

**What.** Pure-Python selection + a compiled `_TITLE_NOISE_RE`.

**Why.** Fixed top-N wastes the token budget on near-duplicate revisions of the same manual and on low-content chunks; an adaptive floor keeps high-confidence queries tight while low-confidence queries still get ≥3 docs.

### 3.8 ChatAgent: cited answers, word cap, JSON + SSE streaming

**How.** `QDMSChatAgent` builds `ChatPromptTemplate` from `CHAT_SYSTEM`/`CHAT_HUMAN` with `{max_response_words}` substituted from `settings.chat_max_response_words` (300). `format_context()` emits `article_1/doc_1/file_1` blocks each with `Title:`/`URL:` (URL via `_build_citation_href`) and strips a `_METADATA_HEADER_RE` of internal header lines so the LLM doesn't confuse "Doc 06" with citation indices. The system prompt demands **inline markdown links `[Title](URL)`** placed after each sentence (never clustered, never a Sources section). Two entry points: `ainvoke()` (non-stream, records token usage + duration as OTel metrics) and `astream()` (async generator yielding tokens; records TTFT). Post-processing in `helpers/chat.py`: `normalize_citations()` repairs LLM deviations (`(doc_1)`→`[doc_1]`, casing, strips appended "Sources:" sections, remaps title-based cites), `convert_markdown_to_html()` (mistune + table plugin, plus `_unescape_llm_html_entities` to recover `&lt;table&gt;`), then `process_citations()` swaps `[doc_N]` for `<a class="citation-link">` tags.

**What.** LangChain LCEL `prompt | llm`, mistune markdown, regex citation pipeline.

**Why.** Inline clickable citations are the product's trust signal; the post-processor is defensive because LLMs frequently deviate from the citation format. Word cap controls cost and latency.

### 3.9 SSE streaming endpoint (`/SmartChat/stream`, `/SmartSearch/stream`)

**How.** `qdms_core/routers/chat.py::chat_stream` returns a `StreamingResponse(media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})`. The async `event_stream()` emits a typed SSE protocol: `status` (progress hints "QDMSbot is thinking…"), `cards` (full metadata), `sources`, `token` (per LLM token), `meta` (`response_id`), `error`, `done`; it opens with `retry: 3000` so browsers reconnect after 3 s. Token streaming has two robustness guards: a **table-row buffer** holds partial `| …` lines until `\n` so markdown renderers never see a broken row, and a **whitespace-runaway guard** aborts after 200 consecutive whitespace chars (LLM padding-loop failure mode) and appends a truncation notice. After streaming it persists the processed HTML and emits `meta`.

**What.** Starlette `StreamingResponse`, manual SSE framing, `X-Accel-Buffering: no` for nginx.

**Why.** Real-time token delivery for chat UX; the guards prevent the two real failure modes observed (broken tables mid-stream, runaway whitespace).

### 3.10 Multi-environment SQL engine factory

**How.** `qdms_core/config/db.py::get_database_engine()` switches on `settings.env`: `local` → SQLite (`sqlite:///.\test.db`, `check_same_thread=False`); `develop`/`qa` → `mssql+pymssql://user:pass@server:port/db` (SQL auth, password URL-quoted); `uat`/`prod` → `mssql+pyodbc:///?odbc_connect=...` with an ODBC string using `Authentication=ActiveDirectoryIntegrated;Encrypt=yes;TrustServerCertificate=yes` and **no username/password**. All non-SQLite engines share `pool_size=100, max_overflow=100, pool_timeout=30, pool_recycle=3600, pool_pre_ping=True`. Engine is created at import; `SessionLocal`/`Base` + `get_db()` generator follow.

**What.** SQLAlchemy 2.x `create_engine`, `pymssql`, `pyodbc` (ODBC Driver 18), `declarative_base`.

**Why.** Dev runs zero-dependency on SQLite; lower envs use simple SQL credentials; production uses Azure AD integrated auth so **no DB password is stored anywhere** — a security posture choice. `pool_pre_ping` + `pool_recycle` survive idle-connection drops on Azure SQL.

### 3.11 Dual Mongo topology: source DB + Cosmos vector DB, each sync + async

**How.** `qdms_core/config/mongodb.py` builds **four** clients sharing `_CONN_PARAMS` (5 s connect/serverSelection, 30 s socket, `maxPoolSize=settings.mongo_max_pool_size(500)`, `appName="qdms-ai"`): `sync_client`/`async_client` for the **source** DB (`MONGO_DB_NAME`), and `vector_sync_client`/`async_vector_client` for the **Cosmos vector** DB (`MONGO_VECTOR_DB`/`_NAME`). `get_async_mongo_db()` is the FastAPI dependency for source-DB endpoints; `get_vector_collection()` (sync, for LangChain's vector store) and `get_async_vector_collection()` (Motor, for direct vector queries inside `_mmr_search`) cover the vector DB.

**What.** `pymongo.MongoClient` (sync) + `motor.AsyncIOMotorClient` (async), Cosmos DB vCore Mongo API.

**Why.** Two physically separate databases: the source DB holds raw documents + `$text` indexes (content search lives here), the Cosmos DB holds chunk embeddings + HNSW vector index (vector search lives here). Each needs a *sync* client (LangChain's `AzureCosmosDBVectorSearch` requires a sync pymongo collection; `run_in_threadpool` DB work) and an *async* Motor client (non-blocking direct queries in `async def` endpoints). Sharing one vector sync client across the three embedders (instead of `from_connection_string` per embedder) cut monitor threads from 3→1 (per the in-file comment).

### 3.12 Two httpx pools (LLM vs embedding)

**How.** `qdms_core/config/httpx_client.py` splits the connection budget (`azure_http_max_connections=500`): **60 % → `azure_http_client`** (chat+filter LLM), **40 % → `azure_embed_http_client`** (embeddings). Both use HTTP/2, 120 s keepalive (vs OpenAI SDK default 5 s), and a shared response hook `_log_azure_backend_health()` that warns on `x-envoy-upstream-service-time > 5 s` and TPM remaining `< 5 %` (`x-ratelimit-remaining-tokens`). The agents pass these as `http_async_client=...` to `AzureChatOpenAI`/`AzureOpenAIEmbeddings`.

**What.** Two `httpx.AsyncClient` pools, HTTP/2 multiplexing, response event hook.

**Why.** Embedding load is bursty (3 collections × N queries); on a single shared pool a flood of LLM calls would starve embeddings (the comment estimates "+40 s queue time"). Separate TCP pools isolate the two workloads. The header hook gives early warning of Azure backend slowness / token-budget exhaustion in logs.

### 3.13 GZipCompressMiddleware that skips SSE + RequestTimingMiddleware

**How.** `qdms_core/middleware.py`. Both are **pure ASGI** (`async def __call__(scope, receive, send)`), not Starlette `BaseHTTPMiddleware`. `GZipCompressMiddleware` wraps `send`, buffers non-streaming bodies, and gzips (level 1) anything ≥ `minimum_size` (1000) when the client sends `Accept-Encoding: gzip` — logging `[COMPRESS] original/compressed/saved%`. Crucially, in `http.response.start` it inspects `content-type`; if it sees **`text/event-stream`** it sets `is_streaming=True` and passes every chunk straight through uncompressed. `RequestTimingMiddleware` records monotonic elapsed and logs `[REQUEST] METHOD path -> status | duration`, skipping noisy prefixes (`/docs`, `/health`, `/metrics`, …).

**What.** Raw ASGI middleware, `gzip` stdlib.

**Why.** The module docstring documents the reason: `BaseHTTPMiddleware` buffers the full body via an anyio channel and spawns an extra Task per request, which under 20+ heavy concurrent workflows delays lightweight requests — raw ASGI wraps only `send` with zero buffering and zero extra tasks. Skipping `text/event-stream` is essential: buffering an SSE response to gzip it would destroy the real-time chunked delivery. `RequestTimingMiddleware` replaces uvicorn's access log (disabled via `access_log=False` in `run_qdms.py`) with a single consistent line.

### 3.14 Access control: external API or local table, fail-open in non-prod

**How.** `QDMSSearchAgent.invoke()` resolves allowed document IDs through `qdms_core/helpers/access_cache.py` (`AccessCache`, per-user TTL 300 s, `time.monotonic` expiry, per-user `asyncio.Lock` to prevent thundering-herd). On a cache **miss** it launches `get_all_items()` (`qdms_core/helpers/user_access_api.py`) and the filter agent **concurrently** via `asyncio.gather`. `get_all_items()` POSTs to `USER_ACCESS_API_URL` (or reads `user_documents.json` when the URL is literally `"local"`) and maps `ItemType` `ART/DOC/FLS` → `articles/documents/files` lists. If access lookup fails: in **non-prod** (`env ∈ {local,develop,qa}`) `_is_acl_fail_open_enabled()` returns True → `allowed_items = {all: None}` (unrestricted, explicit fail-open sentinel); in **prod/uat** it fails **closed** to empty allow-lists. When `qdms_access_control_enabled=false` the whole step is skipped. A `UserDocumentAccess` SQLAlchemy table (`models/user_document_access.py`) provides an alternative local source via `get_allowed_items(session)`.

**What.** httpx external API, in-memory TTL cache with per-key locks, SQLAlchemy fallback table, OTel cache hit/miss counter + duration histogram.

**Why.** Document-level permissions live in the upstream maritime system; caching them per-user removes a blocking round-trip from the hot path, and running the fetch in parallel with the filter LLM hides its latency. Fail-open in dev keeps local work unblocked; fail-closed in prod is the safe default.

### 3.15 Client plugin system + dynamic root path

**How.** `qdms_core/plugins/loader.py` reads `QDMS_CLIENT`. On `core` everything returns safe defaults (no-op routes, `ClientPolicy()`, `None` hooks). On a non-core value it `importlib.import_module("qdms_client.<module>")` and **raises** on ImportError (deployment error, not silent fallback). Hooks (`SearchHook`/`ChatHook` Protocols in `plugins/contracts.py`) are validated via `runtime_checkable isinstance`. `main.py::_load_version()` derives `root_path = "/qdms"` for core or `"/qdms-{client}"` otherwise, and `display_name`/`full_version` from `pyproject.toml` `[tool.poetry].version`. Client routers are registered **before** core routers so they win on duplicate paths (FastAPI first-registered-wins). Prompt overrides are loaded + validated by `plugins/prompts.py::load_validated_prompt()` (checks required template vars, falls back to core prompt). `qdms_client/` ships a template (`config.py POLICY`, `hooks.py search_hook`, `routers.py /template/health`).

**What.** `importlib` dynamic import, `typing.Protocol` + `runtime_checkable`, frozen `ClientPolicy` dataclass, FastAPI router precedence.

**Why.** One codebase, many branded client deployments (per `wiki/client_customization.md`). Per-client policy (token caps, endpoint roles, monthly cost limit), prompt tweaks, hooks, and extra routes without forking core. Distinct root paths let multiple client instances sit behind one reverse proxy.

### 3.16 Observability: OTel traces/metrics/logs, Prometheus, Alloy→Loki

**How.** `qdms_core/observability/telemetry.py::setup_telemetry()` (called first in `main.py`) registers a `TracerProvider` (`BatchSpanProcessor` → OTLP `/v1/traces`), a `MeterProvider` with **both** `PrometheusMetricReader` (feeds `/metrics`) and a `PeriodicExportingMetricReader` → OTLP `/v1/metrics`, and a `LoggerProvider` (`BatchLogRecordProcessor` → OTLP `/v1/logs`). `instrument_libraries()` (called *before* app imports) auto-instruments SQLAlchemy, httpx, and pymongo (Motor delegates to pymongo); `instrument_app(app)` adds FastAPI instrumentation excluding `/metrics` and `^/$`. `metrics.py` adds psutil system metrics + four DB-pool observable gauges + cache metrics. Custom metrics in `agents.py`: `gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, `qdms.pipeline.stage.duration`, `qdms.pipeline.stage.errors`. `logger.py` attaches an OTel `LoggingHandler` and `LoggingInstrumentor` injects `otelTraceID`/`otelSpanID` into every log line. `dashboards/provisioning/datasources.yaml` wires Grafana Loki↔Tempo↔Prometheus with trace-to-log correlation.

**What.** OpenTelemetry SDK 1.40.0, OTLP/HTTP exporters, Prometheus exporter, Grafana Alloy as the OTLP collector forwarding to Tempo/Prometheus/Loki.

**Why.** Full RED metrics + distributed traces + correlated logs across the multi-stage pipeline, with a single `OTEL_EXPORTER_OTLP_ENDPOINT` and a single kill-switch (`OTEL_TELEMETRY_DISABLED`) that returns NoOp providers at zero overhead.

### 3.17 Cosmos HNSW index + metadata filter indexes + `$text` indexes

**How.** Two index paths. (a) `QDMSEmbedder.__init__` calls `vector_store.create_index(num_lists=1, dimensions=1536, similarity=COS, kind=VECTOR_HNSW, m=16, ef_construction=200)` (best-effort, warns if exists). (b) `qdms_core/scripts/create_cosmos_indexes.py::create_mongodb_indexes()` issues a raw `db.command({"createIndexes": col, "indexes":[{"key":{"vectorContent":"cosmosSearch"}, "cosmosSearchOptions":{"kind":"vector-hnsw","m":16,"efConstruction":200,"similarity":"COS","dimensions":1536}}]})` — deliberately raw because pymongo's `create_index()` strips the unknown `cosmosSearchOptions`. It then samples 100 docs to discover `metadata.*` keys and creates a filter index per key. Run at startup (`lifespan` → `create_mongodb_indexes("vector")`). `qdms_core/scripts/create_text_indexes.py` builds `$text` indexes on the *vector* collections with weighted fields (`metadata.<Name>`:10, `textContent`:1).

**What.** Cosmos DB vCore HNSW vector index (cosine, m=16, efConstruction=200), Mongo metadata `$1` filter indexes, Mongo weighted `$text` indexes.

**Why.** HNSW gives sub-linear ANN search; filter indexes make the `pre_filter` ACL/date clauses fast inside the vector scan; `$text` indexes turn keyword search from a `$regex` full scan into an inverted-index lookup (the script header notes `$text` is "5x faster and finds 15x more results").

### 3.18 Background data sync + embedding pipeline

**How.** `main.py` registers an APScheduler `BackgroundScheduler` interval job (`hours=sync_interval_minutes/60`, `max_instances=1`, `next_run_time=None`) and arms the first run only *after* warmup via `scheduler.reschedule_job(start_date=now+interval)`. The job `run_data_sync()` (`routers/scripts.py`): `download_blobs_from_azure.main()` (pull SAS-URL zips, extract `art/docs/forms/fschunks/categories.json`, bulk-upsert into source Mongo with `embedding_status="pending"`), then `load_and_embed_documents()` per collection. Embedding (`embed_collection.py` → `QDMSEmbedder.embed_documents` → `QDMSMongoLoader.load`) shuffles pending docs, deletes prior chunks, extracts content (HTML via `Html2TextTransformer`, files via `FileLoader`), writes a `SearchText` field (title + first 2 KB) used by `$text` + reranker, splits, and upserts vectors.

**What.** APScheduler, `azure-storage-blob` `ContainerClient`, pymongo `bulk_write(UpdateOne, upsert=True)`, LangChain loaders.

**Why.** Incremental, status-tracked re-embedding (`pending/completed/failed`) so only changed docs are re-processed; the `next_run_time=None` + post-warmup reschedule prevents a sync from firing during cold startup.

### 3.19 Versatile FileLoader (PDF/DOCX/DOC/XLSX/PPTX/RTF/XML/images/zip)

**How.** `qdms_core/ai/files_loader.py::FileLoader.load()` dispatches by extension with **fallback chains**: PDF → `PyPDFLoader` then `PDFPlumberLoader` (handles password-protected → empty doc); legacy `.doc` → bundled **antiword.exe** subprocess (`qdms_core/ai/antiword/`); Excel → `UnstructuredExcelLoader` → pandas (openpyxl/xlrd/odf/pyxlsb) → HTML-disguised-as-Excel detection; `.ppt/.pptx` → python-pptx (replaced Unstructured which "hangs on some files"); images → `ImageCaptionLoader`; video → sample 3 frames via OpenCV then caption; zip → recursive.

**What.** LangChain community loaders, antiword, pandas multi-engine, OpenCV, python-pptx.

**Why.** Maritime document corpora are messy (mislabeled extensions, HTML files saved as `.xls`, password PDFs, legacy `.doc`); layered fallbacks maximize successful extraction without one bad file failing the batch.

---

## 4. ★ Unique & special implementations

These are the things you would *not* see in a generic RAG service:

- **Three model-differentiated agents wired as import-time singletons with cross-injection** — filter (`gpt-4.1-mini`) injected into search (`gpt-4.1`), chat (`gpt-4.1-mini`) separate. The deliberate cheap/strong split + shared single clients is unusual; most projects create LLMs per-request.
- **Prompt-cache-aware system/human split** — the rulebook lives in `*_SYSTEM` specifically to hit Azure's ≥1024-token prefix cache, and the filter agent logs the returned `cached_tokens` to verify it's working.
- **`_fast_mmr` with a precomputed `simsimd` cosine matrix** replacing LangChain's per-iteration recompute — a real algorithmic optimization, plus a mode switch that *skips MMR entirely* in latency mode.
- **Dual retrieval with two different physical databases** — `$text` keyword search on the **source** Mongo DB and vector search on the **Cosmos** vector DB, fused with weighted NSF and (for alt queries) RRF. Two-database split is rare; most hybrid-search systems keep both indexes co-located.
- **Cross-encoder reranker preloaded at startup via `run_in_threadpool` inside `warm_connections()`**, run on a *dedicated* `ThreadPoolExecutor` with `OMP_NUM_THREADS=1`, using a **quantized ONNX** model resolved through an offline HF snapshot. The thread-isolation reasoning (avoiding thread-in-thread contention) is a non-obvious, measured fix.
- **`+`-forced AND `$text` semantics with a stopword carve-out and a dynamic `textScore` floor** (`max(len(significant)*1.5, 2.0)`) — solves the "ISM Code matches Dress code" OR-bug.
- **Per-collection embedders with different chunk sizes (1536 vs 6144)** and a separate larger chunk just for uploaded files.
- **Multi-env SQL factory where prod uses ActiveDirectoryIntegrated with NO password** (pyodbc ODBC 18) while dev/qa use pymssql SQL-auth and local uses SQLite — one factory, three auth strategies.
- **Dual Mongo topology with four clients** (sync+async × source+vector), each justified (LangChain needs sync, async endpoints need Motor).
- **Two separate httpx pools (60/40 LLM vs embed)** with HTTP/2 and an Azure-header health hook warning on backend slowness / TPM exhaustion.
- **Pure-ASGI `GZipCompressMiddleware` that explicitly skips `text/event-stream`** so SSE is never buffered, plus a pure-ASGI timing middleware replacing uvicorn access logs — both chosen to dodge `BaseHTTPMiddleware`'s task/buffer overhead under concurrency.
- **SSE stream with a markdown-table-row buffer and a whitespace-runaway abort** — two production-hardening guards most streaming endpoints lack.
- **Revision-aware title normalization** (`_TITLE_NOISE_RE` / `_normalize_title`) that collapses dated/versioned revisions of the same manual so context isn't wasted on duplicates.
- **Per-operation concurrency semaphores** — module-level `asyncio.Semaphore` globals in `agents.py` (`_chat_semaphore=32`, `_filter_semaphore=32`, `_embedding_semaphore=64`) plus a per-`QDMSSearchAgent` `_vector_semaphore=30`, shared across all requests (not literally per-endpoint), so fast filter calls don't queue behind slow chat calls, plus an `anyio` thread limiter raised from 40→200.
- **Client plugin system + dynamic `/qdms-{client}` root path** with fail-closed import semantics and validated prompt overrides.
- **Per-user `AccessCache` with `asyncio.Lock` anti-stampede**, ACL fetched *in parallel* with the filter LLM on cache miss, and env-gated fail-open/closed.

---

## 5. Easily-missed / subtle details & gotchas

- **Cost tracking is effectively dead code.** `qdms_core/helpers/cost_tracking.py` (`is_monthly_limit_reached`, `update_monthly_cost`) and the `CostTracking` table exist, the README advertises 409/429 enforcement, **but nothing in the live request path calls them** (verified by grep). Every chat/search persists `cost=0.0`, and both agents `return response.content, 0.0`. The README's "OpenAICallbackHandler" cost capture is not implemented. Token usage *is* captured, but only into OTel metrics, not into the cost table.
- **README ↔ code drift.** Fusion weights are 0.4/0.6 in code (README says 0.3/0.7). Default `SEARCH_MODE` is `latency` (so the reranker is *off* by default; `RERANKER_ENABLED = settings.search_mode == "accuracy"`). README mentions a `qdms_core/ai/customs.py` and `config/client.py` — **neither exists** in the tree. Default `min_score_threshold` is 0.5 in settings (README/older `.env` say 0.2). README "root_path `/qdms-imcsm`" is just an example; actual root is derived from `QDMS_CLIENT` (`core` → `/qdms`).
- **Secrets committed to `.env`.** Real-looking Azure OpenAI key, Cosmos connection string (with password), and a SAS URL are checked in at `D:\QDMS\MariApps.ocean.ai.QDMS\.env`, and `ACCESS_TOKEN_EXPIRE_MINUTES=25920000` (~49 years) — directly contradicting `.claude/rules/security.md`. The `SECRET_KEY` value is even a leftover-looking OpenAI-style string.
- **Per-user auth is intentionally trusted, not enforced.** `ChatHistory`/`chat/sessions` endpoints comment that the C# proxy uses a *shared service-account JWT* and authenticates the end user upstream, so `user_id` from the request body is trusted (no ownership check). This is a deliberate trust boundary, easy to miss.
- **User cache detaches ORM objects.** `helpers/auth.py::get_current_user` eagerly touches `user.role`, then `db.expunge` + `make_transient` before caching — otherwise reading `user.role` from a cached, session-detached object raises `DetachedInstanceError`. Cache key is `username`, TTL 60 s.
- **`/token` is a sync `def` on purpose** so FastAPI runs bcrypt's ~300 ms hash in a threadpool, keeping it off the event loop (comment in `routers/auth.py`).
- **`HF_HUB_OFFLINE=1` is forced in `main.py` before any HF import** — must be set early or the reranker load hits HF and 429s.
- **Embedding loader shuffles pending docs** (`random.shuffle`) before processing — spreads load and avoids re-failing the same stuck doc first every run.
- **`_combine_filters` will raise on a non-ISO date** (`strptime("%Y-%m-%d")`) — the prompt is the only guard forcing `YYYY-MM-DD`.
- **`create_filter_index` over sampled keys** — `create_cosmos_indexes.py` only samples the first 100 docs to discover `metadata.*` keys, so a metadata field absent from the first 100 docs gets no filter index.
- **Atomic cost UPDATE pattern** in `update_monthly_cost` uses `update(...).values(total_cost=CostTracking.total_cost + cost)` to avoid lost-update races — good pattern, but unreachable (see first bullet).
- **Greeting detection is conservative** (`≤4 tokens`, exact normalized match) — false negatives fall through to the full pipeline, which is the safe direction.
- **`X-Accel-Buffering: no` + `Cache-Control: no-cache`** on SSE responses defeat nginx buffering; without them the gzip-skip alone wouldn't guarantee real-time delivery behind the proxy.
- **`anyio` default thread limiter raised 40→200** in `lifespan`; with the many `run_in_threadpool` SQL/sync-Mongo calls, the default 40 would bottleneck under concurrency.
- **`DEFAULT_FIELD_MAPPING` mismatch for files** — `date` maps to `LastUpdatedTime` on files but `Date` on articles/docs; `reviewed_date` has no `files` entry, so a reviewed-date filter silently doesn't apply to files.
- **`content_search` strips `metadata.document_id` before applying filters** to the *source* DB (source docs use `_id`, not `metadata.document_id`), and converts allowed-id strings to `ObjectId` when valid.
- **Display vs context split** — every router returns *pre-reranker* `display_*` lists for UI cards but feeds the *reranked* lists to the LLM context, so the cards show a broader set than what the answer cited.

---

## 6. Tech inventory

| Library / tool | Version (pyproject) | Role & why |
|---|---|---|
| FastAPI | ≥0.85.1 | ASGI web framework; all routers/endpoints |
| uvicorn | ^0.30.6 | ASGI server (`run_qdms.py`, keep-alive 669 s, `access_log=False`) |
| Starlette | ≥0.27 | `StreamingResponse` (SSE), `run_in_threadpool` |
| Pydantic / pydantic-settings | ^2.0 / ^2.0 | request/response schemas + `Settings` env binding |
| SQLAlchemy | ^2.0.34 | ORM + multi-env engine, pool metrics |
| pymssql | ^2.2.11 | SQL Server SQL-auth driver (develop/qa) |
| pyodbc (ODBC Driver 18) | (system) | SQL Server ActiveDirectoryIntegrated (uat/prod) |
| pymongo | ^4.8 | sync Mongo (source + Cosmos vector via LangChain) |
| motor | ^3.5.1 | async Mongo (source + vector direct queries) |
| langchain / -core / -openai / -community / -text-splitters | 0.2.x | LLM orchestration, `AzureChatOpenAI`, `AzureOpenAIEmbeddings`, `AzureCosmosDBVectorSearch`, loaders, `RecursiveCharacterTextSplitter` |
| Azure OpenAI (gpt-4.1, gpt-4.1-mini, text-embedding-3-small) | API `2025-01-01-preview` | filter/search/chat LLMs + embeddings |
| Azure Cosmos DB vCore (Mongo API) | — | vector store w/ HNSW index (m=16, efConstruction=200, cosine, 1536-dim) |
| simsimd | ≥5.0 | SIMD cosine for `_fast_mmr` (numpy fallback) |
| sentence-transformers | ≥2.2 | CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| onnxruntime / optimum | 1.19.2 / ≥2.0 | quantized ONNX reranker backend |
| torch (CPU) | ≥2.0,<2.6 | sentence-transformers dependency (CPU wheel source) |
| huggingface_hub | (via st) | offline model snapshot resolution |
| httpx | ≥0.24 | two Azure pools (HTTP/2), access-API client |
| mistune | ^3.1.3 | markdown→HTML (table plugin) for answers |
| html2text | ^2024.2.26 | HTML→text for article/doc embedding |
| pypdf / pdfminer-six / opencv / Pillow / openpyxl / xlrd / docx2txt / python-pptx / antiword(.exe) | various | multi-format file extraction |
| APScheduler | ^3.10.4 | background data-sync scheduler |
| azure-storage-blob | ^12.24.0 | blob/SAS sync of source zips |
| python-jose / passlib[bcrypt] / bcrypt | ^3.3.0 / ^1.7.4 / ^4.2.0 | JWT + password hashing |
| OpenTelemetry SDK + OTLP/HTTP + Prometheus exporter + instrumentations | 1.40.0 / 0.61b0 | traces/metrics/logs; auto-instr FastAPI/SQLAlchemy/httpx/pymongo/system |
| prometheus_client | (via exporter) | `/metrics` endpoint |
| Grafana Alloy / Tempo / Loki / Prometheus | (infra) | OTLP collector → trace/log/metric backends |
| tiktoken | ^0.7.0 | token-aware chunk splitting |
| python-dotenv | ^1.0.1 | `.env` loading |
| tomli | ^2.4.0 | read `pyproject.toml` version at startup |
| Poetry / Ruff / mypy / deptry / pytest | — | build, lint, types, dep-check, tests |

Key env vars (from `settings.py` / `.env`): `ENV`, `QDMS_CLIENT`, `SEARCH_MODE` (latency/accuracy), `CONTENT_SEARCH_ENABLED`, `QDMS_ACCESS_CONTROL_ENABLED`, `MIN_SCORE_THRESHOLD`, `HNSW_EF_SEARCH`, `MMR_K/MMR_FETCH_K/MMR_LAMBDA`, `FUSION_TEXT_WEIGHT/FUSION_VECTOR_WEIGHT`, `RERANKER_*`, `CHAT/FILTER/EMBEDDING/VECTOR_SEARCH_MAX_CONCURRENT`, `AZURE_HTTP_MAX_CONNECTIONS`, `ANYIO_MAX_THREADS`, `COST_LIMIT`, `SYNC_INTERVAL_MINUTES`, `CHAT_HISTORY_LIMIT`, `MAX_CONTEXT_DOCS`, `OTEL_*`, `USER_ACCESS_API_URL`, `QDMS_FRONTEND_URL`, `QDMS_COMPANY_ID`.

---

## 7. Open questions / couldn't determine from the code

- **Cost limit enforcement**: helpers + table exist but are not wired into chat/search; was 409/429 enforcement removed, planned, or only ever in the README? Token usage goes to OTel only.
- **`customs.py` / `config/client.py`**: referenced in README/CLAUDE doc structure but absent from the source tree — stale docs or files removed.
- **`feedback_for_id`, `suggestions`, `cited_sources` columns** on `ChatHistory` are defined (and added by `scripts/migrate_chathistory.py`) but I found no writer for them in any path — likely future features. NOTE: a *working* feedback mechanism exists via the separate `like`/`dislike` boolean columns and the `LikeMessage`/`DislikeMessage` endpoints (see §8); the three NVARCHAR(MAX) columns are unrelated/unused.
- **`get_mongo_db` (sync source dependency)** is defined but I didn't see it injected by any active endpoint (async `get_async_mongo_db` is used everywhere) — possibly only for scripts/`run_in_threadpool`.
- **Real `qdms_client` deployments**: only the in-repo *template* (`config.py`/`hooks.py`/`routers.py`) is present; actual per-branch client packages (referenced in `wiki/client_customization.md`) live outside this checkout, so their concrete prompt/policy overrides can't be verified here.
- **Reranker model availability offline**: `load_reranker` depends on the ONNX model being pre-cached locally (`snapshot_download(local_files_only=True)`); the provisioning step that populates that cache isn't in this repo.
- **Vessel endpoints' role/vessel ACL source**: `VesselSearch/VesselChat` pass `vessel_id`/`role` into `MongoDynamicFilter`, but how those values are authenticated/authorized for a given user (vs. trusting the request body like the office flow) isn't shown.
- **Exact Cosmos index name in prod**: `INDEX_NAME` default is `oceanai_mol_qdms_index_hnsw` in settings but `create_cosmos_indexes.py` defaults the index name to the DB name — which one production actually uses depends on env config not present here.

---

## 8. Addendum — missed details & corrections

This section was produced by an adversarial completeness review of the first pass against the live source. The first pass was largely accurate; below are details it MISSED and a short list of claims re-verified or corrected.

### 8.1 Like/Dislike feedback endpoints (MISSED feature)

**How.** `routers/chat.py` exposes two more endpoints the first pass never documented: `POST /QDMSbot/v1/SmartChat/LikeMessage` and `.../DislikeMessage` (both **sync `def`**, both behind `Depends(get_current_user)`). Each takes a `ReactionInput{response_id}`, looks up the `ChatHistory` row by `id`, and **toggles** the boolean: liking sets `like=True, dislike=False`; clicking an active like un-likes it; like and dislike are mutually exclusive. Returns `ReactionResponse{response_id, like, dislike}`. The `response_id` returned to the client after every chat/search persist (`response_data["response_id"] = chat_history_id`, and the SSE `meta` event) is exactly the `ChatHistory.id` these endpoints mutate.

**What.** SQLAlchemy row toggle, 404 if the row is missing, generic 500 on error.

**Why.** This is the product's thumbs-up/down feedback loop. It is wired through the `ChatHistory.like` / `ChatHistory.dislike` columns — **NOT** the unused `suggestions`/`feedback_for_id`/`cited_sources` columns flagged in §5/§7. So the corpus has *two* feedback storage shapes: one live (like/dislike booleans) and one dormant (the three text columns).

### 8.2 Category map cache with negative caching + thundering-herd guard (MISSED)

**How.** `helpers/search.py::build_category_map()` caches the full `categories` collection (ObjectId→Name) in module-level globals (`_category_cache`, `_category_cache_ts`) with a **300 s TTL**. Two non-obvious tricks: (1) it sets `_category_cache_ts = time.monotonic()` **before** the DB fetch so that concurrent requests on a cold/expired cache don't all stampede the collection scan (explicit "thundering herd on timeout" comment); (2) on fetch failure it **negative-caches** — logs a warning and returns the stale (or empty) map rather than raising, so a transient Mongo blip can't fail every search. `extract_metadata()` consumes this map to turn `CategoryId` ObjectIds into display names across all three collections, deduping by `_id`/`document_id`.

**What.** Module-global TTL cache, `time.monotonic`, negative caching, set-timestamp-before-fetch.

**Why.** The category collection is small and rarely changes; caching it removes a full collection scan from every request's metadata-enrichment step, and the pre-fetch timestamp prevents N concurrent misses from all scanning at once.

### 8.3 `doc_to_dict` deliberately de-async'd (MISSED micro-optimization)

`helpers/search.py::doc_to_dict()` carries an explicit comment that it was made a **plain (non-async) function** to remove "~90 unnecessary event-loop round-trips per request under concurrent load." A small but deliberate hot-path optimization in the same family as the rest of the concurrency tuning.

### 8.4 Idempotent SQL migration script (MISSED)

`scripts/migrate_chathistory.py` is a hand-rolled, **re-runnable** migration that `inspect()`s the `chathistory` table and `ALTER TABLE ... ADD`s the three columns (`suggestions`, `feedback_for_id`, `cited_sources` as `NVARCHAR(MAX)/INT NULL`) only if absent, inside a single `engine.begin()` transaction. There is no Alembic in the project — schema evolution is done with these targeted scripts plus `Base.metadata.create_all` at startup. Other operational scripts present and worth knowing: `generate_search_text.py` (back-fills the `SearchText` field used by `$text`+reranker), `update_user_document_access.py` / `generate_user_access_json.py` (populate the local ACL table / `user_documents.json` fallback), `download_blobs_from_azure.py`, `embed_collection.py`, `create_text_indexes.py`, `create_cosmos_indexes.py`.

### 8.5 Vessel endpoints: ACL source swap, same trust boundary (clarifies §7 open question)

**How.** `routers/vessel.py` (`POST /QDMSbot/v1/VesselSearch/`, `.../VesselChat`) mirror the office SmartSearch/SmartChat but **replace the document-ID access API with `vessel_id` + `role` metadata filtering** on the vector collections (via `MongoDynamicFilter`'s `_universal_or_match` vessel/role clauses, §3.3). They are behind `Depends(get_current_user)` but, like the office flow, **trust the `user_id`/`vessel_id`/`role` in the request body** — the C# proxy is the real authenticator (the shared-service-account trust boundary from §5). So the §7 question "how are vessel_id/role authorized" resolves to: they aren't, by design — same upstream-trust model as the office endpoints.

### 8.6 OTel two-phase init ordering + graceful shutdown (under-documented in §3.16)

**How.** `observability/instrumentation.py` enforces a strict order with documented "pitfalls": `instrument_libraries()` must run **before any app module import** because `config/db.py` builds the SQLAlchemy engine at import time — so the engine is even imported *inside* `instrument_libraries()` (after `SQLAlchemyInstrumentor().instrument()`) to guarantee the event hooks exist first. `PymongoInstrumentor` is registered globally and Motor piggybacks on pymongo, so async Mongo/Cosmos ops are traced without separate Motor instrumentation. On the way down, `main.py`'s lifespan does a **graceful OTel shutdown**: for each of the three providers it calls `force_flush(timeout_millis=5000)` then `shutdown()`, each wrapped in `try/except`, so an unreachable collector can't hang shutdown. All three OTel disable-checks read `OTEL_TELEMETRY_DISABLED` and return NoOp at zero overhead.

**Why.** Provider-first / instrument-before-import is the only ordering that captures the import-time-created engine; bounded flush timeouts make shutdown robust when Alloy is down.

### 8.7 Re-verified claims (first pass was CORRECT)

- **Cost tracking is dead code** — confirmed: `is_monthly_limit_reached` / `update_monthly_cost` (`helpers/cost_tracking.py`) and the `CostTracking` model are never imported by any router or agent (grep-verified). Both agents `return ..., 0.0` (agents.py:1678, :1854) and routers persist `cost=0.0` (chat.py:388, search.py:407). No `OpenAICallbackHandler`/`get_openai_callback` anywhere.
- **Fusion 0.4/0.6, not 0.3/0.7** — confirmed `fusion_text_weight=0.4`, `fusion_vector_weight=0.6` (settings.py:135/139).
- **`SEARCH_MODE` defaults to `latency`** (settings.py:85) and `RERANKER_ENABLED = settings.search_mode == "accuracy"` (agents.py:82) — reranker OFF by default. Confirmed.
- **`customs.py` / `config/client.py` absent** — confirmed missing from the tree.
- **`min_score_threshold=0.5`** (settings.py:108), **`hnsw_ef_search=40`** (:112), **`reranker_threshold=0.0`** (:155) — all confirmed.
- **Semaphore defaults** chat=32 / filter=32 / embedding=64 / vector=30 — confirmed (settings.py:116–131, agents.py:94–96,281).
- **Committed secrets + ~49-year token** — confirmed: `.env` holds real-looking `SECRET_KEY`, `AZURE_OPENAI_API_KEY`, `MONGO_VECTOR_DB` (conn string), `AZURE_STORAGE_SAS_URL`, and `ACCESS_TOKEN_EXPIRE_MINUTES=25920000`; the expiry **is** actually applied (`routers/auth.py:29` reads `access_token_expire_minutes` and passes it to `create_access_token`), so tokens really are minted with ~49-year lifetime. Contradicts `.claude/rules/security.md`.
- **Auth user-cache detach** (`db.expunge`+`make_transient` after touching `user.role`, 60 s TTL, size-bounded eviction) and **`/token` sync-`def`-for-bcrypt-in-threadpool** — both confirmed verbatim in `helpers/auth.py` / `routers/auth.py`.

### 8.8 Minor corrections / nuances

- **`$regex` is still advertised to the LLM.** The `Filter` Pydantic operator description (`ai/output_parsers.py:7`) lists `$regex` as an allowed operator; the §3.3 "`$regex` explicitly forbidden for dates" is right only for the *date* path (`filters.py` blocks regex on date fields and uses `$in` for vessel/role) — the model can still emit `$regex` on non-date fields. Precise statement: `$regex` is permitted in general but deliberately avoided for dates and for vessel/role ACL matching (the measured 30–50 % perf win).
- **Semaphore comment vs. value drift inside agents.py.** The `_vector_semaphore` inline comment says "Default 12 allows 2 full requests," but the actual `settings.vector_search_max_concurrent` default is **30** (settings.py:118, "5 full requests"). The code uses the setting (30); the "12" in the comment is stale. Trust 30.
- **`min_content_chars` is configurable (200 default)** and `max_context_docs=15` — §3.7's hard-coded-looking constants are actually settings fields, so the adaptive-context floor/min/max are env-tunable.
