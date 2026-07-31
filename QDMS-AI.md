---
aliases: [QDMS-AI]
tags: [project/qdms-ai, rag, fastapi, azure-openai, cosmos-db, vector-search, reranker, maritime]
type: project-note
status: reference
---

# QDMS-AI

> [!abstract] In one line
> The RAG brain behind PAL Wiki: three specialist agents (filter, search, chat) plus dual vector+keyword retrieval and a cross-encoder reranker over a maritime document corpus.

Related: [[OceanAI Projects MOC]] · [[EagleAI Platform]], [[PAL Wiki]], [[BSM SOA Agent]], [[Kriya QA Platform]], [[Decarbonization Agent]]

PAL Wiki's chatbot calls [[QDMS-AI]] for every AI search and chat request; QDMS-AI sits behind a C# wiki/reverse proxy and runs entirely on Azure OpenAI + Azure Cosmos DB.

**Source root:** `D:\QDMS\MariApps.ocean.ai.QDMS` · Package `qdms_core` (+ optional `qdms_client` plugin) · Python `>3.10,<3.11`, Poetry, FastAPI/uvicorn.

---

## Architecture at a glance

```mermaid
flowchart TD
    Proxy["C# QDMS Wiki / reverse proxy<br/>(shared service-account JWT)"] -->|"root_path = /qdms or /qdms-{client}"| App

    subgraph App["FastAPI app (qdms_core/main.py)"]
        MW["Middleware (pure ASGI):<br/>CORS → GZip(skips text/event-stream) → RequestTiming"]
        Routers["Routers: auth · users · search · chat · scripts · vessel<br/>+ client routers (registered FIRST, override on dup paths)<br/>/metrics · / (version)"]
    end

    App --> Greet{"is_greeting()?<br/>≤4 tokens"}
    Greet -->|yes| Chat
    Greet -->|no| Filter

    Filter["QDMSFilterAgent (gpt-4.1-mini)<br/>filters[] · follow-up · remaining/alternative query"]
    ACL["AccessCache + user_access_api<br/>(runs in PARALLEL on cache MISS)"]
    Filter -.asyncio.gather.- ACL

    Filter --> Search

    subgraph Search["QDMSSearchAgent (gpt-4.1) — parallel asyncio.gather"]
        Text["$text content search<br/>(source MongoDB, Motor)<br/>+AND forcing, textScore floor, 5s timeout"]
        Vec["vector MMR search (Cosmos vCore)<br/>simsimd cosine, HNSW efSearch=40"]
    end

    Text --> Fuse["Normalized Score Fusion<br/>text 0.4 / vector 0.6"]
    Vec --> Fuse
    Fuse --> Rerank["optional Cross-Encoder rerank<br/>(ONNX, accuracy mode only)"]
    Rerank --> Crop["get_cropped_context()<br/>adaptive, revision-aware dedup"]
    Crop --> Chat["QDMSChatAgent (gpt-4.1-mini)<br/>ainvoke (JSON) or astream (SSE)"]
    Chat --> Out["inline citations → mistune → HTML<br/>persist ChatHistory (SQL) → return cards/sources/answer"]

    Text --> SourceDB[("MongoDB SOURCE DB<br/>oceanai_qdms_data_*<br/>Articles/docs/files, $text indexes")]
    Vec --> CosmosDB[("Cosmos DB vCore VECTOR DB<br/>oceanai_qdms_mol_*<br/>chunk+vector, HNSW index")]
    Out --> SQL[("SQL: SQLite / pymssql / pyodbc<br/>users, roles, chathistory, costtracking")]
```

**Walk-through.** A request enters through the C# proxy carrying a shared service-account JWT and is mounted at a dynamic `root_path` (`/qdms` for core, `/qdms-{client}` for a branded client). Pure-ASGI middleware handles CORS, gzip (but never for SSE), and timing. If the query is a greeting it short-circuits straight to the chat agent. Otherwise the **FilterAgent** (`gpt-4.1-mini`) turns natural language into MongoDB filters + a refined search phrase, running in parallel with the per-user ACL fetch. The **SearchAgent** (`gpt-4.1`) then fans out into **dual retrieval** — `$text` keyword search on the source Mongo DB and vector MMR search on the Cosmos vector DB — fuses them with normalized-score fusion (0.4 text / 0.6 vector), optionally reranks with a cross-encoder, and adaptively crops context. The **ChatAgent** (`gpt-4.1-mini`) synthesizes a cited, word-capped answer, either as non-streaming JSON or an SSE token stream. Everything is instrumented with OpenTelemetry (traces → Tempo, metrics → Prometheus `/metrics`, logs → Alloy→Loki).

---

## How it works, feature by feature

### Three specialist agents as import-time singletons

All three agents are built **once at module import** in `qdms_core/config/deps.py`: `qdms_filter_agent`, then `qdms_search_agent = QDMSSearchAgent(filter_agent=qdms_filter_agent)` (the filter is cross-injected into search and called inside `QDMSSearchAgent.invoke()`), then `qdms_chat_agent`. Routers just import these names. Each wraps its own `AzureChatOpenAI` (LangChain LCEL `prompt | llm | parser`) with a distinct deployment.

- **What:** module-level singletons, not FastAPI `Depends`.
- **Why:** the docstring says it plainly — *"avoids duplicate LLM clients, embedding clients, and vector DB connections."* Building once shares a single httpx pool, single Cosmos connection, preloaded prompts. The **cheap/strong split** is deliberate: `gpt-4.1-mini` for the high-frequency filter + chat steps, the stronger `gpt-4.1` only for search.

### QDMSFilterAgent — filters, follow-ups, greetings, prompt caching

`QDMSFilterAgent` (`qdms_core/ai/agents.py` ~line 1759) splits its prompt into `DYNAMIC_FILTER_SYSTEM` (static rules) + `DYNAMIC_FILTER_HUMAN` (per-request data) in `qdms_core/ai/prompts.py`. Output is parsed with `JsonOutputParser(pydantic_object=DynamicFilter)` exposing `is_search_required`, `is_follow_up`, `remaining_query`, `alternative_query`, `n`, `filters[]`. `ainvoke()` is split into **three timed stages** so only the LLM call holds `_filter_semaphore` (rendering and parsing run outside the concurrency budget). It also reads and logs `cached_tokens` from `response_metadata`.

- **Follow-up detection** (prompt rule 3): catches numerical refs ("step 2"), ordinals, demonstratives ("that/this/it"), sequential words ("explain/elaborate"), process refs. If the answer is already in chat history it sets `is_search_required=false, n=0` to skip retrieval.
- **Greeting short-circuit** is separate and faster: `qdms_core/helpers/greeting.py::is_greeting()` is a pure keyword gate (≤4 tokens, normalized match against `_GREETING_WORDS`) called by routers **before** the filter agent.
- **remaining_query / alternative_query** force official maritime/IMO terminology, expand acronyms (`"ism" → "ISM code safety management system"`), and require the alternative query to use *genuinely different vocabulary* (used later for dual-query RRF).
- **Why the prompt split:** Azure caches identical SystemMessage prefixes ≥1024 tokens, so the big rulebook lives in `_SYSTEM` (cacheable) and per-request data in `_HUMAN`.

### DynamicFilter operators + date expansion + ACL/vessel/role merge

`qdms_core/ai/filters.py`. `MongoDynamicFilter._combine_filters()` folds the LLM `filters[]` into `{field: {op: value}}`. Any field whose name contains "date" is parsed with `datetime.strptime(val, "%Y-%m-%d")`; a `$eq` on a date is **expanded to a full-day range** `{"$gte": dt, "$lt": dt + timedelta(days=1)}` (a bare `$eq` on a timestamp would never match). Operators: `$eq, $ne, $lt, $lte, $gt, $gte, $in, $nin`. `dynamic_filter(filter_type)` maps LLM field names to per-collection Mongo paths via `DEFAULT_FIELD_MAPPING` (e.g. `date` → `LastUpdatedTime` for files but `Date` for docs/articles), then ANDs the ACL (`metadata.document_id: {$in: allowed_ids}`), a vessel filter, and a role filter. Vessel/role use `_universal_or_match()` building an `$or` over exists/None/""/[]/all/value-variants — with **`$in` exact matches instead of `$regex`**, an explicit 30–50% perf win during HNSW traversal (the filter is a `pre_filter` evaluated per-candidate inside the vector scan).

### Dual retrieval: vector MMR (simsimd) + `$text`, fused 0.4/0.6

`QDMSSearchAgent.invoke()` runs both arms in one `asyncio.gather`:

- **Vector arm** `_mmr_search()`: embeds the query, issues a native async Motor `$search.cosmosSearch` aggregate (`path="vectorContent"`, `k=effective_fetch_k`, `efSearch=40`, `min_score_threshold=0.5`). In **accuracy mode** it runs `_fast_mmr()` — a hand-written MMR that precomputes the full similarity matrix **once** with `simsimd.cdist(metric="cosine")` (numpy fallback), replacing LangChain's `maximal_marginal_relevance` which recomputes every loop. In **latency mode** MMR is skipped (Cosmos already ranked), saving ~1.3 MB transfer + numpy work.
- **Content arm** `_content_search()`: a Mongo `$text` query on the *source* collection. Significant words get a `+` prefix to force **AND semantics** (without it "ISM Code" matches "Dress code" via OR on "Code"); a stopword set is excluded from forcing. Results floored by `textScore >= max(len(significant)*1.5, 2.0)`. Uses `find()` not `aggregate()` so the optimizer pushes sort+limit into the text index scan (2–3× faster), with `max_time_ms` + `asyncio.wait_for(timeout=5.0)` as a hard deadline.
- **Fusion** `_score_fuse()`: min-max normalize each list to [0,1], combine `fusion_text_weight=0.4` + `fusion_vector_weight=0.6`, keyed by a stable business id (`ArticleNumber`/`DocumentNumber`/`FileID`) so matches survive reimports. Degenerate all-equal lists fall back to rank-based scoring. `_dedup_by_key()` keeps the best chunk per key; when an `alternative_query` exists, `_merge_vector_results()` fuses primary+alt with **Reciprocal Rank Fusion (`rrf_k=60`)**.

> [!warning] Fusion weights: trust the code, not the README
> The README claims fusion is "0.3/0.7". The live defaults in `settings.py` are **0.4 text / 0.6 vector** (`fusion_text_weight=0.4`, `fusion_vector_weight=0.6`).

### Cross-encoder ONNX reranker, loaded at startup

`QDMSSearchAgent.load_reranker()` loads `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder` with `backend="onnx"`, `file_name="onnx/model_quint8_avx2.onnx"` (quantized AVX2 ONNX), resolved through `huggingface_hub.snapshot_download(local_files_only=True)` so it never hits the HF API. It is **preloaded at startup**: `deps.warm_connections()` calls `await run_in_threadpool(qdms_search_agent.load_reranker)` (only in `SEARCH_MODE=accuracy`). Inference runs in a **dedicated `ThreadPoolExecutor`** (`thread_name_prefix="reranker"`, sized `cpu_count//2`) with `OMP_NUM_THREADS=1`, isolated from the shared anyio pool. `_rerank_batch()` flattens all collections into one `predict()` call (title prepended to content) and filters by `reranker_threshold=0.0`.

- **Why:** model load is 10–15 s, so paying it once at startup keeps the first query fast. `OMP_NUM_THREADS=1` + private executor prevents thread-in-thread contention that caused 0.8–5.6 s latency variance. `HF_HUB_OFFLINE=1` (forced in `main.py` before any HF import) avoids 429 rate limits.

### Per-collection embedders + differing chunk sizes (1536 vs 6144)

`QDMSSearchAgent.__init__` builds three `QDMSEmbedder` (`qdms_core/ai/embedder.py`): `Articles` and `docs` at chunk **1536**, and `files` at chunk **6144**. Each owns its own `AzureOpenAIEmbeddings(model=text-embedding-3-small)` on the dedicated embed httpx client and its own `AzureCosmosDBVectorSearch` store. Chunking happens in `QDMSMongoLoader` via `RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=..., chunk_overlap=100)`.

- **Why:** Articles/docs are short structured HTML (precise small chunks); uploaded files (PDFs/manuals) are long and need bigger 6144-token chunks to preserve context.

### Adaptive context assembly (quality-not-count, revision-aware)

`qdms_core/helpers/chat.py::get_cropped_context()` pools all reranked results, sorts by `similarity_score`, and computes a **dynamic score floor** = `max(top_score - _SCORE_GAP, _SCORE_FLOOR)` that auto-calibrates by mode (`accuracy`: CE-logit gap 8.0, floor 0.0; `latency`: NSF gap 0.8, floor 0.1). It enforces `_MIN_CONTEXT=3` and `_MAX_CONTEXT=max_context_docs(15)`, drops chunks under `MIN_CONTENT_CHARS(200)`, dedups by business key, then **revision-aware dedups** by `_normalize_title()` (`_TITLE_NOISE_RE` strips version/rev numbers, dates, month names, possessives, parenthesized numbers — so `"BS FMD Reporting Procedures _May-2021"` collapses with `"BS FMD Reporting Procedures"`).

### ChatAgent: cited answers, word cap, JSON + SSE

`QDMSChatAgent` builds `ChatPromptTemplate` from `CHAT_SYSTEM`/`CHAT_HUMAN` with `{max_response_words}` from `settings.chat_max_response_words` (**300**). `format_context()` emits `article_1/doc_1/file_1` blocks with `Title:`/`URL:` (via `_build_citation_href`) and strips a `_METADATA_HEADER_RE` so the LLM doesn't confuse "Doc 06" with citation indices. The system prompt demands **inline markdown links `[Title](URL)`** after each sentence (never clustered, never a Sources section). Two entry points: `ainvoke()` (records token usage + duration) and `astream()` (records TTFT). Post-processing: `normalize_citations()` repairs LLM deviations (`(doc_1)`→`[doc_1]`, casing, strips appended "Sources:" sections), `convert_markdown_to_html()` (mistune + table plugin, `_unescape_llm_html_entities`), then `process_citations()` swaps `[doc_N]` for `<a class="citation-link">` tags.

### SSE streaming endpoint (`/SmartChat/stream`, `/SmartSearch/stream`)

`qdms_core/routers/chat.py::chat_stream` returns a `StreamingResponse(media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})`. The async `event_stream()` emits a typed SSE protocol: `status`, `cards`, `sources`, `token`, `meta` (`response_id`), `error`, `done`; it opens with `retry: 3000`. Two robustness guards: a **table-row buffer** holds partial `| …` lines until `\n` so renderers never see a broken row, and a **whitespace-runaway abort** after 200 consecutive whitespace chars (LLM padding-loop failure mode) with a truncation notice.

### Like/Dislike feedback endpoints

`routers/chat.py` exposes `POST /QDMSbot/v1/SmartChat/LikeMessage` and `.../DislikeMessage` (both **sync `def`**, behind `Depends(get_current_user)`). Each takes `ReactionInput{response_id}`, looks up the `ChatHistory` row by `id`, and **toggles** mutually-exclusive booleans (`like`/`dislike`). The `response_id` returned after every chat/search (and in the SSE `meta` event) **is** the `ChatHistory.id` these endpoints mutate. This is the live feedback loop — distinct from the dormant text columns (see Watch out).

### Multi-environment SQL engine factory

`qdms_core/config/db.py::get_database_engine()` switches on `settings.env`: `local` → SQLite (`check_same_thread=False`); `develop`/`qa` → `mssql+pymssql://...` (SQL auth, password URL-quoted); `uat`/`prod` → `mssql+pyodbc:///?odbc_connect=...` with `Authentication=ActiveDirectoryIntegrated;Encrypt=yes;TrustServerCertificate=yes` and **no username/password**. Non-SQLite engines share `pool_size=100, max_overflow=100, pool_timeout=30, pool_recycle=3600, pool_pre_ping=True`. Prod uses Azure AD integrated auth so **no DB password is stored anywhere**.

### Dual Mongo topology: source DB + Cosmos vector DB, each sync + async

`qdms_core/config/mongodb.py` builds **four** clients sharing `_CONN_PARAMS` (5 s connect, 30 s socket, `maxPoolSize=500`, `appName="qdms-ai"`): `sync_client`/`async_client` for the source DB, and `vector_sync_client`/`async_vector_client` for the Cosmos vector DB. Each store needs a *sync* client (LangChain's `AzureCosmosDBVectorSearch` requires sync pymongo) **and** an *async* Motor client (non-blocking direct queries in `async def` endpoints). Sharing one vector sync client across the three embedders cut monitor threads 3→1.

### Two httpx pools (LLM vs embedding)

`qdms_core/config/httpx_client.py` splits the budget (`azure_http_max_connections=500`): **60% → `azure_http_client`** (chat+filter LLM), **40% → `azure_embed_http_client`** (embeddings). Both use HTTP/2, 120 s keepalive (vs OpenAI SDK default 5 s), and a shared hook `_log_azure_backend_health()` warning on `x-envoy-upstream-service-time > 5 s` and TPM remaining `< 5%`. Separate TCP pools stop a flood of LLM calls from starving bursty embedding load (the comment estimates "+40 s queue time" on a shared pool).

### GZip + RequestTiming middleware (pure ASGI)

`qdms_core/middleware.py`. Both are **pure ASGI** (`async def __call__(scope, receive, send)`), not `BaseHTTPMiddleware`. `GZipCompressMiddleware` buffers non-streaming bodies and gzips (level 1) anything ≥ `minimum_size`(1000) when the client accepts gzip — but in `http.response.start` it checks `content-type` and if it sees **`text/event-stream`** it passes every chunk through uncompressed. `RequestTimingMiddleware` logs `[REQUEST] METHOD path -> status | duration`, skipping noisy prefixes (`/docs`, `/health`, `/metrics`). Raw ASGI avoids `BaseHTTPMiddleware`'s body-buffering anyio channel + extra per-request Task that delays lightweight requests under 20+ heavy concurrent workflows. (uvicorn access log disabled via `access_log=False`.)

### Access control: external API or local table, env-gated fail-open

`QDMSSearchAgent.invoke()` resolves allowed document IDs via `qdms_core/helpers/access_cache.py` (`AccessCache`, per-user TTL 300 s, `time.monotonic` expiry, per-user `asyncio.Lock` anti-stampede). On a **miss** it launches `get_all_items()` (`helpers/user_access_api.py`) and the filter agent **concurrently** via `asyncio.gather`. `get_all_items()` POSTs to `USER_ACCESS_API_URL` (or reads `user_documents.json` when the URL is literally `"local"`), mapping `ItemType` `ART/DOC/FLS` → `articles/documents/files`. On failure: **non-prod** (`local/develop/qa`) fails **open** (`allowed_items = {all: None}`); **prod/uat** fails **closed** to empty lists. Skipped entirely when `qdms_access_control_enabled=false`.

### Client plugin system + dynamic root path

`qdms_core/plugins/loader.py` reads `QDMS_CLIENT`. On `core` everything returns safe defaults; on a non-core value it `importlib.import_module("qdms_client.<module>")` and **raises** on ImportError (deployment error, not silent fallback). Hooks (`SearchHook`/`ChatHook` Protocols) are validated via `runtime_checkable isinstance`. `main.py::_load_version()` derives `root_path = "/qdms"` for core or `"/qdms-{client}"` otherwise. **Client routers are registered before core routers** so they win on duplicate paths (FastAPI first-registered-wins). Prompt overrides validated by `plugins/prompts.py::load_validated_prompt()` (checks required vars, falls back to core).

### Observability: OTel traces/metrics/logs

`qdms_core/observability/telemetry.py::setup_telemetry()` registers a `TracerProvider` (→ OTLP `/v1/traces`), a `MeterProvider` with **both** `PrometheusMetricReader` (feeds `/metrics`) and a `PeriodicExportingMetricReader` (→ OTLP `/v1/metrics`), and a `LoggerProvider` (→ OTLP `/v1/logs`). Custom metrics in `agents.py`: `gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, `qdms.pipeline.stage.duration`, `qdms.pipeline.stage.errors`. `LoggingInstrumentor` injects `otelTraceID`/`otelSpanID` into every log line. **Two-phase init ordering matters:** `instrument_libraries()` must run before any app module import (because `config/db.py` builds the SQLAlchemy engine at import time — the engine is even imported *inside* `instrument_libraries()` after `SQLAlchemyInstrumentor().instrument()`). `PymongoInstrumentor` covers Motor (it piggybacks on pymongo). Shutdown does `force_flush(timeout_millis=5000)` then `shutdown()` per provider, each in try/except. Single kill-switch: `OTEL_TELEMETRY_DISABLED` → NoOp at zero overhead.

### Cosmos HNSW + metadata filter + `$text` indexes

Two index paths: (a) `QDMSEmbedder.__init__` calls `vector_store.create_index(num_lists=1, dimensions=1536, similarity=COS, kind=VECTOR_HNSW, m=16, ef_construction=200)` (best-effort). (b) `scripts/create_cosmos_indexes.py` issues a **raw** `db.command({"createIndexes": ..., "cosmosSearchOptions":{...}})` — raw because pymongo's `create_index()` strips the unknown `cosmosSearchOptions`. It then samples 100 docs to discover `metadata.*` keys and creates a filter index per key. `scripts/create_text_indexes.py` builds weighted `$text` indexes (`metadata.<Name>`:10, `textContent`:1). The script header notes `$text` is "5x faster and finds 15x more results" than `$regex`.

### Background data sync + embedding pipeline

`main.py` registers an APScheduler `BackgroundScheduler` interval job (`max_instances=1`, `next_run_time=None`) and arms the first run only *after* warmup via `reschedule_job`. `run_data_sync()` pulls SAS-URL zips (`download_blobs_from_azure.main()`), extracts `art/docs/forms/fschunks/categories.json`, bulk-upserts into source Mongo with `embedding_status="pending"`, then `load_and_embed_documents()` per collection. Embedding shuffles pending docs (`random.shuffle`), deletes prior chunks, extracts content (HTML via `Html2TextTransformer`, files via `FileLoader`), writes a `SearchText` field (title + first 2 KB) used by `$text` + reranker, splits, and upserts vectors. Status-tracked (`pending/completed/failed`) so only changed docs re-process.

### Category map cache (negative caching + herd guard)

`helpers/search.py::build_category_map()` caches the full `categories` collection (ObjectId→Name) in module-level globals with a **300 s TTL**. Two tricks: (1) it sets `_category_cache_ts` **before** the DB fetch so concurrent cold-cache requests don't stampede; (2) on fetch failure it **negative-caches** — returns the stale/empty map rather than raising, so a transient Mongo blip can't fail every search.

### Versatile FileLoader

`qdms_core/ai/files_loader.py::FileLoader.load()` dispatches by extension with **fallback chains**: PDF → `PyPDFLoader` then `PDFPlumberLoader`; legacy `.doc` → bundled **antiword.exe** subprocess; Excel → `UnstructuredExcelLoader` → pandas (openpyxl/xlrd/odf/pyxlsb) → HTML-disguised-as-Excel detection; `.ppt/.pptx` → python-pptx (replaced Unstructured which "hangs on some files"); images → `ImageCaptionLoader`; video → 3 OpenCV frames + caption; zip → recursive. Maritime corpora are messy (mislabeled extensions, HTML saved as `.xls`, password PDFs), so layered fallbacks keep one bad file from failing the batch.

---

## What makes it unusual

> [!tip] Three model-differentiated agents, cross-injected as singletons
> Filter (`gpt-4.1-mini`) is injected into Search (`gpt-4.1`), with Chat (`gpt-4.1-mini`) separate — all built once at import to share clients. The deliberate cheap/strong split plus shared single httpx/Cosmos connections is rare; most projects create LLM clients per request.

> [!tip] Prompt-cache-aware system/human split
> The rulebook lives in `*_SYSTEM` specifically to hit Azure's ≥1024-token prefix cache, and the filter agent logs the returned `cached_tokens` to *verify* the cache is actually working.

> [!tip] `_fast_mmr` with a precomputed simsimd cosine matrix
> Replaces LangChain's per-iteration similarity recompute with one `simsimd.cdist` matrix (numpy fallback), plus a mode switch that skips MMR entirely in latency mode. A real algorithmic optimization, not just config.

> [!tip] Dual retrieval across two physically separate databases
> `$text` keyword search lives on the **source** Mongo DB; vector search lives on the **Cosmos** vector DB; results are fused with weighted NSF and (for alt queries) RRF. Most hybrid-search systems keep both indexes co-located — splitting them is unusual.

> [!tip] Cross-encoder reranker preloaded on an isolated thread pool
> Quantized ONNX model, resolved via an offline HF snapshot, preloaded at startup through `run_in_threadpool` inside `warm_connections()`, and run on a dedicated `ThreadPoolExecutor` with `OMP_NUM_THREADS=1`. The thread-isolation reasoning (avoiding thread-in-thread contention) is a non-obvious, measured fix.

> [!tip] `+`-forced AND `$text` semantics with a stopword carve-out
> A dynamic `textScore` floor `max(len(significant)*1.5, 2.0)` plus `+`-prefixing of significant words solves the "ISM Code matches Dress code" OR-bug that plain `$text` produces.

> [!tip] Per-collection embedders with different chunk sizes
> 1536 tokens for short structured Articles/docs, a separate larger **6144** just for long uploaded files.

> [!tip] One SQL factory, three auth strategies
> Prod uses `ActiveDirectoryIntegrated` with **no password** (pyodbc ODBC 18), dev/qa use pymssql SQL-auth, local uses SQLite — selected purely by `settings.env`.

> [!tip] Dual Mongo topology, four clients
> sync+async × source+vector, each justified: LangChain needs sync pymongo, async endpoints need Motor.

> [!tip] Two httpx pools (60/40 LLM vs embed)
> HTTP/2 with an Azure-header health hook warning on backend slowness and TPM exhaustion — so bursty embedding load never starves LLM calls (or vice versa).

> [!tip] Pure-ASGI gzip that skips SSE
> A raw-ASGI compression middleware that explicitly passes `text/event-stream` through uncompressed, plus a pure-ASGI timing middleware replacing uvicorn access logs — both chosen to dodge `BaseHTTPMiddleware`'s task/buffer overhead under concurrency.

> [!tip] SSE stream hardening
> A markdown-table-row buffer and a whitespace-runaway abort — two production guards most streaming endpoints lack.

> [!tip] Revision-aware title normalization
> `_TITLE_NOISE_RE` / `_normalize_title` collapse dated/versioned revisions of the same manual so token budget isn't wasted on near-duplicates.

> [!tip] Per-operation concurrency semaphores
> Module-level globals in `agents.py` (`_chat_semaphore=32`, `_filter_semaphore=32`, `_embedding_semaphore=64`) plus a per-agent `_vector_semaphore=30`, shared across all requests, so fast filter calls don't queue behind slow chat calls. The anyio thread limiter is raised 40→200.

> [!tip] Client plugin system with fail-closed imports
> Dynamic `/qdms-{client}` root path, validated prompt overrides, and a hard raise (not silent fallback) when a client module fails to import.

---

## Watch out / easy to miss

> [!bug] Cost tracking is effectively dead code
> `helpers/cost_tracking.py` (`is_monthly_limit_reached`, `update_monthly_cost`) and the `CostTracking` table exist, and the README advertises 409/429 enforcement — but **nothing in the live request path calls them** (grep-verified). Both agents `return response.content, 0.0` (agents.py:1678, :1854); routers persist `cost=0.0` (chat.py:388, search.py:407). No `OpenAICallbackHandler`/`get_openai_callback` anywhere. Token usage *is* captured, but only into OTel metrics, not the cost table. The atomic `update(...).values(total_cost=... + cost)` lost-update pattern in `update_monthly_cost` is a good pattern but unreachable.

> [!bug] Secrets committed to `.env` + ~49-year token lifetime
> `D:\QDMS\MariApps.ocean.ai.QDMS\.env` checks in a real-looking `SECRET_KEY`, `AZURE_OPENAI_API_KEY`, a Cosmos connection string (with password), and a SAS URL, plus `ACCESS_TOKEN_EXPIRE_MINUTES=25920000` (~49 years). The expiry **is actually applied** (`routers/auth.py:29` passes it to `create_access_token`), so tokens really are minted with that lifetime. Directly contradicts `.claude/rules/security.md`.

> [!warning] README ↔ code drift
> Fusion is 0.4/0.6 in code (README says 0.3/0.7). Default `SEARCH_MODE=latency`, so the reranker is **OFF by default** (`RERANKER_ENABLED = settings.search_mode == "accuracy"`). README references `qdms_core/ai/customs.py` and `config/client.py` — **neither exists**. Default `min_score_threshold=0.5` (README/older `.env` say 0.2). README's `/qdms-imcsm` root is just an example; actual root derives from `QDMS_CLIENT`.

> [!warning] Per-user auth is trusted, not enforced (by design)
> `ChatHistory`/`chat/sessions` and the vessel endpoints trust the `user_id`/`vessel_id`/`role` in the request body — the C# proxy uses a shared service-account JWT and authenticates the end user upstream. So the §7 "how is vessel ACL authorized?" question resolves to: it isn't, by design — same upstream-trust boundary as the office flow.

> [!warning] Two feedback storage shapes — one live, one dormant
> The **live** feedback loop is the `ChatHistory.like`/`dislike` booleans driven by `LikeMessage`/`DislikeMessage`. The `suggestions` / `feedback_for_id` / `cited_sources` NVARCHAR(MAX) columns (added by `scripts/migrate_chathistory.py`) have **no writer in any path** — likely future features, unrelated to like/dislike.

> [!warning] User cache detaches ORM objects
> `helpers/auth.py::get_current_user` eagerly touches `user.role`, then `db.expunge` + `make_transient` before caching — otherwise reading `user.role` from a cached, session-detached object raises `DetachedInstanceError`. Cache key is `username`, TTL 60 s.

> [!warning] `/token` is a sync `def` on purpose
> So FastAPI runs bcrypt's ~300 ms hash in a threadpool, keeping it off the event loop.

> [!warning] `_combine_filters` raises on a non-ISO date
> `strptime("%Y-%m-%d")` is unguarded — the prompt is the only thing forcing `YYYY-MM-DD`.

> [!warning] Filter index discovery samples only the first 100 docs
> `create_cosmos_indexes.py` samples 100 docs to find `metadata.*` keys, so a metadata field absent from the first 100 gets no filter index.

> [!warning] `DEFAULT_FIELD_MAPPING` mismatch for files
> `date` maps to `LastUpdatedTime` on files but `Date` on articles/docs; `reviewed_date` has **no** `files` entry, so a reviewed-date filter silently doesn't apply to files.

> [!warning] `$regex` is still advertised to the LLM for non-date fields
> The `Filter` Pydantic operator description (`ai/output_parsers.py:7`) lists `$regex` as allowed. It is deliberately avoided only for *dates* and for vessel/role ACL matching (the measured 30–50% win) — but the model can still emit `$regex` on non-date fields.

> [!warning] Stale semaphore comment inside agents.py
> The `_vector_semaphore` inline comment says "Default 12 allows 2 full requests," but `settings.vector_search_max_concurrent` defaults to **30** (the code uses the setting). Trust 30.

> [!warning] Display vs context split
> Routers return *pre-reranker* `display_*` lists for UI cards but feed the *reranked* lists to the LLM context — so the cards show a broader set than what the answer actually cited.

> [!warning] SSE proxy headers + threadpool tuning are load-bearing
> `X-Accel-Buffering: no` + `Cache-Control: no-cache` defeat nginx buffering (gzip-skip alone wouldn't guarantee real-time delivery behind the proxy). The anyio thread limiter is raised 40→200 in `lifespan` because the many `run_in_threadpool` SQL/sync-Mongo calls would bottleneck at the default 40. `HF_HUB_OFFLINE=1` must be set before any HF import or the reranker load 429s.

> [!warning] `doc_to_dict` deliberately de-async'd
> `helpers/search.py::doc_to_dict()` was made a plain (non-async) function to remove "~90 unnecessary event-loop round-trips per request under concurrent load."

---

## Tech used (and why)

| Tool | Role / why it's here |
|---|---|
| FastAPI (≥0.85.1) | ASGI web framework; all routers/endpoints |
| uvicorn (^0.30.6) | ASGI server (`run_qdms.py`, keep-alive 669 s, `access_log=False`) |
| Starlette (≥0.27) | `StreamingResponse` (SSE), `run_in_threadpool` |
| Pydantic / pydantic-settings (^2.0) | request/response schemas + `Settings` env binding |
| SQLAlchemy (^2.0.34) | ORM + multi-env engine factory, pool metrics |
| pymssql (^2.2.11) | SQL Server SQL-auth driver (develop/qa) |
| pyodbc (ODBC Driver 18) | SQL Server ActiveDirectoryIntegrated, no password (uat/prod) |
| pymongo (^4.8) | sync Mongo (source + Cosmos vector via LangChain) |
| motor (^3.5.1) | async Mongo (source + vector direct queries) |
| langchain / -core / -openai / -community / -text-splitters (0.2.x) | `AzureChatOpenAI`, `AzureOpenAIEmbeddings`, `AzureCosmosDBVectorSearch`, loaders, splitter |
| Azure OpenAI — gpt-4.1, gpt-4.1-mini, text-embedding-3-small (API `2025-01-01-preview`) | filter/search/chat LLMs + embeddings |
| Azure Cosmos DB vCore (Mongo API) | vector store, HNSW (m=16, efConstruction=200, cosine, 1536-dim) |
| simsimd (≥5.0) | SIMD cosine for `_fast_mmr` (numpy fallback) |
| sentence-transformers (≥2.2) | CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| onnxruntime / optimum (1.19.2 / ≥2.0) | quantized ONNX reranker backend |
| torch CPU (≥2.0,<2.6) | sentence-transformers dependency |
| huggingface_hub | offline model snapshot resolution |
| httpx (≥0.24) | two Azure pools (HTTP/2 60/40), access-API client |
| mistune (^3.1.3) | markdown→HTML (table plugin) for answers |
| html2text (^2024.2.26) | HTML→text for article/doc embedding |
| pypdf / pdfminer-six / opencv / Pillow / openpyxl / xlrd / docx2txt / python-pptx / antiword.exe | multi-format file extraction |
| APScheduler (^3.10.4) | background data-sync scheduler |
| azure-storage-blob (^12.24.0) | blob/SAS sync of source zips |
| python-jose / passlib[bcrypt] / bcrypt (^3.3.0 / ^1.7.4 / ^4.2.0) | JWT + password hashing |
| OpenTelemetry SDK + OTLP/HTTP + Prometheus exporter + instrumentations (1.40.0 / 0.61b0) | traces/metrics/logs; auto-instr FastAPI/SQLAlchemy/httpx/pymongo/system |
| prometheus_client | `/metrics` endpoint |
| Grafana Alloy / Tempo / Loki / Prometheus (infra) | OTLP collector → trace/log/metric backends |
| tiktoken (^0.7.0) | token-aware chunk splitting |
| python-dotenv (^1.0.1) | `.env` loading |
| tomli (^2.4.0) | read `pyproject.toml` version at startup |
| Poetry / Ruff / mypy / deptry / pytest | build, lint, types, dep-check, tests |

**Key env vars:** `ENV`, `QDMS_CLIENT`, `SEARCH_MODE` (latency/accuracy), `CONTENT_SEARCH_ENABLED`, `QDMS_ACCESS_CONTROL_ENABLED`, `MIN_SCORE_THRESHOLD`, `HNSW_EF_SEARCH`, `MMR_K/MMR_FETCH_K/MMR_LAMBDA`, `FUSION_TEXT_WEIGHT/FUSION_VECTOR_WEIGHT`, `RERANKER_*`, `*_MAX_CONCURRENT`, `AZURE_HTTP_MAX_CONNECTIONS`, `ANYIO_MAX_THREADS`, `COST_LIMIT`, `SYNC_INTERVAL_MINUTES`, `CHAT_HISTORY_LIMIT`, `MAX_CONTEXT_DOCS`, `OTEL_*`, `USER_ACCESS_API_URL`, `QDMS_FRONTEND_URL`, `QDMS_COMPANY_ID`.

---

## Questions / unknowns

- [ ] **Cost limit enforcement** — helpers + table exist but aren't wired into chat/search. Was 409/429 enforcement removed, planned, or only ever in the README? Token usage goes to OTel only.
- [ ] **`customs.py` / `config/client.py`** — referenced in README/CLAUDE docs but absent from the tree: stale docs or removed files?
- [ ] **`feedback_for_id` / `suggestions` / `cited_sources` columns** are defined (and migrated) but have no writer in any path — likely future features (the live feedback path is the separate like/dislike booleans).
- [ ] **`get_mongo_db` (sync source dependency)** is defined but not injected by any active endpoint — possibly only for scripts / `run_in_threadpool`.
- [ ] **Real `qdms_client` deployments** — only the in-repo template is present; actual per-branch client packages (prompts/policies) live outside this checkout and can't be verified.
- [ ] **Reranker model offline availability** — `load_reranker` needs the ONNX model pre-cached locally; the provisioning step that populates that cache isn't in this repo.
- [ ] **Vessel endpoints' ACL source** — `vessel_id`/`role` are trusted from the request body (upstream-trust model); how they're authenticated for a given user isn't shown in-tree.
- [ ] **Exact Cosmos index name in prod** — `INDEX_NAME` default is `oceanai_mol_qdms_index_hnsw` in settings but `create_cosmos_indexes.py` defaults the name to the DB name; which one prod uses depends on env config not present here.

---

## See also

[[OceanAI Projects MOC]] · [[EagleAI Platform]] · [[PAL Wiki]] · [[BSM SOA Agent]] · [[Kriya QA Platform]] · [[Decarbonization Agent]]
