# ProRag — Architecture

> A small, self-hostable RAG platform for mixed corpora (structured + unstructured), with a
> citing chatbot that hands back the actual source PDF, open to the right page, highlighted.
> Since the 2026-07 product-ization arc it is also a multi-user system: identity + ACLs,
> sessions + OIDC, per-user budgets, an S3 connector, and an admin API — a company can
> install it and point it at their document systems.
>
> Design ancestor: **QDMS-AI** (maritime RAG on Azure OpenAI + Cosmos vCore). ProRag keeps its
> genuinely good retrieval and streaming ideas and throws out the enterprise scaffolding.
>
> Target scale: one deployment = one customer, ~10k–500k chunks, one 2-vCPU/4 GB box or a
> laptop. Ops budget: a `docker compose up`.

---

## 1. System overview

One FastAPI process, one ParadeDB Postgres. That is the whole production topology — no
worker process, no Redis, no Celery.

Postgres carries the jobs that QDMS-AI spread across MongoDB + Cosmos vCore + SQL Server:
vector index (pgvector HNSW), keyword index (pg_search BM25, `tsvector` as fallback),
relational metadata (documents/chunks/chat history), the identity + ACL layer
(users/groups/document_acl), and connector sync state. Original files live on disk in a
compose named volume (`blobdata`), addressed by content hash.

Ingestion is **inline in the request handler** (`prorag/ingest/core.py:ingest_bytes()`),
not a job queue — there is no `jobs` table and no worker loop. A very large upload holds
the connection open for the duration; that is the documented remaining gap (the seam for
a future worker is marked in code). The same `ingest_bytes()` is shared by the HTTP
upload route and the S3 connector sync engine.

The request path is a deliberately shortened version of QDMS-AI's: **plan → hybrid retrieve →
fuse → crop → stream**. QDMS-AI's three-agent chain becomes two LLM calls (planner on a
cheap model, answerer on a strong-ish model) because the third agent's job — turning NL
into DB filters — is folded into the planner's single JSON output.

```mermaid
flowchart TD
    UI["Web UI (Next.js 16 + PDF.js)<br/>chat pane · citation chips · PDF viewer"]
    UI -->|"POST /chat/stream (SSE)"| API

    subgraph API["FastAPI app (prorag/)"]
        MW["Middleware: GZip(skips text/event-stream) → RequestTiming"]
        R["Routers: /ingest · /chat · /search · /files · /feedback · /eval<br/>/connectors · /admin · /auth (login/OIDC) · /healthz"]
        Sched["scheduler_loop (lifespan):<br/>connector poll + mandatory sweep"]
    end

    API --> Gate{"trivial turn?<br/>greeting / pure follow-up"}
    Gate -->|yes| Answer
    Gate -->|no| Plan

    Plan["Planner LLM (cheap tier)<br/>→ {search_needed, queries[2], mode}"]
    Plan --> Retr

    subgraph Retr["Hybrid retrieval — sequential on one connection, ACL-filtered in SQL"]
        Vec["pgvector HNSW cosine<br/>halfvec cast when embed_dim > 2000<br/>per-query × 2 queries"]
        Fts["BM25 (pg_search @@@)<br/>tsvector websearch_to_tsquery fallback"]
        Struct["structured row search<br/>tables → JSONB rows (mode=table)"]
    end

    Vec --> Fuse["RRF fusion (k=60)<br/>across all arms + both queries"]
    Fts --> Fuse
    Struct --> Fuse
    Fuse --> Crop["adaptive crop:<br/>dynamic score floor · min 3 / max 12<br/>revision-aware title dedup · token budget"]
    Crop --> Answer

    Answer["Answer LLM (strong tier) via LiteLLM<br/>astream → SSE tokens"]
    Answer --> Post["citation post-processing:<br/>normalize [S3] → resolve → sources[] with<br/>doc_id, page, bbox, file_url"]
    Post --> UI
    UI -->|"GET /files/{doc_id}#page=7&highlight=..."| Files["file server (range requests)"]

    subgraph PG[("ParadeDB Postgres 17<br/>pgvector + pg_search")]
        T1["documents · chunks(embedding halfvec/vector, tsv tsvector)"]
        T2["tables · table_rows(JSONB)"]
        T3["chats · messages · citations · feedback"]
        T4["users · groups · user_groups · document_acl · access_rules · sessions"]
        T5["api_keys · usage · connectors · connector_items · eval_runs"]
    end

    API --- PG
    Sched -->|"sync_incremental / full_sweep"| C["connectors/: S3 poll, download<br/>→ ingest_bytes() (#22)"]
    C --> PG
    Blob[["blobdata volume — original files by sha256"]] --- API
    Blob --- Files
```

The identity layer is a first-class citizen: the deployment is the tenant (no tenant
table); principals are `users` + `groups`; `document_acl` rows mirror grants; group
membership is resolved server-side at query time (never trusted from token claims);
`access_rules` let an admin grant a group by confirming a natural-language rule.
Full rationale on issues #2/#3/#4/#15.

---

## 2. Tech stack

| Choice | One-line justification |
|---|---|
| **Python 3.12 + FastAPI + uvicorn** | Same ergonomics as QDMS-AI; native async matters for SSE + streaming. |
| **ParadeDB Postgres 17 (pgvector + pg_search)** | One database for vectors, real BM25, relational data, identity, and connector state — the single biggest ops saving over QDMS-AI's three-store topology. |
| **pgvector HNSW, `halfvec` cast above 2000 dims** | `0001_initial.py` builds the HNSW index on `embedding::halfvec(embed_dim)` when `embed_dim > 2000` (pgvector's cap); the query casts both sides via `_distance_expr()` so the index actually matches (#16). |
| **pg_search BM25 (`@@@`, `paradedb.score()`)** | Real BM25 as the primary keyword arm; `websearch_to_tsquery` + `ts_rank_cd` remains as the fallback when pg_search is unavailable. |
| **PyMuPDF (`fitz`) with optional Docling** | PyMuPDF is the always-installed parser; Docling (layout/table/bbox) is used when present, falling back silently. |
| **pandas + `openpyxl` for CSV/XLSX** | Structured files never need a document parser; read the frame, that's the job. |
| **LiteLLM (python SDK, not the proxy)** | Provider-agnostic in ~one import: same code hits Ollama, OpenAI, Anthropic, Groq, or OpenRouter by changing a model string. The air-gap answer: point the models at an internal OpenAI-compatible endpoint. |
| **Embeddings: API-only via LiteLLM** | `EMBED_MODEL` (e.g. `text-embedding-3-small`, or `openrouter/google/text-embedding-004` at 2048 dims). No local ONNX embedder is configured in this deployment. |
| **Reranker: `BAAI/bge-reranker-v2-m3`, ONNX int8** | Code default `rerank_enabled=True`, but this deployment runs `RERANK_ENABLED=false` in `.env` — measured at ~130 s per 40-hit CPU query (#11), the honest default for CPU installs is off (#12). The `full` GPU image is the place to turn it on. |
| **SQLAlchemy 2.0 (async) + Alembic** | QDMS-AI's hand-rolled `migrate_*.py` scripts were a known wart; 13 migrations (0001–0013), advisory-locked auto-migrate at container startup. |
| **Authlib (OIDC) + argon2-cffi (local passwords)** | OIDC-first via discovery (4-field config: issuer/client_id/client_secret/redirect_path), local accounts as the no-IdP fallback; SAML only brokered through oauth2-proxy/Keycloak (#2, #5, #19). |
| **boto3 (S3 connector), moto (dev/test)** | The first connector (#22): any S3-compatible endpoint (AWS, MinIO, R2), incremental etag/size poll + mandatory full-ID sweep. |
| **Next.js 16.2.12 + React 19 + Tailwind 4 + pdf.js** | The viewer *is* the feature — pdf.js renders a PDF page with a custom highlight layer in-browser. `web-next/` is the active frontend. |
| **Docker Compose (paradedb + api + caddy)** | Versioned bundle (`${PRORAG_IMAGE}:${PRORAG_VERSION}`), auto-migrate entrypoint, `python -m prorag.doctor` smoke check (#8, #20). |

Explicitly **not** chosen: Qdrant/Weaviate/Milvus (a second stateful service for a capability
Postgres already has), LangChain (the abstraction cost exceeded its value in QDMS-AI —
retrieval here is SQL and a rerank call), Elasticsearch, Kubernetes, Redis, Celery.

---

## 3. Ingestion pipeline

Ingestion is **synchronous and inline** — no job queue. A single request runs the whole
lifecycle, and the same core function powers both the HTTP route and the S3 connector:

```
POST /ingest → sha256 the bytes → dedupe → blobs/{sha256}{ext}
            → parse (route by MIME) → chunk → embed (batched, retry-with-backoff)
            → bulk insert → status='ready'
```

`ingest_bytes(session, data, filename, collection)` in `prorag/ingest/core.py` owns
dedup, parse routing, the table artifacts, embed-with-retry, bulk chunk insert, the
`processing → ready/failed` lifecycle, and auto-admission into confirmed access rules.
`prorag/ingest/router.py` is a thin HTTP wrapper keeping only the trust-boundary
concerns (streaming read with a size ceiling, suffix allowlist, collection-name
validation). The S3 connector calls `ingest_bytes()` directly with bytes it already
has — skipping the HTTP layer entirely.

### 3.1 Routing

| Input | Path |
|---|---|
| `.csv`, `.xlsx`, `.tsv` | **Structured path** — pandas reads the frame, each sheet becomes a table |
| `.pdf`, `.docx`, `.pptx` | **Docling path** when Docling is installed (splits elements into prose or table); PyMuPDF text path otherwise |
| `.md`, `.txt` | **Text path** (markdown-aware splitter, no parser) |

A PDF is never purely one or the other. Docling returns a typed element stream; ProRag sends
`Table` elements down the structured path and everything else down the prose path, so a single
report file produces both narrative chunks and table chunks.

### 3.2 Unstructured path (prose)

1. Parser → elements in reading order with `(page_no, bbox)` provenance.
2. **Structure-aware chunking**: split on heading boundaries first, then pack sections to a
   target of **~700 tokens with 100-token overlap**, never merging across a page boundary
   *and* never merging across an H1/H2.
3. Every chunk carries a **heading breadcrumb** prefix (`# Safety Manual > 4. Drills > 4.2 Fire`)
   in its embedded text but flagged so the UI can strip it.
4. `page_start`/`page_end` and the union bbox come from element provenance and are stored.

### 3.3 Structured path (tables, CSVs, forms)

Tables are not prose and chunking them like prose destroys them. Three artifacts per table:

1. **Rows → `table_rows`** as JSONB, one row per record, with the header map — what makes
   `WHERE data->>'vessel' = 'X'` possible.
2. **A table summary chunk** — an LLM-free synthesized text blob:
   `"Table: <caption>. Columns: a, b, c. 240 rows. Sample: <first 3 rows as markdown>"`.
3. **Row-window chunks** — the table serialized as markdown in windows of ~25 rows, header
   repeated in every window.

Wide tables (>12 columns) are additionally serialized **row-per-chunk as key–value pairs**
(`vessel: X | inspection_date: 2026-03-01 | result: pass`). Forms (label/value PDFs) take
exactly this path. CSV/XLSX skip Docling; `page_no` for a CSV is `null` and the citation
degrades gracefully to a row-range anchor.

### 3.4 Identity, ACL and connector tables

Beyond the core corpus tables (`documents`, `chunks`, `tables`, `table_rows`), the schema
carries the product-ization layer (19 models total, migrations 0001–0013):

- **Identity**: `users` (external_subject nullable+unique for IdP users, email unique,
  password_hash nullable for OIDC-only, is_admin, disabled_at, daily_cap_usd_override),
  `groups` (source `local|idp`), `user_groups` (membership rows — the truth queries read).
- **Access control**: `document_acl` (doc_id, principal_type `user|group|public`, principal_id,
  source — mirrors the *source's* principals per #15), `access_rules` (NL query + target group,
  state `draft|confirmed`, stored query_embedding).
- **Sessions/auth**: `sessions` (sha256-hashed token, user_id, expires_at, revoked_at),
  `api_keys` (bearer keys, now with nullable `user_id` — null = legacy unscoped super-key).
- **Ops**: `usage` (per-call LLM spend, now with nullable `user_id` + `(user_id, created_at)`
  index for the budget query), `feedback`, `eval_runs`, `connectors` (type/config/enabled/
  last_sync_at/last_full_sweep_at/last_error), `connector_items` (remote object diff state).

**ACL default is deny.** A document with no `document_acl` row is visible to admins only —
this is what Tier-C connector content (S3, Notion) enters as until an admin grants access.
The one compatibility exception: migration 0008 backfills a `public` grant for pre-existing
documents so the single-user install keeps working after upgrade; an admin can revoke.

---

## 4. Retrieval

### 4.1 Planning (one cheap LLM call)

A single JSON call on the cheap tier returns:

```json
{ "search_needed": true, "queries": ["fire drill frequency SOLAS", "muster exercise interval requirement"], "mode": "default" }
```

- **Two deliberately different queries** (vocabulary-disjoint, fused by RRF).
- `search_needed=false` skips retrieval on pure greetings/follow-ups answerable from history.
- `mode: "table"` engages the structured arm.
- **Prompt-cache-aware split**: the whole rulebook lives in a static system message and only
  `{query}` varies in the user message.
- **Defensive parsing**: any planner/JSON failure degrades to running the raw query verbatim
  as both queries — a planner failure never fails the request. Filters are validated but not
  yet applied (`ponytail:` note in code) — the filter compiler is deferred.
- The greeting short-circuit (≤4-word gate) skips the planner entirely (#12).

### 4.2 Three arms, sequential on one connection, ACL-filtered

All three arms run **sequentially on one `AsyncSession`** — an `asyncio.gather` over a
single connection was measured identical (0.81 s both ways, one connection serializes
regardless) and removed after it caused `InvalidRequestError` on cold sessions (#11,
HARDENING.md). Every arm takes an optional `user` and applies the same visibility
predicate **inside the SQL, before LIMIT** — never a post-filter:

- **Vector**: `ORDER BY embedding <=> $q LIMIT k`, with `_distance_expr()` casting both
  sides to `halfvec(embed_dim)` when `embed_dim > 2000` so the HNSW expression index
  matches (#16). Two queries per message.
- **Keyword**: real BM25 via pg_search `@@@` against the bm25 index when the extension is
  present (probed once per process); falls back to `websearch_to_tsquery` + `ts_rank_cd`.
- **Structured**: JSONB row search over `table_rows` — engaged only when the planner set
  `mode: "table"`.

The visibility predicate (`prorag/retrieve/visibility.py:visibility_clause(user)`) is an
EXISTS clause over `document_acl`: visible iff `public`, or a direct user grant, or a grant
to any of the user's groups (group ids resolved once per request from `user_groups`).
`is_admin` and legacy unscoped keys bypass. The BM25 arm's raw-SQL twin uses bound
parameters only — no user input interpolation.

### 4.3 Fusion: RRF everywhere

ProRag uses **RRF (k=60) for everything** — arms, queries, all of it.
`score = Σ 1/(60 + rank_i)`. Rank-based fusion needs no score calibration between a cosine
distance and a BM25 score. Arms get weights (`vector 1.0, fts 1.0, structured 1.2`) applied
to their RRF contribution. Deduplicate to the best chunk per `(doc_id, ord)` before cropping.

### 4.4 Rerank + crop

- **Rerank**: top-40 fused → `bge-reranker-v2-m3` (ONNX int8) in a dedicated
  `ThreadPoolExecutor` with `OMP_NUM_THREADS=1`. Off in this deployment (see §2); the
  executor is correct for a GPU install.
- **Adaptive crop** (`retrieve/crop.py`): sort by score, take a **dynamic floor of
  `max(top_score - gap, floor)`**, clamp to min 3 / max 12 chunks, drop chunks under
  150 chars, stop at a hard **token budget** (default 6000).
- **Revision-aware dedup**: collapse chunks whose parent `title_norm` matches, keeping the
  newest `doc_date`. `title_norm` is computed at ingest time (a column), not per-request.
- **Neighbour expansion**: for each surviving prose chunk, if budget allows, pull `ord±1`
  from the same doc.

Context is assembled as numbered blocks the answerer must cite:

```
[S1] Safety Manual (Rev 3) — p.14
Heading: 4. Drills > 4.2 Fire
<text>
```

---

## 5. Chat, citations, and getting the PDF back

### 5.1 The citation contract

The answerer is instructed to write **`[S1]`, `[S2]` inline, immediately after the sentence they
support**, never clustered, never a "Sources:" section. Post-processing still exists because
LLMs: `normalize_citations()` maps `(S1)`, `[s1]`, `[source 1]`, `[1]` → `[S1]`, drops indices
out of range, and strips any trailing sources block.

### 5.2 SSE protocol

`POST /chat/stream` → `text/event-stream`, headers `Cache-Control: no-cache`,
`X-Accel-Buffering: no`, opening `retry: 3000`. Events:

| event | payload |
|---|---|
| `status` | `{"stage":"planning"\|"retrieving"\|"reranking"\|"answering"}` |
| `sources` | `Source[]` — sent **before the first token** so citation chips resolve instantly |
| `budget` | `{"warning":"..."}` — emitted after `sources` when the user is over their soft cap (#21) |
| `token` | `{"t":"..."}` |
| `citation` | `{"n":3}` — emitted the moment `[S3]` completes in the stream |
| `usage` | real token counts from the provider's final `include_usage` chunk (#13) |
| `meta` | `{"message_id":"...","usage":{...},"cost_usd":0.0021}` |
| `error` | `{"message":"..."}` |
| `done` | `{}` |

Hardening carried over from QDMS-AI: a **markdown-table-row buffer** (hold partial `| …`
lines until `\n`), a **whitespace-runaway abort** after 200 consecutive whitespace chars, a
heartbeat comment (`: ping`) every 15 s so idle proxies don't kill the connection, and gzip
middleware that skips `text/event-stream`.

### 5.3 The Source object — this is the product

```json
{
  "n": 3,
  "doc_id": "8f2c…",
  "title": "Safety Manual (Rev 3)",
  "snippet": "Fire drills shall be conducted at intervals not exceeding…",
  "page": 14,
  "bbox": [72.0, 385.5, 523.2, 461.8],
  "kind": "prose",
  "score": 0.91,
  "file_url": "/files/8f2c…/original.pdf#page=14"
}
```

Three ways the user gets the source, in increasing weight:

1. **Hover a chip** → the snippet, from the payload already in hand. Zero requests.
2. **Click a chip** → the right-hand pdf.js viewer loads `file_url`, jumps to `page`, and
   draws a translucent rect over `bbox` on the text layer.
3. **Download** → `GET /files/{doc_id}/original?download=1`, original bytes, original filename.

Non-PDF sources degrade honestly: DOCX/PPTX cite `page` and offer download; CSV rows cite
`Table "Inspections", rows 40–64` and the chip opens an inline data grid (`/tables/{id}/rows`).

Every citation is persisted in a `citations` table joined to the message. Every `Source[]`
returned to the user is ACL-filtered — `/files/{doc_id}/original` and `/tables/{table_id}/rows`
return **404** (not 403) when the document is invisible to the caller, so citations cannot
leak existence (#3, #18).

### 5.4 Cost tracking and budgets

LiteLLM's `completion_cost()` is called on every response; the number lands in
`usage` (and `messages.cost_usd`). Two layers of caps (#9, #21):

- **Install-wide hard cap** (`daily_cost_cap_usd`) — unchanged behavior: over budget →
  `429` before touching the LLM.
- **Per-user daily cap** — soft at 1× (`user_daily_cap_usd`, default $1.00): the request
  still runs, but `/chat` returns a `budget_warning` and `/chat/stream` emits a `budget`
  event. At `user_hard_cap_multiplier` × (default 2×): `429` with
  `"you have used $X of $Y today, resets at midnight UTC"`. Per-user override column
  (`daily_cap_usd_override`, migration 0010) replaces the global default for that user.

Both windows are UTC (`_utc_day_start()`, #13). Token usage comes from the provider's real
`include_usage` chunk; `estimate_tokens()` is fallback only. No reservation machinery — the
accepted one-answer overshoot is documented in the code (#9.4). Retrieval spend (planner +
embeddings) counts against the user.

---

## 6. API surface

All routes except `/healthz`, `/readyz`, `/web` static and `/auth/*` sit behind
`require_auth` — a no-op when `auth_enabled=False` (local dev default).

| Method | Path | Notes |
|---|---|---|
| `POST` | `/ingest` | multipart `file`, `collection?` → `202 {doc_id, status, duplicate_of?}` |
| `POST` | `/chat` | `{message, chat_id?, filters?}` → `{message_id, answer_md, answer_html, sources[], usage, budget_warning?}` |
| `POST` | `/chat/stream` | same body → SSE per §5.2 |
| `GET` | `/search` | `?q&k=10` → `{results: Source[], timings}` — retrieval without the LLM (the debugging endpoint) |
| `POST` | `/feedback` | `{message_id, rating: up\|down, comment?}` → `{ok}` (toggles) |
| `GET` | `/files/{doc_id}/original` | `?download=0\|1` → the file, `Accept-Ranges: bytes`, ETag = sha256; 404 when invisible |
| `GET` | `/tables/{table_id}/rows` | `?limit&offset&where` → JSONB rows; 404 when invisible |
| `GET` | `/stats` | document + ready counts (header stat; intentionally unfiltered, count-only) |
| `POST` | `/eval/run` | runs the golden set; `GET /eval/runs/{id}` → per-question + aggregate |
| `POST` | `/auth/login` | email+password → sets `prorag_session` HttpOnly cookie |
| `POST` | `/auth/logout` | revokes the session |
| `GET` | `/auth/oidc/login` · `/auth/oidc/callback` | Authlib OIDC; both 404 when `oidc_issuer` unset |
| `GET/POST` | `/connectors` · `GET/PATCH/DELETE /connectors/{id}` | S3 connector CRUD (admin) |
| `POST` | `/connectors/{id}/sync` | `?full=` — incremental poll or mandatory sweep; `409` if a sync is already running |
| `GET` | `/admin/documents` | paged, filter by status/collection/label/source, `q=` filename search |
| `GET` | `/admin/documents/{id}/access` | "why can X see Y": grant rows joined to rule/source provenance |
| `POST/GET` | `/admin/rules` · `GET/PATCH/DELETE /admin/rules/{id}` | access rules; `PATCH` only while `draft` |
| `POST` | `/admin/rules/{id}/preview` | runs the NL query through the arms admin-unfiltered → top-10 sample + count |
| `POST` | `/admin/rules/{id}/confirm` | freezes the rule, writes `document_acl` group grants |
| `GET` | `/admin/rules/{id}/grants` | paged audit feed of grants |
| `GET` | `/admin/users` · `PATCH /admin/users/{id}` | users + membership + cap override + today's spend |
| `GET` | `/admin/users/{id}/visible-docs` | reverse ACL: "what can this person see" |
| `POST/GET` | `/admin/groups` · `PATCH/DELETE /admin/groups/{id}` | group CRUD — local-source groups only (idp rows 400) |
| `POST/DELETE` | `/admin/groups/{id}/members/{user_id}` | membership add/remove (local groups) |
| `GET` | `/admin/usage` | `?window=7d` — usage by user/day/model + warned-users list |
| `GET` | `/healthz` · `/readyz` | liveness / (db + migrations) — `readyz` bounded at 5 s |

**Auth**: session cookies (`HttpOnly; SameSite=Lax; Secure` driven by
`session_cookie_secure`) via local login or OIDC, **plus** bearer API keys for machines
(sha256-hashed, `user_id` nullable for legacy unscoped super-keys). Session wins when both
present. OIDC config is four fields (`oidc_issuer`, `oidc_client_id`, `oidc_client_secret`,
`oidc_redirect_path`), discovery + JWKS handle the rest; group claims *seed* membership rows
at login — the stored `user_groups` rows are the truth queries read (#2). Local accounts use
argon2; bootstrap via `scripts/create_admin.py`.

---

## 7. Project layout

```
├─ docker-compose.yml            # paradedb + api + caddy; blobdata/pgdata volumes
├─ docker-entrypoint.sh          # advisory-locked alembic upgrade head, then exec CMD
├─ .gitattributes                # *.sh text eol=lf (entrypoint survives Windows checkouts)
├─ pyproject.toml / alembic.ini  # uv + Alembic (path_separator=os)
├─ alembic/versions/             # 0001–0013
├─ prorag/
│  ├─ main.py                    # app factory, middleware, router mounts, scheduler_loop lifespan
│  ├─ settings.py                # pydantic-settings; every tunable lives here + boot validators
│  ├─ db.py                      # async engine (pool_size/max_overflow/pool_timeout), session dep
│  ├─ models.py                  # 19 SQLAlchemy models (corpus + identity + ACL + ops)
│  ├─ schemas.py                 # Pydantic request/response incl. Source
│  ├─ auth.py                    # argon2 hash/verify, sessions, current_user (cookie OR key)
│  ├─ auth_routes.py             # /auth/login, /auth/logout, /auth/oidc/login, /auth/oidc/callback
│  ├─ llm.py                     # LiteLLM wrapper: answer/embed/plan, cost, include_usage
│  ├─ cost.py                    # UTC-window spend, per-user budgets, budget_decision()
│  ├─ doctor.py                  # `python -m prorag.doctor` — 8 smoke checks, injectable
│  ├─ operations/                # OPERATIONS layer — business logic, no HTTP
│  │  ├─ retrieval.py            # hybrid pipeline: retrieve()/gather_hits()/build_prompt()/SYSTEM_PROMPT
│  │  ├─ chat.py                 # persist_exchange() — user+assistant turns + citations rows
│  │  └─ budget.py               # check_daily_cap() — install-wide + per-user cost gates
│  ├─ ingest/
│  │  ├─ router.py               # POST /ingest — HTTP trust boundary only
│  │  ├─ core.py                 # ingest_bytes(): dedup→parse→chunk→embed→store; shared w/ connectors
│  │  ├─ parse.py                # MIME routing → Docling / pandas / plain text
│  │  ├─ chunk.py                # prose chunker (heading+page aware)
│  │  ├─ tables.py               # summary / window / row-kv serialization + table_rows
│  │  └─ store.py                # blob write, upsert, dedupe by sha256
│  ├─ retrieve/
│  │  ├─ router.py               # GET /search — per-stage debug breakdown; pipeline via operations/
│  │  ├─ plan.py                 # two-query planner, defensive parse, greeting skip
│  │  ├─ arms.py                 # vector (halfvec cast) / BM25 / fts fallback / structured SQL
│  │  ├─ visibility.py           # visibility_clause() + visible_doc_guard() — the ACL predicate
│  │  ├─ fuse.py                 # RRF
│  │  ├─ rerank.py               # OpenRouter hosted cross-encoder + flatness guard
│  │  └─ crop.py                 # adaptive crop, revision dedup, neighbour expansion
│  ├─ chat/
│  │  ├─ router.py               # /chat, /chat/stream, /feedback — HTTP only; ops in operations/
│  │  ├─ stream.py               # SSE framing, table buffer, whitespace guard, heartbeat
│  │  └─ citations.py            # normalize → resolve → Source[] → persist
│  ├─ files/router.py            # original bytes (ranges), table rows, /stats
│  ├─ connectors/
│  │  ├─ router.py               # admin CRUD + POST /{id}/sync
│  │  ├─ sync.py                 # sync_incremental + full_sweep, per-item error isolation
│  │  ├─ s3.py                   # S3Connector (boto3, any S3-compatible endpoint)
│  │  └─ scheduler.py            # asyncio loop: poll / sweep decision matrix, per-connector locks
│  ├─ admin/router.py            # documents, rules (preview/confirm/grants), users, groups, usage
│  └─ eval/
│     ├─ router.py               # POST /eval/run, GET /eval/runs/{id}
│     ├─ runner.py               # deterministic metrics + optional ragas
│     └─ golden.jsonl            # checked-in TEMPLATE (8 placeholder rows, not a real golden set)
├─ web-next/                     # Next.js 16 chat UI: page.tsx, sse.ts, pdf-viewer, citation pills
├─ scripts/
│  ├─ migrate.py                 # advisory-locked alembic upgrade head (entrypoint calls this)
│  ├─ create_admin.py            # bootstrap local admin user, prints password once
│  ├─ create_api_key.py          # mint a bearer key
│  ├─ smoke.py                   # manual end-to-end script
│  ├─ sweep.py                   # golden-set sweep over retrieval knobs
│  └─ bench_scale.py             # scale measurements (#11)
└─ tests/                        # 214 tests — pure + DB-backed integration (visibility,
                                 # connectors, admin, auth, scheduler); conftest disposes
                                 # the engine after each test
```

Flat, ~38 source files. No `plugins/`, no `config/` package with six modules, no
`observability/` tree.

### The three layers

```
Routers (HTTP)  ->  Operations (business logic)  ->  Persistence + Infrastructure
admin/ router      operations/retrieval.py           db.py / models.py (SQL)
chat/ router       operations/budget.py              ingest/store.py (blobs)
ingest/ router     operations/chat.py                llm.py (LiteLLM)
files/ router      ingest/core.py                    cost.py (usage ledger)
retrieve/ router   retrieve/{plan,arms,fuse,         connectors/s3.py
connectors/ router   rerank,crop,visibility}.py
eval/ router       chat/citations.py
                   connectors/sync.py
```

Routers are the composition root: they parse/validate the request, inject
dependencies (`Depends(get_session)`, `Depends(current_user)`), call an
operation, and shape the response. Operations hold the business logic and
never import a router — `prorag/operations/*` is the explicit home for
cross-surface logic (the retrieval pipeline, the cost-cap gate, chat
persistence), which is why the eval surface can share them without depending
on `chat.router`. The retrieve arms / parse / chunk / sync modules are
operations in all but name (kept in their domain packages for cohesion).

---

## 8. Status — what's implemented, what isn't

### Implemented

**Phases 1–6 of the original plan** (all complete): MVP ingestion + citations, hybrid
retrieval + rerank, structured documents/tables, SSE streaming + the pdf.js viewer,
auth/cost-cap/feedback/health, golden-set evaluation + sweep + CI.

**The product-ization arc (issues #1–#24, 2026-07-31 → 08-01):**

- Identity + ACL schema (migration 0008, #17): `users`, `groups`, `user_groups`,
  `document_acl`, `api_keys.user_id`, `usage.user_id` — with a `public` backfill grant for
  pre-existing documents.
- ACL enforcement in all three retrieval arms + file endpoints (migration 0008, #18):
  `visibility_clause()` in SQL pre-LIMIT, 404-not-403 on `/files` and `/tables`.
- Auth: local accounts + server-side sessions + OIDC via Authlib (migration 0009, #19):
  `argon2-cffi`, `prorag_session` HttpOnly cookie, group claims seed membership rows.
- Packaging (migration 0010-era, #20): auto-migrate entrypoint (`scripts/migrate.py` +
  `docker-entrypoint.sh`), `prorag doctor` (8 checks), `blobdata` named volume, versioned
  compose image tag, boot-time settings validators.
- Per-user budgets (migration 0010, #21): soft per-user daily cap + hard install-wide cap,
  `budget_warning` on `/chat`, `budget` SSE event, usage attribution via `user_id`.
- S3 connector (migration 0011, #22): poll/sync/delete-propagation via `ingest_bytes()`
  reuse, Tier-C default-deny ACLs, sha256 dedup linking.
- Connector polling scheduler (migration 0012, #23): `next_action` decision matrix
  (incremental vs mandatory sweep), per-connector locks, `409` on overlap, `last_error` on
  the row, conftest engine-dispose fixture.
- Admin dashboard backing API (migration 0013, #24): documents listing + provenance,
  access-rule preview/confirm/grants + auto-admission on ingest, users/groups/membership,
  reverse-ACL, usage report.
- Bug fixes shipped during the arc: dead HNSW index — halfvec cast in `_distance_expr()`
  (#16, `c4992d5`); cost-window UTC bug + real streamed token usage (#13, `7b1f826`);
  reverse-proxy defects — `--proxy-headers`, API de-published from the host (#14, `5c5dbf9`).

### Not implemented (deferred)

- **#25 (OPEN)**: Admin dashboard UI in `web-next` — documents, rules, people, usage views.
  All backing API endpoints exist (#24); the UI does not. To be designed via **Open Design**
  (MCP) per the 2026-08-03 frontend decision.
- Login/logout UI in `web-next` — sessions come from `/auth/login` via curl/scripts for now.
- Jobs/worker queue for background ingestion — ingestion stays inline (#12.5); the seam is
  marked in `ingest/core.py`.
- Connectors beyond S3 — SharePoint/Graph, Google Drive, Confluence (DC before Cloud),
  Notion are researched (#6) and sequenced but not built; Notion/S3 are Tier C (no ACL
  mirroring).
- SCIM user provisioning — JIT only for now; group *sync* scheduler is also future work.
- Rule edit re-run diffs — #4's pending-diff-on-edit; v1 ships confirm-once (#24).
- Planner filter compilation — filters are validated but not applied (`ponytail:` note).
- slim/full Docker image variants (#8 decided, not built); model weights baked in, not
  downloaded at first use, is the target.
- Upgrade/migration story for existing installs, observability for installs we cannot see,
  licensing/entitlement — listed under "Not yet specified" in #1.

---

## 9. Borrowed / dropped

### Borrowed from QDMS-AI

| Idea | How it lands here |
|---|---|
| Hybrid keyword + vector retrieval | Both arms, but co-located in one Postgres instead of two databases. |
| Fusion of multiple result lists | RRF (k=60) everywhere, replacing the split NSF/RRF scheme. |
| Two-query expansion (vocabulary-disjoint alternative query) | Planner emits exactly two, fused by RRF. |
| Cheap/strong model split | Cheap tier plans, strong tier answers — via LiteLLM tiers, not Azure deployments. |
| Cross-encoder rerank on a dedicated `ThreadPoolExecutor`, `OMP_NUM_THREADS=1` | Kept — now off by default in this deployment (#12). |
| Adaptive context cropping with a dynamic score floor + min/max clamp | Verbatim, plus a token budget instead of a doc count. |
| Revision-aware title dedup | Kept, but `title_norm` is computed once at ingest and stored, not regexed per request. |
| Prompt-cache-aware system/user split | Kept. |
| Greeting / follow-up short-circuit before retrieval | Kept — cheapest possible latency win. |
| SSE hardening: table-row buffer, whitespace-runaway abort, `X-Accel-Buffering: no`, `retry:` | Verbatim, plus a heartbeat. |
| Citation post-processing that repairs LLM deviations | Kept, smaller, because `[Sn]` is a smaller target than `[Title](URL)`. |
| Pure-ASGI gzip that skips `text/event-stream` | Verbatim — `BaseHTTPMiddleware` genuinely does buffer SSE. |
| Like/dislike feedback keyed to a persisted response id | Kept as `/feedback`. |
| Startup warmup (model + connection pool) before serving | Kept. |

### Deliberately dropped

| Dropped | Why |
|---|---|
| Client plugin system + dynamic `/qdms-{client}` root path | One deployment. An abstraction with one implementation. |
| Multi-environment SQL engine factory (SQLite/pymssql/pyodbc-ADIntegrated) | One database everywhere: Postgres. Same code local and in prod. |
| Dual-DB topology + four Mongo clients | One Postgres, one async pool. This single change removes most of the ops burden. |
| MongoDB / Cosmos vCore / Azure entirely | pgvector + BM25 covers it; no cloud lock-in, runs on a laptop. |
| Full OTel stack (Tempo/Loki/Prometheus/Alloy/Grafana) | stdlib logging + `/healthz`. Reintroduce when there's more than one process to correlate. |
| LangChain | Retrieval is SQL; generation is one LiteLLM call. The wrapper cost exceeded its value. |
| Three separate agents | Two LLM calls. The filter agent's job is a field in the planner's JSON. |
| JWT auth, user/role tables, bcrypt, service-account trust boundary | Server-side sessions + OIDC + bearer keys. Nothing to revoke server-side is the wrong default for a product (#2). |
| python-jose | Two unpatched CVE classes; FastAPI migrated off it. Authlib + PyJWT instead (#5). |
| Weighted min-max score fusion (0.4/0.6) | RRF needs no cross-arm score calibration — the exact thing QDMS-AI kept mis-tuning. |
| MMR (`simsimd` `_fast_mmr`) | The reranker plus revision dedup already handles redundancy. |
| 6144-token chunks | A citation pointing at 6144 tokens isn't a citation; ~700 + neighbour expansion beats it. |
| Hand-rolled `migrate_*.py` scripts | Alembic (13 migrations). |
| `$regex` exposed to the LLM, unguarded `strptime` date parsing | Allow-listed operators, validated filters, drop-with-warning on bad input. |
| APScheduler in-process | An asyncio loop in `main.py` lifespan, driven by a pure `next_action()` decision (#23). |
| Webhooks for connectors | A self-hosted install behind a firewall has no reachable inbound endpoint — polling-first is the only mode (#15). |
| Materialised grant lists | ACLs store *principals*; group membership resolves fresh at query time — revocation is instant and correct for free (#15). |
| Display-vs-context split (cards showing pre-rerank, answer using post-rerank) | `sources[]` contains exactly what was in the LLM's context. Showing the user a different set than the model saw is a trust bug. |
| README that drifts from the code | Constants live in `settings.py`; the README quotes it and the eval suite tests it. |
