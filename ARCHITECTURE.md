# ProRag — Architecture

> A small, self-hostable RAG platform for mixed corpora (structured + unstructured), with a
> citing chatbot that hands back the actual source PDF, open to the right page, highlighted.
>
> Design ancestor: **QDMS-AI** (maritime RAG on Azure OpenAI + Cosmos vCore). ProRag keeps its
> genuinely good retrieval and streaming ideas and throws out the enterprise scaffolding.
>
> Target scale: 1 developer, ~10k–500k chunks, one 2-vCPU/4 GB box or a laptop. Ops budget: a
> `docker compose up`.

---

## 1. System overview

One FastAPI process, one Postgres. That's the whole production topology.

Postgres carries four jobs that QDMS-AI spread across MongoDB + Cosmos vCore + SQL Server:
vector index (pgvector HNSW), keyword index (native `tsvector` FTS), relational metadata
(documents/chunks/chat history), and job queue (a `SKIP LOCKED` table — no Redis, no Celery).
Original files live on local disk (or S3-compatible blob if you outgrow disk), addressed by
content hash.

The request path is a deliberately shortened version of QDMS-AI's: **plan → hybrid retrieve →
fuse → rerank → crop → stream**. QDMS-AI's three-agent chain becomes two LLM calls (planner on a
cheap model, answerer on a strong-ish model) because the third agent's job — turning NL into DB
filters — is folded into the planner's single JSON output.

```mermaid
flowchart TD
    UI["Web UI (SvelteKit + PDF.js)<br/>chat pane · citation chips · PDF viewer"]
    UI -->|"POST /chat/stream (SSE)"| API

    subgraph API["FastAPI app (prorag/)"]
        MW["Middleware: CORS → GZip(skips text/event-stream) → RequestTiming"]
        R["Routers: /ingest · /documents · /chat · /search · /files · /feedback · /eval · /healthz"]
    end

    API --> Gate{"trivial turn?<br/>greeting / pure follow-up"}
    Gate -->|yes| Answer
    Gate -->|no| Plan

    Plan["Planner LLM (cheap tier)<br/>→ {search_needed, queries[2], filters[], mode}"]
    Plan --> Retr

    subgraph Retr["Hybrid retrieval — asyncio.gather, all in Postgres"]
        Vec["pgvector HNSW cosine<br/>ef_search=64, per-query × 2 queries"]
        Fts["Postgres FTS<br/>websearch_to_tsquery + ts_rank_cd"]
        Struct["structured row search<br/>tables → JSONB rows + SQL filters"]
    end

    Vec --> Fuse["RRF fusion (k=60)<br/>across all arms + both queries"]
    Fts --> Fuse
    Struct --> Fuse
    Fuse --> Rerank["Cross-encoder rerank<br/>bge-reranker-v2-m3 ONNX int8, local, top-40 → top-k"]
    Rerank --> Crop["adaptive crop:<br/>dynamic score floor · min 3 / max 12<br/>revision-aware title dedup · token budget"]
    Crop --> Answer

    Answer["Answer LLM (strong tier) via LiteLLM<br/>astream → SSE tokens"]
    Answer --> Post["citation post-processing:<br/>normalize [S3] → resolve → sources[] with<br/>doc_id, page, bbox, file_url"]
    Post --> UI
    UI -->|"GET /files/{doc_id}#page=7&highlight=..."| Files["file server (range requests)"]

    subgraph PG[("PostgreSQL 16 + pgvector")]
        T1["documents · chunks(embedding vector, tsv tsvector)"]
        T2["tables · table_rows(JSONB)"]
        T3["chats · messages · citations · feedback"]
        T4["jobs (FOR UPDATE SKIP LOCKED)"]
    end

    API --- PG
    Worker["ingest worker (same image, `prorag worker`)"] --- PG
    Worker --> Parse["Docling: layout + tables + page/bbox<br/>| pandas for CSV/XLSX"]
    Parse --> Emb["embeddings via LiteLLM<br/>(bge-m3 local | text-embedding-3-small)"]
    Emb --> PG
    Blob[["blobs/ — original files by sha256"]] --- Worker
    Blob --- Files
```

---

## 2. Tech stack

| Choice | One-line justification |
|---|---|
| **Python 3.12 + FastAPI + uvicorn** | Same ergonomics as QDMS-AI; native async matters for SSE + parallel retrieval arms. |
| **PostgreSQL 16 + pgvector 0.8 (HNSW)** | One database for vectors, FTS, relational data, and the job queue — the single biggest ops saving over QDMS-AI's three-store topology. |
| **Postgres native FTS (`tsvector` + `websearch_to_tsquery`)** | The keyword arm for free, in the same transaction as the vector arm — no second engine to keep in sync. |
| **Docling (IBM) for PDF/DOCX/PPTX** | Best current open parser that returns layout, reading order, *and* per-element page + bounding box in one pass — the bbox is what makes PDF highlighting possible; `unstructured.io` needs a paid tier for equivalent table quality, PyMuPDF alone has no table structure. |
| **PyMuPDF (`fitz`) as fallback + page renderer** | Fast text extraction when Docling is overkill, and it renders page thumbnails for the citation preview. |
| **pandas + `openpyxl` for CSV/XLSX** | Structured files never need a document parser; read the frame, that's the job. |
| **LiteLLM (python SDK, not the proxy)** | Provider-agnostic in ~one import: same code hits Ollama, OpenAI, Anthropic, Groq, or OpenRouter by changing a model string. Kills QDMS-AI's Azure lock-in. |
| **Embeddings: `BAAI/bge-m3` local (ONNX) by default, `text-embedding-3-small` by config** | bge-m3 is 1024-dim, multilingual, strong on tables/short text, and runs free on CPU; the API path exists for people who don't want the RAM. |
| **Reranker: `BAAI/bge-reranker-v2-m3`, ONNX int8, via `sentence-transformers` CrossEncoder** | Materially better than `ms-marco-MiniLM` (QDMS-AI's choice) on multilingual and long chunks, still CPU-viable at top-40; Cohere Rerank is a config swap for anyone who'd rather pay. |
| **SQLAlchemy 2.0 (async) + Alembic** | QDMS-AI's hand-rolled `migrate_*.py` scripts were a known wart; Alembic is 20 lines of setup and never bites again. |
| **Job queue = Postgres table + `FOR UPDATE SKIP LOCKED`** | Ingestion is the only background work; a table and a `while True` beat adding Redis + Celery to a side project. |
| **SvelteKit + `pdf.js` (`pdfjs-dist`)** | The viewer *is* the feature — pdf.js is the only thing that renders a PDF page with a custom highlight layer in-browser, and SvelteKit keeps the frontend a single small app. |
| **Ragas + a checked-in golden set** | Faithfulness / context-precision / answer-relevancy scored in CI so retrieval tuning has a number attached. |
| **Docker Compose (api + worker + postgres) / `uv` for deps** | Two services and a database; `uv` because Poetry's resolver was the slowest thing in the QDMS-AI dev loop. |
| **structlog + `/healthz` + one Prometheus counter set (optional)** | Replaces the whole OTel/Tempo/Loki/Alloy/Grafana stack. Reintroduce OTel the day you have more than one process to correlate. |

Explicitly **not** chosen: Qdrant/Weaviate/Milvus (a second stateful service for a capability
Postgres already has), LangChain (the abstraction cost exceeded its value in QDMS-AI —
retrieval here is SQL and a rerank call), Elasticsearch, Kubernetes.

---

## 3. Ingestion pipeline

Upload is synchronous only up to "file stored + job queued". Everything else is the worker.

```
POST /ingest → sha256 the bytes → dedupe → blobs/{sha256}{ext}
            → INSERT documents(status='pending') → INSERT jobs → 202 {doc_id}

worker: claim job (SKIP LOCKED) → route by MIME → parse → chunk → embed → index → status='ready'
```

### 3.1 Routing

| Input | Path |
|---|---|
| `.csv`, `.xlsx`, `.tsv`, `.json`(array of objects) | **Structured path** |
| `.pdf`, `.docx`, `.pptx` | **Docling path** — which itself splits every extracted element into prose or table |
| `.md`, `.txt`, `.html` | **Text path** (markdown-aware splitter, no parser) |

A PDF is never purely one or the other. Docling returns a typed element stream; ProRag sends
`Table` elements down the structured path and everything else down the prose path, so a single
report file produces both narrative chunks and table chunks. This is the core improvement over
QDMS-AI, which flattened every PDF into one text blob and lost tables entirely.

### 3.2 Unstructured path (prose)

1. Docling → `DoclingDocument`; take elements in reading order with `(page_no, bbox)` provenance.
2. **Structure-aware chunking**: split on heading boundaries first, then pack sections to a
   target of **~700 tokens with 100-token overlap**, never merging across a page boundary
   *and* never merging across an H1/H2. (700, not QDMS-AI's 1536/6144: modern rerankers do
   better with tighter chunks, and a citation that points at 6144 tokens isn't a citation.)
3. Every chunk carries a **heading breadcrumb** prefix (`# Safety Manual > 4. Drills > 4.2 Fire`)
   in its embedded text but flagged so the UI can strip it. Cheap, and it fixes the classic
   "chunk that says 'this must be done monthly' with no subject" failure.
4. `page_start`/`page_end` and the union bbox come from element provenance and are stored.

### 3.3 Structured path (tables, CSVs, forms)

Tables are not prose and chunking them like prose destroys them. Three artifacts per table:

1. **Rows → `table_rows`** as JSONB, one row per record, with the header map. This is what
   makes `WHERE data->>'vessel' = 'X'` possible — a genuine capability QDMS-AI lacked.
2. **A table summary chunk** — an LLM-free synthesized text blob:
   `"Table: <caption>. Columns: a, b, c. 240 rows. Sample: <first 3 rows as markdown>"`.
   Embedded and FTS-indexed. This is what semantic search hits.
3. **Row-window chunks** — the table serialized as markdown in windows of ~25 rows, header
   repeated in every window. Embedded. This is what gets fed to the LLM when a question needs
   actual cell values.

Wide tables (>12 columns) are additionally serialized **row-per-chunk as key–value pairs**
(`vessel: X | inspection_date: 2026-03-01 | result: pass`), which retrieves far better than a
markdown row that's mostly pipes. Forms (label/value PDFs) take exactly this path.

CSV/XLSX skip Docling: pandas reads them, each sheet becomes a table, same three artifacts.
`page_no` for a CSV is `null` and the citation degrades gracefully to a row-range anchor.

### 3.4 Metadata schema

```sql
CREATE TABLE documents (
  id            uuid PRIMARY KEY,
  sha256        text UNIQUE NOT NULL,       -- content-addressed dedupe
  filename      text NOT NULL,
  mime          text NOT NULL,
  blob_path     text NOT NULL,              -- blobs/{sha256}.pdf
  page_count    int,
  title         text,                       -- parsed or filename
  title_norm    text,                       -- revision-stripped, for dedup (see §4.4)
  doc_date      date,
  revision      text,                       -- "Rev 3", "May-2021" — pulled out of the title
  collection    text NOT NULL DEFAULT 'default',
  meta          jsonb NOT NULL DEFAULT '{}',-- user-supplied tags, filterable
  status        text NOT NULL,              -- pending|parsing|embedding|ready|failed
  error         text,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id            bigserial PRIMARY KEY,
  doc_id        uuid NOT NULL REFERENCES documents ON DELETE CASCADE,
  ord           int NOT NULL,               -- reading order within doc
  kind          text NOT NULL,              -- prose | table_summary | table_window | row
  text          text NOT NULL,              -- what the LLM sees
  embed_text    text NOT NULL,              -- breadcrumb + text; what was embedded
  heading_path  text[],                     -- ['Safety Manual','4. Drills','4.2 Fire']
  page_start    int,
  page_end      int,
  bbox          real[],                     -- [x0,y0,x1,y1] in PDF points, union of elements
  table_id      bigint REFERENCES tables ON DELETE CASCADE,
  token_count   int NOT NULL,
  embedding     vector(1024) NOT NULL,
  tsv           tsvector GENERATED ALWAYS AS (
                  setweight(to_tsvector('english', coalesce(array_to_string(heading_path,' '),'')), 'A')
                  || setweight(to_tsvector('english', text), 'B')) STORED
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200);
CREATE INDEX ON chunks USING gin (tsv);
CREATE INDEX ON chunks (doc_id, ord);

CREATE TABLE tables (
  id uuid PRIMARY KEY, doc_id uuid REFERENCES documents ON DELETE CASCADE,
  caption text, columns text[], row_count int, page_no int, bbox real[]
);
CREATE TABLE table_rows (
  id bigserial PRIMARY KEY, table_id uuid REFERENCES tables ON DELETE CASCADE,
  row_no int, data jsonb NOT NULL
);
CREATE INDEX ON table_rows USING gin (data jsonb_path_ops);
```

`page_start` + `bbox` are the entire citation-to-PDF story. Everything in §5 falls out of these
two columns; ingestion's real job is to never lose them.

Idempotency: re-ingesting the same sha256 is a no-op. A changed file is a new document; the old
one is soft-superseded via `title_norm` so the revision-dedup in §4.4 prefers the newer
`doc_date`.

---

## 4. Retrieval

### 4.1 Planning (one cheap LLM call)

A single JSON call on the cheap tier returns:

```json
{ "search_needed": true, "is_follow_up": false,
  "queries": ["fire drill frequency SOLAS", "muster exercise interval requirement"],
  "filters": {"collection": "safety", "doc_date": {"gte": "2024-01-01"}},
  "wants_table": true }
```

Kept from QDMS-AI: **two deliberately different queries** (vocabulary-disjoint, fused by RRF),
`search_needed=false` to skip retrieval on pure follow-ups, and the **prompt-cache-aware split**
— the whole rulebook lives in a static system message (≥1024 tokens so OpenAI/Anthropic prefix
caching engages) and only `{date, history, query}` go in the user message. Unlike QDMS-AI, the
`cached_tokens` readout is logged *and* asserted in a test, so the cache silently breaking is a
red test rather than a bigger invoice.

Filters are validated against a Pydantic allow-list of fields and operators and compiled to
parameterised SQL. No LLM-authored `$regex`, no `strptime` that raises on bad input (QDMS-AI had
both) — an unparseable filter is dropped with a warning, never a 500.

The greeting short-circuit stays: a ≤4-token keyword gate that skips the planner entirely.

### 4.2 Three arms, one gather

All three run against the same Postgres connection pool via `asyncio.gather`:

- **Vector**: `ORDER BY embedding <=> $q LIMIT 40` per query, `hnsw.ef_search = 64`, filters in
  the `WHERE` (pgvector 0.8 costs pre-filters properly, so no post-filter recall cliff).
- **FTS**: `WHERE tsv @@ websearch_to_tsquery('english', $q) ORDER BY ts_rank_cd(tsv, q) DESC`.
  `websearch_to_tsquery` gives AND-by-default plus quoted phrases for free — this is the clean
  version of QDMS-AI's hand-rolled `+`-prefixing hack, and it doesn't need a stopword carve-out.
- **Structured**: only when `wants_table`, or when the planner emitted a field filter that maps
  onto a known column — a JSONB query over `table_rows` returning matching rows grouped by table.

Every arm has a 3 s `statement_timeout`; a slow arm degrades the result, it never fails the turn.

### 4.3 Fusion: RRF everywhere

QDMS-AI used weighted min-max normalization (0.4/0.6) *and* RRF depending on the code path.
ProRag uses **RRF (k=60) for everything** — arms, queries, all of it. `score = Σ 1/(60 + rank_i)`.
Rank-based fusion needs no score calibration between a cosine distance and a `ts_rank_cd`, which
is exactly the calibration QDMS-AI kept getting wrong (and then documented wrong in its README).
Arms get weights (`vector 1.0, fts 1.0, structured 1.2`) applied to their RRF contribution — one
tunable set of three numbers, all in `settings`, all in the eval harness.

Deduplicate to the best chunk per `(doc_id, ord)` before reranking.

### 4.4 Rerank + crop

- **Rerank**: top-40 fused → `bge-reranker-v2-m3` (ONNX int8) in a **dedicated
  `ThreadPoolExecutor` with `OMP_NUM_THREADS=1`**, loaded once at startup. Straight lift from
  QDMS-AI — the thread-isolation finding was real and hard-won. Unlike QDMS-AI it is **on by
  default**; if it's off by default nobody ever measures it.
- **Adaptive crop** (`context.py`), also lifted: sort by rerank score, take a **dynamic floor of
  `max(top_score - gap, floor)`**, clamp to min 3 / max 12 chunks, drop chunks under 150 chars,
  and stop at a hard **token budget** (default 6000) rather than a doc count.
- **Revision-aware dedup**: collapse chunks whose parent `title_norm` matches, keeping the newest
  `doc_date`. `title_norm` is computed at *ingest* time (a column), not per-request by regex —
  same idea as QDMS-AI, done once instead of on every query.
- **Neighbour expansion**: for each surviving prose chunk, if budget allows, pull `ord±1` from
  the same doc. Cheap, and it fixes answers that get cut mid-procedure.

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
support**, never clustered, never a "Sources:" section. Numbers only — no markdown links, no
titles. QDMS-AI asked for `[Title](URL)` and then needed a repair regex for four different
deviation modes; asking for `[S<n>]` cuts the deviation surface to almost nothing.

Post-processing still exists, because LLMs: `normalize_citations()` maps `(S1)`, `[s1]`,
`[source 1]`, `[1]` → `[S1]`, drops indices out of range, and strips any trailing sources block.
It is small and unit-tested against a corpus of real deviations.

### 5.2 SSE protocol

`GET|POST /chat/stream` → `text/event-stream`, headers `Cache-Control: no-cache`,
`X-Accel-Buffering: no`, opening `retry: 3000`. Events:

| event | payload |
|---|---|
| `status` | `{"stage":"planning"\|"retrieving"\|"reranking"\|"answering"}` |
| `sources` | `Source[]` — sent **before the first token** so citation chips resolve instantly |
| `token` | `{"t":"..."}` |
| `citation` | `{"n":3}` — emitted the moment `[S3]` completes in the stream, so the UI can light the chip live |
| `meta` | `{"message_id":"...","usage":{...},"cost_usd":0.0021}` |
| `error` | `{"message":"..."}` |
| `done` | `{}` |

Hardening carried over verbatim from QDMS-AI, because both failure modes are real: a
**markdown-table-row buffer** (hold partial `| …` lines until `\n`) and a **whitespace-runaway
abort** after 200 consecutive whitespace chars. Added: a heartbeat comment (`: ping`) every 15 s
so idle proxies don't kill the connection, and gzip middleware that skips `text/event-stream`.

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
  "file_url": "/files/8f2c…/original.pdf#page=14",
  "preview_url": "/files/8f2c…/page/14.png?highlight=72,385.5,523.2,461.8"
}
```

Three ways the user gets the source, in increasing weight:

1. **Hover a chip** → the snippet, from the payload already in hand. Zero requests.
2. **Click a chip** → the right-hand pdf.js viewer loads `file_url` (HTTP range requests, so a
   200 MB manual streams the one page), jumps to `page`, and draws a translucent rect over
   `bbox` on the text layer. Instant, no server render.
3. **Download** → `GET /files/{doc_id}/original?download=1`, `Content-Disposition: attachment`,
   original bytes, original filename.

`preview_url` is the fallback for non-PDF sources and for link previews: PyMuPDF renders the page
to PNG at 110 DPI with the bbox stroked, cached on disk by `(doc_id, page, bbox)`.

Non-PDF sources degrade honestly: DOCX/PPTX cite `page` from Docling's page mapping and offer
download + a rendered preview; CSV rows cite `Table "Inspections", rows 40–64` and the chip
opens an inline data grid instead of the PDF viewer. `bbox: null` simply means no highlight.

Every citation is persisted in a `citations` table joined to the message, so "which sources did
this answer actually use" is a query, not a guess — QDMS-AI had a `cited_sources` column with no
writer, and this is that column, wired up.

### 5.4 Cost tracking, wired up

QDMS-AI's most-cited flaw. Here, LiteLLM's `completion_cost()` is called on every response, the
number lands in `messages.cost_usd`, and a daily rollup view backs a soft cap: over budget →
`429` with a clear message. It is ~15 lines, and it exists in the request path or not at all.

---

## 6. API surface

| Method | Path | Body / params → response |
|---|---|---|
| `POST` | `/ingest` | multipart `file`, `collection?`, `meta?` → `202 {doc_id, status, duplicate_of?}` |
| `POST` | `/ingest/url` | `{url, collection?}` → `202 {doc_id}` |
| `GET` | `/documents` | `?collection&status&q&limit&cursor` → `{items: DocumentSummary[], next}` |
| `GET` | `/documents/{id}` | → `Document` + `{chunk_count, table_count, error?}` |
| `DELETE` | `/documents/{id}` | → `204` (cascades chunks/tables; blob GC'd if unreferenced) |
| `POST` | `/documents/{id}/reindex` | → `202 {job_id}` |
| `POST` | `/search` | `{query, filters?, k=10, rerank=true}` → `{results: Source[], timings}` (retrieval without the LLM — the debugging endpoint) |
| `POST` | `/chat` | `{message, chat_id?, filters?}` → `{message_id, answer_md, answer_html, sources: Source[], usage}` |
| `POST` | `/chat/stream` | same body → SSE per §5.2 |
| `GET` | `/chats` · `/chats/{id}` | list / full transcript with sources per message |
| `DELETE` | `/chats/{id}` | → `204` |
| `GET` | `/files/{doc_id}/original` | `?download=0\|1` → the file, `Accept-Ranges: bytes`, ETag = sha256 |
| `GET` | `/files/{doc_id}/page/{n}.png` | `?highlight=x0,y0,x1,y1&dpi=110` → rendered page PNG, cached |
| `GET` | `/tables/{table_id}/rows` | `?limit&offset&where` → JSONB rows (backs the CSV citation grid) |
| `POST` | `/feedback` | `{message_id, rating: up\|down, note?}` → `{ok}` (toggles, like QDMS-AI) |
| `POST` | `/eval/run` | no body → `200 {run_id, aggregate}` (synchronous); `GET /eval/runs/{id}` → per-question + ragas scores |
| `GET` | `/healthz` · `/readyz` | liveness / (db + reranker loaded) |

**Auth**: a single `Authorization: Bearer <api_key>` checked against a hashed key in
`api_keys`, plus per-key `collection` scoping. No JWT, no user table, no password hashing, no
49-year tokens. When multi-user is genuinely needed, the honest answer is an OIDC proxy in
front — not a hand-rolled auth system. Keys come from env/DB only; `.env` is gitignored and
`.env.example` ships with placeholders. This is the fix for QDMS-AI's "trusted, not enforced"
boundary at a scale where a real one is affordable.

---

## 7. Project layout

```
prorag/
├─ docker-compose.yml            # api + worker + postgres(pgvector image)
├─ pyproject.toml                # uv
├─ .env.example
├─ alembic/versions/
├─ prorag/
│  ├─ main.py                    # app factory, lifespan (warm reranker, warm pool), routers
│  ├─ settings.py                # pydantic-settings; every tunable lives here, nowhere else
│  ├─ middleware.py              # pure-ASGI gzip(skip SSE) + request timing
│  ├─ db.py                      # async engine, session dep, pgvector registration
│  ├─ models.py                  # SQLAlchemy: documents, chunks, tables, table_rows,
│  │                             #   chats, messages, citations, feedback, jobs, api_keys
│  ├─ schemas.py                 # Pydantic request/response incl. Source
│  ├─ auth.py                    # bearer key check + collection scoping
│  ├─ llm.py                     # LiteLLM wrapper: cheap/strong tiers, cost, retries, semaphores
│  ├─ embed.py                   # embedding backend (local ONNX | API), batching
│  ├─ ingest/
│  │  ├─ router.py               # /ingest, /documents
│  │  ├─ worker.py               # SKIP LOCKED loop; `python -m prorag.ingest.worker`
│  │  ├─ parse.py                # MIME routing → Docling / pandas / plain text
│  │  ├─ chunk.py                # prose chunker (heading+page aware)
│  │  ├─ tables.py               # summary / window / row-kv serialization + table_rows
│  │  └─ store.py                # blob write, upsert, dedupe by sha256
│  ├─ retrieve/
│  │  ├─ router.py               # /search
│  │  ├─ plan.py                 # planner LLM + filter compilation (allow-listed)
│  │  ├─ arms.py                 # vector / fts / structured SQL
│  │  ├─ fuse.py                 # RRF
│  │  ├─ rerank.py               # ONNX cross-encoder, dedicated executor
│  │  └─ context.py              # adaptive crop, revision dedup, neighbour expansion, budget
│  ├─ chat/
│  │  ├─ router.py               # /chat, /chat/stream, /chats, /feedback
│  │  ├─ prompts.py              # SYSTEM (static, cacheable) / USER (dynamic) split
│  │  ├─ stream.py               # SSE framing, table buffer, whitespace guard, heartbeat
│  │  └─ citations.py            # normalize → resolve → Source[] → persist
│  ├─ files/router.py            # original bytes (ranges), page PNG w/ highlight, table rows
│  └─ eval/
│     ├─ golden.jsonl            # ~50 q/a/expected-doc triples, checked in
│     └─ run.py                  # ragas: faithfulness, context precision/recall, answer relevancy
├─ web/                          # SvelteKit: chat pane, citation chips, pdf.js viewer, upload
└─ tests/
   ├─ test_chunk.py test_tables.py test_fuse.py test_citations.py test_stream.py
   └─ test_retrieval_smoke.py    # docker-compose'd postgres, 3 docs in, 3 questions out
```

Flat, ~25 source files. No `plugins/`, no `config/` package with six modules, no
`observability/` tree.

---

## 8. Phased plan

Each phase is one focused session and ends with something runnable.

**Phase 1 — MVP: text in, cited answer out.**
Compose file (postgres+pgvector), Alembic init, `documents`/`chunks` tables, `/ingest` for
PDF+txt+md via PyMuPDF only, fixed 700-token chunking with page numbers, embeddings via LiteLLM,
vector-only search, `/chat` non-streaming with `[Sn]` citations, `/files/{id}/original`.
*Done when:* upload a PDF, ask a question, get an answer with `[S1]` and a link that opens the
right PDF.

**Phase 2 — Hybrid + rerank.**
`tsv` generated column + GIN index, FTS arm, RRF fusion, two-query planner (cheap tier), ONNX
`bge-reranker-v2-m3` on its dedicated executor, adaptive crop + revision dedup + token budget,
`/search` debug endpoint. *Done when:* an acronym/document-number query that vector-only missed
now comes back first.

**Phase 3 — Structured documents.**
Swap PyMuPDF for Docling on PDF/DOCX/PPTX, split the element stream prose vs table, `tables` +
`table_rows` + JSONB GIN, the three table artifacts, CSV/XLSX path, structured retrieval arm,
`/tables/{id}/rows`. *Done when:* "how many inspections failed in Q1" answers from a CSV and
cites the table.

**Phase 4 — Streaming + the viewer.**
`/chat/stream` SSE with the full event set, table buffer, whitespace guard, heartbeat, gzip
middleware that skips SSE. SvelteKit UI: chat pane, live citation chips, pdf.js viewer with the
bbox highlight layer, upload/document list. *Done when:* clicking `[S3]` scrolls the PDF to
page 14 with the sentence highlighted.

**Phase 5 — Operations.**
Bearer API keys + collection scoping, cost tracking wired into the request path with a daily
cap, `/feedback`, structlog request logs, `/healthz`+`/readyz`, background job retries with
backoff, page-render cache, one-command deploy (Compose on a $6 VPS + Caddy for TLS).

**Phase 6 — Evaluation & tuning.**
`golden.jsonl`, `/eval/run` with ragas (faithfulness, context precision/recall, answer
relevancy), a sweep script over RRF arm weights / `ef_search` / chunk size / rerank top-N, and a
CI job that fails on a faithfulness regression. *Done when:* changing a retrieval constant
produces a number instead of an argument.

*Optional Phase 7, only on evidence:* query decomposition for multi-hop, HyDE for sparse
corpora, per-collection embedding models, ColBERT-style late interaction. All speculative until
Phase 6 says otherwise.

---

## 9. Borrowed / dropped

### Borrowed from QDMS-AI

| Idea | How it lands here |
|---|---|
| Hybrid keyword + vector retrieval | Both arms, but co-located in one Postgres instead of two databases. |
| Fusion of multiple result lists | RRF (k=60) everywhere, replacing the split NSF/RRF scheme. |
| Two-query expansion (vocabulary-disjoint alternative query) | Planner emits exactly two, fused by RRF. |
| Cheap/strong model split | Cheap tier plans, strong tier answers — via LiteLLM tiers, not Azure deployments. |
| Cross-encoder rerank on a dedicated `ThreadPoolExecutor`, `OMP_NUM_THREADS=1`, preloaded at startup | Verbatim. The thread-in-thread contention finding was measured and correct. |
| Adaptive context cropping with a dynamic score floor + min/max clamp | Verbatim, plus a token budget instead of a doc count. |
| Revision-aware title dedup | Kept, but `title_norm` is computed once at ingest and stored, not regexed per request. |
| Prompt-cache-aware system/user split, with `cached_tokens` verified | Kept, and the verification is now a test rather than a log line. |
| Greeting / follow-up short-circuit before retrieval | Kept — cheapest possible latency win. |
| SSE hardening: table-row buffer, whitespace-runaway abort, `X-Accel-Buffering: no`, `retry:` | Verbatim, plus a heartbeat. |
| Citation post-processing that repairs LLM deviations | Kept, smaller, because `[Sn]` is a smaller target than `[Title](URL)`. |
| Pure-ASGI gzip that skips `text/event-stream` | Verbatim — `BaseHTTPMiddleware` genuinely does buffer SSE. |
| Per-operation semaphores around LLM/embedding calls | Kept, three numbers in `settings`. |
| Startup warmup (model + connection pool) before serving | Kept. |
| Like/dislike feedback keyed to a persisted response id | Kept as `/feedback`. |
| Incremental, status-tracked re-embedding | Kept as the `jobs` table + `documents.status`. |
| Negative caching / anti-stampede on hot lookups | Kept where it applies (page renders, key lookups). |

### Deliberately dropped

| Dropped | Why |
|---|---|
| Client plugin system + dynamic `/qdms-{client}` root path | One deployment. An abstraction with one implementation. |
| Multi-environment SQL engine factory (SQLite/pymssql/pyodbc-ADIntegrated) | One database everywhere: Postgres. Same code local and in prod. |
| Dual-DB topology + four Mongo clients | One Postgres, one async pool. This single change removes most of the ops burden. |
| MongoDB / Cosmos vCore / Azure entirely | pgvector + FTS covers it; no cloud lock-in, runs on a laptop. |
| Full OTel stack (Tempo/Loki/Prometheus/Alloy/Grafana) | structlog + `/healthz` + optional counters. Reintroduce when there's more than one process to correlate. |
| LangChain | Retrieval is SQL; generation is one LiteLLM call. The wrapper cost exceeded its value. |
| Three separate agents | Two LLM calls. The filter agent's job is a field in the planner's JSON. |
| Two httpx pools with a 60/40 split | Meaningful at 500 connections; noise at side-project load. |
| JWT auth, user/role tables, bcrypt, service-account trust boundary | Bearer API keys with collection scoping. No 49-year tokens, no upstream-trust hole. |
| Weighted min-max score fusion (0.4/0.6) | RRF needs no cross-arm score calibration — the exact thing QDMS-AI kept mis-tuning. |
| MMR (`simsimd` `_fast_mmr`) | The reranker plus revision dedup already handles redundancy; MMR was compensating for having no reranker on by default. |
| 6144-token chunks | A citation pointing at 6144 tokens isn't a citation; ~700 + neighbour expansion beats it. |
| Hand-rolled `migrate_*.py` scripts | Alembic. |
| `$regex` exposed to the LLM, unguarded `strptime` date parsing | Allow-listed operators, validated filters, drop-with-warning on bad input. |
| Dead cost-tracking module | Replaced by cost tracking that is actually called on every response. |
| APScheduler in-process | A `jobs` table and a worker loop. |
| Display-vs-context split (cards showing pre-rerank, answer using post-rerank) | `sources[]` contains exactly what was in the LLM's context. Showing the user a different set than the model saw is a trust bug. |
| README that drifts from the code | Constants live in `settings.py`; the README quotes it and the eval suite tests it. |
```
