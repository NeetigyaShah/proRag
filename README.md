# ProRag — self-hostable RAG platform

Text in, cited answer out. See `ARCHITECTURE.md` for the full design and current status.

Core scope: PDF/txt/md/DOCX/PPTX/CSV/XLSX ingestion, structure-aware ~700-token chunking,
hybrid retrieval (vector + BM25 + structured), RRF fusion, SSE streaming with `[Sn]`
citations, a pdf.js viewer that opens the cited page with the sentence highlighted, and a
multi-user layer — identity (OIDC + local accounts), ACL enforcement, per-user budgets,
an S3 connector, and an admin API. The one open build item is the admin dashboard UI (#25).

## Setup

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (or point EMBED_MODEL/ANSWER_MODEL at any LiteLLM-supported provider)

docker compose up -d postgres
uv sync   # or: pip install -e ".[dev]"

uv run alembic upgrade head
uv run uvicorn prorag.main:app --reload
```

## Try it

```bash
curl -F "file=@some.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "What does the document say about X?"}'
```

Or run the manual end-to-end script:

```bash
python scripts/smoke.py some.pdf "What does the document say about X?"
```

## Tests

The pure-function smoke test (chunking + citation resolution, no DB/LLM needed):

```bash
uv run pytest tests/test_chunk_citations.py
```

## API surface (summary — full list in ARCHITECTURE.md §6)

| Method | Path | Notes |
|---|---|---|
| `POST` | `/ingest` | multipart `file`, `collection?` → `202 {doc_id, status, duplicate_of?}` |
| `POST` | `/chat` | `{message}` → `{answer, sources[], budget_warning?}` |
| `POST` | `/chat/stream` | SSE: status → sources → budget? → tokens → meta → done |
| `GET` | `/search` | `?q&k` — retrieval without the LLM (debugging endpoint) |
| `GET` | `/files/{doc_id}/original` | serves the stored blob; 404 when invisible to the caller |
| `GET` | `/tables/{table_id}/rows` | JSONB rows backing the CSV citation grid |
| `POST` | `/feedback` | `{message_id, rating: up\|down, comment?}` |
| `POST` | `/auth/login` · `/auth/logout` | local accounts → `prorag_session` HttpOnly cookie |
| `GET` | `/auth/oidc/login` · `/auth/oidc/callback` | Authlib OIDC; 404 when unconfigured |
| `GET/POST` | `/connectors` · `GET/PATCH/DELETE /connectors/{id}` | S3 connector CRUD (admin) |
| `POST` | `/connectors/{id}/sync` | `?full=` — incremental poll or mandatory sweep |
| `GET` | `/admin/documents` · `/admin/documents/{id}/access` | documents + ACL provenance (admin) |
| `POST/GET` | `/admin/rules` · `PATCH/DELETE /admin/rules/{id}` | access rules (admin) |
| `POST` | `/admin/rules/{id}/preview` · `/confirm` | sample+count, then freeze + grant |
| `GET` | `/admin/rules/{id}/grants` | audit feed |
| `GET` | `/admin/users` · `PATCH /admin/users/{id}` | users + cap override + spend |
| `GET` | `/admin/users/{id}/visible-docs` | reverse ACL |
| `POST/GET` | `/admin/groups` · `PATCH/DELETE /admin/groups/{id}` | local groups only |
| `POST/DELETE` | `/admin/groups/{id}/members/{user_id}` | membership |
| `GET` | `/admin/usage` | `?window=7d` by user/day/model |
| `GET` | `/healthz` · `/readyz` | liveness / (db + migrations) |

## Known gaps (see ARCHITECTURE.md §8 for the full list)

- **Admin dashboard UI** — the backing API is complete (#24); the four views in `web-next`
  (documents, rules, people, usage) are the one open issue (#25). To be designed via Open Design.
- **Login/logout UI** — sessions work via `/auth/login` (curl/scripts) but there is no login page yet.
- **Background ingestion** — no jobs/worker queue; ingestion runs inline in the request handler.
- **Connectors beyond S3** — SharePoint/Graph, Google Drive, Confluence, Notion are researched and
  sequenced but not built.
- **SCIM** — JIT user provisioning only; no group sync scheduler yet.

## Operations

- **Auth**: off by default (`AUTH_ENABLED=false`). Turn it on for a human-account install:
  ```bash
  uv run python scripts/create_admin.py --email you@corp.com   # prints the password once
  # then: POST /auth/login with email+password → prorag_session cookie (HttpOnly, SameSite=Lax)
  ```
  Or connect your IdP with the four OIDC settings (`OIDC_ISSUER`, `OIDC_CLIENT_ID`,
  `OIDC_CLIENT_SECRET`, `OIDC_REDIRECT_PATH`) — discovery handles the rest, groups claims
  seed membership. Machine access stays on bearer keys:
  ```bash
  uv run python scripts/create_api_key.py --name "my laptop"
  # then: curl -H "Authorization: Bearer <printed key>" http://localhost:8000/stats
  ```
- **Cost caps**: every planner/answer/embedding call is priced (`litellm.completion_cost()`,
  falling back to `FALLBACK_PRICE_PER_1K_USD`) and logged to the `usage` table with a
  `user_id`. Two layers: an install-wide hard cap (`DAILY_COST_CAP_USD` → `429` before
  touching the LLM) and a per-user soft cap (`USER_DAILY_CAP_USD`, default $1.00 — the
  request still runs but `/chat` carries a `budget_warning`; at `USER_HARD_CAP_MULTIPLIER`×
  it refuses with "you have used $X of $Y today, resets at midnight UTC"). Both windows
  are UTC. Admins override per user via `PATCH /admin/users/{id}`.
- **Connectors**: S3-compatible object storage (AWS, MinIO, R2) via the admin API. The
  scheduler polls every `CONNECTOR_POLL_SECONDS` (default 900 s) and runs a mandatory
  full-ID sweep every `CONNECTOR_SWEEP_HOURS` (default 24) — that sweep is the deletion/
  revocation signal. Manual trigger: `POST /connectors/{id}/sync?full=true`.
- **Admin API**: documents (with the stored `error` string), access rules (preview →
  confirm → grants + auto-admission of new matches), users/groups, reverse ACL, usage —
  all under `/admin/*`, admin-gated.
- **Smoke check**: `python -m prorag.doctor` — 8 checks (settings, db, migrations, blob
  dir, llm, embed, rerank, bm25), exit nonzero on any genuine FAIL, WARNs don't fail it.
- **Feedback**: `POST /feedback {"message_id", "rating": "up"|"down", "comment"?}`.
- **Health**: `/healthz` is liveness; `/readyz` additionally runs `SELECT 1` against Postgres.

## Deploy on a VPS

A $6/mo 1-2 vCPU box is enough for the target scale (§1). Steps:

```bash
# on the VPS
git clone <this repo> && cd ragPro
cp .env.example .env   # fill in provider key(s), set AUTH_ENABLED=true, pick a real DAILY_COST_CAP_USD

docker compose up -d postgres
uv run python scripts/create_admin.py --email you@corp.com   # local admin account, prints the password once
uv run python scripts/create_api_key.py --name prod          # optional: machine access, save the printed key

docker compose up -d   # api's entrypoint runs `alembic upgrade head` itself before serving
docker compose exec api python -m prorag.doctor   # day-one smoke check — OK/WARN/FAIL per setting
```

This brings up `postgres`, `api` (port 8000, not exposed publicly beyond the compose
network once Caddy is in front), and `caddy` (ports 80/443) reverse-proxying to `api`.
Edit `Caddyfile`: replace the `:80 { ... }` block with your real domain
(`example.com { reverse_proxy api:8000 }`) once DNS points at the box — Caddy fetches
and renews the TLS cert automatically, no extra config. Blobs persist in the `blobdata`
named volume (`docker volume inspect prorag_blobdata` for its host path — a compose
volume survives `docker compose down`, just not `down -v`); Postgres data lives in the
`pgdata` named volume the same way.

## Evaluation & tuning (Phase 6)

`prorag/eval/golden.jsonl` is a checked-in **template** — 8 placeholder rows in the
golden-set format `{question, expected_answer_contains[], expected_doc_ids[]?,
expected_source_titles[]?, notes}`. Replace them with real questions against your own
corpus before the numbers below mean anything.

```bash
# run the golden set through the live app (needs Postgres + an LLM key, same as /chat)
curl -X POST http://localhost:8000/eval/run
curl http://localhost:8000/eval/runs/1
```

Each run scores every question two ways, computes an aggregate, and persists both
(per-question JSONB + aggregate) to the `eval_runs` table:

- **Deterministic, always on** — no ragas needed: retrieval hit-rate (did an expected
  doc/title survive the crop?), answer keyword coverage (`expected_answer_contains`
  substrings found in the answer), citation validity (every `[Sn]` resolves in range).
- **ragas (optional)** — faithfulness, answer_relevancy, context_precision. `ragas` is
  deliberately *not* a hard dependency (heavy, and this repo's CI doesn't install it);
  `pip install ragas datasets` to turn it on. Without it, the aggregate includes
  `"ragas": "skipped (ragas not installed)"` and a one-line warning is logged — no error.

**Sweep**: `scripts/sweep.py` runs the golden set once per combination of
`rerank_top_n` / `crop_score_gap` / `crop_max_docs` / `structured_weight` (the grid in
`SWEEP_GRID`), overriding `settings` in-process — no code edits per run — and prints a
ranked table. Chunk `target_tokens` is *not* swept: it's an ingest-time decision
(re-chunking means re-ingesting), not a per-request knob.

```bash
uv run python scripts/sweep.py
uv run python scripts/sweep.py --golden path/to/your_golden.jsonl
```

**CI** (`.github/workflows/ci.yml`): lint (`ruff check` + `ruff format --check`) and
`pytest` (the pure-function suite — no DB/LLM needed) run on every push/PR. An
`eval-regression` job is included but commented out, since it needs a live Postgres
service container and a real LLM key as a repo secret; the workflow file documents
exactly what to uncomment and set to turn it on.

## Project status

**All original 6 phases are implemented** (MVP ingestion + citations, hybrid retrieval +
rerank, structured documents/tables, SSE streaming + the pdf.js viewer, auth/cost-cap/
feedback/health, golden-set evaluation + sweep + CI), **plus the product-ization arc**
(issues #1–#24, July–Aug 2026): identity + ACL schema and enforcement, OIDC + local
accounts + sessions, per-user budgets, packaging (auto-migrate entrypoint, `prorag doctor`,
blob volume), the S3 connector + polling scheduler, and the admin dashboard backing API.
Bug fixes shipped along the way: the dead HNSW index (#16), the cost-window UTC bug and
real streamed token usage (#13), and the reverse-proxy defects (#14).

**What remains:** the admin dashboard UI in `web-next` (#25, the one open issue — to be
designed via Open Design), a login UI, a background ingestion worker, connectors beyond
S3, and SCIM. Optional Phase 7 ideas (query decomposition, HyDE, per-collection
embeddings, ColBERT) remain speculative until the eval numbers say otherwise.
