# ProRag — all 6 phases complete

Text in, cited answer out. See `ARCHITECTURE.md` for the full design and phased plan.

Phase 1 scope: PDF/txt/md ingestion (PyMuPDF only), fixed ~700-token chunking with
page numbers, embeddings + answering via LiteLLM, vector-only search (pgvector
cosine), non-streaming `/chat` with `[Sn]` citations, and `/files/{doc_id}/original`.

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

## API surface (Phase 1 subset of §6)

| Method | Path | Notes |
|---|---|---|
| `POST` | `/ingest` | multipart `file`, `collection?` → `202 {doc_id, status, duplicate_of?}` |
| `POST` | `/chat` | `{message}` → `{answer, sources[]}` |
| `GET` | `/files/{doc_id}/original` | serves the stored blob |
| `GET` | `/healthz` | liveness |

## What's deliberately not here yet

A real `jobs`/SKIP-LOCKED worker queue is still later work (§8) — ingestion
stays synchronous inline in the request handler, now with a retry-with-backoff
around the embed step and a `status='failed'` path instead of a 500.

## Operations (Phase 5)

- **Auth**: off by default (`AUTH_ENABLED=false`). Turn it on and mint a key:
  ```bash
  uv run python scripts/create_api_key.py --name "my laptop"
  # then: curl -H "Authorization: Bearer <printed key>" http://localhost:8000/documents
  ```
- **Cost cap**: every planner/answer/embedding call is priced (`litellm.completion_cost()`,
  falling back to `FALLBACK_PRICE_PER_1K_USD` for models litellm doesn't price) and
  logged to the `usage` table. Once today's summed cost reaches `DAILY_COST_CAP_USD`,
  `/chat` and `/chat/stream` return `429` before touching the LLM.
- **Feedback**: `POST /feedback {"message_id", "rating": "up"|"down", "comment"?}`.
- **Health**: `/healthz` is liveness; `/readyz` additionally runs `SELECT 1` against Postgres.

## Deploy on a VPS

A $6/mo 1-2 vCPU box is enough for the target scale (§1). Steps:

```bash
# on the VPS
git clone <this repo> && cd ragPro
cp .env.example .env   # fill in provider key(s), set AUTH_ENABLED=true, pick a real DAILY_COST_CAP_USD

docker compose up -d postgres
docker compose run --rm api alembic upgrade head
uv run python scripts/create_api_key.py --name prod   # save the printed key

docker compose up -d
```

This brings up `postgres`, `api` (port 8000, not exposed publicly beyond the compose
network once Caddy is in front), and `caddy` (ports 80/443) reverse-proxying to `api`.
Edit `Caddyfile`: replace the `:80 { ... }` block with your real domain
(`example.com { reverse_proxy api:8000 }`) once DNS points at the box — Caddy fetches
and renews the TLS cert automatically, no extra config. Blobs persist on the host at
`./blobs`; Postgres data lives in the `pgdata` named volume.

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

All 6 phases from `ARCHITECTURE.md` §8 are implemented: MVP ingestion + citations
(Phase 1), hybrid retrieval + rerank (Phase 2), structured documents/tables (Phase 3),
SSE streaming + the pdf.js viewer (Phase 4), auth/cost-cap/feedback/health (Phase 5),
and golden-set evaluation + a tuning sweep + CI (Phase 6, this section). Optional
Phase 7 ideas (query decomposition, HyDE, per-collection embeddings, ColBERT) remain
speculative, per §8, until the eval numbers here say otherwise.
