# Universal RAG Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn ProRag into a universal-document RAG package that ingests any text/scan document, installs in one command, and runs production-hardened out of the box.

**Architecture:** Three independent workstreams on the existing stack: (A) ingestion universality — OCR for scanned PDFs/images plus re-ingesting documents that missed the section-aware chunker; (B) one-command Docker Compose setup with auto .env, migrations, admin seeding, health output; (C) production hardening — backups, resource limits, deployment docs. No new runtime components.

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy async / asyncpg, docling 2.117 (OCR via RapidOcrOptions), Docker Compose (paradedb + api + caddy), Next.js 16 web-next, bash + PowerShell setup scripts.

## Global Constraints

- **No new test files.** The sole developer does not run tests (standing directive). Verification steps are run-based commands with expected output; the existing pytest suite must stay green wherever a task touches code it covers.
- Config lives in `prorag/settings.py` only; env vars flow through pydantic-settings from `.env`.
- Frontend file-picker `accept` in `web-next/components/top-bar.tsx` must mirror `ALLOWED_SUFFIXES` in `prorag/ingest/router.py` (known coupling).
- Docling stays a lazy import (probed via `find_spec`, imported once on first parse — torch cost).
- Python ≥3.12, line length 119 (ruff), repo doc style: module docstrings, `When called / What / Returns` triplets.
- Every task ends with a commit.

---

## Task 1: Floor-only crop, junk-doc removal, commit in-flight retrieval fixes

Completes ADR 0002 (flatness guard is already implemented in `rerank.py`/`settings.py`, uncommitted) and lands it with the crop change in one commit.

**Files:**
- Modify: `prorag/settings.py` (crop fields)
- Modify: `prorag/retrieve/crop.py` (remove `score_gap` param and its use)
- Modify: `prorag/operations/retrieval.py` (call site)
- Modify: `tests/test_retrieval.py` (existing crop tests passing `score_gap` — update calls only)
- Modify: `prorag/retrieve/rerank.py` (already edited — include in commit)
- Modify: `docs/adr/0002-flatness-guard-and-floor-only-crop.md` (already written — include)

**Interfaces:**
- Consumes: `settings.crop_score_floor` (new default `0.02`), `settings.rerank_flat_spread` (`0.03`)
- Produces: `crop_context(hits, *, min_docs, max_docs, max_chunks_per_doc, score_floor, token_budget, min_chars)` — no `score_gap`; `rerank()` unchanged signature

- [ ] **Step 1: Crop — drop the dynamic gap**

In `prorag/retrieve/crop.py`, delete the `score_gap` parameter from `crop_context` and replace the floor computation:

```python
def crop_context(
    hits: list[dict],
    *,
    min_docs: int = 3,
    max_docs: int = 12,
    max_chunks_per_doc: int = 3,
    score_floor: float = 0.02,
    token_budget: int = 6000,
    min_chars: int = 150,
) -> list[dict]:
    """... (docstring: floor-only — no dynamic gap; a single erratic high
    score must never starve the tail, ADR 0002)"""
```

and replace:

```python
    top_score = pool[0]["score"]
    floor = max(top_score - score_gap, score_floor)
    kept = [h for h in pool if h["score"] >= floor]
```

with:

```python
    kept = [h for h in pool if h["score"] >= score_floor]
```

- [ ] **Step 2: Settings — floor 0.02, remove the gap**

In `prorag/settings.py`:

```python
    crop_min_docs: int = 3
    crop_max_docs: int = 12
    crop_max_chunks_per_doc: int = 3  # sections of one PDF all answerable
    # Floor-only crop (ADR 0002): everything below an absolute relevance bar
    # is cut; no dynamic top-score gap — a single erratic spike must never
    # starve the rest of the context.
    crop_score_floor: float = 0.02
    crop_token_budget: int = 6000
```

Delete `crop_score_gap: float = 0.15  # dynamic floor = max(top_score - gap, crop_score_floor)`.

- [ ] **Step 3: Call site**

In `prorag/operations/retrieval.py`, remove `score_gap=settings.crop_score_gap,` from the `crop_context(...)` call.

- [ ] **Step 4: Existing tests stay green**

In `tests/test_retrieval.py`, find every `crop_context(...)` call that passes `score_gap=` and delete that argument. Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -q
```

Expected: all pass (crop assertions still hold — the floor logic they test is unchanged).

- [ ] **Step 5: Delete the junk document**

Find and remove the upload-test artifact ("Speed Check …") — chunks, document row, and blob file:

```bash
.venv/Scripts/python.exe - <<'EOF'
import asyncio
from sqlalchemy import select, text
from prorag.db import SessionLocal
from prorag.models import Document

async def main():
    async with SessionLocal() as s:
        docs = (await s.execute(select(Document).where(Document.filename.ilike("speed check%")))).scalars().all()
        for d in docs:
            print("deleting", d.filename, d.blob_path)
            await s.delete(d)
        await s.commit()

asyncio.run(main())
EOF
```

Also remove the matching blob file under `blobs/` (path printed above).

- [ ] **Step 6: Verify the failing case is fixed**

Run the stream probe for both person queries:

```bash
.venv/Scripts/python.exe - <<'EOF'
import json, httpx
for q in ["tell me about neetigy shahs skills", "tell me about neetigya shah and his projects"]:
    with httpx.stream("POST", "http://127.0.0.1:8000/chat/stream", json={"message": q}, timeout=180) as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: ") and ev == "sources":
                for s in json.loads(line[6:]):
                    tag = "RESUME" if s["doc_id"].startswith("f8150f10") else ("AIEFS" if s["doc_id"].startswith("3be5268e") else "OTHER")
                    print(q[:25], "S%d" % s["n"], tag, "p.%s" % s.get("page"), round(s.get("score", 0), 3))
                break
EOF
```

Expected: no `OTHER` chunk in any context; resume chunks present for both queries; resume skills chunk in the top-5 of the skills query.

- [ ] **Step 7: Commit**

```bash
git add prorag/settings.py prorag/retrieve/crop.py prorag/operations/retrieval.py prorag/retrieve/rerank.py tests/test_retrieval.py docs/adr CONTEXT.md
git commit -m "retrieval: floor-only crop (ADR 0002), flatness guard, remove junk test doc"
```

---

## Task 2: OCR for scanned PDFs and image uploads

**Files:**
- Modify: `prorag/ingest/parse.py` (image + OCR paths)
- Modify: `prorag/ingest/router.py` (`ALLOWED_SUFFIXES`)
- Modify: `web-next/components/top-bar.tsx` (`accept` attribute)

**Interfaces:**
- Consumes: `sniff_mime(data)` (extend), `Element`/`StructuredChunk` from `ingest/chunk.py`
- Produces: image formats flow through the same `ingest_bytes` parse → chunk → embed → store pipeline with page/bbox anchors

- [ ] **Step 1: Extend the signature + suffix lists**

In `prorag/ingest/parse.py`, add magic bytes for the image families:

```python
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "zip"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
]
```

In `prorag/ingest/router.py`:

```python
ALLOWED_SUFFIXES = (".pdf", ".txt", ".md", ".docx", ".pptx", ".csv", ".xlsx", ".tsv",
                    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp")
```

In `web-next/components/top-bar.tsx`, mirror it:

```tsx
accept=".pdf,.txt,.md,.docx,.pptx,.csv,.xlsx,.tsv,.png,.jpg,.jpeg,.tiff,.tif,.webp"
```

- [ ] **Step 2: Route images through docling with OCR**

In `prorag/ingest/parse.py`, add an image parse path beside the docling PDF path. Probe the installed API first (docling 2.117):

```bash
.venv/Scripts/python.exe -c "from docling.datamodel.pipeline_options import RapidOcrOptions; print('ok')"
```

Then add:

```python
def _get_image_converter():
    """docling converter for raster formats: OCR turned on (rapidocr).
    Lazy: docling import is torch-heavy; same idiom as _get_docling_converter."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, FormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.ocr_options = RapidOcrOptions()
    return DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={InputFormat.IMAGE: FormatOption(pipeline_options=opts)},
    )
```

`parse_document_bytes` (or the equivalent router in `parse.py`) dispatches: `sniff_mime` returning `image/*` → image converter on a temp file; the converter's `.document` yields docling elements that feed the existing `chunk_elements()` path (page = 1, bbox from element boxes). If docling or OCR is unavailable, degrade to a single plain-text chunk carrying no bbox (graceful, same contract as the PDF fallback).

- [ ] **Step 3: Verify with a real scan**

Upload a scanned PDF (image-only pages) and a `.png` photo through the UI. Then:

```bash
.venv/Scripts/python.exe - <<'EOF'
import asyncio
from sqlalchemy import text
from prorag.db import SessionLocal
async def main():
    async with SessionLocal() as s:
        rows = (await s.execute(text("""
            SELECT d.filename, count(c.id) chunks, count(c.bbox) with_bbox
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
            WHERE d.filename ILIKE ANY (ARRAY['%scan%', '%png%', '%jpg%', '%jpeg%', '%tiff%', '%webp%'])
            GROUP BY d.filename
        """))).all()
        print(rows)
asyncio.run(main())
EOF
```

Expected: the scan/image document exists with chunks; text is searchable (a keyword query returns it); bbox count > 0 for docling-OCR'd elements where available.

- [ ] **Step 4: Commit**

```bash
git add prorag/ingest/parse.py prorag/ingest/router.py web-next/components/top-bar.tsx
git commit -m "ingest: OCR for scanned PDFs and images (docling rapidocr)"
```

---

## Task 3: Re-ingest documents that missed the section-aware chunker

The aiefs book (216 chunks, every one spanning pages, mid-sentence starts) proves it was ingested via the `chunk_pages` fallback. `chunk_elements()` — page- and H1/H2-aware, with breadcrumbs and bboxes — already exists and produces the exact anchors the product wants. This task finds fallback-ingested documents and re-ingests their blobs.

**Files:**
- Create: `scripts/reingest.py`

**Interfaces:**
- Consumes: `prorag.ingest.core.ingest_bytes(session, data, filename, collection)`, `prorag.models.Document.blob_path`
- Produces: documents re-ingested through docling; chunk sets page-aligned with `heading_path` and `bbox`

- [ ] **Step 1: Write the re-ingest script**

```python
"""Re-ingest documents whose chunks were cut by the fallback chunker.

Detects them structurally: fallback chunks can span pages and carry no
heading_path (docling chunks never cross a page). Re-runs each blob through
ingest_bytes() so docling + chunk_elements() produce section-aware chunks.
Safe to re-run: documents are replaced, chunks cascade-deleted.

Usage: python scripts/reingest.py [--doc-id UUID ...] [--all]
"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import func, select

from prorag.db import SessionLocal
from prorag.ingest.core import ingest_bytes
from prorag.models import Chunk, Document


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    async with SessionLocal() as s:
        stmt = select(Document)
        if args.doc_id:
            stmt = stmt.where(Document.id.in_(args.doc_id))
        elif not args.all:
            # Default: only fallback-chunked documents — any chunk whose
            # page_start != page_end or whose heading_path is empty.
            stmt = stmt.join(Chunk).group_by(Document.id).having(
                func.bool_or(Chunk.page_start != Chunk.page_end)
                | func.bool_or(func.cardinality(Chunk.heading_path) == 0)
            )
        docs = (await s.execute(stmt)).scalars().unique().all()
        print(f"re-ingesting {len(docs)} documents")
        for d in docs:
            data = Path(d.blob_path).read_bytes()
            print(f"  {d.filename} ({len(data)} bytes)")
            await ingest_bytes(s, data, d.filename, d.collection)
            await s.commit()

asyncio.run(main())
```

(If `cardinality` differs in the installed Postgres, use `func.coalesce(func.array_length(Chunk.heading_path, 1), 0) == 0`.)

- [ ] **Step 2: Run it on the book**

```bash
.venv/Scripts/python.exe scripts/reingest.py --doc-id 3be5268e-aa09-469e-a27c-b7a5b6c38ff2
```

Expected: the aiefs document is replaced; chunks now single-page (or tightly page-bounded), `heading_path` populated, bbox present.

- [ ] **Step 3: Verify a multi-page topic retrieves cleanly**

```bash
.venv/Scripts/python.exe - <<'EOF'
import asyncio
from sqlalchemy import text
from prorag.db import SessionLocal
async def main():
    async with SessionLocal() as s:
        rows = (await s.execute(text("""
            SELECT count(*) FILTER (WHERE page_start <> page_end) multi,
                   count(*) FILTER (WHERE cardinality(heading_path) > 0) with_heading
            FROM chunks WHERE doc_id = '3be5268e-aa09-469e-a27c-b7a5b6c38ff2'
        """))).one()
        print(rows)
asyncio.run(main())
EOF
```

Expected: `multi = 0`, `with_heading > 0`. Then re-run the Task 1 stream probe on "explain matrix multiplication intuition" — citations should land on exact pages (p.75–78 region) with coherent chunk boundaries.

- [ ] **Step 4: Commit**

```bash
git add scripts/reingest.py
git commit -m "scripts: re-ingest fallback-chunked documents through docling"
```

---

## Task 4: One-command Docker setup

**Files:**
- Create: `setup.sh`
- Modify: `docker-compose.yml` (api healthcheck)
- Modify: `docker-entrypoint.sh` (idempotent admin seeding, optional)
- Modify: `start.ps1` (delegate to the same flow; keep for contributors)

**Interfaces:**
- Consumes: `.env.example`, `scripts/create_admin.py`, `prorag doctor`
- Produces: `setup.sh` exits 0 with the app healthy at `http://localhost`, URLs + credentials printed

- [ ] **Step 1: Add the api healthcheck**

In `docker-compose.yml`, under `api:`:

```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 20s
```

- [ ] **Step 2: Write setup.sh**

```bash
#!/usr/bin/env bash
# One-command installer: docker check -> .env generation -> compose up ->
# migrations (entrypoint) -> admin seed -> health summary.
set -euo pipefail
cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then re-run ./setup.sh" >&2
  exit 1
fi

if [ ! -f .env ]; then
  echo "Generating .env from .env.example…"
  POSTGRES_PASSWORD="$(openssl rand -hex 16)"
  SESSION_SECRET="$(openssl rand -hex 32)"
  sed "s/POSTGRES_PASSWORD=prorag/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/; s/^SESSION_COOKIE_SECURE=.*/SESSION_COOKIE_SECURE=false/" .env.example > .env
  echo "SESSION_SECRET=$SESSION_SECRET" >> .env
fi

docker compose up -d --build
echo "Waiting for postgres…"
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-prorag}" >/dev/null 2>&1; do sleep 2; done
echo "Waiting for api…"
until docker compose exec -T api python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)" >/dev/null 2>&1; do sleep 2; done

echo "Seeding admin…"
docker compose exec -T api python scripts/create_admin.py --email "${ADMIN_EMAIL:-admin@example.com}" || true

echo ""
echo "ProRag is up: http://localhost"
echo "Backend docs: http://localhost/api/docs"
docker compose exec -T api python -m prorag.doctor
```

(`start.ps1` keeps its role for contributors; add a comment pointing at `setup.sh` / WSL.)

- [ ] **Step 3: Verify on a clean slate**

```bash
docker compose down -v && ./setup.sh
```

Expected: `.env` generated with random password, postgres + api + caddy healthy, admin credentials printed, `prorag doctor` reports no FAILs, `http://localhost` serves the web UI, upload + query work.

- [ ] **Step 4: Commit**

```bash
git add setup.sh docker-compose.yml start.ps1
git commit -m "ops: one-command docker setup (setup.sh)"
```

---

## Task 5: QUICKSTART.md and .env.example polish

**Files:**
- Create: `QUICKSTART.md`
- Modify: `.env.example` (working defaults; secrets documented as overridable)

- [ ] **Step 1: Write QUICKSTART.md**

Three sections: (1) prerequisites (Docker Desktop), (2) `./setup.sh` walkthrough with the printed output annotated, (3) day-two commands (`docker compose logs -f api`, `docker compose exec postgres pg_dump …`, admin reset via `create_admin.py --reset`). Keep it under 60 lines — the setup script already prints everything.

- [ ] **Step 2: .env.example — make defaults runnable**

Ensure every unset-with-default variable actually works out of the box: `POSTGRES_PASSWORD` default `prorag` (setup.sh overrides), `SESSION_COOKIE_SECURE=false` for local http, `OPENROUTER_API_KEY=` empty (doctor warns, doesn't fail), `AUTH_ENABLED=false`. Remove any commented-out stale options.

- [ ] **Step 3: Commit**

```bash
git add QUICKSTART.md .env.example
git commit -m "docs: quickstart and runnable env example"
```

---

## Task 6: Backups and resource limits

**Files:**
- Create: `scripts/backup.sh`, `scripts/restore.sh`
- Modify: `docker-compose.yml` (resource limits)

- [ ] **Step 1: Write backup.sh**

```bash
#!/usr/bin/env bash
# pg_dump of the postgres service + copy of the blob volume, timestamped.
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="backups/$STAMP"
mkdir -p "$OUT"
docker compose exec -T postgres pg_dump -U "${POSTGRES_USER:-prorag}" -d "${POSTGRES_DB:-prorag}" -Fc > "$OUT/db.dump"
docker compose exec -T api tar -C /app/blobs -czf - . > "$OUT/blobs.tar.gz" 2>/dev/null || docker cp "$(docker compose ps -q api)":/app/blobs "$OUT/blobs"
echo "backup written to $OUT"
```

And `restore.sh` (pg_restore into a fresh volume; blobs extracted back). Both must be idempotent and safe to re-run.

- [ ] **Step 2: Resource limits**

In `docker-compose.yml`, add to postgres and api:

```yaml
    deploy:
      resources:
        limits:
          memory: 2g
```

(postgres) and `memory: 4g` (api — docling/torch parses in-process).

- [ ] **Step 3: Verify a round trip**

```bash
./scripts/backup.sh && docker compose down && docker compose up -d && ./scripts/restore.sh
```

Expected: documents count unchanged (`GET /api/stats` returns the same number), a query still answers.

- [ ] **Step 4: Commit**

```bash
git add scripts/backup.sh scripts/restore.sh docker-compose.yml
git commit -m "ops: pg_dump/restore scripts and compose resource limits"
```

---

## Task 7: DEPLOYMENT.md — managed Postgres, blob storage, secrets

**Files:**
- Create: `DEPLOYMENT.md`

- [ ] **Step 1: Write DEPLOYMENT.md**

Document the migration path with no code changes required (all config-driven):

1. **Managed Postgres**: set `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db` in `.env`; the provider must support `pgvector` and the FTS/BM25 features used (ParadeDB or Postgres + pgvector + a `websearch_to_tsquery`-compatible setup); run `alembic upgrade head` (or the entrypoint's auto-migrate) once.
2. **Blob storage**: today blobs live on a volume (`BLOB_DIR=./blobs`). For object storage, the existing S3 connector already stores/retrieves external objects — document that a `BLOB_DIR` pointing at a mounted object-storage filesystem (rclone/s3fs) is the supported path; no code change.
3. **Secrets**: `.env` permissions (`chmod 600`), rotate `POSTGRES_PASSWORD`/`SESSION_SECRET`, `SESSION_COOKIE_SECURE=true` behind HTTPS, Caddy TLS via the existing `Caddyfile`.
4. **Backups**: point Task 6's scripts at the managed instance (`pg_dump` remote) — note `-h` host flag.

- [ ] **Step 2: Cross-check every claim against the code** (DATABASE_URL parsing in `prorag/db.py`, `BLOB_DIR` in `settings.py`, Caddyfile) — the doc must be executable by someone who has never seen the repo.

- [ ] **Step 3: Commit**

```bash
git add DEPLOYMENT.md
git commit -m "docs: deployment guide (managed postgres, blobs, secrets)"
```

---

## Task 8: README feature close-out

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the feature list**

Add: OCR ingestion (scanned PDFs, images), one-command setup (`./setup.sh`), backup/restore scripts, deployment guide link, retrieval quality notes (hosted cross-encoder rerank, flatness guard, floor-only crop). Remove any stale references to the free-tier reranker or the legacy `web/` UI.

- [ ] **Step 2: Verify no stale claims** — grep the README for "free tier", "rerank_llm", "web/chat.js"; delete leftovers.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: feature close-out"
```

---

## Self-review notes

- Spec coverage: OCR (Task 2) ✓, current formats unchanged ✓, one-command setup (Tasks 4–5) ✓, hardening-only scale (Tasks 6–7) ✓, floor-only crop + junk doc (Task 1) ✓, ADRs/glossary (Task 1 commit) ✓.
- No placeholders: every code step contains real code; doc steps name the exact files and claims to make.
- Type consistency: `crop_context` signature change is threaded through Task 1 steps 1–4; `ALLOWED_SUFFIXES` ↔ `accept` coupling is called out in both Task 2 steps; `ingest_bytes(session, data, filename, collection)` matches `prorag/ingest/core.py`'s existing call in `router.py`.
