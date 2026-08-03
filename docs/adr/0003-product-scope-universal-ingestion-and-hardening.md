# ADR 0003: Product scope — universal ingestion, one-command setup, hardening-only scale

- Status: accepted (2026-08-04)
- Deciders: Neetigya Shah

## Context

The product vision: a package that ingests *any* document, installs in a few
commands, and scales from personal use to production without redesign. The
corpus is currently resumes + books, but the target is arbitrary documents.

## Decision

Three scoping choices:

1. **Universal ingestion = current formats + OCR.** Supported: PDF, DOCX,
   PPTX, XLSX, CSV, TSV, TXT, MD (+ tables). New: scanned PDFs and images
   (`.png/.jpg/.jpeg/.tiff/.webp`) via docling's OCR pipeline. Deliberately
   **not** in scope: HTML/EPUB, URL fetching, email, audio.
2. **Setup = one-command Docker Compose.** A `setup.sh`/`setup.ps1`
   entrypoint: check Docker, generate `.env` (random secrets) if missing,
   `docker compose up -d`, wait for health, seed an admin, print URLs and a
   `prorag doctor` summary. No no-Docker local mode (start.ps1 stays for
   contributors).
3. **Scale = production hardening of the single stack, not new components.**
   Backups, health checks, resource limits, and documented migration paths
   (managed Postgres, blob storage to S3). No job queue, no worker
   processes, no multi-node serving. The config-flag scaling design is
   deferred unless a real deployment needs it.

## Consequences

- The ingestion surface stays within docling's proven formats + OCR.
- One documented path into the product; contributors keep start.ps1.
- Scaling means operating the same stack better, not re-architecting it.
