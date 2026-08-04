# Deployment guide

ProRag is one Compose stack (postgres + api + caddy) that runs on a single
box out of the box. This guide documents the production-hardening path —
**no code changes needed, all configuration** (ADR 0003: hardening only, no
new components).

## 1. Managed Postgres

Set `DATABASE_URL` in `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
```

Requirements for the provider:

- **pgvector** extension (chunk embeddings) — enable it in the provider UI or
  `CREATE EXTENSION IF NOT EXISTS vector;`
- **FTS/BM25**: the keyword arm uses `websearch_to_tsquery` + `ts_rank_cd`
  (standard Postgres FTS — works everywhere). If the provider offers
  ParadeDB's BM25 (`@@@`, `paradedb.score`), the keyword arm upgrades itself
  automatically; plain Postgres is fine.

Run migrations once (the API's entrypoint auto-migrates on boot, or):

```bash
docker compose run --rm api alembic upgrade head
```

Blobs stay on the local volume — see next section for object storage.

## 2. Blob storage

Blobs live at `BLOB_DIR` (`./blobs` by default, a volume in Compose). For
object storage, mount an object-storage filesystem at `BLOB_DIR` — the API
only ever reads/writes files through that path:

```bash
# example: rclone mount of an S3 bucket
rclone mount my-bucket:prorag-blobs /data/blobs --daemon
BLOB_DIR=/data/blobs
```

The S3 **connector** (admin UI) is the built-in way to pull documents from
any S3-compatible endpoint into ProRag — the mount above is for ProRag's own
stored blobs, not for syncing.

## 3. Secrets

- `.env` permissions: `chmod 600 .env`
- Rotate `POSTGRES_PASSWORD` and `SESSION_SECRET` (compose healthcheck +
  session signing both depend on them)
- Behind HTTPS set `SESSION_COOKIE_SECURE=true`
- The OpenRouter key (`OPENROUTER_API_KEY`) is the only paid-service secret;
  everything else is local

## 4. TLS

The bundled Caddyfile terminates TLS for a configured domain. Behind a
reverse proxy (nginx/cloudflare), point Caddy at the proxy or disable it and
serve the `api` + `caddy` ports yourself. `SESSION_COOKIE_SECURE=true`
requires HTTPS at the edge.

## 5. Backups

`./scripts/backup.sh` / `./scripts/restore.sh` dump postgres (custom format)
and the blob volume. For a managed database, run the equivalent remotely:

```bash
pg_dump "$DATABASE_URL" -Fc -f db.dump
```

Restore into a fresh database with `pg_restore --no-owner`.

## 6. Sizing

| Resource | Baseline | Why |
|---|---|---|
| CPU | 2 cores | docling layout model during ingest; rerank/OCR/embeddings are all API calls |
| RAM | 4 GB | docling + Postgres; 16 GB comfortably handles 400+ page PDFs |
| Disk | ~2× corpus | blobs + Postgres (chunks + vectors) |
