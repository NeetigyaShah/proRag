# Quickstart

ProRag is a self-hosted RAG platform: upload any document (PDF, DOCX, PPTX,
XLSX, CSV, TSV, TXT, MD — plus scanned PDFs and images via OCR), then ask
questions and get answers with cited page highlights.

## Requirements

- Docker Desktop (with the Linux engine)
- An OpenRouter API key with a little credit (rerank + OCR + embeddings + answers all bill against it — a few dollars lasts a long time)

## Install

```bash
# 1. Clone or copy the project, then from its root:
export OPENROUTER_API_KEY=sk-or-v1-...   # also add it to .env later

# 2. One command:
./setup.sh
```

That one command:

1. Checks Docker is running.
2. Generates `.env` from `.env.example` with a random `POSTGRES_PASSWORD` and `SESSION_SECRET` (your `OPENROUTER_API_KEY` still needs adding: `echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env`).
3. Builds and starts postgres + api + caddy.
4. Waits for the database and API (migrations run automatically in the API's entrypoint).
5. Seeds an admin user and prints the one-time password.
6. Runs `prorag doctor` so you can see every service check at a glance.

Done — the app is at **http://localhost**. Upload a document, ask a question,
click the citation chips to see the highlighted source pages.

## Day-two commands

| Task | Command |
|---|---|
| Logs | `docker compose logs -f api` |
| Stop / start | `docker compose down` / `docker compose up -d` |
| Reset admin password | `docker compose exec api python scripts/create_admin.py --email admin@example.com --reset` |
| Backup | `./scripts/backup.sh` (writes `backups/<timestamp>/`) |
| Restore | `./scripts/restore.sh backups/<timestamp>` |
| Health | `docker compose exec api python -m prorag.doctor` |

## Windows note

`setup.sh` needs a bash (WSL or Git Bash). On PowerShell, run the same steps
manually: copy `.env.example` to `.env`, set `POSTGRES_PASSWORD`/`SESSION_SECRET`,
`docker compose up -d`, then `docker compose exec api python scripts/create_admin.py --email admin@example.com`.
