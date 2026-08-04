#!/usr/bin/env bash
# One-command installer: docker check -> .env generation -> compose up ->
# migrations (entrypoint) -> admin seed -> health summary.
#
# Usage: ./setup.sh            (Linux/macOS/WSL)
#        .\setup.ps1           (PowerShell — Windows)
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
  echo "  .env written (random POSTGRES_PASSWORD + SESSION_SECRET)"
fi

docker compose up -d --build

echo "Waiting for postgres…"
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-prorag}" >/dev/null 2>&1; do sleep 2; done

echo "Waiting for api (migrations run in the entrypoint)…"
until docker compose exec -T api python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)" >/dev/null 2>&1; do sleep 2; done

echo "Seeding admin…"
docker compose exec -T api python scripts/create_admin.py --email "${ADMIN_EMAIL:-admin@example.com}" || true

echo ""
echo "ProRag is up: http://localhost"
echo "Backend docs: http://localhost/api/docs"
echo ""
docker compose exec -T api python -m prorag.doctor || true
