#!/usr/bin/env bash
# Restore a backup created by backup.sh into a fresh stack.
# Usage: ./scripts/restore.sh backups/<timestamp>
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:?usage: ./scripts/restore.sh backups/<timestamp>}"
[ -f "$SRC/db.dump" ] || { echo "no db.dump in $SRC" >&2; exit 1; }

echo "dropping and recreating the database…"
docker compose exec -T postgres psql -U "${POSTGRES_USER:-prorag}" -d postgres -c "DROP DATABASE IF EXISTS ${POSTGRES_DB:-prorag};"
docker compose exec -T postgres psql -U "${POSTGRES_USER:-prorag}" -d postgres -c "CREATE DATABASE ${POSTGRES_DB:-prorag};"

echo "restoring db.dump…"
docker compose exec -T postgres pg_restore -U "${POSTGRES_USER:-prorag}" -d "${POSTGRES_DB:-prorag}" --no-owner --role="${POSTGRES_USER:-prorag}" < "$SRC/db.dump"

echo "restoring blobs…"
API_ID="$(docker compose ps -q api)"
if [ -f "$SRC/blobs.tar.gz" ]; then
  docker compose exec -T api rm -rf /app/blobs && docker compose exec -T api mkdir -p /app/blobs
  docker compose exec -T api tar -xzf - -C /app/blobs < "$SRC/blobs.tar.gz"
else
  docker cp "$SRC/blobs/." "$API_ID:/app/blobs/"
fi

echo "restored from $SRC"
