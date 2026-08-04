#!/usr/bin/env bash
# pg_dump of the postgres service + a copy of the blob volume, timestamped.
# Usage: ./scripts/backup.sh            -> backups/<timestamp>/db.dump + blobs/
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="backups/$STAMP"
mkdir -p "$OUT"

echo "dumping postgres…"
docker compose exec -T postgres pg_dump -U "${POSTGRES_USER:-prorag}" -d "${POSTGRES_DB:-prorag}" -Fc > "$OUT/db.dump"

echo "copying blobs…"
if docker compose exec -T api test -d /app/blobs 2>/dev/null; then
  docker compose exec -T api tar -C /app/blobs -czf - . > "$OUT/blobs.tar.gz"
else
  API_ID="$(docker compose ps -q api)"
  docker cp "$API_ID:/app/blobs" "$OUT/blobs"
fi

echo "backup written to $OUT"
echo "restore with: ./scripts/restore.sh $OUT"
