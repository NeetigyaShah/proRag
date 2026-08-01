#!/bin/sh
set -e

python scripts/migrate.py

exec "$@"
