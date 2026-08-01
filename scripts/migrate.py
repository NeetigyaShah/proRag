"""Auto-migrate entrypoint (#20): `alembic upgrade head` behind a Postgres
advisory lock, so N replicas booting at once don't race each other's DDL.

Run directly (`python scripts/migrate.py`) or via docker-entrypoint.sh before
uvicorn starts. Exits 0 with one clear line on success, 1 with one clear line
on failure — a container that fails to migrate must never come up half-applied.

The lock connection is a plain asyncpg connection (asyncpg is already a
dependency via SQLAlchemy's async driver — no new sync driver needed). Alembic
itself is invoked programmatically via `alembic.command.upgrade`, which this
repo's alembic/env.py runs against its own `asyncio.run(...)` — that can't
nest inside the lock connection's already-running loop, so it's dispatched to
a worker thread via `asyncio.to_thread`, which has no event loop of its own.
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from alembic.config import Config

from alembic import command
from prorag.settings import settings

ROOT = Path(__file__).resolve().parent.parent
ADVISORY_LOCK_KEY = "prorag_migrate"


def _sync_dsn(url: str) -> str:
    # asyncpg.connect() wants a plain postgres DSN, not SQLAlchemy's
    # "+asyncpg" driver-qualified URL.
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _run_alembic_upgrade() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(cfg, "head")


async def _migrate() -> None:
    conn = await asyncpg.connect(_sync_dsn(settings.database_url), timeout=settings.db_connect_timeout)
    try:
        await conn.execute("SELECT pg_advisory_lock(hashtext($1))", ADVISORY_LOCK_KEY)
        try:
            await asyncio.to_thread(_run_alembic_upgrade)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", ADVISORY_LOCK_KEY)
    finally:
        await conn.close()


def main() -> None:
    try:
        asyncio.run(_migrate())
    except Exception as exc:
        print(f"migrate: FAILED — {exc}", file=sys.stderr)
        sys.exit(1)
    print("migrate: OK — database is at head")
    sys.exit(0)


if __name__ == "__main__":
    main()
