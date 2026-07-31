"""CLI helper: mints a fresh API key and prints it once (§6, §8 Phase 5).
Only the hash is ever stored — if you lose the printed key, make a new one.

Usage:
    python scripts/create_api_key.py [--name label] [--collection safety]
"""

import argparse
import asyncio

from prorag.auth import hash_key, new_api_key
from prorag.db import SessionLocal
from prorag.models import ApiKey


async def _create(name: str | None, collection: str | None) -> None:
    raw = new_api_key()
    async with SessionLocal() as session:
        session.add(ApiKey(key_hash=hash_key(raw), name=name, collection=collection))
        await session.commit()
    print("API key created — save it now, it will not be shown again:")
    print(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None, help="human-readable label")
    parser.add_argument("--collection", default=None, help="scope this key to one collection (default: unscoped)")
    args = parser.parse_args()
    asyncio.run(_create(args.name, args.collection))


if __name__ == "__main__":
    main()
