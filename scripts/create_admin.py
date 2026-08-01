"""CLI helper: bootstraps a local admin user and prints the password once
(#19, mirrors scripts/create_api_key.py). Only the argon2 hash is ever
stored — if you lose the printed password, run this again with --reset.

Usage:
    python scripts/create_admin.py --email admin@example.com [--password secret]
    # omit --password to generate a random one
"""

import argparse
import asyncio
import secrets

from sqlalchemy import select

from prorag.auth import hash_password
from prorag.db import SessionLocal
from prorag.models import User


async def _create(email: str, password: str | None, reset: bool) -> None:
    raw = password or secrets.token_urlsafe(16)
    async with SessionLocal() as session:
        user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            user = User(email=email, display_name="Admin", is_admin=True, password_hash=hash_password(raw))
            session.add(user)
        elif reset:
            user.password_hash = hash_password(raw)
            user.is_admin = True
        else:
            print(f"user {email} already exists — pass --reset to set a new password")
            return
        await session.commit()

    print(f"Admin user {email} ready — password (save it now, it will not be shown again):")
    print(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="admin's login email")
    parser.add_argument("--password", default=None, help="password (default: generate a random one)")
    parser.add_argument("--reset", action="store_true", help="overwrite the password if the user already exists")
    args = parser.parse_args()
    asyncio.run(_create(args.email, args.password, args.reset))


if __name__ == "__main__":
    main()
