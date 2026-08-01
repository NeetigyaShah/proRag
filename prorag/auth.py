"""Bearer API-key auth (§6, §8 Phase 5). The presented key is hashed and looked
up by hash — raw keys are never stored, so there is no raw-to-raw comparison
to time; the hash-then-lookup *is* the constant-time compare.

`settings.auth_enabled=False` (default) skips the check entirely so the local
web UI keeps working without a key.
"""

import hashlib
import secrets

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.db import get_session
from prorag.models import ApiKey, User
from prorag.settings import settings


def new_api_key() -> str:
    return secrets.token_urlsafe(32)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def require_api_key(request: Request, session: AsyncSession = Depends(get_session)) -> ApiKey | None:
    """FastAPI dependency for every router except /healthz, /readyz, /web."""
    if not settings.auth_enabled:
        return None

    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    presented = header.removeprefix("Bearer ").strip()

    key = (await session.execute(select(ApiKey).where(ApiKey.key_hash == hash_key(presented)))).scalar_one_or_none()
    if key is None:
        raise HTTPException(401, "invalid api key")

    # ponytail: collection scoping only checks the query string here — the
    # request body (e.g. ChatRequest.collection) would need a body-parsing
    # dance to read from a dependency. Add when a scoped key is actually
    # issued for a JSON-body route.
    requested_collection = request.query_params.get("collection")
    if key.collection and requested_collection and requested_collection != key.collection:
        raise HTTPException(403, "api key not scoped to this collection")

    return key


async def current_user(
    api_key: ApiKey | None = Depends(require_api_key), session: AsyncSession = Depends(get_session)
) -> User | None:
    """Resolves the bearer key's user_id to a User row (#18). None is the
    same "super-principal, no filtering" meaning visibility_clause() gives
    None for: auth disabled, or a legacy unscoped key (user_id IS NULL, #2)."""
    if api_key is None or api_key.user_id is None:
        return None
    return (await session.execute(select(User).where(User.id == api_key.user_id))).scalar_one_or_none()
