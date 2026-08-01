"""Bearer API-key + session-cookie auth (§6, §8 Phase 5; sessions/#19). Both
credential types are hashed and looked up by hash — raw values are never
stored, so there is no raw-to-raw comparison to time; the hash-then-lookup
*is* the constant-time compare.

`settings.auth_enabled=False` (default) skips the router-level guard
(`require_auth`) entirely so the local web UI keeps working without a key.
`current_user` is independent of that flag — it always tries to resolve who's
asking (session cookie first, then bearer key), for visibility filtering and
personalization even when the guard is off.
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.db import get_session
from prorag.models import ApiKey, Session, User
from prorag.settings import settings

SESSION_COOKIE_NAME = "prorag_session"

_ph = PasswordHasher()


def new_api_key() -> str:
    return secrets.token_urlsafe(32)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def hash_password(raw_password: str) -> str:
    return _ph.hash(raw_password)


def verify_password(password_hash: str, raw_password: str) -> bool:
    """True only for a matching password against a well-formed hash — a
    corrupt/foreign hash format fails closed rather than raising."""
    try:
        return _ph.verify(password_hash, raw_password)
    except (VerificationError, InvalidHashError):
        return False


async def create_session(user_id: uuid.UUID, session: AsyncSession) -> str:
    """Mints a session token, stores it hashed (same idiom as api_keys), and
    returns the raw value — callers put that in the cookie, never in the DB."""
    raw = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(days=settings.session_ttl_days)
    session.add(Session(token_hash=hash_key(raw), user_id=user_id, expires_at=expires_at))
    await session.commit()
    return raw


async def resolve_session_token(raw_token: str, session: AsyncSession) -> Session | None:
    """Looks up a raw session token; returns None (not an exception) for
    unknown/expired/revoked tokens — the caller decides what that means."""
    row = (
        await session.execute(select(Session).where(Session.token_hash == hash_key(raw_token)))
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.revoked_at is not None or row.expires_at <= datetime.now(UTC):
        return None
    return row


async def _session_from_cookie(request: Request, session: AsyncSession) -> Session | None:
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        return None
    return await resolve_session_token(raw, session)


async def _api_key_from_header(request: Request, session: AsyncSession) -> ApiKey | None:
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return None
    presented = header.removeprefix("Bearer ").strip()
    return (await session.execute(select(ApiKey).where(ApiKey.key_hash == hash_key(presented)))).scalar_one_or_none()


async def require_api_key(request: Request, session: AsyncSession = Depends(get_session)) -> ApiKey | None:
    """Bearer-only guard. Used directly as a fallback inside require_auth, and
    still usable standalone wherever only machine (key) auth makes sense."""
    if not settings.auth_enabled:
        return None

    key = await _api_key_from_header(request, session)
    if key is None:
        header = request.headers.get("authorization", "")
        raise HTTPException(401, "missing bearer token" if not header else "invalid api key")

    # ponytail: collection scoping only checks the query string here — the
    # request body (e.g. ChatRequest.collection) would need a body-parsing
    # dance to read from a dependency. Add when a scoped key is actually
    # issued for a JSON-body route.
    requested_collection = request.query_params.get("collection")
    if key.collection and requested_collection and requested_collection != key.collection:
        raise HTTPException(403, "api key not scoped to this collection")

    return key


async def require_auth(request: Request, session: AsyncSession = Depends(get_session)) -> ApiKey | None:
    """Router-level guard (#19): passes when auth is disabled, OR a valid
    session cookie is present, OR a valid bearer key is presented — checked in
    that order. A session-authenticated request carries no ApiKey row (None,
    same "no scope to check" meaning as an unscoped key); a bearer-
    authenticated one returns the ApiKey so require_api_key's collection-scope
    check still applies. Expired/revoked sessions fall through to the bearer
    check rather than failing outright, and only 401/403 if neither works."""
    if not settings.auth_enabled:
        return None
    if await _session_from_cookie(request, session) is not None:
        return None
    return await require_api_key(request, session)


async def current_user(request: Request, session: AsyncSession = Depends(get_session)) -> User | None:
    """Resolves the caller's identity: session cookie first, then bearer key,
    else None (#19). None is the same "super-principal, no filtering" meaning
    visibility_clause() gives for: no credentials at all, or a legacy
    unscoped key (user_id IS NULL, #2). Deliberately independent of
    settings.auth_enabled — see module docstring."""
    sess = await _session_from_cookie(request, session)
    if sess is not None:
        return (await session.execute(select(User).where(User.id == sess.user_id))).scalar_one_or_none()

    api_key = await _api_key_from_header(request, session)
    if api_key is not None and api_key.user_id is not None:
        return (await session.execute(select(User).where(User.id == api_key.user_id))).scalar_one_or_none()

    return None
