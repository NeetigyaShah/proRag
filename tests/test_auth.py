"""Tests for local accounts, sessions, and OIDC login (#19).

Password hashing is pure and needs no DB. Everything else is DB-backed and
follows tests/test_identity_schema.py's pattern: skip cleanly if Postgres is
unreachable, clean up rows in a `finally`, dispose() the engine afterward so
pooled connections don't stay bound to this test's (closing) event loop.

Endpoint tests dispatch through `httpx.AsyncClient` over `ASGITransport`
rather than `fastapi.testclient.TestClient`: TestClient drives the ASGI app
from its own event loop, which would hand asyncpg connections opened by this
test's direct SessionLocal() calls to a different loop than the request
handler's — ASGITransport shares this test's loop, so a real DB and a real
request can be exercised in the same test safely.

The OIDC callback test monkeypatches `_exchange_code_for_claims` — the single
function that does discovery + code exchange + JWKS fetch + signature check —
so it never touches the network; only the upsert/group-sync/session logic
after that boundary is under test.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, text

import prorag.auth_routes as auth_routes
from prorag.auth import (
    SESSION_COOKIE_NAME,
    create_session,
    current_user,
    hash_key,
    hash_password,
    new_api_key,
    require_auth,
    resolve_session_token,
    verify_password,
)
from prorag.db import SessionLocal, engine
from prorag.main import app
from prorag.models import ApiKey, Group, Session, User, UserGroup
from prorag.settings import settings

# ---- password hashing (pure) ---------------------------------------------------


def test_hash_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password(h, "correct horse battery staple") is True


def test_verify_password_rejects_wrong_password():
    h = hash_password("correct horse battery staple")
    assert verify_password(h, "wrong password") is False


def test_hash_password_never_stores_the_raw_password():
    raw = "super-secret-value"
    assert raw not in hash_password(raw)


def test_verify_password_fails_closed_on_a_malformed_hash():
    assert verify_password("not-a-real-argon2-hash", "anything") is False


# ---- DB helper (same pattern as test_identity_schema.py) -----------------------


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


# ---- session create / resolve / expire / revoke (DB) ---------------------------


async def test_session_create_resolve_expire_revoke():
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    try:
        async with session:
            user = User(email=f"{tag}@example.com")
            session.add(user)
            await session.flush()

            raw = await create_session(user.id, session)
            try:
                row = await resolve_session_token(raw, session)
                assert row is not None
                assert row.user_id == user.id

                row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
                await session.commit()
                assert await resolve_session_token(raw, session) is None

                row.expires_at = datetime.now(UTC) + timedelta(days=1)
                row.revoked_at = datetime.now(UTC)
                await session.commit()
                assert await resolve_session_token(raw, session) is None

                assert await resolve_session_token("token-that-was-never-issued", session) is None
            finally:
                await session.execute(delete(Session).where(Session.user_id == user.id))
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
    finally:
        await engine.dispose()


# ---- current_user precedence: cookie > key > none -------------------------------


def _whoami_app():
    from fastapi import Depends, FastAPI

    whoami_app = FastAPI()

    @whoami_app.get("/whoami")
    async def whoami(user: User | None = Depends(current_user)):
        return {"email": user.email if user else None}

    return whoami_app


async def test_current_user_precedence_cookie_over_key_over_none():
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    try:
        async with session:
            cookie_user = User(email=f"{tag}-cookie@example.com")
            key_user = User(email=f"{tag}-key@example.com")
            session.add_all([cookie_user, key_user])
            await session.flush()

            raw_session_token = await create_session(cookie_user.id, session)
            raw_key = new_api_key()
            session.add(ApiKey(key_hash=hash_key(raw_key), user_id=key_user.id))
            await session.commit()

            try:
                transport = ASGITransport(app=_whoami_app())
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/whoami")
                    assert resp.json() == {"email": None}

                    resp = await client.get("/whoami", headers={"Authorization": f"Bearer {raw_key}"})
                    assert resp.json() == {"email": key_user.email}

                    resp = await client.get(
                        "/whoami",
                        headers={"Authorization": f"Bearer {raw_key}"},
                        cookies={SESSION_COOKIE_NAME: raw_session_token},
                    )
                    assert resp.json() == {"email": cookie_user.email}  # cookie wins when both present

                    resp = await client.get("/whoami", cookies={SESSION_COOKIE_NAME: raw_session_token})
                    assert resp.json() == {"email": cookie_user.email}
            finally:
                await session.execute(delete(ApiKey).where(ApiKey.user_id == key_user.id))
                await session.execute(delete(Session).where(Session.user_id == cookie_user.id))
                await session.execute(delete(User).where(User.id.in_([cookie_user.id, key_user.id])))
                await session.commit()
    finally:
        await engine.dispose()


# ---- require_auth guard: session OR key when enabled, wide open when not --------


def _protected_app():
    from fastapi import Depends, FastAPI

    protected_app = FastAPI()

    @protected_app.get("/protected")
    async def protected(_=Depends(require_auth)):
        return {"ok": True}

    return protected_app


async def test_require_auth_session_or_key_and_auth_disabled_is_wide_open(monkeypatch):
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    try:
        async with session:
            user = User(email=f"{tag}-guard@example.com")
            session.add(user)
            await session.flush()

            raw_session_token = await create_session(user.id, session)
            raw_key = new_api_key()
            session.add(ApiKey(key_hash=hash_key(raw_key), user_id=user.id))
            await session.commit()

            try:
                transport = ASGITransport(app=_protected_app())
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    monkeypatch.setattr(settings, "auth_enabled", False)
                    resp = await client.get("/protected")
                    assert resp.status_code == 200

                    monkeypatch.setattr(settings, "auth_enabled", True)
                    resp = await client.get("/protected")
                    assert resp.status_code == 401

                    resp = await client.get("/protected", cookies={SESSION_COOKIE_NAME: raw_session_token})
                    assert resp.status_code == 200

                    resp = await client.get("/protected", headers={"Authorization": f"Bearer {raw_key}"})
                    assert resp.status_code == 200

                    row = (await session.execute(select(Session).where(Session.user_id == user.id))).scalar_one()
                    row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
                    await session.commit()

                    # expired session cookie alone -> 401
                    resp = await client.get("/protected", cookies={SESSION_COOKIE_NAME: raw_session_token})
                    assert resp.status_code == 401

                    # expired session cookie + a valid key -> falls through to the key, 200
                    resp = await client.get(
                        "/protected",
                        cookies={SESSION_COOKIE_NAME: raw_session_token},
                        headers={"Authorization": f"Bearer {raw_key}"},
                    )
                    assert resp.status_code == 200
            finally:
                await session.execute(delete(ApiKey).where(ApiKey.user_id == user.id))
                await session.execute(delete(Session).where(Session.user_id == user.id))
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
    finally:
        await engine.dispose()


# ---- /auth/login and /auth/logout (DB, real app) --------------------------------


async def test_login_rejects_wrong_password_and_succeeds_then_logout_revokes():
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    email = f"{tag}@example.com"
    try:
        async with session:
            user = User(email=email, password_hash=hash_password("s3cret-pass"))
            session.add(user)
            await session.commit()

            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/auth/login", json={"email": email, "password": "wrong"})
                    assert resp.status_code == 401

                    resp = await client.post("/auth/login", json={"email": email, "password": "s3cret-pass"})
                    assert resp.status_code == 200
                    raw_cookie = resp.cookies.get(SESSION_COOKIE_NAME)
                    assert raw_cookie

                    row = (await session.execute(select(Session).where(Session.user_id == user.id))).scalar_one()
                    assert row.revoked_at is None

                    resp = await client.post("/auth/logout", cookies={SESSION_COOKIE_NAME: raw_cookie})
                    assert resp.status_code == 200

                    await session.refresh(row)
                    assert row.revoked_at is not None
            finally:
                await session.execute(delete(Session).where(Session.user_id == user.id))
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
    finally:
        await engine.dispose()


async def test_login_rejects_a_user_with_no_local_password():
    """An OIDC-only user (password_hash is None) can't log in with a password
    even if they somehow guess a matching plaintext."""
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    email = f"{tag}-oidc-only@example.com"
    try:
        async with session:
            user = User(email=email, external_subject=f"sub-{tag}")
            session.add(user)
            await session.commit()

            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/auth/login", json={"email": email, "password": "anything"})
                    assert resp.status_code == 401
            finally:
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
    finally:
        await engine.dispose()


# ---- OIDC: 404 when unconfigured, callback upserts + seeds groups --------------


async def test_oidc_endpoints_404_when_issuer_unset(monkeypatch):
    monkeypatch.setattr(settings, "oidc_issuer", None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/auth/oidc/login")).status_code == 404
        assert (await client.get("/auth/oidc/callback")).status_code == 404


async def test_oidc_callback_upserts_user_and_replaces_idp_group_membership(monkeypatch):
    monkeypatch.setattr(settings, "oidc_issuer", "https://idp.example.com")
    monkeypatch.setattr(settings, "oidc_client_id", "test-client")
    monkeypatch.setattr(settings, "oidc_client_secret", "test-secret")

    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    sub = f"sub-{tag}"
    email = f"{tag}-oidc@example.com"
    group_a, group_b = f"group-a-{tag}", f"group-b-{tag}"

    claims = {"sub": sub, "email": email, "name": "OIDC User", "groups": [group_a, group_b]}

    async def fake_exchange(code, redirect_uri, nonce):
        assert code == "auth-code-1"
        return claims

    monkeypatch.setattr(auth_routes, "_exchange_code_for_claims", fake_exchange)

    try:
        async with session:
            try:
                transport = ASGITransport(app=app)
                state_cookie = "state-value:nonce-value"
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get(
                        "/auth/oidc/callback",
                        params={"code": "auth-code-1", "state": "state-value"},
                        cookies={auth_routes.OIDC_STATE_COOKIE: state_cookie},
                        follow_redirects=False,
                    )
                assert resp.status_code == 302
                assert resp.headers["location"] == "/"
                assert resp.cookies.get(SESSION_COOKIE_NAME)

                user = (await session.execute(select(User).where(User.external_subject == sub))).scalar_one()
                assert user.email == email

                idp_memberships = (
                    await session.execute(
                        select(UserGroup).join(Group, Group.id == UserGroup.group_id).where(
                            UserGroup.user_id == user.id, Group.source == "idp"
                        )
                    )
                ).scalars().all()
                assert len(idp_memberships) == 2

                # second login drops group_b -> idp-sourced membership is replaced, not merged
                claims["groups"] = [group_a]

                async def fake_exchange_2(code, redirect_uri, nonce):
                    return claims

                monkeypatch.setattr(auth_routes, "_exchange_code_for_claims", fake_exchange_2)

                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp2 = await client.get(
                        "/auth/oidc/callback",
                        params={"code": "auth-code-2", "state": "state-value"},
                        cookies={auth_routes.OIDC_STATE_COOKIE: state_cookie},
                        follow_redirects=False,
                    )
                assert resp2.status_code == 302

                idp_memberships_after = (
                    await session.execute(
                        select(UserGroup).join(Group, Group.id == UserGroup.group_id).where(
                            UserGroup.user_id == user.id, Group.source == "idp"
                        )
                    )
                ).scalars().all()
                assert len(idp_memberships_after) == 1
            finally:
                await session.execute(delete(User).where(User.external_subject == sub))  # cascades sessions/user_groups
                await session.execute(delete(Group).where(Group.source == "idp", Group.external_id.in_([group_a, group_b])))
                await session.commit()
    finally:
        await engine.dispose()


async def test_oidc_callback_rejects_state_mismatch(monkeypatch):
    monkeypatch.setattr(settings, "oidc_issuer", "https://idp.example.com")
    monkeypatch.setattr(settings, "oidc_client_id", "test-client")
    monkeypatch.setattr(settings, "oidc_client_secret", "test-secret")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/auth/oidc/callback",
            params={"code": "auth-code", "state": "wrong-state"},
            cookies={auth_routes.OIDC_STATE_COOKIE: "expected-state:nonce"},
            follow_redirects=False,
        )
    assert resp.status_code == 400
