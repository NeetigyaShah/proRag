"""Endpoint gating for /connectors (#22): admin-only CRUD + sync. Follows
tests/test_auth.py's pattern — ASGITransport+AsyncClient sharing this test's
event loop with a real DB session, so a real request and direct SessionLocal
calls can safely mix.

Two guards stack here: require_auth (router-level, prorag/main.py) gates on
"authenticated at all", then require_admin (connectors/router.py) gates on
is_admin. With auth disabled, both no-op — consistent with the rest of the
app (prorag/auth.py's module docstring)."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, text

from prorag.auth import SESSION_COOKIE_NAME, create_session
from prorag.db import SessionLocal, engine
from prorag.main import app
from prorag.models import Session, User
from prorag.settings import settings


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def test_connectors_wide_open_when_auth_disabled(monkeypatch):
    # The route hits the shared engine via get_session even with no explicit
    # SessionLocal() call here — dispose in a finally so this test's event
    # loop doesn't leave pooled connections a later test's loop can't reuse
    # (tests/test_identity_schema.py's documented pattern).
    try:
        monkeypatch.setattr(settings, "auth_enabled", False)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/connectors")
        assert resp.status_code == 200
    finally:
        await engine.dispose()


async def test_connectors_401_with_no_credentials_when_auth_enabled(monkeypatch):
    try:
        monkeypatch.setattr(settings, "auth_enabled", True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/connectors")
        assert resp.status_code == 401
    finally:
        await engine.dispose()


async def test_connectors_403_for_authenticated_non_admin_then_200_for_admin(monkeypatch):
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    try:
        async with session:
            plain_user = User(email=f"{tag}-plain@example.com", is_admin=False)
            admin_user = User(email=f"{tag}-admin@example.com", is_admin=True)
            session.add_all([plain_user, admin_user])
            await session.flush()

            plain_token = await create_session(plain_user.id, session)
            admin_token = await create_session(admin_user.id, session)
            await session.commit()

            try:
                monkeypatch.setattr(settings, "auth_enabled", True)
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/connectors", cookies={SESSION_COOKIE_NAME: plain_token})
                    assert resp.status_code == 403

                    resp = await client.get("/connectors", cookies={SESSION_COOKIE_NAME: admin_token})
                    assert resp.status_code == 200
            finally:
                await session.execute(
                    delete(Session).where(Session.user_id.in_([plain_user.id, admin_user.id]))
                )
                await session.execute(delete(User).where(User.id.in_([plain_user.id, admin_user.id])))
                await session.commit()
    finally:
        await engine.dispose()
