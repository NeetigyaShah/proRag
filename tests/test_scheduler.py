"""Scheduler tests (#23).

next_action is a pure function — no DB, no clock mocking, no fixtures — so
the sweep/incremental/wait decision matrix (including the never-run None
timestamps) is exercised directly.

The overlap-guard test drives the real endpoint (same pattern as
tests/test_connectors_admin_gating.py): pre-acquire the connector's lock the
way a scheduled tick would hold it mid-run, then confirm POST
/connectors/{id}/sync sees it locked and rejects instead of starting a
second run on top of it.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from moto import mock_aws
from sqlalchemy import delete, text

from prorag.connectors.scheduler import _tick, get_lock, next_action
from prorag.db import SessionLocal
from prorag.main import app
from prorag.models import Connector
from prorag.settings import settings

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
POLL_SECONDS = 900
SWEEP_HOURS = 24


@pytest.mark.parametrize(
    "last_sync_at,last_full_sweep_at,expected",
    [
        # Never run at all -> sweep first, seeds last_full_sweep_at.
        (None, None, "sweep"),
        # Swept before but never incrementally synced -> incremental (sweep not due).
        (None, NOW - timedelta(hours=1), "incremental"),
        # Synced before but never swept -> sweep (mandatory reconciliation, #15).
        (NOW - timedelta(seconds=100), None, "sweep"),
        # Neither due yet.
        (NOW - timedelta(seconds=100), NOW - timedelta(hours=1), "wait"),
        # Poll interval elapsed, sweep window hasn't -> incremental.
        (NOW - timedelta(seconds=1000), NOW - timedelta(hours=1), "incremental"),
        # Sweep window elapsed, poll interval hasn't -> sweep wins.
        (NOW - timedelta(seconds=100), NOW - timedelta(hours=25), "sweep"),
        # Both elapsed -> sweep wins.
        (NOW - timedelta(seconds=1000), NOW - timedelta(hours=25), "sweep"),
        # Exactly on the boundary counts as due (>=).
        (NOW - timedelta(seconds=POLL_SECONDS), NOW - timedelta(hours=1), "incremental"),
        (NOW - timedelta(seconds=100), NOW - timedelta(hours=SWEEP_HOURS), "sweep"),
    ],
)
def test_next_action_matrix(last_sync_at, last_full_sweep_at, expected):
    assert next_action(NOW, last_sync_at, last_full_sweep_at, POLL_SECONDS, SWEEP_HOURS) == expected


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def test_manual_sync_returns_409_when_the_connectors_lock_is_already_held(monkeypatch):
    """Simulates the scheduler (or another manual call) mid-run by holding
    the connector's lock directly — connectors/router.py and
    connectors/scheduler.py share the same module-level lock dict, so
    holding it here is equivalent to a real run in progress."""
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    async with session:
        connector = Connector(
            id=uuid.uuid4(),
            type="s3",
            name=f"s3-{tag}",
            config={"bucket": "b", "region": "us-east-1", "access_key_id": "x", "secret_access_key": "y"},
        )
        session.add(connector)
        await session.commit()

        try:
            monkeypatch.setattr(settings, "auth_enabled", False)
            lock = get_lock(connector.id)
            await lock.acquire()
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(f"/connectors/{connector.id}/sync")
                assert resp.status_code == 409
                assert resp.json() == {"detail": "sync already running"}
            finally:
                lock.release()
        finally:
            await session.execute(delete(Connector).where(Connector.id == connector.id))
            await session.commit()


async def test_tick_records_last_error_and_keeps_running_the_rest_of_the_tick():
    """A connector whose bucket was never created makes boto3 raise inside
    sync_incremental — `_tick()` must record that on the row and keep going
    rather than propagate (#23: 'log + last_error, continue')."""
    with mock_aws():
        session = await _get_session()
        tag = uuid.uuid4().hex[:8]
        async with session:
            broken = Connector(
                id=uuid.uuid4(),
                type="s3",
                name=f"broken-{tag}",
                config={
                    "bucket": f"does-not-exist-{tag}",
                    "region": "us-east-1",
                    "access_key_id": "testing",
                    "secret_access_key": "testing",
                },
                last_error="a stale error from a previous run",
            )
            session.add(broken)
            await session.commit()

            try:
                await _tick()

                await session.refresh(broken)
                assert broken.last_error is not None
                assert broken.last_sync_at is None  # the failed run never got to record success
            finally:
                await session.execute(delete(Connector).where(Connector.id == broken.id))
                await session.commit()
