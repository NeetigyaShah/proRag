"""Polling scheduler (#23, on top of #22's sync engine, per #15's cadence:
15-min incremental poll, mandatory nightly full sweep as the only reliable
deletion/move signal).

`next_action` is a pure decision function — no DB, no clock mocking needed,
just datetimes in and a string out — so the sweep-vs-incremental-vs-wait
matrix is unit-testable without a running loop.

`scheduler_loop` ticks in-process every `min(poll_seconds, 60)` seconds and
walks enabled connectors *sequentially* — this is a single-box product (#15),
one connector at a time is plenty; the per-connector-asyncio.Task upgrade
path is straightforward if that ever stops being true.

The per-connector asyncio.Lock here is the same one POST /connectors/{id}/sync
(connectors/router.py) uses for its overlap guard — a scheduled tick skips a
connector that a manual sync already has locked, and vice versa. A plain
module-level dict is enough at this scale: it lives for the process's
lifetime, one lock per connector id, created on first use.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import select

from prorag.connectors.sync import full_sweep, sync_incremental
from prorag.db import SessionLocal
from prorag.models import Connector
from prorag.settings import settings

logger = logging.getLogger(__name__)

Action = Literal["sweep", "incremental", "wait"]

# ponytail: process-lifetime dict, never pruned — connectors are an
# admin-configured handful (single digits to low tens), not a per-request
# entity, so this doesn't grow unbounded the way a per-user dict would.
_locks: dict[uuid.UUID, asyncio.Lock] = defaultdict(asyncio.Lock)


def get_lock(connector_id: uuid.UUID) -> asyncio.Lock:
    return _locks[connector_id]


def next_action(
    now: datetime,
    last_sync_at: datetime | None,
    last_full_sweep_at: datetime | None,
    poll_seconds: int,
    sweep_hours: int,
) -> Action:
    """Sweep wins when due — it's the mandatory reconciliation signal (#15),
    so it preempts a merely-due incremental poll rather than queuing behind
    it. A connector that has never run (both timestamps None) is swept
    first, which also seeds last_full_sweep_at."""
    sweep_due = last_full_sweep_at is None or now - last_full_sweep_at >= timedelta(hours=sweep_hours)
    if sweep_due:
        return "sweep"
    incremental_due = last_sync_at is None or now - last_sync_at >= timedelta(seconds=poll_seconds)
    if incremental_due:
        return "incremental"
    return "wait"


async def _run_one(connector_id: uuid.UUID, now: datetime) -> None:
    """Fresh session for this one connector's run — not the tick's listing
    session, and not shared across connectors — so a slow/failed run can't
    hold a transaction open across the whole sweep of connectors."""
    async with SessionLocal() as session:
        row = (await session.execute(select(Connector).where(Connector.id == connector_id))).scalar_one_or_none()
        if row is None or not row.enabled:
            return
        action = next_action(
            now, row.last_sync_at, row.last_full_sweep_at, settings.connector_poll_seconds, settings.connector_sweep_hours
        )
        if action == "wait":
            return
        try:
            await (full_sweep(row, session) if action == "sweep" else sync_incremental(row, session))
        except Exception as exc:
            # One connector's failure must not crash the loop or the app
            # (#23) — log, record on the row, move on to the next connector.
            logger.exception("scheduled %s failed for connector %s", action, connector_id)
            await session.rollback()
            row.last_error = str(exc)
            await session.commit()
            return
        row.last_error = None
        await session.commit()


async def _tick() -> None:
    async with SessionLocal() as session:
        connector_ids = (
            await session.execute(select(Connector.id).where(Connector.enabled.is_(True)))
        ).scalars().all()

    now = datetime.now(UTC)
    for connector_id in connector_ids:
        lock = get_lock(connector_id)
        if lock.locked():
            # A manual POST /connectors/{id}/sync is already running this
            # connector — skip it this tick rather than queue behind it.
            continue
        async with lock:
            await _run_one(connector_id, now)


async def scheduler_loop() -> None:
    """Runs until cancelled (main.py's lifespan does that on shutdown). With
    zero connectors configured — the local-dev default — each tick is one
    cheap empty SELECT."""
    interval = min(settings.connector_poll_seconds, 60)
    while True:
        try:
            await _tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("scheduler tick failed")
        await asyncio.sleep(interval)
