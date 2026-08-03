"""Sync engine for connectors (#22): incremental poll + full-ID sweep, per
#15's polling-first architecture ("the sweep is mandatory, it is the only
reliable deletion/move signal found"). Both are plain async functions over a
connector row — not methods on a class — so they're testable without HTTP or
a running server, and callable from POST /connectors/{id}/sync (router.py).

Tier C (#15): nothing here ever creates a document_acl row. Documents
ingested through a connector are invisible to non-admins until an admin
grants access (#18's default-deny) — that's the deliberate behaviour, not a
gap to fill in later.

Dedup interplay: ingest_bytes() already resolves a sha256 collision against
an existing document (returns duplicate_of instead of re-ingesting) — the
only connector-side work for that path is linking connector_items.doc_id to
whatever doc_id it returns, which the normal per-item bookkeeping below does
unconditionally.

boto3 calls in S3Connector are synchronous; each one here is pushed onto a
worker thread via asyncio.to_thread so a large bucket listing or object
download doesn't stall the event loop for concurrent chat/search requests.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.connectors.s3 import RemoteObject, S3Connector
from prorag.ingest.core import _safe_error, ingest_bytes
from prorag.ingest.router import ALLOWED_SUFFIXES
from prorag.models import Connector, ConnectorItem, Document
from prorag.settings import settings

logger = logging.getLogger(__name__)

# ponytail: S3-compatible LastModified has ~1s granularity and clocks across
# providers aren't trustworthy (s3.py's docstring) — an object touched in the
# same second as the previous watermark could otherwise be silently missed by
# the `since` pre-filter. A few seconds of overlap costs nothing: diff_objects
# still resolves anything already-synced back down to "unchanged" via
# etag+size, so widen the window rather than trust the clock down to the wire.
_SINCE_SAFETY_BUFFER = timedelta(seconds=5)


def _build_connector(connector_row: Connector) -> S3Connector:
    """When called: at the start of sync_incremental and full_sweep. What:
    constructs the concrete connector client for the row's type+config (only
    "s3" is supported today), raising ValueError on unknown types. Returns:
    the S3Connector."""
    if connector_row.type == "s3":
        return S3Connector(connector_row.config)
    raise ValueError(f"unknown connector type: {connector_row.type}")


def diff_objects(
    remote: list[RemoteObject], known: dict[str, tuple[str | None, int | None]]
) -> tuple[list[RemoteObject], list[RemoteObject], list[str]]:
    """Pure new/changed/unchanged classification (#22) — `known` maps
    external_id -> (etag, size) as last recorded in connector_items. Split
    out from _process_objects so the new/changed/unchanged matrix is testable
    without a DB or an S3 fixture."""
    new, changed, unchanged = [], [], []
    for obj in remote:
        prior = known.get(obj.key)
        if prior is None:
            new.append(obj)
        elif prior != (obj.etag, obj.size):
            changed.append(obj)
        else:
            unchanged.append(obj.key)
    return new, changed, unchanged


def find_deleted(remote_keys: set[str], known: dict[str, str]) -> set[str]:
    """Pure deletion detection for the full sweep (#22, #15's mandatory
    reconciliation) — `known` maps external_id -> its current connector_item
    status. Anything not in this listing and not already marked deleted is
    newly missing."""
    return {key for key, status in known.items() if key not in remote_keys and status != "deleted"}


async def _upsert_item(
    session: AsyncSession, connector_id, key: str, obj: RemoteObject, status: str, doc_id, error: str | None = None
) -> None:
    """Fresh get-or-create + commit per item, deliberately not reusing an
    ORM instance loaded earlier in the run: ingest_bytes() may have rolled
    back the shared session on this exact item (its own failure path), which
    expires every object attached to the session — a fresh SELECT sidesteps
    that instead of relying on in-memory state that may no longer be valid."""
    row = (
        await session.execute(
            select(ConnectorItem).where(ConnectorItem.connector_id == connector_id, ConnectorItem.external_id == key)
        )
    ).scalar_one_or_none()
    if row is None:
        row = ConnectorItem(connector_id=connector_id, external_id=key)
        session.add(row)
    row.etag = obj.etag
    row.size = obj.size
    row.last_modified = obj.last_modified
    row.last_seen_at = datetime.now(UTC)
    row.status = status
    row.doc_id = doc_id
    row.error = error
    await session.commit()


async def _process_objects(
    session: AsyncSession, connector: S3Connector, connector_id, collection: str, remote: list[RemoteObject]
) -> dict:
    """New/changed/unchanged classification + ingest for one listing. Shared
    by sync_incremental and full_sweep (the sweep just adds deletion
    detection on top, since it has the full key universe)."""
    known_rows = (
        await session.execute(
            select(ConnectorItem.external_id, ConnectorItem.etag, ConnectorItem.size).where(
                ConnectorItem.connector_id == connector_id
            )
        )
    ).all()
    known = {r.external_id: (r.etag, r.size) for r in known_rows}
    new_objs, changed_objs, unchanged_keys = diff_objects(remote, known)

    report = {"new": 0, "changed": 0, "skipped": 0, "errors": 0}

    for obj, is_new in [(o, True) for o in new_objs] + [(o, False) for o in changed_objs]:
        if not obj.key.lower().endswith(ALLOWED_SUFFIXES):
            await _upsert_item(
                session, connector_id, obj.key, obj, status="skipped", doc_id=None, error="unsupported file type"
            )
            report["skipped"] += 1
            continue
        if obj.size > settings.max_upload_bytes:
            await _upsert_item(
                session, connector_id, obj.key, obj, status="skipped", doc_id=None, error="exceeds max_upload_bytes"
            )
            report["skipped"] += 1
            continue

        try:
            data = await asyncio.to_thread(connector.fetch, obj)
            filename = obj.key.rsplit("/", 1)[-1] or obj.key
            resp = await ingest_bytes(session, data, filename=filename, collection=collection)
        except HTTPException as exc:
            logger.warning("connector item failed: key=%s reason=%s", obj.key, exc.detail)
            await _upsert_item(session, connector_id, obj.key, obj, status="error", doc_id=None, error=_safe_error(exc))
            report["errors"] += 1
            continue
        except Exception as exc:
            # One bad object must not kill the run (#22) — record and move on.
            logger.exception("connector item failed: key=%s", obj.key)
            await _upsert_item(session, connector_id, obj.key, obj, status="error", doc_id=None, error=_safe_error(exc))
            report["errors"] += 1
            continue

        await _upsert_item(session, connector_id, obj.key, obj, status="synced", doc_id=resp.doc_id)
        report["new" if is_new else "changed"] += 1

    if unchanged_keys:
        await session.execute(
            update(ConnectorItem)
            .where(ConnectorItem.connector_id == connector_id, ConnectorItem.external_id.in_(unchanged_keys))
            .values(last_seen_at=datetime.now(UTC))
        )
        await session.commit()

    return report


async def sync_incremental(connector_row: Connector, session: AsyncSession) -> dict:
    """Poll: list the bucket, ingest new/changed objects. Does not detect
    deletions or moves — that needs the full key universe, which is
    full_sweep's job."""
    connector_id = connector_row.id
    collection = connector_row.config.get("collection", "default")
    connector = _build_connector(connector_row)

    since = connector_row.last_sync_at
    if since is not None:
        since = since - _SINCE_SAFETY_BUFFER
    remote = await asyncio.to_thread(connector.list_changed, since)
    report = await _process_objects(session, connector, connector_id, collection, remote)
    report["deleted"] = 0

    connector_row.last_sync_at = datetime.now(UTC)
    await session.commit()
    return report


async def full_sweep(connector_row: Connector, session: AsyncSession) -> dict:
    """Full-ID reconciliation (#15: mandatory, the only reliable deletion/move
    signal). Lists every key, ingests new/changed the same as
    sync_incremental, then deletes any Document whose connector_item was
    previously synced but is no longer in the listing (cascade cleans
    chunks/tables/document_acl per models.Document's relationships)."""
    connector_id = connector_row.id
    collection = connector_row.config.get("collection", "default")
    connector = _build_connector(connector_row)

    remote = await asyncio.to_thread(connector.list_changed, None)
    report = await _process_objects(session, connector, connector_id, collection, remote)

    remote_keys = {o.key for o in remote}
    known_rows = (
        await session.execute(
            select(ConnectorItem.external_id, ConnectorItem.doc_id, ConnectorItem.status).where(
                ConnectorItem.connector_id == connector_id
            )
        )
    ).all()
    known_status = {r.external_id: r.status for r in known_rows}
    doc_id_by_key = {r.external_id: r.doc_id for r in known_rows}

    deleted_keys = find_deleted(remote_keys, known_status)
    for key in deleted_keys:
        doc_id = doc_id_by_key.get(key)
        if doc_id is not None:
            doc = (await session.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            if doc is not None:
                await session.delete(doc)
        item = (
            await session.execute(
                select(ConnectorItem).where(ConnectorItem.connector_id == connector_id, ConnectorItem.external_id == key)
            )
        ).scalar_one()
        item.status = "deleted"
        item.doc_id = None
        await session.commit()

    report["deleted"] = len(deleted_keys)
    connector_row.last_sync_at = datetime.now(UTC)
    connector_row.last_full_sweep_at = datetime.now(UTC)
    await session.commit()
    return report
