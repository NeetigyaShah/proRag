"""GET /files/{doc_id}/original — serve the stored blob.
GET /tables/{table_id}/rows — JSONB rows, backs the CSV citation grid (§6)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import current_user
from prorag.db import get_session
from prorag.models import Document, Table, TableRow, User
from prorag.retrieve.visibility import visible_doc_guard

router = APIRouter()


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_session)):
    """Header stat: how many documents are ingested (and how many are ready).
    # ponytail: left unfiltered by ACL (#18) — a count is low-sensitivity
    # (no titles/content) and it's the dashboard header every user sees;
    # add per-user filtering if that stops being true."""
    total = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
    ready = (
        await session.execute(select(func.count()).select_from(Document).where(Document.status == "ready"))
    ).scalar_one()
    return {"documents": total, "ready": ready}


@router.get("/files/{doc_id}/original")
async def get_original(
    doc_id: uuid.UUID,
    download: int = 0,
    session: AsyncSession = Depends(get_session),
    user: User | None = Depends(current_user),
):
    """Serve the stored blob. Inline by default so pdf.js can render it in the
    viewer — passing `filename=` would set content-disposition: attachment,
    which makes the browser download the file instead of handing bytes to the
    viewer. `?download=1` opts back into the attachment behaviour.

    Blobs are content-addressed (path = sha256), so they can be cached hard.

    A doc that's invisible to `user` (#18) 404s exactly like a missing one —
    existence must not leak (#3) — so the guard runs before the real lookup.
    """
    if not await visible_doc_guard(session, user, doc_id):
        raise HTTPException(404, "document not found")

    doc = (await session.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "document not found")

    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{doc.filename}"'
    return FileResponse(doc.blob_path, media_type=doc.mime, headers=headers)


@router.get("/tables/{table_id}/rows")
async def get_table_rows(
    table_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    user: User | None = Depends(current_user),
):
    """Rows inherit their owning document's visibility (#18) — a table has no
    ACL of its own, so the guard checks table.doc_id and 404s the same way as
    a missing table_id when it's invisible."""
    table = (await session.execute(select(Table).where(Table.id == table_id))).scalar_one_or_none()
    if table is None or not await visible_doc_guard(session, user, table.doc_id):
        raise HTTPException(404, "table not found")

    stmt = select(TableRow).where(TableRow.table_id == table_id).order_by(TableRow.row_no).offset(offset).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return {"rows": [{"row_no": r.row_no, "data": r.data} for r in rows]}
