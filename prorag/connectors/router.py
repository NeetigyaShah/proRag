"""Admin-only CRUD for connectors + POST /connectors/{id}/sync (#22).

Gating: current_user must be present AND is_admin when auth is enabled, else
403 — same "wide open when settings.auth_enabled is False" convenience the
rest of the app gives local dev (prorag/auth.py's module docstring). This is
an admin surface (config includes a secret, sync can delete documents), so
unlike files/router.py's visible_doc_guard there's no need to 404-style
existence away from a caller who's already been let through the admin gate.

Tier C (#15): connector-ingested docs get no document_acl rows. That's
enforced in connectors/sync.py, not here — this module only decides who may
configure/trigger a sync.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import current_user
from prorag.connectors.sync import full_sweep, sync_incremental
from prorag.db import get_session
from prorag.models import Connector, User
from prorag.schemas import ConnectorCreate, ConnectorOut, ConnectorUpdate, SyncReport
from prorag.settings import settings

router = APIRouter(prefix="/connectors", tags=["connectors"])


async def require_admin(user: User | None = Depends(current_user)) -> None:
    if not settings.auth_enabled:
        return
    if user is None or not user.is_admin:
        raise HTTPException(403, "admin access required")


async def _get_or_404(session: AsyncSession, connector_id: uuid.UUID) -> Connector:
    row = (await session.execute(select(Connector).where(Connector.id == connector_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "connector not found")
    return row


@router.post("", response_model=ConnectorOut, status_code=201, dependencies=[Depends(require_admin)])
async def create_connector(body: ConnectorCreate, session: AsyncSession = Depends(get_session)):
    row = Connector(id=uuid.uuid4(), type=body.type, name=body.name, config=body.config, enabled=body.enabled)
    session.add(row)
    await session.commit()
    return row


@router.get("", response_model=list[ConnectorOut], dependencies=[Depends(require_admin)])
async def list_connectors(session: AsyncSession = Depends(get_session)):
    return (await session.execute(select(Connector))).scalars().all()


@router.get("/{connector_id}", response_model=ConnectorOut, dependencies=[Depends(require_admin)])
async def get_connector(connector_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _get_or_404(session, connector_id)


@router.patch("/{connector_id}", response_model=ConnectorOut, dependencies=[Depends(require_admin)])
async def update_connector(
    connector_id: uuid.UUID, body: ConnectorUpdate, session: AsyncSession = Depends(get_session)
):
    row = await _get_or_404(session, connector_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(row, field, value)
    await session.commit()
    return row


@router.delete("/{connector_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_connector(connector_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    row = await _get_or_404(session, connector_id)
    await session.delete(row)
    await session.commit()


@router.post("/{connector_id}/sync", response_model=SyncReport, dependencies=[Depends(require_admin)])
async def sync_connector(connector_id: uuid.UUID, full: bool = False, session: AsyncSession = Depends(get_session)):
    """`?full=true` runs the mandatory reconciliation sweep (deletion
    propagation, #15) instead of the incremental poll. Manual trigger v1 —
    the background scheduler is the next ticket (#22's scope note)."""
    row = await _get_or_404(session, connector_id)
    try:
        report = await (full_sweep(row, session) if full else sync_incremental(row, session))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return report
