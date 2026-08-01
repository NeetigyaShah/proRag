"""Query-time visibility enforcement (#18), on top of #15/#17's document_acl
schema. One predicate, reused by every retrieval arm and the file/table
endpoints so enforcement can't drift between call sites — per #3's
resolution, this is a SQL predicate applied *inside* each arm, before LIMIT,
never a post-filter (post-filtering both leaks via crop/source counts and
wrecks ranking/recall).
"""

import uuid

from sqlalchemy import ColumnElement, and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.models import Document, DocumentAcl, User, UserGroup


def visibility_clause(user: User | None) -> ColumnElement | None:
    """None means "no filtering": user is None (auth disabled, or a legacy
    unscoped API key — the super-principal per #2) or user.is_admin.
    Otherwise an EXISTS over document_acl, correlated to Document.id: public,
    or a direct user grant, or a grant to one of the user's groups. The group
    membership check is a correlated subquery *inside* the EXISTS rather than
    resolved to a Python list first — one round trip, and consistent within
    the same transaction (#3)."""
    if user is None or user.is_admin:
        return None

    group_ids = select(UserGroup.group_id).where(UserGroup.user_id == user.id).scalar_subquery()

    return exists(
        select(1).where(
            DocumentAcl.doc_id == Document.id,
            or_(
                DocumentAcl.principal_type == "public",
                and_(DocumentAcl.principal_type == "user", DocumentAcl.principal_id == user.id),
                and_(DocumentAcl.principal_type == "group", DocumentAcl.principal_id.in_(group_ids)),
            ),
        )
    )


async def visible_doc_guard(session: AsyncSession, user: User | None, doc_id: uuid.UUID) -> bool:
    """Single-document visibility check for the file/table endpoints. Callers
    turn False into a plain 404 — existence must not leak (#3)."""
    stmt = select(Document.id).where(Document.id == doc_id)
    clause = visibility_clause(user)
    if clause is not None:
        stmt = stmt.where(clause)
    return (await session.execute(stmt)).scalar_one_or_none() is not None
