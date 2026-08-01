"""Admin dashboard backing API (#24): "the dashboard is a client of normal
API endpoints, never a backdoor" (#10's resolution) — every mutation here is
a plain FastAPI route behind the same admin gate as /connectors
(prorag/connectors/router.py's `require_admin`, reused rather than
reimplemented).

Four surfaces, matching #10's four dashboard views:

1. Documents — GET /admin/documents, paged/filtered.
2. Access rules (#4's model) — CRUD + preview/confirm/grants on
   access_rules (migration 0013). Preview and confirm share
   `_rule_candidates()`, an admin-unfiltered vector search grouped by
   document with a similarity floor (`settings.rule_similarity_floor`);
   confirm additionally persists the rule's embedding and writes
   document_acl group grants via an INSERT...SELECT...WHERE NOT EXISTS (no
   duplicate identical grants). Auto-admission of *new* documents into
   confirmed rules lives in ingest/core.py, not here — it reuses the same
   floor against chunk embeddings the ingest pipeline already computed.
3. People & groups — users (with group membership + spend), group CRUD +
   membership (local-source groups only; idp rows belong to the sync),
   reverse-ACL ("what can this person see") and per-document provenance
   ("why can X see Y").
4. Usage — GET /admin/usage, aggregated by user/day(UTC)/model, plus the
   soft-cap-warned user list.

Revocation takes effect on the next query (#10's resolution, #12.4) — there
is no cache to purge here.
"""

import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.connectors.router import require_admin
from prorag.cost import _utc_day_start
from prorag.db import get_session
from prorag.llm import embed_texts
from prorag.models import (
    AccessRule,
    Chunk,
    Connector,
    ConnectorItem,
    Document,
    DocumentAcl,
    Group,
    Usage,
    User,
    UserGroup,
)
from prorag.retrieve.arms import _distance_expr
from prorag.retrieve.visibility import visibility_clause
from prorag.schemas import (
    AccessRuleCreate,
    AccessRuleOut,
    AccessRuleUpdate,
    GroupCreate,
    GroupOut,
    UserPatch,
)
from prorag.settings import settings

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# ---- documents --------------------------------------------------------------


@router.get("/documents")
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: str | None = None,
    collection: str | None = None,
    label: str | None = None,
    source: str | None = None,
    q: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    conditions = []
    if status:
        conditions.append(Document.status == status)
    if collection:
        conditions.append(Document.collection == collection)
    if q:
        conditions.append(Document.filename.ilike(f"%{q}%"))
    if label:
        conditions.append(
            select(1)
            .select_from(DocumentAcl)
            .join(Group, Group.id == DocumentAcl.principal_id)
            .where(DocumentAcl.doc_id == Document.id, DocumentAcl.principal_type == "group", Group.name == label)
            .exists()
        )
    if source:
        conditions.append(
            select(1)
            .select_from(ConnectorItem)
            .join(Connector, Connector.id == ConnectorItem.connector_id)
            .where(ConnectorItem.doc_id == Document.id, Connector.name == source)
            .exists()
        )

    total = (await session.execute(select(func.count()).select_from(Document).where(*conditions))).scalar_one()
    rows = (
        (
            await session.execute(
                select(Document)
                .where(*conditions)
                .order_by(Document.created_at.desc())
                .limit(page_size)
                .offset((page - 1) * page_size)
            )
        )
        .scalars()
        .all()
    )

    doc_ids = [d.id for d in rows]
    labels_by_doc: dict[uuid.UUID, list[str]] = defaultdict(list)
    connector_by_doc: dict[uuid.UUID, str] = {}
    if doc_ids:
        label_rows = await session.execute(
            select(DocumentAcl.doc_id, Group.name)
            .join(Group, Group.id == DocumentAcl.principal_id)
            .where(DocumentAcl.doc_id.in_(doc_ids), DocumentAcl.principal_type == "group")
        )
        for doc_id, name in label_rows:
            labels_by_doc[doc_id].append(name)

        conn_rows = await session.execute(
            select(ConnectorItem.doc_id, Connector.name)
            .join(Connector, Connector.id == ConnectorItem.connector_id)
            .where(ConnectorItem.doc_id.in_(doc_ids))
        )
        connector_by_doc = dict(conn_rows.all())

    items = [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "error": d.error,
            "collection": d.collection,
            "page_count": d.page_count,
            "created_at": d.created_at,
            "labels": labels_by_doc.get(d.id, []),
            "connector": connector_by_doc.get(d.id),
        }
        for d in rows
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/documents/{doc_id}/access")
async def document_access(doc_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """ "Why can X see Y" (#4's resolution point 5): grant rows joined to
    rule/source provenance."""
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")

    acl_rows = (
        (
            await session.execute(
                select(DocumentAcl).where(DocumentAcl.doc_id == doc_id).order_by(DocumentAcl.created_at)
            )
        )
        .scalars()
        .all()
    )

    group_ids = {r.principal_id for r in acl_rows if r.principal_type == "group" and r.principal_id}
    user_ids = {r.principal_id for r in acl_rows if r.principal_type == "user" and r.principal_id}
    rule_ids = {uuid.UUID(r.source.removeprefix("rule:")) for r in acl_rows if r.source.startswith("rule:")}

    group_names = (
        dict((await session.execute(select(Group.id, Group.name).where(Group.id.in_(group_ids)))).all())
        if group_ids
        else {}
    )
    user_emails = (
        dict((await session.execute(select(User.id, User.email).where(User.id.in_(user_ids)))).all())
        if user_ids
        else {}
    )
    rule_names = (
        dict((await session.execute(select(AccessRule.id, AccessRule.name).where(AccessRule.id.in_(rule_ids)))).all())
        if rule_ids
        else {}
    )

    items = []
    for r in acl_rows:
        principal_name = None
        if r.principal_type == "group":
            principal_name = group_names.get(r.principal_id)
        elif r.principal_type == "user":
            principal_name = user_emails.get(r.principal_id)
        rule_id = uuid.UUID(r.source.removeprefix("rule:")) if r.source.startswith("rule:") else None
        items.append(
            {
                "principal_type": r.principal_type,
                "principal_id": r.principal_id,
                "principal_name": principal_name,
                "source": r.source,
                "rule_id": rule_id,
                "rule_name": rule_names.get(rule_id) if rule_id else None,
                "created_at": r.created_at,
            }
        )
    return {"doc_id": doc_id, "access": items}


# ---- access rules -------------------------------------------------------------


async def _get_rule_or_404(session: AsyncSession, rule_id: uuid.UUID) -> AccessRule:
    rule = await session.get(AccessRule, rule_id)
    if rule is None:
        raise HTTPException(404, "rule not found")
    return rule


async def _rule_candidates(session: AsyncSession, query_embedding: list[float]):
    """Admin-unfiltered candidate search for a rule's nl_query, one query:
    grouped by document (functional dependency on Document.id lets filename/
    title ride along without being in GROUP BY), kept only if the document's
    best chunk clears settings.rule_similarity_floor, ordered by that best
    similarity. Shared by preview (ad hoc embedding) and confirm (the
    embedding about to be persisted on the rule)."""
    best_sim = func.max(1 - _distance_expr(query_embedding))
    stmt = (
        select(Document.id.label("doc_id"), Document.filename, Document.title, best_sim.label("best_similarity"))
        .select_from(Chunk)
        .join(Document, Chunk.doc_id == Document.id)
        .where(Document.status == "ready")
        .group_by(Document.id)
        .having(best_sim >= settings.rule_similarity_floor)
        .order_by(best_sim.desc())
    )
    return (await session.execute(stmt)).all()


@router.post("/rules", response_model=AccessRuleOut, status_code=201)
async def create_rule(body: AccessRuleCreate, session: AsyncSession = Depends(get_session)):
    if await session.get(Group, body.group_id) is None:
        raise HTTPException(404, "group not found")
    rule = AccessRule(id=uuid.uuid4(), name=body.name, nl_query=body.nl_query, group_id=body.group_id)
    session.add(rule)
    await session.commit()
    return rule


@router.get("/rules", response_model=list[AccessRuleOut])
async def list_rules(session: AsyncSession = Depends(get_session)):
    return (await session.execute(select(AccessRule).order_by(AccessRule.created_at.desc()))).scalars().all()


@router.get("/rules/{rule_id}", response_model=AccessRuleOut)
async def get_rule(rule_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _get_rule_or_404(session, rule_id)


@router.patch("/rules/{rule_id}", response_model=AccessRuleOut)
async def update_rule(rule_id: uuid.UUID, body: AccessRuleUpdate, session: AsyncSession = Depends(get_session)):
    rule = await _get_rule_or_404(session, rule_id)
    if rule.state != "draft":
        # v1 ships confirm-once (#10's out-of-scope note): no pending-diff
        # re-run on edit, so editing a confirmed rule is refused instead of
        # silently leaving its membership stale.
        raise HTTPException(400, "rule is confirmed; delete and recreate it to change it")
    if body.group_id is not None and await session.get(Group, body.group_id) is None:
        raise HTTPException(404, "group not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(rule, field, value)
    await session.commit()
    return rule


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(rule_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    rule = await _get_rule_or_404(session, rule_id)
    # #4: "deleting a rule revokes its grants immediately" — every grant this
    # rule ever wrote is tagged source='rule:{id}', so revocation is one
    # scoped DELETE, no diff against current membership needed.
    await session.execute(delete(DocumentAcl).where(DocumentAcl.source == f"rule:{rule.id}"))
    await session.delete(rule)
    await session.commit()


@router.post("/rules/{rule_id}/preview")
async def preview_rule(rule_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    rule = await _get_rule_or_404(session, rule_id)
    [embedding] = await embed_texts([rule.nl_query], session=session)
    rows = await _rule_candidates(session, embedding)
    sample = [
        {"doc_id": r.doc_id, "filename": r.filename, "title": r.title, "similarity": float(r.best_similarity)}
        for r in rows[:10]
    ]
    return {"count": len(rows), "sample": sample, "similarity_floor": settings.rule_similarity_floor}


@router.post("/rules/{rule_id}/confirm", response_model=AccessRuleOut)
async def confirm_rule(rule_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    rule = await _get_rule_or_404(session, rule_id)
    if rule.state == "confirmed":
        raise HTTPException(400, "rule already confirmed")

    [embedding] = await embed_texts([rule.nl_query], session=session)
    rows = await _rule_candidates(session, embedding)
    doc_ids = [r.doc_id for r in rows]

    rule.query_embedding = embedding
    rule.state = "confirmed"
    rule.confirmed_at = datetime.now(UTC)

    if doc_ids:
        # INSERT...SELECT...WHERE NOT EXISTS: one round trip for every member
        # doc, and re-confirming (or a doc this rule already granted through
        # auto-admission) never writes a duplicate identical grant.
        await session.execute(
            text(
                """
                INSERT INTO document_acl (doc_id, principal_type, principal_id, source)
                SELECT v.doc_id, 'group', :group_id, CAST(:src AS text)
                FROM unnest(CAST(:doc_ids AS uuid[])) AS v(doc_id)
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_acl a
                    WHERE a.doc_id = v.doc_id AND a.principal_type = 'group'
                      AND a.principal_id = :group_id AND a.source = CAST(:src AS text)
                )
                """
            ),
            {"group_id": rule.group_id, "src": f"rule:{rule.id}", "doc_ids": doc_ids},
        )
    await session.commit()
    return rule


@router.get("/rules/{rule_id}/grants")
async def rule_grants(
    rule_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """The "recently auto-labelled" audit feed (#4's resolution point 2)."""
    rule = await _get_rule_or_404(session, rule_id)
    src = f"rule:{rule.id}"
    total = (
        await session.execute(select(func.count()).select_from(DocumentAcl).where(DocumentAcl.source == src))
    ).scalar_one()
    rows = (
        await session.execute(
            select(DocumentAcl.doc_id, DocumentAcl.created_at, Document.filename)
            .join(Document, Document.id == DocumentAcl.doc_id)
            .where(DocumentAcl.source == src)
            .order_by(DocumentAcl.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    items = [{"doc_id": r.doc_id, "filename": r.filename, "granted_at": r.created_at} for r in rows]
    return {"items": items, "total": total}


# ---- people & groups ----------------------------------------------------------


@router.get("/users")
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    total = (await session.execute(select(func.count()).select_from(User))).scalar_one()
    users = (
        (
            await session.execute(
                select(User).order_by(User.created_at.desc()).limit(page_size).offset((page - 1) * page_size)
            )
        )
        .scalars()
        .all()
    )
    user_ids = [u.id for u in users]

    groups_by_user: dict[uuid.UUID, list[str]] = defaultdict(list)
    spend_by_user: dict[uuid.UUID, float] = {}
    if user_ids:
        group_rows = await session.execute(
            select(UserGroup.user_id, Group.name)
            .join(Group, Group.id == UserGroup.group_id)
            .where(UserGroup.user_id.in_(user_ids))
        )
        for user_id, name in group_rows:
            groups_by_user[user_id].append(name)

        spend_rows = await session.execute(
            select(Usage.user_id, func.coalesce(func.sum(Usage.cost_usd), 0.0))
            .where(Usage.user_id.in_(user_ids), Usage.created_at >= _utc_day_start())
            .group_by(Usage.user_id)
        )
        spend_by_user = dict(spend_rows.all())

    items = [
        {
            "id": u.id,
            "email": u.email,
            "display_name": u.display_name,
            "is_admin": u.is_admin,
            "disabled_at": u.disabled_at,
            "daily_cap_usd_override": u.daily_cap_usd_override,
            "groups": groups_by_user.get(u.id, []),
            "today_spend_usd": spend_by_user.get(u.id, 0.0),
        }
        for u in users
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.patch("/users/{user_id}")
async def update_user(user_id: uuid.UUID, body: UserPatch, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    await session.commit()
    return {
        "id": user.id,
        "email": user.email,
        "is_admin": user.is_admin,
        "disabled_at": user.disabled_at,
        "daily_cap_usd_override": user.daily_cap_usd_override,
    }


@router.get("/users/{user_id}/visible-docs")
async def user_visible_docs(
    user_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """Reverse ACL: "what can this person see" (#4's resolution point 5),
    reusing the exact predicate query-time enforcement uses (#18's
    visibility_clause) rather than a second filtering dialect."""
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(404, "user not found")

    conditions = [Document.status == "ready"]
    clause = visibility_clause(target)
    if clause is not None:
        conditions.append(clause)

    total = (await session.execute(select(func.count()).select_from(Document).where(*conditions))).scalar_one()
    rows = (
        (
            await session.execute(
                select(Document).where(*conditions).order_by(Document.created_at.desc()).limit(limit).offset(offset)
            )
        )
        .scalars()
        .all()
    )
    items = [{"id": d.id, "filename": d.filename, "collection": d.collection} for d in rows]
    return {"items": items, "total": total}


@router.post("/groups", response_model=GroupOut, status_code=201)
async def create_group(body: GroupCreate, session: AsyncSession = Depends(get_session)):
    group = Group(id=uuid.uuid4(), name=body.name, source="local")
    session.add(group)
    await session.commit()
    return group


@router.get("/groups", response_model=list[GroupOut])
async def list_groups(session: AsyncSession = Depends(get_session)):
    return (await session.execute(select(Group).order_by(Group.name))).scalars().all()


async def _get_local_group_or_404(session: AsyncSession, group_id: uuid.UUID) -> Group:
    group = await session.get(Group, group_id)
    if group is None:
        raise HTTPException(404, "group not found")
    if group.source != "local":
        # idp rows are the sync's (#24 scope note) — this endpoint only
        # mutates locally-created groups.
        raise HTTPException(400, "idp-synced groups are managed by the sync, not this endpoint")
    return group


@router.patch("/groups/{group_id}", response_model=GroupOut)
async def update_group(group_id: uuid.UUID, body: GroupCreate, session: AsyncSession = Depends(get_session)):
    group = await _get_local_group_or_404(session, group_id)
    group.name = body.name
    await session.commit()
    return group


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(group_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    group = await _get_local_group_or_404(session, group_id)
    await session.delete(group)
    await session.commit()


@router.post("/groups/{group_id}/members/{user_id}", status_code=204)
async def add_group_member(group_id: uuid.UUID, user_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    await _get_local_group_or_404(session, group_id)
    if await session.get(User, user_id) is None:
        raise HTTPException(404, "user not found")
    if await session.get(UserGroup, (user_id, group_id)) is None:
        session.add(UserGroup(user_id=user_id, group_id=group_id))
        await session.commit()


@router.delete("/groups/{group_id}/members/{user_id}", status_code=204)
async def remove_group_member(group_id: uuid.UUID, user_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    await _get_local_group_or_404(session, group_id)
    await session.execute(delete(UserGroup).where(UserGroup.user_id == user_id, UserGroup.group_id == group_id))
    await session.commit()


# ---- usage --------------------------------------------------------------------

_WINDOW_RE = re.compile(r"^(\d+)d$")


@router.get("/usage")
async def usage_report(window: str = "7d", session: AsyncSession = Depends(get_session)):
    m = _WINDOW_RE.match(window)
    if not m:
        raise HTTPException(400, "window must look like '7d'")
    start = datetime.now(UTC) - timedelta(days=int(m.group(1)))

    rows = (
        await session.execute(
            text(
                """
                SELECT user_id,
                       (created_at AT TIME ZONE 'UTC')::date AS day,
                       model,
                       SUM(cost_usd) AS cost_usd,
                       SUM(prompt_tokens + completion_tokens) AS tokens,
                       COUNT(*) AS calls
                FROM usage
                WHERE created_at >= :start
                GROUP BY user_id, day, model
                ORDER BY day DESC
                """
            ),
            {"start": start},
        )
    ).all()
    items = [
        {
            "user_id": r.user_id,
            "day": r.day,
            "model": r.model,
            "cost_usd": float(r.cost_usd),
            "tokens": int(r.tokens),
            "calls": int(r.calls),
        }
        for r in rows
    ]

    # warned-users = today_user_cost >= their effective cap (override, else
    # the install-wide settings.user_daily_cap_usd) — same threshold
    # cost.budget_decision()'s 'warn' branch uses, computed in bulk here
    # instead of one today_user_cost_usd() call per user.
    today_start = _utc_day_start()
    spend_rows = await session.execute(
        select(Usage.user_id, func.coalesce(func.sum(Usage.cost_usd), 0.0))
        .where(Usage.user_id.is_not(None), Usage.created_at >= today_start)
        .group_by(Usage.user_id)
    )
    today_spend = dict(spend_rows.all())

    caps: dict[uuid.UUID, float] = {}
    if today_spend:
        cap_rows = await session.execute(
            select(User.id, User.daily_cap_usd_override).where(User.id.in_(today_spend.keys()))
        )
        caps = {uid: (override if override is not None else settings.user_daily_cap_usd) for uid, override in cap_rows}

    warned_users = [uid for uid, spent in today_spend.items() if spent >= caps.get(uid, settings.user_daily_cap_usd)]

    return {"items": items, "warned_users": warned_users}
