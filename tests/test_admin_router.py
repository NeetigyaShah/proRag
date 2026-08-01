"""Tests for the admin dashboard backing API (#24): endpoint gating, the
access-rule preview/confirm/grants/delete flow (#4's model) plus
auto-admission of new documents into a confirmed rule, reverse-ACL
correctness ("what can this person see"), documents-listing filters, group
membership's local-source-only guard, and usage aggregation.

DB-backed tests follow tests/test_identity_schema.py's pattern: skip cleanly
if Postgres is unreachable, clean up rows in `finally`. Embeddings are
stubbed with random-but-seeded vectors (prorag.llm never called), the same
idea as tests/test_connectors_sync.py's embed stub — no live LLM provider
needed. tests/conftest.py's autouse fixture disposes the engine after every
test, so nothing here does that itself.
"""

import random
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, text

import prorag.admin.router as admin_router
import prorag.ingest.core as ingest_core
from prorag.auth import SESSION_COOKIE_NAME, create_session
from prorag.db import SessionLocal
from prorag.main import app
from prorag.models import (
    AccessRule,
    Chunk,
    Connector,
    ConnectorItem,
    Document,
    DocumentAcl,
    Group,
    Session,
    User,
    UserGroup,
)
from prorag.settings import settings


def _vec(seed: int) -> list[float]:
    # ponytail: uniform(-1, 1), not random() — [0,1)-only components share a
    # "positive orthant" bias that gives *independent* random vectors an
    # expected cosine similarity of ~0.75 in high dimensions (n/4 over n/3),
    # which clears rule_similarity_floor by accident. Zero-mean components
    # make unrelated vectors land near-orthogonal (~0) as the test intends.
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(settings.embed_dim)]


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def _make_ready_doc(session, tag: str, suffix: str, embedding: list[float]) -> Document:
    doc = Document(
        sha256=f"{tag}-{suffix}",
        filename=f"{suffix}.txt",
        mime="text/plain",
        blob_path=f"/tmp/{tag}-{suffix}.txt",
        status="ready",
    )
    session.add(doc)
    await session.flush()
    session.add(
        Chunk(
            doc_id=doc.id,
            ord=0,
            kind="prose",
            text=f"{suffix} marker content",
            embed_text=f"{suffix} marker content",
            token_count=3,
            embedding=embedding,
        )
    )
    return doc


# ---- endpoint gating ----------------------------------------------------------


async def test_admin_documents_wide_open_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/documents")
    assert resp.status_code == 200


async def test_admin_401_with_no_credentials_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin/documents")
    assert resp.status_code == 401


async def test_admin_403_for_authenticated_non_admin_then_200_for_admin(monkeypatch):
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    async with session:
        plain = User(email=f"{tag}-plain@example.com", is_admin=False)
        admin = User(email=f"{tag}-admin@example.com", is_admin=True)
        session.add_all([plain, admin])
        await session.flush()
        plain_token = await create_session(plain.id, session)
        admin_token = await create_session(admin.id, session)
        await session.commit()

        try:
            monkeypatch.setattr(settings, "auth_enabled", True)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/admin/documents", cookies={SESSION_COOKIE_NAME: plain_token})
                assert resp.status_code == 403

                resp = await client.get("/admin/documents", cookies={SESSION_COOKIE_NAME: admin_token})
                assert resp.status_code == 200
        finally:
            await session.execute(delete(Session).where(Session.user_id.in_([plain.id, admin.id])))
            await session.execute(delete(User).where(User.id.in_([plain.id, admin.id])))
            await session.commit()


# ---- documents listing ---------------------------------------------------------


async def test_documents_listing_filters_by_label_source_and_q(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]

    async with session:
        group = Group(name=f"grp-{tag}")
        connector = Connector(id=uuid.uuid4(), type="s3", name=f"conn-{tag}", config={})
        session.add_all([group, connector])
        await session.flush()

        labelled = await _make_ready_doc(session, tag, f"labelled-{tag}", _vec(1))
        synced = await _make_ready_doc(session, tag, f"synced-{tag}", _vec(2))
        plain = await _make_ready_doc(session, tag, f"plain-{tag}", _vec(3))
        session.add(DocumentAcl(doc_id=labelled.id, principal_type="group", principal_id=group.id, source="local"))
        session.add(
            ConnectorItem(connector_id=connector.id, external_id=f"{tag}.txt", doc_id=synced.id, status="synced")
        )
        await session.commit()

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/admin/documents", params={"label": group.name})
                assert resp.status_code == 200
                ids = {row["id"] for row in resp.json()["items"]}
                assert str(labelled.id) in ids
                assert str(plain.id) not in ids

                resp = await client.get("/admin/documents", params={"source": connector.name})
                ids = {row["id"] for row in resp.json()["items"]}
                assert str(synced.id) in ids
                assert str(labelled.id) not in ids

                resp = await client.get("/admin/documents", params={"q": f"plain-{tag}"})
                body = resp.json()
                assert body["total"] == 1
                assert body["items"][0]["id"] == str(plain.id)
                assert body["items"][0]["labels"] == []
                assert body["items"][0]["connector"] is None

                resp = await client.get("/admin/documents", params={"q": f"labelled-{tag}"})
                assert resp.json()["items"][0]["labels"] == [group.name]

                resp = await client.get("/admin/documents", params={"q": f"synced-{tag}"})
                assert resp.json()["items"][0]["connector"] == connector.name
        finally:
            doc_ids = [labelled.id, synced.id, plain.id]
            await session.execute(delete(DocumentAcl).where(DocumentAcl.doc_id.in_(doc_ids)))
            await session.execute(delete(ConnectorItem).where(ConnectorItem.connector_id == connector.id))
            await session.execute(delete(Chunk).where(Chunk.doc_id.in_(doc_ids)))
            await session.execute(delete(Document).where(Document.id.in_(doc_ids)))
            await session.execute(delete(Connector).where(Connector.id == connector.id))
            await session.execute(delete(Group).where(Group.id == group.id))
            await session.commit()


# ---- rule preview / confirm / grants / delete + reverse ACL -------------------


async def test_rule_preview_confirm_grants_delete_and_reverse_acl(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    rule_vec = _vec(101)

    async with session:
        group = Group(name=f"grp-{tag}")
        member_user = User(email=f"{tag}-member@example.com")
        session.add_all([group, member_user])
        await session.flush()

        # matching doc's chunk embedding IS the rule's embedding (similarity
        # 1.0, clears the floor deterministically); the other doc's chunk
        # embedding is an independent random vector, which in 1536-dim space
        # is near-orthogonal (similarity << settings.rule_similarity_floor).
        matching = await _make_ready_doc(session, tag, f"match-{tag}", rule_vec)
        other = await _make_ready_doc(session, tag, f"other-{tag}", _vec(202))

        rule = AccessRule(id=uuid.uuid4(), name=f"rule-{tag}", nl_query="networking docs", group_id=group.id)
        session.add(rule)
        session.add(UserGroup(user_id=member_user.id, group_id=group.id))
        await session.commit()

        async def _fake_embed(texts, session=None):
            assert texts == [rule.nl_query]
            return [rule_vec]

        monkeypatch.setattr(admin_router, "embed_texts", _fake_embed)

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # preview: sees only the matching doc, doesn't persist anything
                resp = await client.post(f"/admin/rules/{rule.id}/preview")
                assert resp.status_code == 200
                body = resp.json()
                assert body["count"] == 1
                assert body["similarity_floor"] == settings.rule_similarity_floor
                assert body["sample"][0]["doc_id"] == str(matching.id)
                assert body["sample"][0]["similarity"] == pytest.approx(1.0, abs=1e-6)

                await session.refresh(rule)
                assert rule.state == "draft"
                assert rule.query_embedding is None

                # confirm: persists the embedding, writes the group grant
                resp = await client.post(f"/admin/rules/{rule.id}/confirm")
                assert resp.status_code == 200
                confirmed = resp.json()
                assert confirmed["state"] == "confirmed"
                assert confirmed["confirmed_at"] is not None

                # confirming again is refused (v1 ships confirm-once)
                resp = await client.post(f"/admin/rules/{rule.id}/confirm")
                assert resp.status_code == 400

                grant = (
                    await session.execute(
                        select(DocumentAcl).where(
                            DocumentAcl.doc_id == matching.id,
                            DocumentAcl.principal_type == "group",
                            DocumentAcl.principal_id == group.id,
                            DocumentAcl.source == f"rule:{rule.id}",
                        )
                    )
                ).scalar_one_or_none()
                assert grant is not None

                no_grant = (
                    await session.execute(select(DocumentAcl).where(DocumentAcl.doc_id == other.id))
                ).scalar_one_or_none()
                assert no_grant is None

                # audit feed
                resp = await client.get(f"/admin/rules/{rule.id}/grants")
                grants_body = resp.json()
                assert grants_body["total"] == 1
                assert grants_body["items"][0]["doc_id"] == str(matching.id)

                # reverse ACL: the group member sees exactly the granted doc
                resp = await client.get(f"/admin/users/{member_user.id}/visible-docs", params={"limit": 200})
                visible_ids = {row["id"] for row in resp.json()["items"]}
                assert str(matching.id) in visible_ids
                assert str(other.id) not in visible_ids

                # provenance: "why can X see Y"
                resp = await client.get(f"/admin/documents/{matching.id}/access")
                access = resp.json()["access"]
                assert len(access) == 1
                assert access[0]["source"] == f"rule:{rule.id}"
                assert access[0]["rule_name"] == rule.name
                assert access[0]["principal_name"] == group.name

                # deleting the rule revokes its grants immediately (#4)
                resp = await client.delete(f"/admin/rules/{rule.id}")
                assert resp.status_code == 204

            revoked = (
                await session.execute(select(DocumentAcl).where(DocumentAcl.doc_id == matching.id))
            ).scalar_one_or_none()
            assert revoked is None
        finally:
            doc_ids = [matching.id, other.id]
            await session.execute(delete(DocumentAcl).where(DocumentAcl.doc_id.in_(doc_ids)))
            await session.execute(delete(AccessRule).where(AccessRule.id == rule.id))
            await session.execute(delete(UserGroup).where(UserGroup.user_id == member_user.id))
            await session.execute(delete(Chunk).where(Chunk.doc_id.in_(doc_ids)))
            await session.execute(delete(Document).where(Document.id.in_(doc_ids)))
            await session.execute(delete(User).where(User.id == member_user.id))
            await session.execute(delete(Group).where(Group.id == group.id))
            await session.commit()


# ---- auto-admission on ingest --------------------------------------------------


async def test_auto_admission_grants_new_document_matching_confirmed_rule(monkeypatch):
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    rule_vec = _vec(303)

    async with session:
        group = Group(name=f"grp-auto-{tag}")
        session.add(group)
        await session.flush()

        rule = AccessRule(
            id=uuid.uuid4(),
            name=f"rule-auto-{tag}",
            nl_query="auto admission docs",
            group_id=group.id,
            state="confirmed",
            query_embedding=rule_vec,
            confirmed_at=datetime.now(UTC),
        )
        session.add(rule)
        await session.commit()

        async def _fake_embed_batched(texts, session=None):
            # the doc's one chunk embeds to exactly the rule's stored
            # embedding, so the floor clears deterministically (per the
            # task's guidance for this test).
            return [rule_vec for _ in texts]

        monkeypatch.setattr(ingest_core, "embed_texts_batched", _fake_embed_batched)

        doc_id = None
        try:
            data = f"auto admission marker document body {tag}".encode()
            result = await ingest_core.ingest_bytes(session, data, f"auto-{tag}.txt", collection="default")
            assert result.status == "ready"
            doc_id = result.doc_id

            grant = (
                await session.execute(
                    select(DocumentAcl).where(
                        DocumentAcl.doc_id == doc_id,
                        DocumentAcl.principal_type == "group",
                        DocumentAcl.principal_id == group.id,
                        DocumentAcl.source == f"rule:{rule.id}",
                    )
                )
            ).scalar_one_or_none()
            assert grant is not None
        finally:
            if doc_id is not None:
                await session.execute(delete(DocumentAcl).where(DocumentAcl.doc_id == doc_id))
                await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
                await session.execute(delete(Document).where(Document.id == doc_id))
            await session.execute(delete(AccessRule).where(AccessRule.id == rule.id))
            await session.execute(delete(Group).where(Group.id == group.id))
            await session.commit()


# ---- group membership: local-source only ---------------------------------------


async def test_group_membership_rejected_for_idp_synced_group(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]

    async with session:
        idp_group = Group(name=f"idp-{tag}", source="idp", external_id=f"ext-{tag}")
        user = User(email=f"{tag}-u@example.com")
        session.add_all([idp_group, user])
        await session.commit()

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(f"/admin/groups/{idp_group.id}/members/{user.id}")
                assert resp.status_code == 400

                resp = await client.delete(f"/admin/groups/{idp_group.id}")
                assert resp.status_code == 400
        finally:
            await session.execute(delete(UserGroup).where(UserGroup.user_id == user.id))
            await session.execute(delete(User).where(User.id == user.id))
            await session.execute(delete(Group).where(Group.id == idp_group.id))
            await session.commit()


# ---- usage aggregation ----------------------------------------------------------


async def test_usage_report_aggregates_and_flags_warned_users(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]

    async with session:
        from prorag.models import Usage

        warned = User(email=f"{tag}-warned@example.com", daily_cap_usd_override=0.01)
        ok = User(email=f"{tag}-ok@example.com", daily_cap_usd_override=1000.0)
        session.add_all([warned, ok])
        await session.flush()
        session.add_all(
            [
                Usage(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50, cost_usd=0.5, user_id=warned.id),
                Usage(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5, cost_usd=0.01, user_id=ok.id),
            ]
        )
        await session.commit()

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/admin/usage", params={"window": "7d"})
                assert resp.status_code == 200
                body = resp.json()
                by_user = {
                    row["user_id"]: row for row in body["items"] if row["user_id"] in (str(warned.id), str(ok.id))
                }
                assert by_user[str(warned.id)]["cost_usd"] == pytest.approx(0.5)
                assert by_user[str(ok.id)]["cost_usd"] == pytest.approx(0.01)
                assert str(warned.id) in body["warned_users"]
                assert str(ok.id) not in body["warned_users"]

                resp = await client.get("/admin/usage", params={"window": "bogus"})
                assert resp.status_code == 400
        finally:
            await session.execute(delete(Usage).where(Usage.user_id.in_([warned.id, ok.id])))
            await session.execute(delete(User).where(User.id.in_([warned.id, ok.id])))
            await session.commit()
