"""Tests for the ACL enforcement layer (#18), built on #17's document_acl
schema (#2, #15 decisions). Two halves, same split as test_retrieval.py:

- pure unit tests for visibility_clause() (no DB)
- one DB-backed round trip proving a grant/revoke/group-grant cycle actually
  changes what vector_search/keyword_search/visible_doc_guard return; skips
  cleanly if the DB is unreachable, same pattern as test_identity_schema.py.
"""

import random
import uuid

import pytest
from sqlalchemy import delete, text

from prorag.db import SessionLocal, engine
from prorag.models import Chunk, Document, DocumentAcl, Group, User, UserGroup
from prorag.retrieve.arms import keyword_search, vector_search
from prorag.retrieve.visibility import visibility_clause, visible_doc_guard
from prorag.settings import settings

# ---- visibility_clause (pure) ----------------------------------------------


def test_visibility_clause_none_for_unscoped_user():
    """user=None is the super-principal (auth disabled / legacy unscoped
    key, #2) — no filtering."""
    assert visibility_clause(None) is None


def test_visibility_clause_none_for_admin():
    admin = User(email="admin@example.com", is_admin=True)
    assert visibility_clause(admin) is None


def test_visibility_clause_is_exists_over_document_acl_with_three_branches():
    user = User(email="user@example.com", is_admin=False)
    user.id = uuid.uuid4()  # normally set by the DB default; a plain literal here is fine for compile()

    clause = visibility_clause(user)
    compiled = str(clause.compile(compile_kwargs={"literal_binds": True}))

    assert compiled.strip().startswith("EXISTS")
    assert "document_acl" in compiled
    # public branch
    assert "principal_type = 'public'" in compiled
    # direct user-grant branch, bound to this user's id (dialect renders the
    # literal UUID without dashes)
    assert "principal_type = 'user'" in compiled
    assert user.id.hex in compiled
    # group-grant branch, resolved via a correlated user_groups subquery
    assert "principal_type = 'group'" in compiled
    assert "user_groups" in compiled
    assert "user_groups.user_id" in compiled


# ---- DB-backed grant/revoke/group-grant cycle -------------------------------


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def test_document_acl_grant_revoke_group_grant_cycle():
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]
    unique_word = f"zzvisacl{tag}"

    try:
        async with session:
            user = User(email=f"{tag}@example.com")
            admin = User(email=f"{tag}-admin@example.com", is_admin=True)
            group = Group(name=f"group-{tag}")
            session.add_all([user, admin, group])
            await session.flush()

            doc = Document(
                sha256=tag,
                filename="visibility.txt",
                mime="text/plain",
                blob_path="/tmp/visibility.txt",
                status="ready",
            )
            session.add(doc)
            await session.flush()

            embedding = [random.random() for _ in range(settings.embed_dim)]
            chunk = Chunk(
                doc_id=doc.id,
                ord=0,
                kind="prose",
                text=f"the {unique_word} marker sentence",
                embed_text=f"the {unique_word} marker sentence",
                token_count=5,
                embedding=embedding,
            )
            session.add(chunk)

            # This doc is new (not part of migration 0008's pre-existing-doc
            # backfill), so it starts with zero ACL rows — add the same kind
            # of 'public' grant that backfill gave pre-existing documents, to
            # exercise that branch explicitly.
            public_grant = DocumentAcl(doc_id=doc.id, principal_type="public", principal_id=None)
            session.add(public_grant)
            await session.commit()

            try:
                # public grant -> visible to a plain user, an admin, and user=None
                assert await visible_doc_guard(session, user, doc.id) is True
                assert await visible_doc_guard(session, admin, doc.id) is True
                assert await visible_doc_guard(session, None, doc.id) is True

                vec_hits = {h["doc_id"] for h in await vector_search(session, embedding, 5, user=user)}
                assert doc.id in vec_hits
                kw_hits = {h["doc_id"] for h in await keyword_search(session, unique_word, 5, user=user)}
                assert doc.id in kw_hits

                # revoke the public grant -> invisible to the plain user, still
                # visible to admin and to user=None (super-principals bypass)
                await session.execute(delete(DocumentAcl).where(DocumentAcl.id == public_grant.id))
                await session.commit()

                assert await visible_doc_guard(session, user, doc.id) is False
                assert await visible_doc_guard(session, admin, doc.id) is True
                assert await visible_doc_guard(session, None, doc.id) is True

                vec_hits = {h["doc_id"] for h in await vector_search(session, embedding, 5, user=user)}
                assert doc.id not in vec_hits
                kw_hits = {h["doc_id"] for h in await keyword_search(session, unique_word, 5, user=user)}
                assert doc.id not in kw_hits

                admin_vec_hits = {h["doc_id"] for h in await vector_search(session, embedding, 5, user=admin)}
                assert doc.id in admin_vec_hits
                unscoped_vec_hits = {h["doc_id"] for h in await vector_search(session, embedding, 5, user=None)}
                assert doc.id in unscoped_vec_hits

                # grant via group membership -> visible again to the plain user
                session.add(UserGroup(user_id=user.id, group_id=group.id))
                session.add(DocumentAcl(doc_id=doc.id, principal_type="group", principal_id=group.id))
                await session.commit()

                assert await visible_doc_guard(session, user, doc.id) is True
                vec_hits = {h["doc_id"] for h in await vector_search(session, embedding, 5, user=user)}
                assert doc.id in vec_hits
                kw_hits = {h["doc_id"] for h in await keyword_search(session, unique_word, 5, user=user)}
                assert doc.id in kw_hits
            finally:
                await session.execute(delete(DocumentAcl).where(DocumentAcl.doc_id == doc.id))
                await session.execute(delete(UserGroup).where(UserGroup.user_id == user.id))
                await session.execute(delete(Chunk).where(Chunk.doc_id == doc.id))
                await session.execute(delete(Document).where(Document.id == doc.id))
                await session.execute(delete(Group).where(Group.id == group.id))
                await session.execute(delete(User).where(User.id.in_([user.id, admin.id])))
                await session.commit()
    finally:
        # Own event loop per test (pytest-asyncio); dispose so pooled
        # connections don't stay bound to this loop and break the next DB
        # test's checkout — same reasoning as test_identity_schema.py.
        await engine.dispose()
