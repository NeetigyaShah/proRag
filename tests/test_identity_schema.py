"""DB-backed round-trip for the identity + ACL schema (migration 0008, #17):
create a user, a group, a membership row, and a document_acl row, then query
them back. Skips cleanly if the DB is unreachable, same pattern as the EXPLAIN
test in tests/test_retrieval.py."""

import uuid

import pytest
from sqlalchemy import delete, select, text

from prorag.db import SessionLocal
from prorag.models import Document, DocumentAcl, Group, User, UserGroup


async def _get_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def test_identity_and_acl_round_trip():
    session = await _get_session()
    tag = uuid.uuid4().hex[:8]

    async with session:
        user = User(email=f"{tag}@example.com", display_name="Test User")
        group = Group(name=f"group-{tag}")
        session.add_all([user, group])
        await session.flush()

        session.add(UserGroup(user_id=user.id, group_id=group.id))

        doc = Document(sha256=tag, filename="f.txt", mime="text/plain", blob_path="/tmp/f.txt", status="ready")
        session.add(doc)
        await session.flush()

        session.add(DocumentAcl(doc_id=doc.id, principal_type="user", principal_id=user.id, source="local"))
        await session.commit()

        try:
            got_user = (await session.execute(select(User).where(User.id == user.id))).scalar_one()
            assert got_user.email == f"{tag}@example.com"
            assert got_user.is_admin is False

            got_membership = (
                await session.execute(
                    select(UserGroup).where(UserGroup.user_id == user.id, UserGroup.group_id == group.id)
                )
            ).scalar_one()
            assert got_membership.group_id == group.id

            got_acl = (await session.execute(select(DocumentAcl).where(DocumentAcl.doc_id == doc.id))).scalar_one()
            assert got_acl.principal_type == "user"
            assert got_acl.principal_id == user.id
        finally:
            await session.execute(delete(DocumentAcl).where(DocumentAcl.doc_id == doc.id))
            await session.execute(delete(UserGroup).where(UserGroup.user_id == user.id))
            await session.execute(delete(Document).where(Document.id == doc.id))
            await session.execute(delete(Group).where(Group.id == group.id))
            await session.execute(delete(User).where(User.id == user.id))
            await session.commit()
