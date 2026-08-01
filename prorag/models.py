"""SQLAlchemy models. Phase 1: documents + chunks. Phase 3 adds tables/table_rows
and structured-document columns on chunks (bbox, heading_path, title_norm, table_id).
Phase 4 adds chats/messages/citations. Phase 5 adds api_keys, usage, feedback.
0008 adds identity + ACL (users, groups, user_groups, document_acl) per the
decisions in #2 and #15 — schema only, no enforcement (that's #18). 0011 adds
connectors + connector_items (#22): the first sync-engine source (S3), Tier C
per #15 (no ACLs to mirror, so no document_acl rows — admin-only by default).

# ponytail: a `jobs` table (§7, SKIP LOCKED queue) still isn't implemented —
# ingestion stays inline; Phase 5's retry loop lives in ingest/router.py
# instead. Add the real queue if/when ingestion needs to run out-of-request.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    FetchedValue,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from prorag.settings import settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sha256: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime: Mapped[str] = mapped_column(String, nullable=False)
    blob_path: Mapped[str] = mapped_column(String, nullable=False)
    page_count: Mapped[int | None] = mapped_column(nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    title_norm: Mapped[str | None] = mapped_column(String, nullable=True)  # §4.4 revision dedup key
    collection: Mapped[str] = mapped_column(String, nullable=False, default="default")
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    tables: Mapped[list["Table"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ord: Mapped[int] = mapped_column(nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False, default="prose")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embed_text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    page_start: Mapped[int | None] = mapped_column(nullable=True)
    page_end: Mapped[int | None] = mapped_column(nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    table_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tables.id", ondelete="CASCADE"), nullable=True
    )
    token_count: Mapped[int] = mapped_column(nullable=False)
    embedding = mapped_column(Vector(settings.embed_dim), nullable=False)
    # generated column (0002) — FetchedValue keeps SQLAlchemy from including it in INSERTs
    tsv = mapped_column(TSVECTOR, FetchedValue(), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    table: Mapped["Table | None"] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_chunks_doc_id_ord", "doc_id", "ord"),)


class Table(Base):
    __tablename__ = "tables"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    columns: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="tables")
    rows: Mapped[list["TableRow"]] = relationship(back_populates="table", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="table")


class TableRow(Base):
    __tablename__ = "table_rows"

    id: Mapped[int] = mapped_column(primary_key=True)
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tables.id", ondelete="CASCADE"), nullable=False
    )
    row_no: Mapped[int] = mapped_column(nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)

    table: Mapped["Table"] = relationship(back_populates="rows")

    __table_args__ = (Index("ix_table_rows_table_id", "table_id"),)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chat: Mapped["Chat"] = relationship(back_populates="messages")
    citations: Mapped[list["Citation"]] = relationship(back_populates="message", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_messages_chat_id", "chat_id"),)


class Citation(Base):
    """One row per [Sn] the answerer actually cited — the writer QDMS-AI's
    dead `cited_sources` column never had (§5.3)."""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    n: Mapped[int] = mapped_column(nullable=False)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    page: Mapped[int | None] = mapped_column(nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)

    message: Mapped["Message"] = relationship(back_populates="citations")

    __table_args__ = (Index("ix_citations_message_id", "message_id"),)


class User(Base):
    """A principal (#2): the deployment is the tenant, the user is who budgets
    and grants attach to. `external_subject` is the IdP `sub`, null for local
    accounts. Group membership is never read from token claims per-request —
    `user_groups` rows are the truth (#2). `password_hash` (0009, #19) is
    nullable because OIDC-only users have none."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_subject: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    password_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    # Admin override for the per-user soft cap (0010, #21) — null means "use
    # settings.user_daily_cap_usd". The dashboard edits this later (#10).
    daily_cap_usd_override: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    disabled_at: Mapped[datetime | None] = mapped_column(nullable=True)


class Group(Base):
    """A grantable principal (#2, #15) — `source` distinguishes locally-created
    groups from ones synced from an IdP; `external_id` is only set for the
    latter. Unique on (source, external_id) is a partial index (migration
    0008) since local groups have no external_id to collide on."""

    __tablename__ = "groups"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False, default="local")  # local | idp
    external_id: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (CheckConstraint("source IN ('local', 'idp')", name="ck_groups_source"),)


class UserGroup(Base):
    """Membership row — the truth queries read (#2); IdP claims may seed this
    at login but are never trusted per-request."""

    __tablename__ = "user_groups"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (PrimaryKeyConstraint("user_id", "group_id"),)


class DocumentAcl(Base):
    """Mirrors the *source's* stated principals (#15) — not materialised
    grants. `principal_id` is null only for `principal_type='public'`. Default
    deny: no row for a document means no access, except the one-time
    compatibility backfill migration 0008 performs for pre-existing documents
    (see that migration's docstring)."""

    __tablename__ = "document_acl"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    principal_type: Mapped[str] = mapped_column(String, nullable=False)  # user | group | public
    principal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="local")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint("principal_type IN ('user', 'group', 'public')", name="ck_document_acl_principal_type"),
        Index("ix_document_acl_doc_id", "doc_id"),
        Index("ix_document_acl_principal", "principal_type", "principal_id"),
    )


class Session(Base):
    """A server-side session (0009, #19) backing the `prorag_session` cookie.
    Same idiom as `ApiKey`: only `token_hash` is stored, the raw token is
    handed to the browser once. `revoked_at` lets logout invalidate a session
    without waiting for `expires_at` — this is why sessions exist instead of a
    self-verifying JWT (#2's resolution: "nothing to revoke server-side is the
    wrong default")."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # explicit timezone=True: unlike created_at/revoked_at's server-side
    # defaults elsewhere in this file, expires_at is always a Python-supplied
    # value (create_session()) — without this, SQLAlchemy binds it as
    # TIMESTAMP WITHOUT TIME ZONE and asyncpg rejects the tz-aware datetime.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ApiKey(Base):
    """Bearer API keys (§6, §8 Phase 5). Only the hash is stored; the raw key
    is printed once by scripts/create_api_key.py and never persisted.
    `user_id` is nullable — null is the legacy unscoped super-key (#2)."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    collection: Mapped[str | None] = mapped_column(String, nullable=True)  # null = unscoped
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Usage(Base):
    """One row per planner/answer/embedding LLM call (§5.4). `message_id` is
    nullable because planner + embedding calls happen before a message exists.
    `user_id` is nullable (pre-#2 rows, machine calls) and indexed together
    with `created_at` for the per-user daily budget query (#2)."""

    __tablename__ = "usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    # explicit timezone=True (bug found while building #21's UTC-window query):
    # the DB column is TIMESTAMPTZ (migration 0005), but without this SQLAlchemy
    # infers a naive TIMESTAMP on the Python side and asyncpg rejects the
    # tz-aware `datetime` today_cost_usd()/today_user_cost_usd() compare
    # against — same footgun Session.expires_at's docstring already documents.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_usage_created_at", "created_at"),
        Index("ix_usage_user_id_created_at", "user_id", "created_at"),
    )


class Feedback(Base):
    """Like/dislike on a message (§6 /feedback). One row per message; posting
    the same rating again toggles it off, same as QDMS-AI."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[str] = mapped_column(String, nullable=False)  # up | down
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_feedback_message_id", "message_id"),
        UniqueConstraint("message_id", name="uq_feedback_message_id"),
    )


class Connector(Base):
    """A configured source (#22, #15's polling-first architecture). `type` is
    currently only 's3' (S3-compatible object storage — AWS/MinIO/R2, the
    plumbing connector per #6: cheapest to stand up, no source ACLs to
    mirror). `config` holds endpoint_url/bucket/prefix/access key id/secret
    plus an optional `collection` — the secret is stored as plain JSONB v1;
    an env-ref indirection (`{"$env": "MY_SECRET"}`-style) is the natural
    upgrade path once this surface is wider than admin-only. `last_sync_at`
    drives the next incremental poll's client-side since-filter (S3 has no
    server-side "changed since" list filter); `last_full_sweep_at` is
    informational (the mandatory reconciliation sweep per #15 doesn't use it
    for filtering — it always lists everything to catch deletes/moves)."""

    __tablename__ = "connectors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_full_sweep_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # #23: the scheduler has no HTTP caller to surface an exception to, so a
    # failed scheduled run records it here instead; cleared on the next
    # success. Manual syncs (POST /connectors/{id}/sync) raise instead.
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ConnectorItem(Base):
    """One row per remote object a connector has ever seen (#22) — the diff
    baseline for both the incremental poll and the full sweep. `external_id`
    is the source's own item identity (S3: the object key). `etag`+`size`
    are the change signal compared against the live listing each sync
    (last_modified is trusted less across S3-compatible providers' clocks,
    so it isn't part of the diff, only informational). `doc_id` is null for
    skipped/errored items and is nulled back out by the full sweep when the
    backing Document is deleted. `status`: synced | skipped | error |
    deleted. No document_acl rows are ever created for a connector-ingested
    doc (#15 Tier C — S3 has no source ACLs to mirror): it's invisible to
    non-admins until an admin grants access (#18), by design, not an
    oversight."""

    __tablename__ = "connector_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    connector_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False
    )
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    etag: Mapped[str | None] = mapped_column(String, nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    doc_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        UniqueConstraint("connector_id", "external_id", name="uq_connector_items_connector_id_external_id"),
        Index("ix_connector_items_connector_id", "connector_id"),
    )


class EvalRun(Base):
    """One row per `POST /eval/run` (§6, §8 Phase 6). `questions` is the
    per-question JSONB (answer + deterministic scores); `aggregate` is the
    mean of those plus ragas scores when ragas is installed."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    questions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    aggregate: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
