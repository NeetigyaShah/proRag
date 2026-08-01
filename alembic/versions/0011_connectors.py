"""add connectors + connector_items (#22)

First sync-engine source (S3-compatible object storage, #6's plumbing
connector) per #15's polling-first architecture. `connectors.config` (JSONB)
holds endpoint_url/bucket/prefix/access key id/secret as plain values v1 —
see prorag/models.py's Connector docstring for the env-ref-indirection
upgrade path this defers.

`connector_items` is the diff baseline both the incremental poll and the
full-ID reconciliation sweep read/write: external_id (source item identity,
e.g. an S3 key) + etag/size (change signal) + doc_id (nullable — null for
skipped/errored items, and nulled by the sweep when it deletes the backing
Document). No document_acl rows are created for anything ingested through a
connector (#15 Tier C: S3 has no source ACLs to mirror) — those documents
are invisible to non-admins until an admin grants access (#18); this
migration lays down no compatibility backfill, unlike 0008's one-time public
grant for pre-existing documents, because there are no pre-existing
connector-ingested documents yet.

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-02

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "connectors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("type", sa.String, nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("last_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_full_sweep_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "connector_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("connector_id", UUID(as_uuid=True), sa.ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.String, nullable=False),
        sa.Column("etag", sa.String, nullable=True),
        sa.Column("size", sa.Integer, nullable=True),
        sa.Column("last_modified", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("last_seen_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.String, nullable=False, server_default="pending"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_unique_constraint(
        "uq_connector_items_connector_id_external_id", "connector_items", ["connector_id", "external_id"]
    )
    op.create_index("ix_connector_items_connector_id", "connector_items", ["connector_id"])


def downgrade() -> None:
    op.drop_index("ix_connector_items_connector_id", table_name="connector_items")
    op.drop_constraint("uq_connector_items_connector_id_external_id", "connector_items", type_="unique")
    op.drop_table("connector_items")
    op.drop_table("connectors")
