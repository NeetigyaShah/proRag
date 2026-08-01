"""add access_rules (#4, #24): the admin dashboard API's rule model

A rule is a natural-language query (`nl_query`) that grants `group_id`
access to matching documents once an admin confirms it — #4's "confirm the
rule, not 20k documents" compromise. `query_embedding` starts NULL: preview
(POST /admin/rules/{id}/preview) embeds nl_query ad hoc without storing it,
so browsing draft rules costs nothing extra; confirm (POST
/admin/rules/{id}/confirm) is what embeds and persists it, so auto-admission
of future documents (ingest/core.py) never needs another LLM call.

v1 ships confirm-once (#10's out-of-scope note: re-run/pending-diff on edit
is future work) — state is just 'draft' | 'confirmed', no third state.

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-02

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

from alembic import op
from prorag.settings import settings

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "access_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("nl_query", sa.Text, nullable=False),
        sa.Column("group_id", UUID(as_uuid=True), sa.ForeignKey("groups.id"), nullable=False),
        sa.Column("state", sa.String, nullable=False, server_default="draft"),
        sa.Column("query_embedding", Vector(settings.embed_dim), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("confirmed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint("state IN ('draft', 'confirmed')", name="ck_access_rules_state"),
    )
    op.create_index("ix_access_rules_group_id", "access_rules", ["group_id"])


def downgrade() -> None:
    op.drop_index("ix_access_rules_group_id", table_name="access_rules")
    op.drop_table("access_rules")
