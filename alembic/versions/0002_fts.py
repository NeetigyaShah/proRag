"""add tsv generated column + GIN index for FTS

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-30

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # generated column, per architecture §3.4 (heading_path doesn't exist yet in
    # Phase 2's chunks table, so weight A collapses to just the text for now —
    # heading_path lands with Docling in Phase 3).
    op.execute("ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED")
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")


def downgrade() -> None:
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_column("chunks", "tsv")
