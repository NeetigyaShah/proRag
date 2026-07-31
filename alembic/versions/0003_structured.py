"""add tables/table_rows + structured-document columns on chunks

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-30

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tables",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("caption", sa.Text, nullable=True),
        sa.Column("columns", ARRAY(sa.String), nullable=True),
        sa.Column("row_count", sa.Integer, nullable=True),
        sa.Column("page_no", sa.Integer, nullable=True),
        sa.Column("bbox", ARRAY(sa.Float), nullable=True),
    )

    op.create_table(
        "table_rows",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("table_id", UUID(as_uuid=True), sa.ForeignKey("tables.id", ondelete="CASCADE"), nullable=False),
        sa.Column("row_no", sa.Integer, nullable=False),
        sa.Column("data", JSONB, nullable=False),
    )
    op.create_index("ix_table_rows_table_id", "table_rows", ["table_id"])
    op.execute("CREATE INDEX ix_table_rows_data_gin ON table_rows USING gin (data jsonb_path_ops)")

    op.add_column("documents", sa.Column("title_norm", sa.String, nullable=True))

    op.add_column("chunks", sa.Column("heading_path", ARRAY(sa.String), nullable=True))
    op.add_column("chunks", sa.Column("bbox", ARRAY(sa.Float), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("table_id", UUID(as_uuid=True), sa.ForeignKey("tables.id", ondelete="CASCADE"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chunks", "table_id")
    op.drop_column("chunks", "bbox")
    op.drop_column("chunks", "heading_path")
    op.drop_column("documents", "title_norm")
    op.drop_index("ix_table_rows_data_gin", table_name="table_rows")
    op.drop_index("ix_table_rows_table_id", table_name="table_rows")
    op.drop_table("table_rows")
    op.drop_table("tables")
