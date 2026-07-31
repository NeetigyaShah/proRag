"""add pg_search BM25 index on chunks (ParadeDB)

The keyword arm of hybrid retrieval moves from Postgres `ts_rank_cd` to a real
BM25 index. The `tsv` column and its GIN index stay in place as a fallback for
deployments running plain pgvector without pg_search (see arms.fts_search).

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-30

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No-op on a stock pgvector image: the app falls back to tsvector ranking.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.execute("CREATE INDEX ix_chunks_bm25 ON chunks USING bm25 (id, text) WITH (key_field='id')")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_bm25")
