"""initial: documents + chunks

Revision ID: 0001
Revises:
Create Date: 2026-07-30

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op
from prorag.settings import settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sha256", sa.String, nullable=False, unique=True),
        sa.Column("filename", sa.String, nullable=False),
        sa.Column("mime", sa.String, nullable=False),
        sa.Column("blob_path", sa.String, nullable=False),
        sa.Column("page_count", sa.Integer, nullable=True),
        sa.Column("title", sa.String, nullable=True),
        sa.Column("collection", sa.String, nullable=False, server_default="default"),
        sa.Column("meta", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("kind", sa.String, nullable=False, server_default="prose"),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embed_text", sa.Text, nullable=False),
        sa.Column("page_start", sa.Integer, nullable=True),
        sa.Column("page_end", sa.Integer, nullable=True),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(settings.embed_dim), nullable=False),
    )
    op.create_index("ix_chunks_doc_id_ord", "chunks", ["doc_id", "ord"])
    # pgvector caps hnsw at 2000 dims on `vector`; halfvec raises that to 4000,
    # which is what lets 2048-dim embedding models (nemotron-3-embed) index.
    if settings.embed_dim > 2000:
        op.execute(
            f"CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
            f"USING hnsw ((embedding::halfvec({settings.embed_dim})) halfvec_cosine_ops) "
            f"WITH (m=16, ef_construction=200)"
        )
    else:
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200)"
        )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
