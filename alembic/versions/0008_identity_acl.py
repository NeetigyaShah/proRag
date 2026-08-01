"""add identity + ACL schema (users, groups, membership, document_acl)

Implements the schema decided in #2 (identity: users are the principal, groups
are synced/stored rather than trusted from token claims) and #15 (ACL storage:
document_acl mirrors source principals, default-deny). No enforcement logic
here — that's the next ticket (#18); this migration only lays down the tables
`user_groups` rows and `document_acl` rows will be read from.

Compatibility backfill: default-deny means new `document_acl` rows are NOT
backfilled for pre-existing documents in general (per #15, "no ACL row = no
access" is the rule going forward). The one exception, called out explicitly
in #17: this upgrade inserts a single principal_type='public' grant per
existing document, so a single-user install that upgrades keeps working
exactly as before (everything was implicitly visible pre-ACL). An admin can
revoke that by deleting the corresponding `document_acl` rows once real
users/groups are in place — it is a deliberate, reversible compatibility
choice, not a security default for new documents.

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-31

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("external_subject", sa.String, nullable=True, unique=True),
        sa.Column("email", sa.String, nullable=False, unique=True),
        sa.Column("display_name", sa.String, nullable=True),
        sa.Column("is_admin", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("disabled_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    op.create_table(
        "groups",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String, nullable=False, unique=True),
        sa.Column("source", sa.String, nullable=False, server_default="local"),
        sa.Column("external_id", sa.String, nullable=True),
        sa.CheckConstraint("source IN ('local', 'idp')", name="ck_groups_source"),
    )
    # Partial unique index: only enforce (source, external_id) uniqueness for
    # IdP-synced groups — local groups have no external_id to collide on.
    op.create_index(
        "uq_groups_source_external_id",
        "groups",
        ["source", "external_id"],
        unique=True,
        postgresql_where=sa.text("external_id IS NOT NULL"),
    )

    op.create_table(
        "user_groups",
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("group_id", UUID(as_uuid=True), sa.ForeignKey("groups.id", ondelete="CASCADE"), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "group_id"),
    )

    op.create_table(
        "document_acl",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("principal_type", sa.String, nullable=False),
        # nullable — a 'public' grant has no principal_id
        sa.Column("principal_id", UUID(as_uuid=True), nullable=True),
        sa.Column("source", sa.String, nullable=False, server_default="local"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("principal_type IN ('user', 'group', 'public')", name="ck_document_acl_principal_type"),
    )
    op.create_index("ix_document_acl_doc_id", "document_acl", ["doc_id"])
    op.create_index("ix_document_acl_principal", "document_acl", ["principal_type", "principal_id"])

    op.add_column("api_keys", sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True))

    op.add_column("usage", sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True))
    op.create_index("ix_usage_user_id_created_at", "usage", ["user_id", "created_at"])

    # Compatibility backfill (see module docstring): one 'public' grant per
    # existing document, admin-revocable, so a pre-ACL single-user install
    # keeps seeing everything it already could after this upgrade.
    op.execute(
        """
        INSERT INTO document_acl (doc_id, principal_type, principal_id, source)
        SELECT id, 'public', NULL, 'local' FROM documents
        """
    )


def downgrade() -> None:
    op.drop_index("ix_usage_user_id_created_at", table_name="usage")
    op.drop_column("usage", "user_id")
    op.drop_column("api_keys", "user_id")
    op.drop_index("ix_document_acl_principal", table_name="document_acl")
    op.drop_index("ix_document_acl_doc_id", table_name="document_acl")
    op.drop_table("document_acl")
    op.drop_table("user_groups")
    op.drop_index("uq_groups_source_external_id", table_name="groups")
    op.drop_table("groups")
    op.drop_table("users")
