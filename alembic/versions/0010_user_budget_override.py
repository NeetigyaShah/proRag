"""add users.daily_cap_usd_override (#21)

Per-user admin override for the soft daily cap (#9's resolution: soft cap per
user, hard cap per install). Nullable — null means "use settings.user_daily_cap_usd",
same convention as every other nullable override column in this schema.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-02

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("daily_cap_usd_override", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("users", "daily_cap_usd_override")
