"""add connectors.last_error (#23)

Polling scheduler (#15's cadence, on top of #22's sync engine): a scheduled
run's failure must be visible somewhere durable, since there's no HTTP caller
around to see an exception the way the manual POST /connectors/{id}/sync
endpoint has. Nullable Text, cleared back to NULL on the next successful run
(scheduler.py) — no history, just "is the last attempt still failing".

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-02

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("connectors", sa.Column("last_error", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("connectors", "last_error")
