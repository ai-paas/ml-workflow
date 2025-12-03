"""Make repo_id nullable in model table

Revision ID: 0017
Revises: 0016
Create Date: 2025-01-20 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make repo_id nullable in model table
    op.alter_column(
        "model",
        "repo_id",
        existing_type=sa.String(length=500),
        nullable=True,
    )


def downgrade() -> None:
    # Make repo_id not nullable in model table
    # Note: This will fail if there are any NULL values in the repo_id column
    op.alter_column(
        "model",
        "repo_id",
        existing_type=sa.String(length=500),
        nullable=False,
    )
