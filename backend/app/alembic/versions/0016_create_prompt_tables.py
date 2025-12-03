"""Create prompt and prompt_variable tables

Revision ID: 0016
Revises: 0015
Create Date: 2025-01-20 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create prompt table
    op.create_table(
        "prompt",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        # TimestampMixin 컬럼들
        sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_by", sa.String(40), nullable=True, server_default=""),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP,
            nullable=True,
            server_default=sa.text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        ),
        sa.Column("updated_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("deleted_at", sa.TIMESTAMP, nullable=True),
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
    )

    # Create prompt_variable table
    op.create_table(
        "prompt_variable",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column(
            "prompt_id",
            sa.Integer(),
            sa.ForeignKey("prompt.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    # Create index for prompt_variable.prompt_id
    op.create_index(
        "ix_prompt_variable_prompt_id",
        "prompt_variable",
        ["prompt_id"],
    )


def downgrade() -> None:
    # Drop index
    op.drop_index("ix_prompt_variable_prompt_id", "prompt_variable")

    # Drop tables
    op.drop_table("prompt_variable")
    op.drop_table("prompt")
