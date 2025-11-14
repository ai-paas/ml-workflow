"""Create knowledge_base_search_test_record table

Revision ID: 0015
Revises: 0014
Create Date: 2025-01-20 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create knowledge_base_search_test_record table
    op.create_table(
        "knowledge_base_search_test_record",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "knowledge_base_id", sa.Integer(), sa.ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("source", sa.String(500), nullable=False),  # collection_name
        sa.Column("text", sa.Text(), nullable=False),  # query 내용
        sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    # Create index
    op.create_index(
        "ix_knowledge_base_search_test_record_knowledge_base_id",
        "knowledge_base_search_test_record",
        ["knowledge_base_id"],
    )


def downgrade() -> None:
    # Drop index
    op.drop_index("ix_knowledge_base_search_test_record_knowledge_base_id", "knowledge_base_search_test_record")

    # Drop table
    op.drop_table("knowledge_base_search_test_record")
