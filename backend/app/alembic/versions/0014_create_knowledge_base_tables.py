"""Create knowledge_base tables

Revision ID: 0014
Revises: 0013
Create Date: 2025-01-20 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create chunk_type table
    op.create_table(
        "chunk_type",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
    )

    # Create language table
    op.create_table(
        "language",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(10), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
    )

    # Create search_method table
    op.create_table(
        "search_method",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
    )

    # Create knowledge_base table
    op.create_table(
        "knowledge_base",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("embedding_model_id", sa.Integer(), sa.ForeignKey("model.id"), nullable=False),
        sa.Column("language_id", sa.Integer(), sa.ForeignKey("language.id"), nullable=False),
        sa.Column("collection_name", sa.String(500), nullable=False, unique=True),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("chunk_type_id", sa.Integer(), sa.ForeignKey("chunk_type.id"), nullable=False),
        sa.Column("search_method_id", sa.Integer(), sa.ForeignKey("search_method.id"), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
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

    # Create knowledge_base_file table
    op.create_table(
        "knowledge_base_file",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("object_storage_uri", sa.String(4000), nullable=True),
        sa.Column(
            "knowledge_base_id", sa.Integer(), sa.ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("chunk_number", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("partition_name", sa.String(500), nullable=False),
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

    # Create indexes
    op.create_index("ix_knowledge_base_embedding_model_id", "knowledge_base", ["embedding_model_id"])
    op.create_index("ix_knowledge_base_language_id", "knowledge_base", ["language_id"])
    op.create_index("ix_knowledge_base_collection_name", "knowledge_base", ["collection_name"])
    op.create_index("ix_knowledge_base_file_knowledge_base_id", "knowledge_base_file", ["knowledge_base_id"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_knowledge_base_file_knowledge_base_id", "knowledge_base_file")
    op.drop_index("ix_knowledge_base_collection_name", "knowledge_base")
    op.drop_index("ix_knowledge_base_language_id", "knowledge_base")
    op.drop_index("ix_knowledge_base_embedding_model_id", "knowledge_base")

    # Drop tables
    op.drop_table("knowledge_base_file")
    op.drop_table("knowledge_base")
    op.drop_table("search_method")
    op.drop_table("language")
    op.drop_table("chunk_type")
