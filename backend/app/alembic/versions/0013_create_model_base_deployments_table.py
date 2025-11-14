"""Create model_base_deployments table

Revision ID: 0013
Revises: 0012
Create Date: 2025-01-15 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create model_base_deployments table
    op.create_table(
        "model_base_deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.Integer(), sa.ForeignKey("model.id"), nullable=False),
        sa.Column("service_name", sa.String(255), nullable=False, unique=True),
        sa.Column("service_hostname", sa.String(500), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("internal_url", sa.String(500), nullable=True),
        sa.Column(
            "status",
            sa.Enum("DEPLOYING", "DEPLOYED", "FAILED", "DELETED", name="basedeploymentstatus"),
            nullable=False,
            server_default="DEPLOYING",
        ),
        sa.Column("deployed_at", sa.TIMESTAMP, nullable=True),
        sa.Column("error_message", sa.String(1000), nullable=True),
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
    op.create_index("ix_model_base_deployments_model_id", "model_base_deployments", ["model_id"])
    op.create_index("ix_model_base_deployments_service_name", "model_base_deployments", ["service_name"])
    op.create_index("ix_model_base_deployments_status", "model_base_deployments", ["status"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_model_base_deployments_status", "model_base_deployments")
    op.drop_index("ix_model_base_deployments_service_name", "model_base_deployments")
    op.drop_index("ix_model_base_deployments_model_id", "model_base_deployments")

    # Drop table (MySQL ENUM은 테이블과 함께 자동으로 삭제됨)
    op.drop_table("model_base_deployments")
