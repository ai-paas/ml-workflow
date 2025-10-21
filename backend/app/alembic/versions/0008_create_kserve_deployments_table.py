"""Create kserve_deployments table

Revision ID: 0008_create_kserve_deployments
Revises: 0007_add_service_workflow_tables
Create Date: 2025-10-20 05:30:00.000000

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # MySQL에서는 ENUM을 VARCHAR로 처리하거나 직접 ENUM 정의
    # Create kserve_deployments table
    op.create_table(
        "kserve_deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("workflow_id", sa.String(36), sa.ForeignKey("workflows.id"), nullable=False),
        sa.Column("component_id", sa.String(255), nullable=False),
        sa.Column("service_name", sa.String(255), nullable=False, unique=True),
        sa.Column("service_hostname", sa.String(500), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("internal_url", sa.String(500), nullable=True),
        sa.Column(
            "status",
            sa.Enum("DEPLOYING", "DEPLOYED", "FAILED", "DELETED", name="deploymentstatus"),
            nullable=False,
            server_default="DEPLOYING",
        ),
        sa.Column("deployed_at", sa.TIMESTAMP, nullable=True),
        sa.Column("deleted_at", sa.TIMESTAMP, nullable=True),
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
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
    )

    # Create indexes
    op.create_index("ix_kserve_deployments_workflow_id", "kserve_deployments", ["workflow_id"])
    op.create_index("ix_kserve_deployments_component_id", "kserve_deployments", ["component_id"])
    op.create_index("ix_kserve_deployments_service_name", "kserve_deployments", ["service_name"])
    op.create_index("ix_kserve_deployments_status", "kserve_deployments", ["status"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_kserve_deployments_status", "kserve_deployments")
    op.drop_index("ix_kserve_deployments_service_name", "kserve_deployments")
    op.drop_index("ix_kserve_deployments_component_id", "kserve_deployments")
    op.drop_index("ix_kserve_deployments_workflow_id", "kserve_deployments")

    # Drop table (MySQL ENUM은 테이블과 함께 자동으로 삭제됨)
    op.drop_table("kserve_deployments")
