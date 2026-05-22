"""service and monitoring FK ondelete set null and cascade

Revision ID: 0035
Revises: 0034
Create Date: 2026-05-22 11:21:48.879041

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0035"
down_revision: Union[str, None] = "0034"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # service_monitoring FK: service_id/workflow_id → CASCADE, user_id → SET NULL
    op.drop_constraint("service_monitoring_ibfk_1", "service_monitoring", type_="foreignkey")
    op.drop_constraint("service_monitoring_ibfk_2", "service_monitoring", type_="foreignkey")
    op.drop_constraint("fk_service_monitoring_user_id", "service_monitoring", type_="foreignkey")
    op.create_foreign_key(
        "fk_service_monitoring_service_id", "service_monitoring", "services", ["service_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_service_monitoring_workflow_id",
        "service_monitoring",
        "workflows",
        ["workflow_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_service_monitoring_user_id", "service_monitoring", "user", ["user_id"], ["id"], ondelete="SET NULL"
    )

    # workflows.service_id → SET NULL (서비스 삭제 시 워크플로우 보존)
    op.drop_constraint("workflows_ibfk_2", "workflows", type_="foreignkey")
    op.create_foreign_key(
        "fk_workflows_service_id", "workflows", "services", ["service_id"], ["id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    op.drop_constraint("fk_workflows_service_id", "workflows", type_="foreignkey")
    op.create_foreign_key("workflows_ibfk_2", "workflows", "services", ["service_id"], ["id"])

    op.drop_constraint("fk_service_monitoring_service_id", "service_monitoring", type_="foreignkey")
    op.drop_constraint("fk_service_monitoring_workflow_id", "service_monitoring", type_="foreignkey")
    op.drop_constraint("fk_service_monitoring_user_id", "service_monitoring", type_="foreignkey")
    op.create_foreign_key("service_monitoring_ibfk_1", "service_monitoring", "services", ["service_id"], ["id"])
    op.create_foreign_key("service_monitoring_ibfk_2", "service_monitoring", "workflows", ["workflow_id"], ["id"])
    op.create_foreign_key("fk_service_monitoring_user_id", "service_monitoring", "user", ["user_id"], ["id"])
