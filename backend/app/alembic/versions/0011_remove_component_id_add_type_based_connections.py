"""remove component_id add type based connections

Revision ID: 0011
Revises: 0010
Create Date: 2024-01-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # workflow_components 테이블에서 component_id 컬럼 제거
    op.drop_column("workflow_components", "component_id")

    # component_connections 테이블은 변경 없음
    # source_component_id와 target_component_id는 그대로 유지 (요청 시 타입으로 찾아서 ID 저장)


def downgrade() -> None:
    # workflow_components 테이블에 component_id 컬럼 추가
    op.add_column("workflow_components", sa.Column("component_id", sa.String(255), nullable=False, server_default=""))
