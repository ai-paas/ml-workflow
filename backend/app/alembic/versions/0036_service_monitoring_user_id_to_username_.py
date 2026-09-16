"""service_monitoring user_id to username string drop user fk

Revision ID: 0036
Revises: 0035
Create Date: 2026-05-22 13:57:04.052478

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "0036"
down_revision: Union[str, None] = "0035"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # user_id 를 user.id FK(int) → username 문자열 저장으로 변경.
    # FK 가 걸린 컬럼은 타입 변경이 막히므로 FK 를 먼저 제거한다.
    op.drop_constraint("fk_service_monitoring_user_id", "service_monitoring", type_="foreignkey")
    op.alter_column(
        "service_monitoring",
        "user_id",
        existing_type=mysql.BIGINT(display_width=20),
        type_=sa.String(length=100),
        existing_nullable=True,
    )


def downgrade() -> None:
    # 컬럼 타입을 BIGINT 로 원복한 뒤 FK 를 재생성한다(타입이 맞아야 FK 가능).
    # 주의: username 문자열은 BIGINT 로 무손실 복원되지 않는다.
    op.alter_column(
        "service_monitoring",
        "user_id",
        existing_type=sa.String(length=100),
        type_=mysql.BIGINT(display_width=20),
        existing_nullable=True,
    )
    op.create_foreign_key(
        "fk_service_monitoring_user_id",
        "service_monitoring",
        "user",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
