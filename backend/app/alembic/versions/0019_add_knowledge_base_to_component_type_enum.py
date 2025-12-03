"""Add KNOWLEDGE_BASE to componenttype enum

Revision ID: 0019
Revises: 0018
Create Date: 2025-11-18 10:54:24.163194

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # MySQL에서 ENUM에 값을 추가하려면 컬럼을 수정해야 함
    # componenttype ENUM에 KNOWLEDGE_BASE 추가
    op.execute(
        "ALTER TABLE workflow_components MODIFY COLUMN type ENUM('START', 'END', 'MODEL', 'KNOWLEDGE_BASE') NOT NULL"
    )


def downgrade() -> None:
    # ENUM에서 KNOWLEDGE_BASE 제거 (기존 값이 KNOWLEDGE_BASE인 경우 에러 발생 가능)
    # 주의: KNOWLEDGE_BASE 타입의 컴포넌트가 있으면 다운그레이드 실패
    op.execute("ALTER TABLE workflow_components MODIFY COLUMN type ENUM('START', 'END', 'MODEL') NOT NULL")
