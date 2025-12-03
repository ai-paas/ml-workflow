"""add model columns

Revision ID: 0012
Revises: 0011
Create Date: 2025-01-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # description 컬럼 타입 변경: String(500) -> Text
    op.alter_column(
        "model", "description", existing_type=sa.String(length=500), type_=sa.Text(), existing_nullable=True
    )

    # repo_id 컬럼 추가: String(500), nullable=False
    op.add_column("model", sa.Column("repo_id", sa.String(length=500), nullable=False, server_default=""))

    # task 컬럼 추가: String(500), nullable=True
    op.add_column("model", sa.Column("task", sa.String(length=500), nullable=True))

    # parameter 컬럼 추가: String(100), nullable=True
    op.add_column("model", sa.Column("parameter", sa.String(length=100), nullable=True))

    # sample_code 컬럼 추가: Text, nullable=True
    op.add_column("model", sa.Column("sample_code", sa.Text(), nullable=True))


def downgrade() -> None:
    # sample_code 컬럼 제거
    op.drop_column("model", "sample_code")

    # parameter 컬럼 제거
    op.drop_column("model", "parameter")

    # task 컬럼 제거
    op.drop_column("model", "task")

    # repo_id 컬럼 제거
    op.drop_column("model", "repo_id")

    # description 컬럼 타입 복원: Text -> String(500)
    op.alter_column(
        "model", "description", existing_type=sa.Text(), type_=sa.String(length=500), existing_nullable=True
    )
