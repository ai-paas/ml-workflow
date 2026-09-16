"""modify about dataset and hyperparameter

Revision ID: 0037
Revises: 0036
Create Date: 2026-06-22 09:32:10.930241

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "0037"
down_revision: Union[str, None] = "0036"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # hyperparameter_type 를 참조하는 FK·컬럼을 먼저 제거해야 부모 테이블을 드롭할 수 있다.
    # (autogenerate 는 drop_table 을 먼저 배치해 FK 위반을 일으키므로 순서를 보정한다.)
    op.drop_constraint("hyperparameter_ibfk_1", "hyperparameter", type_="foreignkey")
    op.drop_column("hyperparameter", "hyperparameter_type_id")
    op.drop_table("hyperparameter_type")
    op.add_column("dataset", sa.Column("kind", sa.String(length=50), nullable=True))
    op.add_column("hyperparameter", sa.Column("param_name", sa.String(length=100), nullable=False))
    op.create_unique_constraint("uq_hparam_experiment_param", "hyperparameter", ["experiment_id", "param_name"])
    op.drop_index("fk_service_monitoring_user_id", table_name="service_monitoring")


def downgrade() -> None:
    op.create_index("fk_service_monitoring_user_id", "service_monitoring", ["user_id"], unique=False)
    op.drop_constraint("uq_hparam_experiment_param", "hyperparameter", type_="unique")
    op.drop_column("hyperparameter", "param_name")
    op.drop_column("dataset", "kind")
    # FK 를 걸기 전에 부모 테이블을 먼저 만들고, 그 다음 컬럼·FK 를 추가한다.
    op.create_table(
        "hyperparameter_type",
        sa.Column("id", mysql.BIGINT(display_width=20), autoincrement=True, nullable=False),
        sa.Column("param_name", mysql.VARCHAR(length=500), nullable=False),
        sa.Column("param_type", mysql.VARCHAR(length=100), nullable=False),
        sa.Column("default_value", mysql.VARCHAR(length=500), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        mysql_collate="utf8mb4_general_ci",
        mysql_default_charset="utf8mb4",
        mysql_engine="InnoDB",
    )
    op.add_column(
        "hyperparameter",
        sa.Column("hyperparameter_type_id", mysql.BIGINT(display_width=20), autoincrement=False, nullable=False),
    )
    op.create_foreign_key(
        "hyperparameter_ibfk_1", "hyperparameter", "hyperparameter_type", ["hyperparameter_type_id"], ["id"]
    )
