"""move_train_msg_to_experiment

Revision ID: 0025
Revises: 0024
Create Date: 2026-04-07 13:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0025"
down_revision: Union[str, None] = "0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("experiment", sa.Column("train_msg", sa.String(length=1000), nullable=True))

    op.execute(
        """
        UPDATE experiment e
        JOIN experiment_metrics em ON e.id = em.experiment_id
        SET e.train_msg = em.train_msg
        WHERE em.train_msg IS NOT NULL
        """
    )

    op.drop_column("experiment_metrics", "train_msg")


def downgrade() -> None:
    op.add_column("experiment_metrics", sa.Column("train_msg", sa.String(length=1000), nullable=True))

    op.execute(
        """
        UPDATE experiment_metrics em
        JOIN experiment e ON em.experiment_id = e.id
        SET em.train_msg = e.train_msg
        WHERE e.train_msg IS NOT NULL
        """
    )

    op.drop_column("experiment", "train_msg")
