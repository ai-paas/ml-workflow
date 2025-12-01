"""Remove unnecessary fields from workflows table

Revision ID: 0010
Revises: 0009_remove_service_unnecessary_fields
Create Date: 2025-10-31

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove columns from workflows table
    op.drop_column("workflows", "workflow_definition")
    op.drop_column("workflows", "kubeflow_pipeline_id")
    op.drop_column("workflows", "public_url")
    op.drop_column("workflows", "backend_api_url")


def downgrade() -> None:
    # Add columns back to workflows table
    op.add_column("workflows", sa.Column("workflow_definition", sa.JSON(), nullable=True))
    op.add_column("workflows", sa.Column("kubeflow_pipeline_id", sa.String(length=255), nullable=True))
    op.add_column("workflows", sa.Column("public_url", sa.String(length=500), nullable=True))
    op.add_column("workflows", sa.Column("backend_api_url", sa.String(length=500), nullable=True))
