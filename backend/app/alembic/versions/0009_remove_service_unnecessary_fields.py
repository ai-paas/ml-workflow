"""Remove unnecessary fields from services table

Revision ID: 0009
Revises: 0008_create_kserve_deployments
Create Date: 2025-10-30

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop legacy tables that are no longer used
    op.drop_table("service_endpoint")
    op.drop_table("service_param_type")

    # Remove columns from services table
    op.drop_column("services", "monitoring_data")
    op.drop_column("services", "backend_api_url")
    op.drop_column("services", "public_url")
    op.drop_column("services", "kserve_endpoint")
    op.drop_column("services", "status")


def downgrade() -> None:
    # Add columns back to services table (in reverse order)
    op.add_column(
        "services",
        sa.Column(
            "status",
            sa.Enum("DRAFT", "ACTIVE", "INACTIVE", "DEPRECATED", name="servicestatus"),
            nullable=False,
            server_default="DRAFT",
        ),
    )
    op.add_column("services", sa.Column("kserve_endpoint", sa.String(length=500), nullable=True))
    op.add_column("services", sa.Column("public_url", sa.String(length=500), nullable=True))
    op.add_column("services", sa.Column("backend_api_url", sa.String(length=500), nullable=True))
    op.add_column("services", sa.Column("monitoring_data", sa.JSON(), nullable=True))

    # Recreate legacy tables
    op.create_table(
        "service_param_type",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("param_name", sa.String(length=500), nullable=False),
        sa.Column("param_type", sa.String(length=100), nullable=False),
        sa.Column("default_value", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("created_by", sa.String(length=40), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("updated_by", sa.String(length=40), nullable=True),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(length=40), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "service_endpoint",
        sa.Column("uuid", sa.String(length=36), nullable=False),
        sa.Column("url", sa.String(length=4000), nullable=False),
        sa.Column("service_param_type_id", sa.Integer(), nullable=False),
        sa.Column("service_param_value", sa.String(length=500), nullable=False),
        sa.Column("reference_model_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("created_by", sa.String(length=40), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("updated_by", sa.String(length=40), nullable=True),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(length=40), nullable=True),
        sa.ForeignKeyConstraint(
            ["reference_model_id"],
            ["model.id"],
        ),
        sa.ForeignKeyConstraint(
            ["service_param_type_id"],
            ["service_param_type.id"],
        ),
        sa.PrimaryKeyConstraint("uuid"),
    )
