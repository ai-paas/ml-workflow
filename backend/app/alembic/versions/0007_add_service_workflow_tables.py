"""Add service and workflow tables

Revision ID: 0007
Revises: 0006_experiment_column_changed
Create Date: 2025-10-02
"""

import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create services table
    op.create_table(
        "services",
        sa.Column("id", sa.String(length=36), nullable=False, server_default=sa.text("(UUID())")),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("creator_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Enum("DRAFT", "ACTIVE", "INACTIVE", "DEPRECATED", name="servicestatus"), nullable=False),
        sa.Column("kserve_endpoint", sa.String(length=500), nullable=True),
        sa.Column("public_url", sa.String(length=500), nullable=True),
        sa.Column("backend_api_url", sa.String(length=500), nullable=True),
        sa.Column("monitoring_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("updated_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
        sa.ForeignKeyConstraint(
            ["creator_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_services_id"), "services", ["id"], unique=False)

    # Create workflows table
    op.create_table(
        "workflows",
        sa.Column("id", sa.String(length=36), nullable=False, server_default=sa.text("(UUID())")),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("status", sa.Enum("DRAFT", "ACTIVE", "ERROR", name="workflowstatus"), nullable=False),
        sa.Column("service_id", sa.String(length=36), nullable=True),
        sa.Column("creator_id", sa.BigInteger(), nullable=False),
        sa.Column("is_template", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("template_id", sa.String(length=36), nullable=True),
        sa.Column("kubeflow_pipeline_id", sa.String(length=255), nullable=True),
        sa.Column("kubeflow_run_id", sa.String(length=255), nullable=True),
        sa.Column("workflow_definition", sa.JSON(), nullable=True),
        sa.Column("public_url", sa.String(length=500), nullable=True),
        sa.Column("backend_api_url", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("updated_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
        sa.ForeignKeyConstraint(
            ["creator_id"],
            ["user.id"],
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["services.id"],
        ),
        sa.ForeignKeyConstraint(
            ["template_id"],
            ["workflows.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_workflows_id"), "workflows", ["id"], unique=False)

    # Create workflow_components table
    op.create_table(
        "workflow_components",
        sa.Column("id", sa.String(length=36), nullable=False, server_default=sa.text("(UUID())")),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("component_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("type", sa.Enum("START", "END", "MODEL", name="componenttype"), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("model_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("updated_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["model.id"],
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_workflow_components_id"), "workflow_components", ["id"], unique=False)

    # Create component_connections table
    op.create_table(
        "component_connections",
        sa.Column("id", sa.String(length=36), nullable=False, server_default=sa.text("(UUID())")),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("source_component_id", sa.String(length=36), nullable=False),
        sa.Column("target_component_id", sa.String(length=36), nullable=False),
        sa.Column("connection_type", sa.String(length=50), nullable=False, server_default="DATA"),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("updated_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
        sa.ForeignKeyConstraint(
            ["source_component_id"],
            ["workflow_components.id"],
        ),
        sa.ForeignKeyConstraint(
            ["target_component_id"],
            ["workflow_components.id"],
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_component_connections_id"), "component_connections", ["id"], unique=False)

    # Create service_monitoring table
    op.create_table(
        "service_monitoring",
        sa.Column("id", sa.String(length=36), nullable=False, server_default=sa.text("(UUID())")),
        sa.Column("service_id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=True),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("message_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("active_users", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("token_usage", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("avg_interaction_count", sa.Float(), nullable=True, server_default="0"),
        sa.Column("response_time_ms", sa.Float(), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("success_rate", sa.Float(), nullable=True, server_default="100"),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=True, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("updated_by", sa.String(40), nullable=True, server_default=""),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("deleted_by", sa.String(40), nullable=True, server_default=""),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["services.id"],
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_service_monitoring_id"), "service_monitoring", ["id"], unique=False)
    op.create_index("ix_service_monitoring_timestamp", "service_monitoring", ["timestamp"], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index("ix_service_monitoring_timestamp", table_name="service_monitoring")
    op.drop_index(op.f("ix_service_monitoring_id"), table_name="service_monitoring")
    op.drop_table("service_monitoring")

    op.drop_index(op.f("ix_component_connections_id"), table_name="component_connections")
    op.drop_table("component_connections")

    op.drop_index(op.f("ix_workflow_components_id"), table_name="workflow_components")
    op.drop_table("workflow_components")

    op.drop_index(op.f("ix_workflows_id"), table_name="workflows")
    op.drop_table("workflows")

    op.drop_index(op.f("ix_services_id"), table_name="services")
    op.drop_table("services")
