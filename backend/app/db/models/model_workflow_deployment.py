"""워크플로 모델 컴포넌트 서빙 배포 정보 (KServe·Ollama 등 공통)."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import TIMESTAMP, Enum, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, TimestampMixin

if TYPE_CHECKING:
    from .service import Workflow


class DeploymentStatus(PyEnum):
    """배포 상태 열거형"""

    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class WorkflowServingDeploymentType(PyEnum):
    """MODEL 컴포넌트 서빙 배포 유형 (§2.5)."""

    KSERVE = "KSERVE"
    OLLAMA = "OLLAMA"
    REMOTE = "REMOTE"


class ServingDeviceType(PyEnum):
    """서빙 시 요청된 디바이스 타입 (§2.3)."""

    GPU = "GPU"
    CPU = "CPU"


class ModelWorkflowDeployment(BaseModel, TimestampMixin):
    """워크플로 내 MODEL 컴포넌트 배포 정보 테이블 (구 kserve_deployments)."""

    __tablename__ = "model_workflow_deployments"
    __table_args__ = (
        Index("ix_model_workflow_deployments_component_id", "component_id"),
        Index("ix_model_workflow_deployments_service_name", "service_name"),
        Index("ix_model_workflow_deployments_status", "status"),
        Index("ix_model_workflow_deployments_workflow_id", "workflow_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=False)
    component_id: Mapped[str] = mapped_column(String(255), nullable=False)

    service_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    service_hostname: Mapped[str] = mapped_column(String(500), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    internal_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    deployment_type: Mapped[WorkflowServingDeploymentType] = mapped_column(
        Enum(WorkflowServingDeploymentType, values_callable=lambda x: [e.value for e in x]),
        default=WorkflowServingDeploymentType.KSERVE,
        nullable=False,
    )
    pvc: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    device_type: Mapped[Optional[ServingDeviceType]] = mapped_column(
        Enum(ServingDeviceType, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    remote_api_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus), default=DeploymentStatus.DEPLOYING, nullable=False
    )

    deployed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="model_deployments")
