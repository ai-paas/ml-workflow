"""KServe 배포 정보 관리 모델"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import TIMESTAMP, Enum, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, TimestampMixin

if TYPE_CHECKING:
    from .service import Workflow, WorkflowComponent


class DeploymentStatus(PyEnum):
    """배포 상태 열거형"""

    DEPLOYING = "DEPLOYING"  # 배포 중
    DEPLOYED = "DEPLOYED"  # 배포 완료
    FAILED = "FAILED"  # 배포 실패
    DELETED = "DELETED"  # 삭제됨


class KServeDeployment(BaseModel, TimestampMixin):
    """KServe 배포 정보 테이블"""

    __tablename__ = "kserve_deployments"
    __table_args__ = (
        Index("ix_kserve_deployments_component_id", "component_id"),
        Index("ix_kserve_deployments_service_name", "service_name"),
        Index("ix_kserve_deployments_status", "status"),
        Index("ix_kserve_deployments_workflow_id", "workflow_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # 워크플로우 및 컴포넌트 정보
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=False)
    component_id: Mapped[str] = mapped_column(String(255), nullable=False)  # WorkflowComponent.id (UUID)

    # KServe 서비스 정보
    service_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    service_hostname: Mapped[str] = mapped_column(String(500), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)  # 정제된 모델 이름 (슬래시 제거)
    internal_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # 상태 관리
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus), default=DeploymentStatus.DEPLOYING, nullable=False
    )

    # 타임스탬프
    deployed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)

    # 에러 정보
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="kserve_deployments")
