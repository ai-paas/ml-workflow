"""모델 기본 배포 정보 관리 모델"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import TIMESTAMP, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, TimestampMixin

if TYPE_CHECKING:
    from .model import Model


class BaseDeploymentStatus(PyEnum):
    """배포 상태 열거형"""

    DEPLOYING = "DEPLOYING"  # 배포 중
    DEPLOYED = "DEPLOYED"  # 배포 완료
    FAILED = "FAILED"  # 배포 실패
    DELETED = "DELETED"  # 삭제됨


class ModelBaseDeployment(BaseModel, TimestampMixin):
    """모델 기본 배포 정보 테이블"""

    __tablename__ = "model_base_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # 모델 정보
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("model.id"), nullable=False)

    # 서비스 정보
    service_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    service_hostname: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)  # 정제된 모델 이름 (슬래시 제거)
    internal_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # 상태 관리
    status: Mapped[BaseDeploymentStatus] = mapped_column(
        Enum(BaseDeploymentStatus), default=BaseDeploymentStatus.DEPLOYING, nullable=False
    )

    # 타임스탬프
    deployed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)

    # 에러 정보
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Relationships
    model: Mapped["Model"] = relationship("Model")
