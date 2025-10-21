"""Service and Workflow related database models"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, TIMESTAMP, Boolean, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, TimestampMixin

if TYPE_CHECKING:
    from .kserve_deployment import KServeDeployment
    from .model import Model
    from .user import UserModel


class ServiceStatus(PyEnum):
    """서비스 상태 열거형"""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"


class WorkflowStatus(PyEnum):
    """워크플로우 상태 열거형"""

    DRAFT = "DRAFT"  # 임시저장 상태
    ACTIVE = "ACTIVE"  # 정상 상태 (사용 가능)
    ERROR = "ERROR"  # 오류 상태


class ComponentType(PyEnum):
    """컴포넌트 타입 열거형"""

    START = "START"  # 시작 노드
    END = "END"  # 종료 노드
    MODEL = "MODEL"  # 모델 노드


class Service(BaseModel, TimestampMixin):
    """서비스 테이블"""

    __tablename__ = "services"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)  # ["tag1", "tag2", ...]
    creator_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
    status: Mapped[ServiceStatus] = mapped_column(Enum(ServiceStatus), default=ServiceStatus.DRAFT, nullable=False)

    # KServe 관련 정보
    kserve_endpoint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    public_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    backend_api_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # 모니터링 데이터 (최근 집계 데이터 캐싱용)
    monitoring_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    creator: Mapped["UserModel"] = relationship("UserModel", back_populates="services")
    workflows: Mapped[List["Workflow"]] = relationship(
        "Workflow", back_populates="service", cascade="all, delete-orphan"
    )


class Workflow(BaseModel, TimestampMixin):
    """워크플로우 테이블"""

    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[WorkflowStatus] = mapped_column(Enum(WorkflowStatus), default=WorkflowStatus.DRAFT, nullable=False)

    # 서비스 연결
    service_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("services.id"), nullable=True)

    # 생성자
    creator_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    # 템플릿 관련
    is_template: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    template_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("workflows.id"), nullable=True
    )  # 템플릿으로부터 생성된 경우

    # Kubeflow 관련
    kubeflow_pipeline_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    kubeflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # 워크플로우 정의 (컴포넌트와 연결 정보)
    workflow_definition: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # URL 정보
    public_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    backend_api_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Relationships
    service: Mapped[Optional["Service"]] = relationship("Service", back_populates="workflows")
    creator: Mapped["UserModel"] = relationship("UserModel", back_populates="workflows")
    template: Mapped[Optional["Workflow"]] = relationship("Workflow", remote_side=[id], backref="derived_workflows")
    components: Mapped[List["WorkflowComponent"]] = relationship(
        "WorkflowComponent", back_populates="workflow", cascade="all, delete-orphan"
    )
    component_connections: Mapped[List["ComponentConnection"]] = relationship(
        "ComponentConnection",
        foreign_keys="[ComponentConnection.workflow_id]",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )
    kserve_deployments: Mapped[List["KServeDeployment"]] = relationship(
        "KServeDeployment", back_populates="workflow", cascade="all, delete-orphan"
    )


class WorkflowComponent(BaseModel, TimestampMixin):
    """워크플로우 컴포넌트 테이블"""

    __tablename__ = "workflow_components"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=False)
    component_id: Mapped[str] = mapped_column(String(255), nullable=False)  # 워크플로우 내 고유 ID
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[ComponentType] = mapped_column(Enum(ComponentType), nullable=False)

    # 컴포넌트 설정 정보
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # 모델 컴포넌트인 경우 모델 ID
    model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model.id"), nullable=True)

    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="components")
    model: Mapped[Optional["Model"]] = relationship("Model")
    outgoing_connections: Mapped[List["ComponentConnection"]] = relationship(
        "ComponentConnection",
        foreign_keys="[ComponentConnection.source_component_id]",
        back_populates="source_component",
    )
    incoming_connections: Mapped[List["ComponentConnection"]] = relationship(
        "ComponentConnection",
        foreign_keys="[ComponentConnection.target_component_id]",
        back_populates="target_component",
    )


class ComponentConnection(BaseModel, TimestampMixin):
    """컴포넌트 간 연결 정보 테이블"""

    __tablename__ = "component_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=False)

    # 연결 정보
    source_component_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflow_components.id"), nullable=False)
    target_component_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflow_components.id"), nullable=False)

    # 연결 속성
    connection_type: Mapped[str] = mapped_column(String(50), default="DATA", nullable=False)  # DATA, CONTROL, etc.
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # 추가 연결 설정

    # Relationships
    workflow: Mapped["Workflow"] = relationship(
        "Workflow", foreign_keys=[workflow_id], back_populates="component_connections"
    )
    source_component: Mapped["WorkflowComponent"] = relationship(
        "WorkflowComponent", foreign_keys=[source_component_id], back_populates="outgoing_connections"
    )
    target_component: Mapped["WorkflowComponent"] = relationship(
        "WorkflowComponent", foreign_keys=[target_component_id], back_populates="incoming_connections"
    )


class ServiceMonitoring(BaseModel, TimestampMixin):
    """서비스 모니터링 데이터 테이블"""

    __tablename__ = "service_monitoring"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    service_id: Mapped[str] = mapped_column(String(36), ForeignKey("services.id"), nullable=False)
    workflow_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=True)

    # 모니터링 메트릭
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)  # 메시지 수
    active_users: Mapped[int] = mapped_column(Integer, default=0)  # 활성 사용자 수
    token_usage: Mapped[int] = mapped_column(Integer, default=0)  # 토큰 사용량
    avg_interaction_count: Mapped[float] = mapped_column(Float, default=0.0)  # 평균 사용자 상호작용 수

    # 추가 메트릭
    response_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 응답 시간 (ms)
    error_count: Mapped[int] = mapped_column(Integer, default=0)  # 오류 수
    success_rate: Mapped[float] = mapped_column(Float, default=100.0)  # 성공률 (%)

    # Relationships
    service: Mapped["Service"] = relationship("Service")
    workflow: Mapped[Optional["Workflow"]] = relationship("Workflow")
