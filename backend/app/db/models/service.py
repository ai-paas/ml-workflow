"""Service and Workflow related database models"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, TIMESTAMP, Boolean, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel, TimestampMixin

if TYPE_CHECKING:
    from .model import Model
    from .model_workflow_deployment import ModelWorkflowDeployment
    from .user import UserModel


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
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"  # 지식 베이스 노드


class Service(BaseModel, TimestampMixin):
    """서비스 테이블"""

    __tablename__ = "services"
    __table_args__ = (Index("ix_services_id", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)  # ["tag1", "tag2", ...]
    creator_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    # Relationships
    creator: Mapped["UserModel"] = relationship("UserModel", back_populates="services")
    # 서비스 삭제 시 워크플로우는 DB FK(ON DELETE SET NULL)로 보존·연결만 해제한다.
    # (ORM delete-orphan 캐스케이드를 쓰지 않고 passive_deletes 로 DB 제약에 위임)
    workflows: Mapped[List["Workflow"]] = relationship(
        "Workflow", back_populates="service", cascade="save-update, merge", passive_deletes=True
    )


class Workflow(BaseModel, TimestampMixin):
    """워크플로우 테이블"""

    __tablename__ = "workflows"
    __table_args__ = (Index("ix_workflows_id", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[WorkflowStatus] = mapped_column(Enum(WorkflowStatus), default=WorkflowStatus.DRAFT, nullable=False)

    # 서비스 연결 (서비스 삭제 시 SET NULL 로 워크플로우 보존)
    service_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("services.id", ondelete="SET NULL"), nullable=True
    )

    # 생성자
    creator_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    # 템플릿 관련
    is_template: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    template_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("workflows.id"), nullable=True
    )  # 템플릿으로부터 생성된 경우

    # Kubeflow 관련
    kubeflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

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
    model_deployments: Mapped[List["ModelWorkflowDeployment"]] = relationship(
        "ModelWorkflowDeployment", back_populates="workflow", cascade="all, delete-orphan"
    )


class WorkflowComponent(BaseModel, TimestampMixin):
    """워크플로우 컴포넌트 테이블"""

    __tablename__ = "workflow_components"
    __table_args__ = (Index("ix_workflow_components_id", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[ComponentType] = mapped_column(Enum(ComponentType), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 컴포넌트 설정 정보
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # 모델 컴포넌트인 경우 모델 ID
    model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model.id"), nullable=True)

    # Knowledge Base 컴포넌트인 경우 Knowledge Base ID
    knowledge_base_id: Mapped[Optional[int]] = mapped_column(ForeignKey("knowledge_base.id"), nullable=True)

    # 모델 컴포넌트인 경우 프롬프트 ID
    prompt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("prompt.id"), nullable=True)

    # 프론트 캔버스 좌표 (음수 허용)
    x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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
    __table_args__ = (Index("ix_component_connections_id", "id"),)

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
    __table_args__ = (
        Index("ix_service_monitoring_id", "id"),
        Index("ix_service_monitoring_timestamp", "timestamp"),
        Index("ix_service_monitoring_service_id_timestamp", "service_id", "timestamp"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # 서비스/워크플로우 삭제 시 모니터링 행은 CASCADE 로 함께 삭제된다.
    service_id: Mapped[str] = mapped_column(String(36), ForeignKey("services.id", ondelete="CASCADE"), nullable=False)
    workflow_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=True
    )
    # 요청 사용자 username (FK 아님 — 사용자 PK가 int 라 문자열 username 을 그대로 저장)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 추론 요청 1건 = 행 1개 (per-event 로그). 집계 메트릭은 조회 시점에 유도한다.
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    token_usage: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 요청별 토큰 사용량
    response_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 응답 시간 (ms)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)  # 성공 여부(success_yn)

    # Relationships
    service: Mapped["Service"] = relationship("Service")
    workflow: Mapped[Optional["Workflow"]] = relationship("Workflow")
