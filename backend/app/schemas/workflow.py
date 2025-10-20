"""Workflow 관련 Pydantic 스키마"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from schemas.base import TimeStampSchemaMixin
from schemas.model import ModelBriefReadSchema
from schemas.user import UserSchema


class WorkflowStatus(str, Enum):
    """워크플로우 상태"""

    DRAFT = "DRAFT"  # 임시저장 상태
    ACTIVE = "ACTIVE"  # 정상 상태 (사용 가능)
    ERROR = "ERROR"  # 오류 상태


class ComponentType(str, Enum):
    """컴포넌트 타입"""

    START = "START"  # 시작 노드
    END = "END"  # 종료 노드
    MODEL = "MODEL"  # 모델 노드


# ============= Component 스키마 =============


class ComponentCreateRequest(BaseModel):
    """컴포넌트 생성 요청"""

    component_id: str = Field(..., description="워크플로우 내 고유 ID")
    name: str
    type: ComponentType
    config: Optional[dict] = None
    model_id: Optional[int] = Field(None, description="모델 컴포넌트인 경우 모델 ID")


class ComponentUpdateRequest(BaseModel):
    """컴포넌트 수정 요청"""

    name: Optional[str] = None
    config: Optional[dict] = None
    model_id: Optional[int] = None


class ComponentReadSchema(TimeStampSchemaMixin):
    """컴포넌트 조회 응답"""

    id: str
    workflow_id: str
    component_id: str
    name: str
    type: ComponentType
    config: Optional[Dict[str, Any]] = None
    model_id: Optional[int] = None
    model: Optional[ModelBriefReadSchema] = None

    class Config:
        from_attributes = True


# ============= Connection 스키마 =============
class ConnectionCreateRequest(BaseModel):
    """컴포넌트 연결 생성 요청"""

    source_component_id: str = Field(..., description="소스 컴포넌트 ID")
    target_component_id: str = Field(..., description="타겟 컴포넌트 ID")
    connection_type: str = Field("DATA", description="연결 타입")
    config: Optional[Dict[str, Any]] = None


class ConnectionReadSchema(BaseModel):
    """컴포넌트 연결 조회 응답"""

    id: str
    workflow_id: str
    source_component_id: str
    target_component_id: str
    source_component: ComponentReadSchema
    target_component: ComponentReadSchema
    connection_type: str
    config: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============= Workflow Definition 스키마 =============
class WorkflowDefinition(BaseModel):
    """워크플로우 정의 (컴포넌트와 연결 정보)"""

    components: List[ComponentCreateRequest]
    connections: List[ConnectionCreateRequest]


# ============= Workflow 스키마 =============
class WorkflowCreateRequest(BaseModel):
    """워크플로우 생성 요청"""

    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    service_id: Optional[str] = None
    is_template: bool = Field(False, description="템플릿 여부")
    template_id: Optional[str] = Field(None, description="템플릿 ID (템플릿으로부터 생성시)")
    workflow_definition: Optional[WorkflowDefinition] = None


class WorkflowCreateInternal(WorkflowCreateRequest):
    """워크플로우 내부 생성용 - status와 creator_id 포함"""

    status: WorkflowStatus = WorkflowStatus.DRAFT
    creator_id: int


class WorkflowUpdateRequest(BaseModel):
    """워크플로우 수정 요청"""

    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[WorkflowStatus] = None
    service_id: Optional[str] = None
    workflow_definition: Optional[WorkflowDefinition] = None


class WorkflowBaseSchema(TimeStampSchemaMixin):
    """워크플로우 기본 정보"""

    id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    status: WorkflowStatus
    service_id: Optional[str] = None
    creator_id: int
    is_template: bool
    template_id: Optional[str] = None

    class Config:
        from_attributes = True


class WorkflowReadSchema(WorkflowBaseSchema):
    """워크플로우 상세 조회 응답"""

    creator: UserSchema
    service_name: Optional[str] = None
    template_name: Optional[str] = None
    kubeflow_pipeline_id: Optional[str] = None
    kubeflow_run_id: Optional[str] = None
    public_url: Optional[str] = None
    backend_api_url: Optional[str] = None
    workflow_definition: Optional[Dict[str, Any]] = None
    components: List[ComponentReadSchema] = Field(default_factory=list)
    connections: List[ConnectionReadSchema] = Field(default_factory=list)

    class Config:
        from_attributes = True


class WorkflowListSchema(BaseModel):
    """워크플로우 목록 조회 응답"""

    total: int
    items: List[WorkflowBaseSchema]


# ============= Template 스키마 =============
class WorkflowTemplateCreateRequest(WorkflowCreateRequest):
    """워크플로우 템플릿 생성 요청"""

    is_template: bool = Field(True, description="템플릿 여부 (항상 True)")


class WorkflowTemplateReadSchema(WorkflowReadSchema):
    """워크플로우 템플릿 조회 응답"""

    usage_count: int = Field(0, description="템플릿 사용 횟수")

    class Config:
        from_attributes = True


# ============= Execution 스키마 =============
class WorkflowExecuteRequest(BaseModel):
    """워크플로우 실행 요청"""

    parameters: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecuteResponse(BaseModel):
    """워크플로우 실행 응답"""

    workflow_id: str
    kubeflow_run_id: str
    status: str
    message: str
