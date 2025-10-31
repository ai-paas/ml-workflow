"""Workflow 관련 Pydantic 스키마"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from pydantic import BaseModel, Field, computed_field, model_serializer
from schemas.base import TimeStampSchemaMixin
from schemas.kserve_deployment import KServeDeploymentReadSchema
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


class ComponentTypeInfo(BaseModel):
    """컴포넌트 타입 정보"""

    type: str = Field(..., description="컴포넌트 타입 (START, END, MODEL)")
    component_id: str = Field(..., description="자동 생성되는 component_id")
    name: str = Field(..., description="타입 표시명")
    description: str = Field(..., description="타입 설명")


# ============= Component 스키마 =============


class ComponentCreateRequest(BaseModel):
    """컴포넌트 생성 요청"""

    component_id: str = Field(
        ...,
        description="워크플로우 내 고유 ID \
(START, END, MODEL 등 - /component-types API로 확인 가능)",
    )
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


class WorkflowCreateInternal(BaseModel):
    """워크플로우 내부 생성용 - status와 creator_id 포함, workflow_definition 제외"""

    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    service_id: Optional[str] = None
    is_template: bool = Field(False, description="템플릿 여부")
    template_id: Optional[str] = Field(None, description="템플릿 ID (템플릿으로부터 생성시)")
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


class WorkflowUpdateInternal(BaseModel):
    """워크플로우 내부 수정용 - WorkflowUpdateRequest와 동일하지만 workflow_definition 제외"""

    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[WorkflowStatus] = None
    service_id: Optional[str] = None


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
    kubeflow_run_id: Optional[str] = None
    components: List[ComponentReadSchema] = Field(default_factory=list)
    component_connections: List[ConnectionReadSchema] = Field(default_factory=list)
    kserve_deployments: List[KServeDeploymentReadSchema] = Field(default_factory=list, exclude=True)
    service: Optional[Any] = Field(None, exclude=True)  # Service 관계 객체 (직접 접근용, 응답에서 제외)
    template: Optional[Any] = Field(None, exclude=True)  # Workflow 관계 객체 (직접 접근용, 응답에서 제외)

    @computed_field
    @property
    def service_name(self) -> Optional[str]:
        """서비스 이름 (service 관계에서 동적으로 가져옴)"""
        if self.service:
            return self.service.name
        return None

    @computed_field
    @property
    def template_name(self) -> Optional[str]:
        """템플릿 이름 (template 관계에서 동적으로 가져옴)"""
        if self.template:
            return self.template.name
        return None

    @computed_field
    @property
    def public_url(self) -> Optional[str]:
        """KServe 배포 정보를 기반으로 동적으로 생성된 공개 URL"""
        if not self.kserve_deployments:
            return None

        # 첫 번째 배포된 모델의 정보를 사용하여 URL 생성
        first_deployment = self.kserve_deployments[0]
        settings = get_settings()
        gateway_url = settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80"
        model_name = first_deployment.model_name

        # KServe V2 Protocol inference 엔드포인트
        return f"{gateway_url}/v2/models/{model_name}/infer"

    @computed_field
    @property
    def backend_api_url(self) -> Optional[str]:
        """KServe 배포 정보를 기반으로 동적으로 생성된 백엔드 API URL"""
        if not self.kserve_deployments:
            return None

        # 첫 번째 배포된 모델의 정보를 사용하여 URL 생성
        first_deployment = self.kserve_deployments[0]
        settings = get_settings()
        gateway_url = settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80"
        model_name = first_deployment.model_name

        # KServe V2 Protocol inference 엔드포인트
        return f"{gateway_url}/v2/models/{model_name}/infer"

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
