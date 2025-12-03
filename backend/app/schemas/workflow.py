"""Workflow 관련 Pydantic 스키마"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from config.settings import get_settings
from db.models.service import ComponentType
from pydantic import BaseModel, Field, computed_field, model_serializer
from schemas.base import TimeStampSchemaMixin
from schemas.kserve_deployment import KServeDeploymentReadSchema
from schemas.model import ModelBriefReadSchema
from schemas.user import UserSchema

if TYPE_CHECKING:
    from db.models.service import ComponentConnection


class WorkflowStatus(str, Enum):
    """워크플로우 상태"""

    DRAFT = "DRAFT"  # 임시저장 상태
    ACTIVE = "ACTIVE"  # 정상 상태 (사용 가능)
    ERROR = "ERROR"  # 오류 상태


class ComponentTypeInfo(BaseModel):
    """컴포넌트 타입 정보"""

    type: str = Field(..., description="컴포넌트 타입 (START, END, MODEL, KNOWLEDGE_BASE)")
    component_id: str = Field(..., description="자동 생성되는 component_id")
    name: str = Field(..., description="타입 표시명")
    description: str = Field(..., description="타입 설명")


# ============= Component 스키마 =============


class ComponentCreateRequest(BaseModel):
    """컴포넌트 생성 요청"""

    name: str
    type: ComponentType
    model_id: Optional[int] = Field(None, description="모델 컴포넌트인 경우 모델 ID")
    knowledge_base_id: Optional[int] = Field(None, description="Knowledge Base 컴포넌트인 경우 Knowledge Base ID")
    prompt_id: Optional[int] = Field(None, description="모델 컴포넌트인 경우 프롬프트 ID")


class ComponentUpdateRequest(BaseModel):
    """컴포넌트 수정 요청 (config 제외)"""

    name: str
    type: ComponentType
    model_id: Optional[int] = Field(None, description="모델 컴포넌트인 경우 모델 ID")
    knowledge_base_id: Optional[int] = Field(None, description="Knowledge Base 컴포넌트인 경우 Knowledge Base ID")
    prompt_id: Optional[int] = Field(None, description="모델 컴포넌트인 경우 프롬프트 ID")


class ComponentReadSchema(TimeStampSchemaMixin):
    """컴포넌트 조회 응답"""

    id: str
    workflow_id: str
    name: str
    type: ComponentType
    model_id: Optional[int] = None
    model: Optional[ModelBriefReadSchema] = None
    knowledge_base_id: Optional[int] = None
    prompt_id: Optional[int] = None

    class Config:
        from_attributes = True


# ============= Connection 스키마 =============
class ConnectionCreateRequest(BaseModel):
    """컴포넌트 연결 생성 요청"""

    source_component_type: ComponentType = Field(..., description="소스 컴포넌트 타입")
    target_component_type: ComponentType = Field(..., description="타겟 컴포넌트 타입")


class ConnectionUpdateRequest(BaseModel):
    """컴포넌트 연결 수정 요청 (connection_type, config 제외)"""

    source_component_type: ComponentType = Field(..., description="소스 컴포넌트 타입")
    target_component_type: ComponentType = Field(..., description="타겟 컴포넌트 타입")


class ConnectionReadSchema(BaseModel):
    """컴포넌트 연결 조회 응답"""

    id: str
    workflow_id: str
    source_component_id: str
    target_component_id: str
    source_component: ComponentReadSchema
    target_component: ComponentReadSchema
    created_at: datetime

    class Config:
        from_attributes = True


# ============= Workflow Definition 스키마 =============
class WorkflowDefinition(BaseModel):
    """워크플로우 정의 (컴포넌트와 연결 정보)"""

    components: List[ComponentCreateRequest]
    connections: List[ConnectionCreateRequest]


class WorkflowUpdateDefinition(BaseModel):
    """워크플로우 수정 정의 (components의 config, connections의 connection_type/config 제외)"""

    components: List[ComponentUpdateRequest]
    connections: List[ConnectionUpdateRequest]


# ============= Workflow 스키마 =============
class WorkflowCreateRequest(BaseModel):
    """워크플로우 생성 요청"""

    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    service_id: Optional[str] = None
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
    workflow_definition: Optional[WorkflowUpdateDefinition] = None


class WorkflowTemplateUpdateRequest(BaseModel):
    """워크플로우 템플릿 수정 요청 (service_id 제외)"""

    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[WorkflowStatus] = None
    workflow_definition: Optional[WorkflowUpdateDefinition] = None


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
class WorkflowTemplateCreateRequest(BaseModel):
    """워크플로우 템플릿 생성 요청"""

    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    workflow_definition: Optional[WorkflowDefinition] = None
    # service_id와 template_id는 템플릿에 포함되지 않음
    # 템플릿은 서비스에 연결되지 않으며, 다른 템플릿으로부터 복사하지 않음


class WorkflowTemplateBriefSchema(WorkflowBaseSchema):
    """워크플로우 템플릿 목록 조회용 간략 정보"""

    creator: UserSchema
    usage_count: int = Field(0, description="템플릿 사용 횟수")

    class Config:
        from_attributes = True


class WorkflowTemplateListSchema(BaseModel):
    """워크플로우 템플릿 목록 조회 응답"""

    total: int
    items: List[WorkflowTemplateBriefSchema]


class WorkflowTemplateReadSchema(WorkflowReadSchema):
    """워크플로우 템플릿 상세 조회 응답"""

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


# ============= Workflow Test 스키마 =============
class KnowledgeBaseTestResult(BaseModel):
    """지식베이스 테스트 결과"""

    search_result: str = Field(..., description="검색 결과 문자열")
    total: int = Field(..., description="검색 결과 총 개수")
    search_method: str = Field(..., description="검색 방법")


class ModelODMTestResult(BaseModel):
    """ODM 모델 테스트 결과"""

    predictions: List[Dict[str, Any]] = Field(..., description="추론 결과 목록")
    image_info: Optional[Dict[str, Any]] = Field(None, description="이미지 메타데이터")


class ModelLLMTestResult(BaseModel):
    """LLM 모델 테스트 결과"""

    response: str = Field(..., description="LLM 응답 텍스트")
    full_response: Optional[Dict[str, Any]] = Field(None, description="전체 응답 (Ollama API 응답)")


class ComponentTestResultBase(BaseModel):
    """컴포넌트 테스트 결과 기본"""

    component_id: str
    component_name: str
    component_type: str  # KNOWLEDGE_BASE 또는 MODEL
    model_type: str  # embedding, ODM, LLM


class KnowledgeBaseComponentTestResult(ComponentTestResultBase):
    """지식베이스 컴포넌트 테스트 결과"""

    component_type: str = Field(default="KNOWLEDGE_BASE", description="컴포넌트 타입")
    model_type: str = Field(default="embedding", description="모델 타입")
    result: KnowledgeBaseTestResult


class ModelComponentTestResult(ComponentTestResultBase):
    """모델 컴포넌트 테스트 결과"""

    component_type: str = Field(default="MODEL", description="컴포넌트 타입")
    model_type: str  # ODM 또는 LLM
    result: Union[ModelODMTestResult, ModelLLMTestResult]


class ComponentTestErrorResult(BaseModel):
    """컴포넌트 테스트 오류 결과"""

    component_id: str
    component_name: str
    component_type: str
    model_type: Optional[str] = None
    error: str


ComponentTestResult = Union[KnowledgeBaseComponentTestResult, ModelComponentTestResult, ComponentTestErrorResult]


class WorkflowRAGTestResponse(BaseModel):
    """RAG 워크플로우 테스트 응답"""

    workflow_id: str
    execution_order: List[str]
    results: List[ComponentTestResult]
    final_result: Optional[str] = Field(None, description="최종 결과 문자열 (LLM 응답 또는 검색 결과)")


class WorkflowMLTestResponse(BaseModel):
    """ML 워크플로우 테스트 응답"""

    workflow_id: str
    execution_order: List[str]
    results: List[ComponentTestResult]
    final_result: Optional[str] = Field(
        None, description="최종 결과 이미지 (bbox와 label이 그려진 이미지를 base64로 인코딩한 문자열)"
    )
