"""Application Service 관련 Pydantic 스키마 (Service와 Workflow를 관리)"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from schemas.base import TimeStampSchemaMixin
from schemas.user import UserSchema
from schemas.workflow import WorkflowBaseSchema, WorkflowReadSchema


class ServiceStatus(str, Enum):
    """서비스 상태"""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"


# ============= Monitoring 스키마 =============
class MonitoringMetrics(BaseModel):
    """모니터링 메트릭"""

    message_count: int = Field(0, description="최근 1시간 총 메시지 수")
    active_users: int = Field(0, description="최근 1시간 활성 사용자 수")
    token_usage: int = Field(0, description="최근 1시간 토큰 사용량")
    avg_interaction_count: float = Field(0.0, description="최근 1시간 평균 사용자 상호작용 수")
    response_time_ms: Optional[float] = Field(None, description="평균 응답 시간(ms)")
    error_count: int = Field(0, description="최근 1시간 오류 수")
    success_rate: float = Field(100.0, description="최근 1시간 성공률(%)")


class WorkflowMonitoring(BaseModel):
    """워크플로우별 모니터링 정보"""

    workflow_id: str
    workflow_name: str
    metrics: MonitoringMetrics
    last_updated: datetime


class ServiceMonitoringData(BaseModel):
    """서비스 모니터링 데이터"""

    total_metrics: MonitoringMetrics = Field(..., description="전체 서비스 메트릭")
    workflow_metrics: List[WorkflowMonitoring] = Field(default_factory=list, description="워크플로우별 메트릭")
    period_start: datetime = Field(..., description="집계 시작 시간")
    period_end: datetime = Field(..., description="집계 종료 시간")


# ============= Service 스키마 =============
class ServiceCreateRequest(BaseModel):
    """서비스 생성 요청"""

    name: str = Field(..., min_length=1, max_length=255, description="서비스 이름")
    description: Optional[str] = Field(None, description="서비스 설명")
    tags: List[str] = Field(default_factory=list, description="서비스 태그")


class ServiceCreateInternal(ServiceCreateRequest):
    """서비스 내부 생성용 - creator_id와 status 포함"""

    creator_id: int
    status: ServiceStatus = ServiceStatus.DRAFT


class ServiceUpdateRequest(BaseModel):
    """서비스 수정 요청"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[ServiceStatus] = None


class ServiceBaseSchema(TimeStampSchemaMixin):
    """서비스 기본 정보 (대표정보)"""

    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    creator_id: int
    status: ServiceStatus

    class Config:
        from_attributes = True


class ServiceBriefSchema(ServiceBaseSchema):
    """서비스 간략 정보 (리스트용)"""

    creator: UserSchema
    workflow_count: int = Field(0, description="연결된 워크플로우 수")

    class Config:
        from_attributes = True


class ServiceDetailSchema(ServiceBaseSchema):
    """서비스 상세 정보"""

    creator: UserSchema
    workflows: List[WorkflowBaseSchema] = Field(default_factory=list, description="연결된 워크플로우 목록")
    monitoring_data: Optional[ServiceMonitoringData] = Field(None, description="모니터링 데이터")
    kserve_endpoint: Optional[str] = Field(None, description="KServe 엔드포인트")
    public_url: Optional[str] = Field(None, description="공개 URL")
    backend_api_url: Optional[str] = Field(None, description="백엔드 서비스 API URL")

    class Config:
        from_attributes = True


class ServiceListResponse(BaseModel):
    """서비스 목록 조회 응답"""

    total: int
    items: List[ServiceBriefSchema]


# ============= Service Deployment 스키마 =============
class ServiceDeployRequest(BaseModel):
    """서비스 배포 요청"""

    min_replicas: int = Field(1, ge=0, description="최소 복제본 수")
    max_replicas: int = Field(3, ge=1, description="최대 복제본 수")
    cpu_request: str = Field("100m", description="CPU 요청량")
    memory_request: str = Field("128Mi", description="메모리 요청량")
    cpu_limit: str = Field("500m", description="CPU 제한")
    memory_limit: str = Field("512Mi", description="메모리 제한")
    enable_gpu: bool = Field(False, description="GPU 사용 여부")
    gpu_count: int = Field(0, description="GPU 수")


class ServiceDeployResponse(BaseModel):
    """서비스 배포 응답"""

    service_id: str
    status: str
    kserve_endpoint: str
    public_url: str
    message: str
