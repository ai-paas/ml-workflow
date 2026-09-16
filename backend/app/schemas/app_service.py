"""Application Service 관련 Pydantic 스키마 (Service와 Workflow를 관리)"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from schemas.base import TimeStampSchemaMixin
from schemas.user import UserBriefSchema
from schemas.workflow import WorkflowBaseSchema


# ============= Monitoring 스키마 =============
class PeriodMetrics(BaseModel):
    """단일 기간 집계 메트릭"""

    message_count: int = Field(0, description="총 메시지 수")
    active_users: int = Field(0, description="활성 사용자 수")
    token_usage: int = Field(0, description="토큰 사용량")
    avg_interaction_count: float = Field(0.0, description="평균 사용자 상호작용 수")
    response_time_ms: Optional[float] = Field(None, description="평균 응답 시간(ms). 요청 없으면 null")
    error_count: int = Field(0, description="오류 수")
    success_rate: Optional[float] = Field(None, description="성공률(%). 요청 없으면 null")


class MonitoringMetrics(BaseModel):
    """기간별 모니터링 메트릭 (1h / 1d / 1w)"""

    # JSON 키를 "1h"/"1d"/"1w"로 노출 (숫자로 시작하는 식별자는 alias로 처리)
    model_config = ConfigDict(populate_by_name=True)

    period_1h: PeriodMetrics = Field(default_factory=PeriodMetrics, alias="1h", description="최근 1시간 집계")
    period_1d: PeriodMetrics = Field(default_factory=PeriodMetrics, alias="1d", description="최근 1일 집계")
    period_1w: PeriodMetrics = Field(default_factory=PeriodMetrics, alias="1w", description="최근 1주일 집계")


class WorkflowMonitoring(BaseModel):
    """워크플로우별 모니터링 정보"""

    workflow_id: str
    workflow_name: str
    metrics: MonitoringMetrics
    last_updated: datetime


class ServiceMonitoringData(BaseModel):
    """서비스 모니터링 데이터"""

    total_metrics: MonitoringMetrics = Field(..., description="전체 서비스 기간별 메트릭")
    workflow_metrics: List[WorkflowMonitoring] = Field(default_factory=list, description="워크플로우별 기간별 메트릭")
    aggregated_at: datetime = Field(..., description="집계 기준 시각(UTC). 각 기간의 끝점")


# ============= Service 스키마 =============
class ServiceCreateRequest(BaseModel):
    """서비스 생성 요청"""

    name: str = Field(..., min_length=1, max_length=255, description="서비스 이름")
    description: Optional[str] = Field(None, description="서비스 설명")
    tags: List[str] = Field(default_factory=list, description="서비스 태그")


class ServiceCreateInternal(ServiceCreateRequest):
    """서비스 내부 생성용 - creator_id 포함"""

    creator_id: int


class ServiceUpdateRequest(BaseModel):
    """서비스 수정 요청"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ServiceBaseSchema(TimeStampSchemaMixin):
    """서비스 기본 정보 (대표정보)"""

    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    creator_id: int

    class Config:
        from_attributes = True


class ServiceBriefSchema(ServiceBaseSchema):
    """서비스 간략 정보 (리스트용)"""

    creator: UserBriefSchema
    workflow_count: int = Field(0, description="연결된 워크플로우 수")

    class Config:
        from_attributes = True


class ServiceDetailSchema(ServiceBaseSchema):
    """서비스 상세 정보"""

    creator: UserBriefSchema
    workflows: List[WorkflowBaseSchema] = Field(default_factory=list, description="연결된 워크플로우 목록")
    monitoring_data: Optional[ServiceMonitoringData] = Field(None, description="모니터링 데이터")

    class Config:
        from_attributes = True


class ServiceListResponse(BaseModel):
    """서비스 목록 조회 응답"""

    total: int
    items: List[ServiceBriefSchema]


# ============= Resource Usage 스키마 =============
class ResourceUsage(BaseModel):
    """리소스 사용량 정보"""

    cpu_usage_millicores: Optional[float] = Field(None, description="CPU 사용량 (밀리코어 단위)")
    cpu_request_millicores: Optional[float] = Field(None, description="CPU 요청량 (밀리코어 단위)")
    cpu_limit_millicores: Optional[float] = Field(None, description="CPU 제한량 (밀리코어 단위)")
    memory_usage_bytes: Optional[int] = Field(None, description="메모리 사용량 (바이트 단위)")
    memory_request_bytes: Optional[int] = Field(None, description="메모리 요청량 (바이트 단위)")
    memory_limit_bytes: Optional[int] = Field(None, description="메모리 제한량 (바이트 단위)")
    gpu_usage_percent: Optional[float] = Field(None, description="GPU 사용률 (%)")
    gpu_memory_usage_bytes: Optional[int] = Field(None, description="GPU 메모리 사용량 (바이트 단위)")


class PodResourceUsage(BaseModel):
    """Pod별 리소스 사용량 정보"""

    pod_name: str = Field(..., description="Pod 이름")
    namespace: str = Field(..., description="네임스페이스")
    deployment_type: str = Field(..., description="배포 타입 (inferenceservice 또는 service)")
    resource_usage: ResourceUsage = Field(..., description="리소스 사용량")
    status: Optional[str] = Field(None, description="Pod 상태")


class DeploymentResourceUsage(BaseModel):
    """배포별 리소스 사용량 정보"""

    deployment_id: str = Field(..., description="KServe 배포 ID")
    service_name: str = Field(..., description="서비스 이름")
    workflow_id: str = Field(..., description="워크플로우 ID")
    component_id: str = Field(..., description="컴포넌트 ID")
    model_name: str = Field(..., description="모델 이름")
    pods: List[PodResourceUsage] = Field(default_factory=list, description="Pod별 리소스 사용량 목록")


class ServiceResourceUsageResponse(BaseModel):
    """서비스 리소스 사용량 조회 응답"""

    service_id: str = Field(..., description="서비스 ID")
    service_name: str = Field(..., description="서비스 이름")
    deployments: List[DeploymentResourceUsage] = Field(default_factory=list, description="배포별 리소스 사용량 목록")
    total_cpu_usage_millicores: Optional[float] = Field(None, description="전체 CPU 사용량 (밀리코어 단위)")
    total_memory_usage_bytes: Optional[int] = Field(None, description="전체 메모리 사용량 (바이트 단위)")
    total_gpu_usage_percent: Optional[float] = Field(None, description="전체 GPU 사용률 (%)")
