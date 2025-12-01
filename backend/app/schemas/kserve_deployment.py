"""KServe 배포 관련 스키마"""

from datetime import datetime
from typing import Optional

from db.models.kserve_deployment import DeploymentStatus
from pydantic import BaseModel


class KServeDeploymentBaseSchema(BaseModel):
    """KServe 배포 기본 스키마"""

    workflow_id: str
    component_id: str
    service_name: str
    service_hostname: str
    model_name: str
    internal_url: Optional[str] = None
    status: DeploymentStatus = DeploymentStatus.DEPLOYING
    deployed_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    error_message: Optional[str] = None


class KServeDeploymentCreateSchema(KServeDeploymentBaseSchema):
    """KServe 배포 생성 스키마"""

    pass


class KServeDeploymentUpdateSchema(BaseModel):
    """KServe 배포 업데이트 스키마"""

    service_name: Optional[str] = None
    service_hostname: Optional[str] = None
    model_name: Optional[str] = None
    internal_url: Optional[str] = None
    status: Optional[DeploymentStatus] = None
    error_message: Optional[str] = None


class KServeDeploymentReadSchema(KServeDeploymentBaseSchema):
    """KServe 배포 읽기 스키마"""

    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KServeDeploymentInfoSchema(BaseModel):
    """KServe 배포 정보 스키마"""

    component_id: str
    service_name: str
    service_hostname: str
    model_name: str
    sanitized_model_name: str
    internal_url: Optional[str] = None
    gateway_url: str
    status: str
    deployed_at: Optional[str] = None
    error_message: Optional[str] = None
    model_id: Optional[int] = None
    model_name: Optional[str] = None
