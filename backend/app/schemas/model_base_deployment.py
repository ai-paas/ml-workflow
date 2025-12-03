"""모델 기본 배포 관련 스키마"""

from datetime import datetime
from typing import Optional

from db.models.model_base_deployment import BaseDeploymentStatus
from pydantic import BaseModel


class ModelBaseDeploymentBaseSchema(BaseModel):
    """모델 기본 배포 기본 스키마"""

    model_id: int
    service_name: str
    service_hostname: Optional[str] = None
    model_name: str
    internal_url: Optional[str] = None
    status: BaseDeploymentStatus = BaseDeploymentStatus.DEPLOYING
    deployed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ModelBaseDeploymentCreateSchema(ModelBaseDeploymentBaseSchema):
    """모델 기본 배포 생성 스키마"""

    pass


class ModelBaseDeploymentUpdateSchema(BaseModel):
    """모델 기본 배포 업데이트 스키마"""

    service_name: Optional[str] = None
    service_hostname: Optional[str] = None
    model_name: Optional[str] = None
    internal_url: Optional[str] = None
    status: Optional[BaseDeploymentStatus] = None
    error_message: Optional[str] = None


class ModelBaseDeploymentReadSchema(ModelBaseDeploymentBaseSchema):
    """모델 기본 배포 읽기 스키마"""

    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
