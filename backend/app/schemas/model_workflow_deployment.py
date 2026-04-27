"""워크플로 모델 배포 관련 스키마."""

from datetime import datetime
from typing import Optional

from db.models.model_workflow_deployment import DeploymentStatus, ServingDeviceType, WorkflowServingDeploymentType
from pydantic import BaseModel


class ModelWorkflowDeploymentBaseSchema(BaseModel):
    workflow_id: str
    component_id: str
    serving_node_name: Optional[str] = None
    service_name: str
    service_hostname: str
    model_name: str
    internal_url: Optional[str] = None
    deployment_type: WorkflowServingDeploymentType = WorkflowServingDeploymentType.KSERVE
    pvc: Optional[str] = None
    device_type: Optional[ServingDeviceType] = None
    remote_api_url: Optional[str] = None
    status: DeploymentStatus = DeploymentStatus.DEPLOYING
    deployed_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ModelWorkflowDeploymentCreateSchema(ModelWorkflowDeploymentBaseSchema):
    pass


class ModelWorkflowDeploymentUpdateSchema(BaseModel):
    service_name: Optional[str] = None
    service_hostname: Optional[str] = None
    model_name: Optional[str] = None
    serving_node_name: Optional[str] = None
    internal_url: Optional[str] = None
    deployment_type: Optional[WorkflowServingDeploymentType] = None
    pvc: Optional[str] = None
    device_type: Optional[ServingDeviceType] = None
    remote_api_url: Optional[str] = None
    status: Optional[DeploymentStatus] = None
    error_message: Optional[str] = None


class ModelWorkflowDeploymentReadSchema(ModelWorkflowDeploymentBaseSchema):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelWorkflowDeploymentInfoSchema(BaseModel):
    component_id: str
    service_name: str
    service_hostname: str
    model_name: str
    sanitized_model_name: str
    deployment_type: str
    internal_url: Optional[str] = None
    gateway_url: Optional[str] = None
    public_url: Optional[str] = None
    backend_api_url: Optional[str] = None
    status: str
    deployed_at: Optional[str] = None
    error_message: Optional[str] = None
    model_id: Optional[int] = None
