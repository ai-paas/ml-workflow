"""워크플로 모델 컴포넌트 배포 관리 Service."""

from datetime import datetime
from typing import Dict, List, Optional

from config.settings import get_settings
from core.serving.serving_workflow_deployment_policy import backend_api_url_from_internal, kserve_public_infer_url
from db.models.model_workflow_deployment import (
    DeploymentStatus,
    ModelWorkflowDeployment,
    ServingDeviceType,
    WorkflowServingDeploymentType,
)
from repos.model_workflow_deployment import model_workflow_deployment_repository
from schemas.model_workflow_deployment import ModelWorkflowDeploymentBaseSchema
from services.workflow import WorkflowService
from sqlalchemy.orm import Session


class ModelWorkflowDeploymentService:
    """워크플로 MODEL 배포 레코드 관리 (KServe·Ollama 등)."""

    @staticmethod
    def _urls_for(
        deployment: ModelWorkflowDeployment,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """(gateway_url, public_url, backend_api_url) — §2.6."""
        s = get_settings()
        gw_raw = (s.KSERVE_GATEWAY_URL or "").strip()
        gateway_url = gw_raw if gw_raw else None
        public_url = None
        if deployment.deployment_type == WorkflowServingDeploymentType.KSERVE:
            public_url = kserve_public_infer_url(s.KSERVE_GATEWAY_URL or "", deployment.model_name)
        backend_api_url = backend_api_url_from_internal(deployment.internal_url)
        return gateway_url, public_url, backend_api_url

    @staticmethod
    def create_deployment(
        db: Session,
        workflow_id: str,
        component_id: str,
        model_name: str,
        service_name: Optional[str] = None,
        service_hostname: Optional[str] = None,
        *,
        deployment_type: WorkflowServingDeploymentType = WorkflowServingDeploymentType.KSERVE,
        pvc: Optional[str] = None,
        device_type: Optional[ServingDeviceType] = None,
        remote_api_url: Optional[str] = None,
        serving_node_name: Optional[str] = None,
    ) -> ModelWorkflowDeployment:
        existing = model_workflow_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)
        if existing:
            return existing

        if not service_name:
            service_name = f"pending-{workflow_id[:8]}-{component_id[:8]}"
        if not service_hostname:
            service_hostname = "pending"

        deployment_data = ModelWorkflowDeploymentBaseSchema(
            workflow_id=workflow_id,
            component_id=component_id,
            serving_node_name=(serving_node_name or None),
            service_name=service_name,
            service_hostname=service_hostname,
            model_name=model_name.replace("/", "-"),
            deployment_type=deployment_type,
            pvc=pvc,
            device_type=device_type,
            remote_api_url=remote_api_url,
            status=DeploymentStatus.DEPLOYING,
        )

        return model_workflow_deployment_repository.create(db, obj_in=deployment_data)

    @staticmethod
    def update_deployment_status(
        db: Session,
        workflow_id: str,
        component_id: str,
        service_name: str,
        service_hostname: str,
        model_name: str,
        status: str,
        internal_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> ModelWorkflowDeployment:
        deployment = model_workflow_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)

        if deployment:
            deployment.service_name = service_name
            deployment.service_hostname = service_hostname
            deployment.model_name = model_name
            deployment.internal_url = internal_url

            if status == "deployed":
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.deployed_at = datetime.utcnow()
                deployment.error_message = None
            elif status == "failed":
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = error_message
            elif status == "deleted":
                deployment.status = DeploymentStatus.DELETED
                deployment.deleted_at = datetime.utcnow()

            db.commit()
            db.refresh(deployment)
        else:
            if status == "deployed":
                dep_status = DeploymentStatus.DEPLOYED
                deployed_at = datetime.utcnow()
            elif status == "failed":
                dep_status = DeploymentStatus.FAILED
                deployed_at = None
            elif status == "deleted":
                dep_status = DeploymentStatus.DELETED
                deployed_at = None
            else:
                dep_status = DeploymentStatus.DEPLOYING
                deployed_at = None

            deployment_data = ModelWorkflowDeploymentBaseSchema(
                workflow_id=workflow_id,
                component_id=component_id,
                service_name=service_name,
                service_hostname=service_hostname,
                model_name=model_name,
                internal_url=internal_url,
                status=dep_status,
                deployed_at=deployed_at,
                error_message=error_message if status == "failed" else None,
            )
            deployment = model_workflow_deployment_repository.create(db, obj_in=deployment_data)

        return deployment

    @staticmethod
    def get_deployment_info(db: Session, workflow_id: str, component_id: str) -> Optional[Dict]:
        deployment = model_workflow_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)

        if not deployment:
            return None

        gateway_url, public_url, backend_api_url = ModelWorkflowDeploymentService._urls_for(deployment)

        return {
            "service_name": deployment.service_name,
            "service_hostname": deployment.service_hostname,
            "model_name": deployment.model_name,
            "deployment_type": deployment.deployment_type.value,
            "internal_url": deployment.internal_url,
            "gateway_url": gateway_url,
            "public_url": public_url,
            "backend_api_url": backend_api_url,
            "status": deployment.status.value,
            "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
            "error_message": deployment.error_message,
        }

    @staticmethod
    def get_deployed_models(db: Session, workflow_id: str, include_component_info: bool = True) -> List[Dict]:
        deployments = model_workflow_deployment_repository.get_by_workflow(db, workflow_id)

        deployed_models = []
        for deployment in deployments:
            gateway_url, public_url, backend_api_url = ModelWorkflowDeploymentService._urls_for(deployment)

            model_info = {
                "component_id": deployment.component_id,
                "service_name": deployment.service_name,
                "service_hostname": deployment.service_hostname,
                "model_name": deployment.model_name,
                "sanitized_model_name": deployment.model_name,
                "deployment_type": deployment.deployment_type.value,
                "internal_url": deployment.internal_url,
                "gateway_url": gateway_url,
                "public_url": public_url,
                "backend_api_url": backend_api_url,
                "status": deployment.status.value,
                "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
                "error_message": deployment.error_message,
            }

            if include_component_info:
                component = WorkflowService.get_component_by_id_and_workflow_id(
                    db, deployment.component_id, workflow_id
                )

                if component:
                    model_info.update(
                        {
                            "model_id": component.model_id,
                            "model_name": component.name,
                        }
                    )

            deployed_models.append(model_info)

        return deployed_models

    @staticmethod
    def validate_deployment_ready(
        db: Session, workflow_id: str, component_id: str
    ) -> tuple[bool, str, Optional[ModelWorkflowDeployment]]:
        deployment = model_workflow_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)

        if not deployment:
            return False, f"Model component {component_id} not deployed", None

        if deployment.status != DeploymentStatus.DEPLOYED:
            return False, f"Model is in {deployment.status.value} state", deployment

        return True, "", deployment

    @staticmethod
    def has_active_deployment(db: Session, workflow_id: str) -> bool:
        deployments = model_workflow_deployment_repository.get_by_workflow(db, workflow_id)
        return any(d.status in (DeploymentStatus.DEPLOYING, DeploymentStatus.DEPLOYED) for d in deployments)

    @staticmethod
    def has_deploying(db: Session, workflow_id: str) -> bool:
        deployments = model_workflow_deployment_repository.get_by_workflow(db, workflow_id)
        return any(d.status == DeploymentStatus.DEPLOYING for d in deployments)

    @staticmethod
    def cleanup_workflow_deployments(db: Session, workflow_id: str) -> int:
        deployments = model_workflow_deployment_repository.get_by_workflow(db, workflow_id)
        for deployment in deployments:
            model_workflow_deployment_repository.update_status(db, deployment, DeploymentStatus.DELETED)

        return len(deployments)

    @staticmethod
    def delete_workflow_deployments(db: Session, workflow_id: str) -> int:
        from core.serving.serving_model_workflow_pvc import process_deployments_before_hard_delete

        process_deployments_before_hard_delete(db, workflow_id)
        return model_workflow_deployment_repository.cleanup_workflow_deployments(db, workflow_id)
