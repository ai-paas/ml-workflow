"""워크플로 모델 배포 Repository."""

from datetime import datetime
from typing import List, Optional

from db.models.model_workflow_deployment import DeploymentStatus, ModelWorkflowDeployment
from repos.base import CRUDBase
from schemas.model_workflow_deployment import ModelWorkflowDeploymentBaseSchema
from sqlalchemy.orm import Session


class ModelWorkflowDeploymentRepository(
    CRUDBase[ModelWorkflowDeployment, ModelWorkflowDeploymentBaseSchema, ModelWorkflowDeploymentBaseSchema]
):
    def get_by_workflow_component(
        self, db: Session, workflow_id: str, component_id: str
    ) -> Optional[ModelWorkflowDeployment]:
        return (
            db.query(self.model)
            .filter(self.model.workflow_id == workflow_id, self.model.component_id == component_id)
            .first()
        )

    def get_by_service_name(self, db: Session, service_name: str) -> Optional[ModelWorkflowDeployment]:
        return db.query(self.model).filter(self.model.service_name == service_name).first()

    def get_by_workflow(
        self, db: Session, workflow_id: str, status: Optional[DeploymentStatus] = None
    ) -> List[ModelWorkflowDeployment]:
        query = db.query(self.model).filter(self.model.workflow_id == workflow_id)
        if status:
            query = query.filter(self.model.status == status)
        return query.all()

    def get_deployed_models(self, db: Session, workflow_id: str) -> List[ModelWorkflowDeployment]:
        return self.get_by_workflow(db, workflow_id, DeploymentStatus.DEPLOYED)

    def update_status(
        self,
        db: Session,
        deployment: ModelWorkflowDeployment,
        status: DeploymentStatus,
        error_message: Optional[str] = None,
    ) -> ModelWorkflowDeployment:
        deployment.status = status

        if status == DeploymentStatus.DEPLOYED:
            deployment.deployed_at = datetime.utcnow()
            deployment.error_message = None
        elif status == DeploymentStatus.FAILED:
            deployment.error_message = error_message
        elif status == DeploymentStatus.DELETED:
            deployment.deleted_at = datetime.utcnow()

        db.commit()
        db.refresh(deployment)
        return deployment

    def cleanup_workflow_deployments(self, db: Session, workflow_id: str) -> int:
        count = db.query(self.model).filter(self.model.workflow_id == workflow_id).delete()
        db.commit()
        return count


model_workflow_deployment_repository = ModelWorkflowDeploymentRepository(ModelWorkflowDeployment)
