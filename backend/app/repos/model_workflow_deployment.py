"""워크플로 모델 배포 Repository."""

from datetime import datetime
from typing import List, Optional, Sequence

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

    def cleanup_workflow_deployments(
        self, db: Session, workflow_id: str, allowed_ids: Optional[Sequence[str]] = None
    ) -> int:
        """워크플로의 배포 행 삭제. allowed_ids 를 주면 그 행만 지운다.

        정리가 도는 동안 같은 워크플로가 다시 실행되면 새 배포 행이 생긴다. 그것까지 지우면
        방금 만든 복제 PVC 가 회수돼 실행 중인 파이프라인이 볼륨을 잃는다.
        """
        q = db.query(self.model).filter(self.model.workflow_id == workflow_id)
        if allowed_ids is not None:
            ids = list(allowed_ids)
            if not ids:
                return 0
            q = q.filter(self.model.id.in_(ids))
        count = q.delete(synchronize_session=False)
        db.commit()
        return count


model_workflow_deployment_repository = ModelWorkflowDeploymentRepository(ModelWorkflowDeployment)
