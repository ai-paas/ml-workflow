"""KServe 배포 정보 Repository"""

from typing import List, Optional

from db.models.kserve_deployment import DeploymentStatus, KServeDeployment
from repos.base import CRUDBase
from schemas.kserve_deployment import KServeDeploymentBaseSchema
from sqlalchemy.orm import Session


class KServeDeploymentRepository(CRUDBase[KServeDeployment, KServeDeploymentBaseSchema, KServeDeploymentBaseSchema]):
    def get_by_workflow_component(self, db: Session, workflow_id: str, component_id: str) -> Optional[KServeDeployment]:
        """워크플로우와 컴포넌트 ID로 배포 정보 조회"""
        return (
            db.query(self.model)
            .filter(self.model.workflow_id == workflow_id, self.model.component_id == component_id)
            .first()
        )

    def get_by_service_name(self, db: Session, service_name: str) -> Optional[KServeDeployment]:
        """서비스 이름으로 배포 정보 조회"""
        return db.query(self.model).filter(self.model.service_name == service_name).first()

    def get_by_workflow(
        self, db: Session, workflow_id: str, status: Optional[DeploymentStatus] = None
    ) -> List[KServeDeployment]:
        """워크플로우의 모든 배포 정보 조회"""
        query = db.query(self.model).filter(self.model.workflow_id == workflow_id)

        if status:
            query = query.filter(self.model.status == status)

        return query.all()

    def get_deployed_models(self, db: Session, workflow_id: str) -> List[KServeDeployment]:
        """워크플로우의 배포된 모델 목록 조회"""
        return self.get_by_workflow(db, workflow_id, DeploymentStatus.DEPLOYED)

    def update_status(
        self, db: Session, deployment: KServeDeployment, status: DeploymentStatus, error_message: Optional[str] = None
    ) -> KServeDeployment:
        """배포 상태 업데이트"""
        from datetime import datetime

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
        """워크플로우의 모든 배포 정보 삭제"""
        count = db.query(self.model).filter(self.model.workflow_id == workflow_id).delete()
        db.commit()
        return count


# Repository 인스턴스
kserve_deployment_repository = KServeDeploymentRepository(KServeDeployment)
