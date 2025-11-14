"""모델 기본 배포 정보 Repository"""

from typing import List, Optional

from db.models.model_base_deployment import BaseDeploymentStatus, ModelBaseDeployment
from repos.base import CRUDBase
from schemas.model_base_deployment import ModelBaseDeploymentBaseSchema
from sqlalchemy.orm import Session


class ModelBaseDeploymentRepository(
    CRUDBase[ModelBaseDeployment, ModelBaseDeploymentBaseSchema, ModelBaseDeploymentBaseSchema]
):
    def get_by_model_id(self, db: Session, model_id: int) -> Optional[ModelBaseDeployment]:
        """모델 ID로 배포 정보 조회"""
        return db.query(self.model).filter(self.model.model_id == model_id).first()

    def get_by_service_name(self, db: Session, service_name: str) -> Optional[ModelBaseDeployment]:
        """서비스 이름으로 배포 정보 조회"""
        return db.query(self.model).filter(self.model.service_name == service_name).first()

    def get_by_status(self, db: Session, status: Optional[BaseDeploymentStatus] = None) -> List[ModelBaseDeployment]:
        """상태별 배포 정보 조회"""
        query = db.query(self.model)

        if status:
            query = query.filter(self.model.status == status)

        return query.all()

    def get_deployed_models(self, db: Session) -> List[ModelBaseDeployment]:
        """배포된 모델 목록 조회"""
        return self.get_by_status(db, BaseDeploymentStatus.DEPLOYED)

    def update_status(
        self,
        db: Session,
        deployment: ModelBaseDeployment,
        status: BaseDeploymentStatus,
        error_message: Optional[str] = None,
    ) -> ModelBaseDeployment:
        """배포 상태 업데이트"""
        from datetime import datetime

        deployment.status = status

        if status == BaseDeploymentStatus.DEPLOYED:
            deployment.deployed_at = datetime.utcnow()
            deployment.error_message = None
        elif status == BaseDeploymentStatus.FAILED:
            deployment.error_message = error_message
        elif status == BaseDeploymentStatus.DELETED:
            from datetime import datetime

            deployment.deployed_at = None  # deleted_at은 TimestampMixin에서 처리

        db.commit()
        db.refresh(deployment)
        return deployment


# Repository 인스턴스
model_base_deployment_repository = ModelBaseDeploymentRepository(ModelBaseDeployment)
