from typing import Optional

from db.models.experiment import ExperimentModel, Hyperparameter
from repos.base import CRUDBase
from schemas.experiment import ExperimentBaseSchema, ExperimentUpdateRequest, HyperparameterBaseSchema
from sqlalchemy.orm import Session


class ExperimentRepository(CRUDBase[ExperimentModel, ExperimentBaseSchema, ExperimentUpdateRequest]):
    def get_by_reference_model_id(self, db: Session, reference_model_id: int) -> list[ExperimentModel]:
        """reference_model_id로 실험 목록 조회"""
        return db.query(self.model).filter(self.model.reference_model_id == reference_model_id).all()

    def get_by_name(self, db: Session, name: str) -> Optional[ExperimentModel]:
        """이름으로 실험 1건 조회(이름 중복 방지용)."""
        return db.query(self.model).filter(self.model.name == name).first()


class HyperparameterRepository(CRUDBase[Hyperparameter, HyperparameterBaseSchema, HyperparameterBaseSchema]):
    pass


experiment_repository = ExperimentRepository(ExperimentModel)
hyperparameter_repository = HyperparameterRepository(Hyperparameter)
