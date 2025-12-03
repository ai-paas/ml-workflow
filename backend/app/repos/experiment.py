from db.models.experiment import ExperimentModel, Hyperparameter, HyperparameterType
from repos.base import CRUDBase
from schemas.experiment import (
    ExperimentBaseSchema,
    ExperimentUpdateRequest,
    HyperparameterBaseSchema,
    HyperparameterTypeBaseSchema,
    HyperparameterTypeReadSchema,
)
from sqlalchemy.orm import Session


class ExperimentRepository(CRUDBase[ExperimentModel, ExperimentBaseSchema, ExperimentUpdateRequest]):
    def get_by_reference_model_id(self, db: Session, reference_model_id: int) -> list[ExperimentModel]:
        """reference_model_id로 실험 목록 조회"""
        return db.query(self.model).filter(self.model.reference_model_id == reference_model_id).all()


class HyperparameterTypeRepository(
    CRUDBase[HyperparameterType, HyperparameterTypeBaseSchema, HyperparameterTypeBaseSchema]
):
    def get_by_param_name(self, db: Session, param_name: str) -> HyperparameterTypeReadSchema:
        return db.query(self.model).filter(self.model.param_name == param_name).first()


class HyperparameterRepository(CRUDBase[Hyperparameter, HyperparameterBaseSchema, HyperparameterBaseSchema]):
    pass


experiment_repository = ExperimentRepository(ExperimentModel)
hyperparameter_type_repository = HyperparameterTypeRepository(HyperparameterType)
hyperparameter_repository = HyperparameterRepository(Hyperparameter)
