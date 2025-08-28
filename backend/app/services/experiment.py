from repos.experiment import experiment_repository, hyperparameter_repository, hyperparameter_type_repository
from schemas.experiment import (
    ExperimentBaseSchema,
    ExperimentReadSchema,
    ExperimentUpdateRequest,
    HyperparameterBaseSchema,
    HyperparameterReadSchema,
    HyperparameterTypeBaseSchema,
    HyperparameterTypeReadSchema,
)
from sqlalchemy.orm import Session


class ExperimentService:
    @staticmethod
    def create(db: Session, *, obj_in: ExperimentBaseSchema):
        experiment_db_obj = experiment_repository.create(db, obj_in=obj_in)
        db.commit()
        return experiment_db_obj

    @staticmethod
    def get(db: Session, pk: int) -> ExperimentReadSchema:
        return experiment_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[ExperimentReadSchema]:
        return experiment_repository.get_multi(db, skip=skip, limit=limit)


class HyperparameterService:
    @staticmethod
    def create(db: Session, *, obj_in: HyperparameterBaseSchema):
        hyperparameter_db_obj = hyperparameter_repository.create(db, obj_in=obj_in)
        db.commit()
        return hyperparameter_db_obj

    @staticmethod
    def get(db: Session, pk: int) -> HyperparameterReadSchema:
        return hyperparameter_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[HyperparameterReadSchema]:
        return hyperparameter_repository.get_multi(db, skip=skip, limit=limit)


class HyperparameterTypeService:
    @staticmethod
    def create(db: Session, *, obj_in: HyperparameterTypeBaseSchema):
        return hyperparameter_type_repository.create(db, obj_in=obj_in)

    @staticmethod
    def get(db: Session, pk: int) -> HyperparameterTypeReadSchema:
        return hyperparameter_type_repository.get(db, pk)

    @staticmethod
    def get_by_param_name(db: Session, param_name: str) -> HyperparameterTypeReadSchema:
        return hyperparameter_type_repository.get_by_param_name(db, param_name)

    @staticmethod
    def update(db: Session, *, obj_in: ExperimentUpdateRequest):
        return experiment_repository.update(db, obj_in=obj_in)
