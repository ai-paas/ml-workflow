from pydantic import BaseModel
from schemas.base import TimeStampSchemaMixin
from schemas.dataset import DatasetReadSchema
from schemas.model import ModelReadSchema


class ExperimentBaseSchema(TimeStampSchemaMixin):
    name: str
    reference_model_id: int
    dataset_id: int
    kubeflow_run_id: str
    mlflow_run_id: str
    status: str
    learning_yn: bool


class ExperimentReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    reference_model_id: int
    dataset_id: int
    kubeflow_run_id: str
    mlflow_run_id: str
    status: str
    learning_yn: bool
    reference_model: "ModelReadSchema"
    dataset: "DatasetReadSchema"

    class Config:
        from_attributes = True


class HyperparameterTypeBaseSchema(BaseModel):
    param_name: str
    param_type: str
    default_value: str


class HyperparameterTypeReadSchema(BaseModel):
    id: int
    param_name: str
    param_type: str
    default_value: str

    class Config:
        from_attributes = True


class HyperparameterBaseSchema(BaseModel):
    value: str
    reference_model_id: int
    hyperparameter_type_id: int


class HyperparameterReadSchema(BaseModel):
    id: int
    value: str
    reference_model_id: int
    hyperparameter_type_id: int
    hyperparameter_type: "HyperparameterTypeReadSchema"

    class Config:
        from_attributes = True
