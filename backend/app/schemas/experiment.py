from typing import Any, Dict, Optional

from pydantic import BaseModel
from schemas.base import TimeStampSchemaMixin
from schemas.dataset import DatasetReadSchema
from schemas.model import ModelReadSchema


class ExperimentCreateRequest(BaseModel):
    name: str
    description: str
    reference_model_id: int
    dataset_id: int
    epochs: int
    batch_size: int
    weight_decay: float
    lr0: float
    lrf: float
    gpus: int
    save_period: int


class ExperimentBaseSchema(TimeStampSchemaMixin):
    name: str
    description: Optional[str] = None
    reference_model_id: int
    dataset_id: int
    kubeflow_run_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    status: str


class ExperimentReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str
    reference_model_id: int
    dataset_id: int
    kubeflow_run_id: str
    mlflow_run_id: str
    status: str
    reference_model: "ModelReadSchema"
    dataset: "DatasetReadSchema"
    hyperparameters: list["HyperparameterReadSchema"]

    class Config:
        from_attributes = True


class ExperimentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    reference_model_id: Optional[int] = None
    dataset_id: Optional[int] = None
    kubeflow_run_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    status: Optional[str] = None


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
    experiment_id: int
    hyperparameter_type_id: int


class HyperparameterReadSchema(BaseModel):
    id: int
    value: str
    experiment_id: int
    hyperparameter_type_id: int
    hyperparameter_type: "HyperparameterTypeReadSchema"

    class Config:
        from_attributes = True


# Training Status 관련 Pydantic 모델들
class TrainingStatusResponse(BaseModel):
    status: str  # RUNNING, FINISHED, FAILED
    start_time: int  # unix timestamp
    end_time: Optional[int] = None  # unix timestamp
    max_epoch: int
    current_epoch: int
    loss_history: list[Any]
    epoch_history: list[Any]
    average_precision_50_history: list[Any]
    average_precision_75_history: list[Any]
    best_average_precision_history: list[Any]
    average_precision_50_95_history: list[Any]
