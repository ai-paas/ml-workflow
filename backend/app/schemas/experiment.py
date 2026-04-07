import json
from datetime import datetime
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
    registration_kubeflow_run_id: Optional[str] = None


class ExperimentReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str
    reference_model_id: int
    dataset_id: int
    kubeflow_run_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    status: str
    reference_model: "ModelReadSchema"
    dataset: "DatasetReadSchema"
    hyperparameters: list["HyperparameterReadSchema"]

    class Config:
        from_attributes = True


class ExperimentUpdateRequest(BaseModel):
    """
    실험 수정 요청 스키마

    학습이 진행 중이거나 완료된 실험에서는 name과 description만 수정 가능합니다.
    다른 필드들(model_id, dataset_id, hyperparameters 등)은 학습 결과에 영향을 주므로
    수정할 수 없습니다.
    """

    name: Optional[str] = None
    description: Optional[str] = None


class ExperimentInternalUpdateRequest(BaseModel):
    """
    내부 통신 전용 실험 수정 요청 스키마

    시스템 내부 통신에서 사용하는 스키마로, status, mlflow_run_id, kubeflow_run_id를 수정할 수 있습니다.
    """

    status: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    kubeflow_run_id: Optional[str] = None
    registration_kubeflow_run_id: Optional[str] = None


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
    status: str
    start_time: int
    end_time: Optional[int] = None
    elapsed_time: int
    max_epoch: int
    current_epoch: int
    loss_history: list[Any]
    epoch_history: list[Any]
    average_precision_50_history: list[Any]
    average_precision_75_history: list[Any]
    best_average_precision_history: list[Any]
    average_precision_50_95_history: list[Any]


# ── 학습 요청 Body 스키마 ──


class TrainingRequest(BaseModel):
    """POST /pipeline/training 요청 바디"""

    model_id: int
    dataset_id: int
    train_name: str = ""
    description: str = ""
    gpus: str = "1"
    batch_size: str = "32"
    epochs: str = "5"
    save_period: str = "1"
    weight_decay: str = "5e-4"
    lr0: str = "0.01"
    lrf: str = "0.05"


# ── 모델 등록 요청/응답 스키마 ──


class ModelRegistrationRequest(BaseModel):
    """POST /pipeline/model/registration 요청 바디"""

    model_name: str
    description: str
    experiment_id: int


class ModelRegistrationResponse(BaseModel):
    """POST /pipeline/model/registration 응답"""

    accepted: bool
    experiment_id: int
    message: str


# ── Experiment 메트릭 스키마 ──


class ExperimentMetricsSchema(BaseModel):
    elapsed_time: int | None = None
    end_time: datetime | None = None
    max_epoch: int = 0
    current_epoch: int = 0
    loss: float | None = None
    loss_history: list[dict] | None = None
    average_precision: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None

    class Config:
        from_attributes = True

    @classmethod
    def from_orm_model(cls, m) -> "ExperimentMetricsSchema":
        """ORM의 precision_value -> API의 precision으로 매핑"""
        return cls(
            elapsed_time=m.elapsed_time,
            end_time=m.end_time,
            max_epoch=m.max_epoch,
            current_epoch=m.current_epoch,
            loss=m.loss,
            loss_history=json.loads(m.loss_history) if m.loss_history else None,
            average_precision=m.average_precision,
            accuracy=m.accuracy,
            precision=m.precision_value,
            recall=m.recall,
        )


# ── Experiment 목록/상세 응답 스키마 ──


class ExperimentListResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    reference_model_id: int
    dataset_id: int
    status: str
    registration_status: str = "NOT_REQUESTED"
    registered_model_id: int | None = None
    elapsed_time: int | None = None
    end_time: datetime | None = None
    reference_model: dict | None = None
    dataset: dict | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExperimentDetailResponse(ExperimentListResponse):
    mlflow_run_id: str | None = None
    train_msg: str | None = None
    model_register_msg: str | None = None
    max_epoch: int = 0
    hyperparameters: list = []
    current_epoch: int = 0
    loss: float | None = None
    loss_history: list[dict] | None = None
    average_precision: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None

    class Config:
        from_attributes = True
