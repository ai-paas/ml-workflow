from datetime import datetime

from db.models.base import BaseModel, TimestampMixin
from db.models.dataset import Dataset
from db.models.model import Model
from sqlalchemy import TIMESTAMP, BigInteger, Boolean, Double, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class ExperimentModel(BaseModel, TimestampMixin):
    __tablename__ = "experiment"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    reference_model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    kubeflow_run_id: Mapped[str] = mapped_column(String(500), nullable=True)
    mlflow_run_id: Mapped[str] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    registration_kubeflow_run_id: Mapped[str | None] = mapped_column(String(500), nullable=True)
    registration_status: Mapped[str] = mapped_column(String(50), default="NOT_REQUESTED")
    registered_model_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    model_register_msg: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    train_msg: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    reference_model: Mapped["Model"] = relationship("Model")
    dataset: Mapped["Dataset"] = relationship("Dataset")
    hyperparameters: Mapped[list["Hyperparameter"]] = relationship(
        "Hyperparameter", back_populates="experiment", cascade="all, delete-orphan"
    )
    metrics: Mapped["ExperimentMetricsModel | None"] = relationship(
        "ExperimentMetricsModel", uselist=False, back_populates="experiment"
    )


class ExperimentMetricsModel(BaseModel):
    __tablename__ = "experiment_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("experiment.id"), unique=True, nullable=False)
    elapsed_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    max_epoch: Mapped[int] = mapped_column(Integer, default=0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    loss: Mapped[float | None] = mapped_column(Double, nullable=True)
    loss_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    average_precision: Mapped[float | None] = mapped_column(Double, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Double, nullable=True)
    precision_value: Mapped[float | None] = mapped_column(Double, nullable=True)
    recall: Mapped[float | None] = mapped_column(Double, nullable=True)

    experiment: Mapped["ExperimentModel"] = relationship("ExperimentModel", back_populates="metrics")


class HyperparameterType(BaseModel):
    __tablename__ = "hyperparameter_type"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    param_name: Mapped[int] = mapped_column(String(500), nullable=False)
    param_type: Mapped[str] = mapped_column(String(100), nullable=False)
    default_value: Mapped[str] = mapped_column(String(500), nullable=False)


class Hyperparameter(BaseModel):
    __tablename__ = "hyperparameter"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id", ondelete="CASCADE"))
    hyperparameter_type_id: Mapped[int] = mapped_column(ForeignKey("hyperparameter_type.id"))

    experiment: Mapped["ExperimentModel"] = relationship("ExperimentModel", back_populates="hyperparameters")
    hyperparameter_type: Mapped["HyperparameterType"] = relationship("HyperparameterType")
