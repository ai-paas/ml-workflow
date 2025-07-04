from datetime import datetime

from db.models.base import BaseModel, TimestampMixin
from db.models.dataset import Dataset
from db.models.model import Model
from sqlalchemy import TIMESTAMP, BigInteger, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class ExperimentModel(BaseModel, TimestampMixin):
    __tablename__ = "experiment"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    reference_model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    kubeflow_run_id: Mapped[str] = mapped_column(String(500), nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    learning_yn: Mapped[str] = mapped_column(Boolean, nullable=False)

    reference_model: Mapped["Model"] = relationship("Model")
    dataset: Mapped["Dataset"] = relationship("Dataset")


class HyperparameterType(BaseModel):
    __tablename__ = "hyperparameter_type"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    param_name: Mapped[int] = mapped_column(String(500), nullable=False)
    param_type: Mapped[str] = mapped_column(String(100), nullable=False)
    default_value: Mapped[str] = mapped_column(String(500), nullable=False)


class Hyperparamter(BaseModel):
    __tablename__ = "hyperparameter"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    reference_model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    hyperparameter_type_id: Mapped[int] = mapped_column(ForeignKey("hyperparameter_type.id"))

    reference_model: Mapped["Model"] = relationship("Model")
    hyperparameter_type: Mapped["HyperparameterType"] = relationship("HyperparameterType")
