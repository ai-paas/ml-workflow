from datetime import datetime

from db.models.base import BaseModel, TimestampMixin
from db.models.dataset import Dataset
from db.models.model import Model
from sqlalchemy import TIMESTAMP, BigInteger, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class ImageRegistry(BaseModel, TimestampMixin):
    __tablename__ = "image_registry"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    train_image_name: Mapped[str] = mapped_column(String(300), nullable=False)
    train_description: Mapped[str] = mapped_column(String(500), nullable=True)
    train_harbor_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    train_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    train_size: Mapped[int] = mapped_column(Integer, nullable=False)
    train_sha256_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    train_base_image: Mapped[str] = mapped_column(String(300), nullable=False)
    var_image_name: Mapped[str] = mapped_column(String(300), nullable=False)
    var_description: Mapped[str] = mapped_column(String(500), nullable=True)
    var_harbor_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    var_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    var_size: Mapped[int] = mapped_column(Integer, nullable=False)
    var_sha256_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    var_base_image: Mapped[str] = mapped_column(String(300), nullable=False)


class ExperimentModel(BaseModel, TimestampMixin):
    __tablename__ = "experiment"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    image_registry_id: Mapped[int] = mapped_column(ForeignKey("image_registry.id"))
    run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    start_time: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    end_time: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    status: Mapped[str] = mapped_column(String(2), nullable=False)

    model: Mapped["Model"] = relationship("Model")
    image_registry: Mapped["ImageRegistry"] = relationship("ImageRegistry")
    dataset: Mapped["Dataset"] = relationship("Dataset")

    # TODO: backpopulate를 설정하지않아도 cascade 삭제되는지 확인 필요.
    hyper_param: Mapped[list["Hyperparamter"]] = relationship("Hyperparamter", cascade="all, delete-orphan")
    experiment_log: Mapped[list["ExperimentLog"]] = relationship("ExperimentLog", cascade="all, delete-orphan")
    metric: Mapped[list["Metric"]] = relationship("Metric", cascade="all, delete-orphan")
    resource_usage: Mapped[list["ResourceUsage"]] = relationship("ResourceUsage", cascade="all, delete-orphan")


class HyperparameterType(BaseModel, TimestampMixin):
    __tablename__ = "hyperparameter_type"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    param_name: Mapped[int] = mapped_column(String(100), nullable=False)
    param_type: Mapped[str] = mapped_column(String(100), nullable=False)


class Hyperparamter(BaseModel, TimestampMixin):
    __tablename__ = "hyperparameter"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    param_type_id: Mapped[int] = mapped_column(ForeignKey("hyperparameter_type.id"))
    param_value: Mapped[str] = mapped_column(String(1000), nullable=False)

    param_type: Mapped["HyperparameterType"] = relationship("HyperparameterType")


class ExperimentLog(BaseModel, TimestampMixin):
    __tablename__ = "experiment_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    content: Mapped[str] = mapped_column(Text, nullable=False)


class Metric(BaseModel, TimestampMixin):
    __tablename__ = "metric"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[str] = mapped_column(Float, nullable=False)


class ResourceUsage(BaseModel, TimestampMixin):
    __tablename__ = "resource_usage"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    cpu_usage: Mapped[str] = mapped_column(Float, nullable=False)
    memory_usage: Mapped[str] = mapped_column(Float, nullable=False)
    gpu_usage: Mapped[str] = mapped_column(Float, nullable=False)
    gpu_memory_usage: Mapped[str] = mapped_column(Float, nullable=False)
