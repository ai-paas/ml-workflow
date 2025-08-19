from typing import Optional

from db.models.base import BaseModel, TimestampCreateMixin, TimestampMixin, TimestampUpdateMixin
from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Model(BaseModel, TimestampMixin):
    __tablename__ = "model"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("model_provider.id"))
    type_id: Mapped[int] = mapped_column(ForeignKey("model_type.id"))
    format_id: Mapped[int] = mapped_column(ForeignKey("model_format.id"))
    parent_model_id: Mapped[int] = mapped_column(ForeignKey("model.id"), nullable=True)
    learning_enable_yn: Mapped[bool] = mapped_column(Boolean, nullable=False)
    train_image_registry_id: Mapped[int] = mapped_column(ForeignKey("train_image_registry.id"), nullable=True)
    inference_image_registry_id: Mapped[int] = mapped_column(ForeignKey("inference_image_registry.id"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    subversion: Mapped[int] = mapped_column(Integer, nullable=False)

    provider_info: Mapped["ModelProvider"] = relationship("ModelProvider")
    type_info: Mapped["ModelType"] = relationship("ModelType")
    format_info: Mapped["ModelFormat"] = relationship("ModelFormat")
    registry: Mapped["ModelRegistry"] = relationship("ModelRegistry", back_populates="reference_model")

    # Parent-Child relationships
    parent_model: Mapped[Optional["Model"]] = relationship("Model", remote_side=[id], back_populates="child_models")
    child_models: Mapped[list["Model"]] = relationship("Model", back_populates="parent_model")


class TrainImageRegistry(BaseModel):
    __tablename__ = "train_image_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_uri: Mapped[str] = mapped_column(String(4000), nullable=False)


class InferenceImageRegistry(BaseModel):
    __tablename__ = "inference_image_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_uri: Mapped[str] = mapped_column(String(4000), nullable=False)


class ModelRegistry(BaseModel, TimestampCreateMixin, TimestampUpdateMixin):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_path: Mapped[str] = mapped_column(String(4000), nullable=False)
    run_id: Mapped[str] = mapped_column(String(100), nullable=True)
    uri: Mapped[str] = mapped_column(String(4000), nullable=False)
    reference_model_id: Mapped[int] = mapped_column(ForeignKey("model.id", ondelete="CASCADE"))

    reference_model: Mapped["Model"] = relationship("Model", back_populates="registry", passive_deletes=True)


class ModelFormat(BaseModel):
    __tablename__ = "model_format"

    id: Mapped[str] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)


class ModelProvider(BaseModel):
    __tablename__ = "model_provider"

    id: Mapped[str] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)


class ModelType(BaseModel):
    __tablename__ = "model_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
