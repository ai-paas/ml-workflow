from db.models.base import BaseModel, TimestampCreateMixin, TimestampMixin, TimestampUpdateMixin
from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Model(BaseModel, TimestampMixin):
    __tablename__ = "model"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    model_provider_id: Mapped[int] = mapped_column(ForeignKey("model_provider.id"))
    model_type_id: Mapped[int] = mapped_column(ForeignKey("model_type.id"))
    model_format_id: Mapped[int] = mapped_column(ForeignKey("model_format.id"))

    model_provider: Mapped["ModelProvider"] = relationship("ModelProvider")
    model_type: Mapped["ModelType"] = relationship("ModelType")
    model_format: Mapped["ModelFormat"] = relationship("ModelFormat")
    model_registry: Mapped["ModelRegistry"] = relationship("ModelRegistry", back_populates="model")


class ModelRegistry(BaseModel, TimestampCreateMixin, TimestampUpdateMixin):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    model_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id", ondelete="CASCADE"))

    model: Mapped["Model"] = relationship("Model", back_populates="model_registry", passive_deletes=True)


class ModelFormat(BaseModel):
    __tablename__ = "model_format"

    id: Mapped[str] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)


class ModelProvider(BaseModel):
    __tablename__ = "model_provider"

    id: Mapped[str] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    link: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)


class ModelType(BaseModel):
    __tablename__ = "model_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
