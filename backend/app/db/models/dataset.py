from db.models.base import BaseModel, TimestampCreateMixin, TimestampMixin, TimestampUpdateMixin
from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Dataset(BaseModel, TimestampMixin):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    dataset_format_id: Mapped[int] = mapped_column(ForeignKey("dataset_format.id"))

    dataset_format: Mapped["DatasetFormat"] = relationship("DatasetFormat")
    dataset_registry: Mapped["DatasetRegistry"] = relationship("DatasetRegistry", back_populates="dataset")


class DatasetRegistry(BaseModel, TimestampMixin):
    __tablename__ = "dataset_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    dataset_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"))

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="dataset_registry", passive_deletes=True)


class DatasetFormat(BaseModel):
    __tablename__ = "dataset_format"

    id: Mapped[str] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
