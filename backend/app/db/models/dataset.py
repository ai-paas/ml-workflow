from typing import Optional

from db.models.base import BaseModel, TimestampCreateMixin, TimestampMixin, TimestampUpdateMixin
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Dataset(BaseModel, TimestampMixin):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    subversion: Mapped[int] = mapped_column(Integer, nullable=False)
    # 데이터셋 분류(학습 태스크 분류명): object-detection / protein-classification
    kind: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    dataset_registry: Mapped["DatasetRegistry"] = relationship("DatasetRegistry", back_populates="dataset")


class DatasetRegistry(BaseModel, TimestampMixin):
    __tablename__ = "dataset_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_path: Mapped[str] = mapped_column(String(4000), nullable=False)
    uri: Mapped[str] = mapped_column(String(4000), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"))

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="dataset_registry", passive_deletes=True)


# class DatasetFormat(BaseModel):
#     __tablename__ = "dataset_format"

#     id: Mapped[str] = mapped_column(Integer, primary_key=True, autoincrement=True)
#     name: Mapped[str] = mapped_column(String(50), nullable=False)
#     description: Mapped[str] = mapped_column(String(500), nullable=True)
