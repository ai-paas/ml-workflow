from __future__ import annotations

from pydantic import BaseModel
from schemas.base import TimeStampSchemaMixin


class DatasetBaseSchema(TimeStampSchemaMixin):
    name: str
    version: int
    subversion: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float


class DatasetReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    version: int
    subversion: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    dataset_registry: DatasetRegistryReadSchema

    class Config:
        from_attributes = True


class DatasetRegistryBaseSchema(TimeStampSchemaMixin):
    artifact_path: str
    uri: str
    dataset_id: int


class DatasetRegistryReadSchema(TimeStampSchemaMixin):
    id: int
    artifact_path: str
    uri: str
    dataset_id: int

    class Config:
        from_attributes = True
