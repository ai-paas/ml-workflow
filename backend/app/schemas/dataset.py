from __future__ import annotations

from typing import Optional

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


class DatasetValidationResponse(BaseModel):
    """데이터셋 파일 검증 응답"""

    is_valid: bool
    message: str
    root_dir: Optional[str] = None
    details: Optional[dict] = None
