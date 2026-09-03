from __future__ import annotations

from typing import Optional

from config.db.enums import DatasetKindEnum
from pydantic import BaseModel
from schemas.base import TimeStampSchemaMixin


class DatasetBaseSchema(TimeStampSchemaMixin):
    name: str
    description: Optional[str] = None
    version: int
    subversion: int
    kind: Optional[DatasetKindEnum] = None


class DatasetReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: Optional[str] = None
    kind: Optional[DatasetKindEnum] = None
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


class DatasetUpdateSchema(BaseModel):
    """데이터셋 업데이트 스키마

    kind 는 PUT 으로 변경하지 않는다(받더라도 무시).
    """

    name: Optional[str] = None
    description: Optional[str] = None
    kind: Optional[DatasetKindEnum] = None


class DatasetValidationResponse(BaseModel):
    """데이터셋 파일 검증 응답"""

    is_valid: bool
    message: str
    details: Optional[dict] = None


class DatasetKindReadSchema(BaseModel):
    """GET /datasets/kinds 응답 원소 — 데이터셋 분류 카탈로그"""

    name: str
    description: str
    accepted_formats: list[str]
    supported_models: list[str]
