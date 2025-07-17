from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from schemas.base import TimeStampCreateUpdateSchema, TimeStampSchemaMixin


class ModelBaseSchema(TimeStampSchemaMixin):
    name: str
    description: str
    provider_id: int
    type_id: int
    format_id: int
    learning_enable_yn: bool
    version: int
    subversion: int


class ModelReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str
    provider_info: ModelProviderReadSchema
    type_info: ModelTypeReadSchema
    format_info: ModelFormatReadSchema
    registry: ModelRegistryReadSchema

    class Config:
        from_attributes = True


class ModelProviderCreateUpdateSchema(BaseModel):
    name: str
    description: str
    link: str


class ModelProviderReadSchema(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class ModelTypeReadSchema(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class ModelFormatReadSchema(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class ModelRegistryRequestSchema(TimeStampCreateUpdateSchema):
    artifact_path: str
    uri: str


class ModelRegistryBaseSchema(TimeStampCreateUpdateSchema):
    artifact_path: str
    uri: str
    reference_model_id: int


class ModelRegistryReadSchema(TimeStampCreateUpdateSchema):
    id: int
    artifact_path: str
    uri: str
    reference_model_id: int

    class Config:
        from_attributes = True
