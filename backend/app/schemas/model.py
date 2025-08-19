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
    parent_model_id: int | None = None
    learning_enable_yn: bool
    version: int
    subversion: int


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


class ModelTypeCreateUpdateSchema(BaseModel):
    name: str
    description: str


class ModelFormatReadSchema(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class ModelFormatCreateUpdateSchema(BaseModel):
    name: str
    description: str


class ModelRegistryRequestSchema(TimeStampCreateUpdateSchema):
    artifact_path: str
    uri: str
    run_id: str


class ModelRegistryBaseSchema(TimeStampCreateUpdateSchema):
    artifact_path: str
    uri: str
    run_id: str
    reference_model_id: int


class ModelRegistryReadSchema(TimeStampCreateUpdateSchema):
    id: int
    artifact_path: str
    uri: str
    run_id: str
    reference_model_id: int

    class Config:
        from_attributes = True


class ModelBriefReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str
    provider_info: ModelProviderReadSchema
    type_info: ModelTypeReadSchema
    format_info: ModelFormatReadSchema
    parent_model_id: int | None = None
    registry: ModelRegistryReadSchema

    class Config:
        from_attributes = True


class ModelReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str
    provider_info: ModelProviderReadSchema
    type_info: ModelTypeReadSchema
    format_info: ModelFormatReadSchema
    parent_model_id: int | None = None
    registry: ModelRegistryReadSchema

    parent_model: Optional[ModelReadParentSchema]
    child_models: Optional[list[ModelReadChildSchema]]

    class Config:
        from_attributes = True


class ModelReadParentSchema(BaseModel):
    id: int
    name: str
    description: str

    parent_model: Optional[ModelReadParentSchema]

    class Config:
        from_attributes = True


class ModelReadChildSchema(BaseModel):
    id: int
    name: str
    description: str

    child_models: Optional[list[ModelReadChildSchema]]

    class Config:
        from_attributes = True
