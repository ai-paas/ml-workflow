from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelBaseSchema(BaseModel):
    name: str
    description: str
    model_provider_id: int
    model_type_id: int
    model_format_id: int


class ModelReadSchema(BaseModel):
    id: int
    name: str
    description: str
    model_provider: ModelProviderReadSchema
    model_type: ModelTypeReadSchema
    model_format: ModelFormatReadSchema
    model_registry: ModelRegistryReadSchema

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
    link: str

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


class ModelRegistryRequestSchema(BaseModel):
    run_id: str
    versions: int
    artifact_path: str
    model_uri: str


class ModelRegistryBaseSchema(BaseModel):
    run_id: str
    version: int
    artifact_path: str
    model_uri: str
    model_id: int


class ModelRegistryReadSchema(BaseModel):
    id: int
    run_id: str
    version: int
    artifact_path: str
    model_uri: str
    model_id: int

    class Config:
        from_attributes = True
