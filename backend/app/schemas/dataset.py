from __future__ import annotations

from pydantic import BaseModel


class DatasetBaseSchema(BaseModel):
    name: str
    description: str
    dataset_format_id: int


class DatasetReadSchema(BaseModel):
    id: int
    name: str
    description: str
    dataset_format: DatasetFormatReadSchema
    dataset_registry: DatasetRegistryReadSchema

    class Config:
        from_attributes = True


class DatasetFormatReadSchema(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class DatasetRegistryBaseSchema(BaseModel):
    run_id: str
    version: int | None = None
    artifact_path: str
    dataset_uri: str
    dataset_id: int


class DatasetRegistryReadSchema(BaseModel):
    id: int
    run_id: str
    version: int | None = None
    artifact_path: str
    dataset_uri: str
    dataset_id: int

    class Config:
        from_attributes = True
