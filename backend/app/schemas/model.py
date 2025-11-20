from __future__ import annotations

from typing import Optional

from db.models.model import ModelTaskType
from pydantic import BaseModel, ConfigDict, Field, field_validator
from schemas.base import TimeStampCreateUpdateSchema, TimeStampSchemaMixin


class ModelBaseSchema(TimeStampSchemaMixin):
    name: str
    description: str | None = None
    repo_id: str | None = None
    provider_id: int
    type_id: int
    format_id: int
    parent_model_id: int | None = None
    learning_enable_yn: bool
    version: int
    subversion: int
    task: Optional[str] = Field(
        None,
        description="모델 태스크 타입: 'embedding', 'text-generation', 'object-detection' 중 하나",
    )
    parameter: str | None = None
    sample_code: str | None = None

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        valid_values = {e.value for e in ModelTaskType}
        if v not in valid_values:
            raise ValueError(f"task는 다음 값 중 하나여야 합니다: {', '.join(valid_values)}")
        return v


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
    run_id: Optional[str] = None
    pvc: Optional[str] = None


class ModelRegistryBaseSchema(TimeStampCreateUpdateSchema):
    artifact_path: str
    uri: str
    run_id: Optional[str] = None
    pvc: Optional[str] = None
    reference_model_id: int


class ModelRegistryReadSchema(TimeStampCreateUpdateSchema):
    id: int
    artifact_path: str
    uri: str
    run_id: Optional[str] = None
    pvc: Optional[str] = None
    reference_model_id: int

    class Config:
        from_attributes = True


class ModelBriefReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str | None = None
    repo_id: str | None = None
    provider_info: ModelProviderReadSchema
    type_info: ModelTypeReadSchema
    format_info: ModelFormatReadSchema
    parent_model_id: int | None = None
    registry: ModelRegistryReadSchema
    task: Optional[str] = None
    parameter: str | None = None
    sample_code: str | None = None

    class Config:
        from_attributes = True


class ModelReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str | None = None
    repo_id: str | None = None
    provider_info: ModelProviderReadSchema
    type_info: ModelTypeReadSchema
    format_info: ModelFormatReadSchema
    parent_model_id: int | None = None
    registry: ModelRegistryReadSchema
    task: Optional[str] = None
    parameter: str | None = None
    sample_code: str | None = None

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
