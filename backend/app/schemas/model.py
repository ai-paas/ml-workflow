from __future__ import annotations

from enum import Enum as PyEnum
from typing import Optional

from db.models.model import ModelTaskType
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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
    opt_enable_yn: bool = False
    version: int
    subversion: int
    task: Optional[str] = Field(
        None,
        description=(
            "모델 태스크 타입: 'embedding', 'text-generation', 'object-detection', "
            "'fill-mask', 'protein-classification', 'protein-structure-prediction', 'vqa' 중 하나"
        ),
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


class PredefinedModelKey(str, PyEnum):
    """사전 정의된 모델 키"""

    YOLOS_TINY = "hustvl/yolos-tiny"
    YOLOS_SMALL = "hustvl/yolos-small"
    DETR_RESNET_50 = "facebook/detr-resnet-50"
    DETR_RESNET_101 = "facebook/detr-resnet-101"
    MEDLLAMA3 = "ahmgam/medllama3-v20:latest"
    BGE_M3 = "bge-m3"
    ESM2_T6_8M = "facebook/esm2_t6_8M_UR50D"
    RNAFM = "multimolecule/rnafm"
    MOLFORMER_XL = "ibm-research/MoLFormer-XL-both-10pct"
    ESMC_300M = "biohub/ESMC-300M"
    YOLOX_S = "yolox_s"
    YOLOX_M = "yolox_m"
    QWQ_32B = "qwq:32b"
    QWEN3_32B = "qwen3:32b"
    QWEN3_30B = "qwen3:30b"
    GPT_OSS_20B = "gpt-oss:20b"


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
    pvc: Optional[str] = Field(None, exclude=True)
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
    learning_enable_yn: bool
    opt_enable_yn: bool
    visibility: str = ""
    recommended_hparams: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_visibility(self) -> "ModelBriefReadSchema":
        if self.parent_model_id is not None or self.opt_enable_yn:
            self.visibility = "CUSTOM"
        else:
            self.visibility = "CATALOG"
        return self

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

    learning_enable_yn: bool
    opt_enable_yn: bool
    visibility: str = ""
    recommended_hparams: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_visibility(self) -> "ModelReadSchema":
        if self.parent_model_id is not None or self.opt_enable_yn:
            self.visibility = "CUSTOM"
        else:
            self.visibility = "CATALOG"
        return self

    class Config:
        from_attributes = True


class ModelReadParentSchema(BaseModel):
    id: int
    name: str
    description: str | None = None

    parent_model: Optional[ModelReadParentSchema]

    class Config:
        from_attributes = True


class ModelReadChildSchema(BaseModel):
    id: int
    name: str
    description: str | None = None

    child_models: Optional[list[ModelReadChildSchema]]

    class Config:
        from_attributes = True
