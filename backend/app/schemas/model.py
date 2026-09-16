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
    RF_DETR_LARGE = "Roboflow/rf-detr-large"
    RF_DETR_MEDIUM = "Roboflow/rf-detr-medium"
    MEDLLAMA3 = "ahmgam/medllama3-v20:latest"
    BGE_M3 = "bge-m3"
    ESM2_T6_8M = "facebook/esm2_t6_8M_UR50D"
    RNAFM = "multimolecule/rnafm"
    MOLFORMER_XL = "ibm-research/MoLFormer-XL-both-10pct"
    ESMC_300M = "biohub/ESMC-300M"
    ESMC_6B = "biohub/ESMC-6B"
    ESMFOLD2 = "biohub/ESMFold2"
    YOLOX_S = "yolox_s"
    YOLOX_M = "yolox_m"
    QWQ_32B = "qwq:32b"
    GPT_OSS_20B = "gpt-oss:20b"
    # 신규 Ollama LLM(text-generation)
    DEEPSEEK_R1_32B = "deepseek-r1:32b"
    GRANITE4_1_30B = "granite4.1:30b"
    LFM2_24B = "lfm2:24b"
    # 신규 Ollama VQA/멀티모달(task=vqa, 당분간 text-generation 동일 추론)
    GEMMA4_27B = "gemma4:27b"
    QWEN3_6_27B = "qwen3.6:27b"
    NEMOTRON3_33B = "nemotron3:33b"
    # 원격 서버 전용(serving_mode=remote_only). 가중치를 내려받지 않고 DB 행만 만든다.
    REMOTE_DEEPSEEK_R1_1_5B = "deepseek-r1:1.5b"
    REMOTE_GEMMA3_1B = "gemma3:1b"
    REMOTE_LLAMA32_1B = "llama3.2:1b"
    REMOTE_QWEN25_0_5B = "qwen2.5:0.5b"
    REMOTE_PHI3_3_8B = "phi3:3.8b"
    REMOTE_QWEN25_CODER_0_5B = "qwen2.5-coder:0.5b"
    REMOTE_TINYLLAMA_1_1B = "tinyllama:1.1b"
    REMOTE_STARCODER2_3B = "starcoder2:3b"
    REMOTE_GRANITE31_MOE_1B = "granite3.1-moe:1b"
    REMOTE_FALCON3_1B = "falcon3:1b"
    REMOTE_LFM25_8B = "lfm2.5:8b"
    REMOTE_NEMOTRON3_NANO_4B = "nemotron-3-nano:4b"
    REMOTE_RNJ1_8B = "rnj-1:8b"
    REMOTE_OLMO3_7B = "olmo-3:7b"
    REMOTE_GRANITE4_3B = "granite4:3b"
    REMOTE_MEDGEMMA15_4B = "medgemma1.5:4b"
    REMOTE_MINICPM_V46_1B = "minicpm-v4.6:1b"
    REMOTE_TRANSLATEGEMMA_4B = "translategemma:4b"
    REMOTE_QWEN3_VL_2B = "qwen3-vl:2b"
    REMOTE_GLM_OCR_Q8_0 = "glm-ocr:q8_0"


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
