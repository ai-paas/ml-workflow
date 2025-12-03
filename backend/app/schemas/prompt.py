from typing import Optional

from db.models.prompt import PromptVariableType
from pydantic import BaseModel, field_validator


class PromptBaseSchema(BaseModel):
    name: str
    description: str | None = None
    content: str


class PromptVariableBaseSchema(BaseModel):
    name: str  # DB에는 String으로 저장, Python 코드에서는 PromptVariableType Enum 사용
    prompt_id: int


class PromptVariableReadSchema(BaseModel):
    id: int
    name: str
    prompt_id: int

    class Config:
        from_attributes = True


class PromptCreateSchema(BaseModel):
    prompt: PromptBaseSchema
    prompt_variable: Optional[list[PromptVariableType]] = None

    @field_validator("prompt_variable")
    @classmethod
    def validate_prompt_variable(cls, v: Optional[list[PromptVariableType]]) -> Optional[list[PromptVariableType]]:
        if v is None:
            return None
        # context만 허용
        for var_type in v:
            if var_type != PromptVariableType.CONTEXT:
                raise ValueError(f"프롬프트 변수는 'context'만 사용할 수 있습니다. 제공된 값: {var_type}")
        return v


class PromptReadSchema(BaseModel):
    id: int
    name: str
    description: str | None = None
    content: str
    prompt_variable: Optional[list[PromptVariableReadSchema]] = None

    class Config:
        from_attributes = True


class PromptUpdateSchema(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    prompt_variable: Optional[list[PromptVariableType]] = None

    @field_validator("prompt_variable")
    @classmethod
    def validate_prompt_variable(cls, v: Optional[list[PromptVariableType]]) -> Optional[list[PromptVariableType]]:
        if v is None:
            return None
        # context만 허용
        for var_type in v:
            if var_type != PromptVariableType.CONTEXT:
                raise ValueError(f"프롬프트 변수는 'context'만 사용할 수 있습니다. 제공된 값: {var_type}")
        return v


class PromptVariableTypeListSchema(BaseModel):
    """프롬프트 변수 타입 목록 응답"""

    available_types: list[str]
