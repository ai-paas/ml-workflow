from typing import Optional

from pydantic import BaseModel


class PromptBaseSchema(BaseModel):
    name: str
    description: str | None = None
    content: str


class PromptVariableBaseSchema(BaseModel):
    name: str
    prompt_id: int


class PromptVariableReadSchema(BaseModel):
    id: int
    name: str
    prompt_id: int

    class Config:
        from_attributes = True


class PromptCreateSchema(BaseModel):
    prompt: PromptBaseSchema
    prompt_variable: Optional[list[str]] = None


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
    prompt_variable: Optional[list[str]] = None
