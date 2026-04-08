from pydantic import BaseModel


class CreateImprovementRequest(BaseModel):
    source_model_id: int
    task_type: str


class CreateImprovementResponse(BaseModel):
    task_id: str
    status: str
    source_model_id: int
    created_at: str


class ImprovementStatusResponse(BaseModel):
    task_id: str
    status: str
    source_model_id: int
    created_at: str
    updated_at: str
    message: str | None = None
    result_model_id: int | None = None
    error: str | None = None


class TaskTypeResponse(BaseModel):
    name: str
    category: str
    description: str | None = None
