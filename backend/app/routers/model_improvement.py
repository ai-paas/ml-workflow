import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from schemas.user import UserSchema
from utils.authentication import get_current_user

router = APIRouter(prefix="/model-improvements", tags=["Model Improvement"])

_TASK_TYPES = [
    {"name": "tensorrt", "category": "optimization", "description": "nvidia.com/gpu"},
    {"name": "openvino", "category": "optimization", "description": "cpu"},
    {"name": "sklearn-onnx", "category": "optimization", "description": "cpu"},
    {"name": "pruning", "category": "lightweight", "description": "cpu"},
    {"name": "npu", "category": "optimization", "description": "furiosa.ai/warboy"},
    {"name": "ptq", "category": "lightweight", "description": "cpu"},
    {"name": "tpu", "category": "optimization", "description": "cpu"},
]

_VALID_TASK_TYPE_NAMES = {t["name"] for t in _TASK_TYPES}
_VALID_CATEGORIES = {"optimization", "lightweight"}

_mock_tasks: dict[str, dict] = {}


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


@router.post("", response_model=CreateImprovementResponse, status_code=202)
async def create_improvement_task(
    body: CreateImprovementRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    if body.task_type not in _VALID_TASK_TYPE_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"유효하지 않은 task_type입니다: {body.task_type}. "
            f"가능한 값: {', '.join(sorted(_VALID_TASK_TYPE_NAMES))}",
        )

    for task in _mock_tasks.values():
        if task["source_model_id"] == body.source_model_id and task["status"] in ("PENDING", "RUNNING"):
            raise HTTPException(status_code=409, detail="OPTIMIZATION_TASK_ALREADY_RUNNING")

    now = datetime.now(timezone.utc).isoformat()
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "status": "PENDING",
        "source_model_id": body.source_model_id,
        "task_type": body.task_type,
        "created_at": now,
        "updated_at": now,
        "message": "최적화 작업이 큐에 등록되었습니다.",
        "result_model_id": None,
        "error": None,
    }
    _mock_tasks[task_id] = task

    return CreateImprovementResponse(
        task_id=task_id,
        status="PENDING",
        source_model_id=body.source_model_id,
        created_at=now,
    )


@router.get("/status", response_model=ImprovementStatusResponse)
async def get_improvement_status(
    task_id: str = Query(..., description="§6.1 응답의 task_id"),
    current_user: UserSchema = Depends(get_current_user),
):
    task = _mock_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}'를 찾을 수 없습니다.")

    now = datetime.now(timezone.utc).isoformat()

    if task["status"] == "PENDING":
        task["status"] = "RUNNING"
        task["updated_at"] = now
        task["message"] = "최적화 작업이 진행 중입니다."
    elif task["status"] == "RUNNING":
        task["status"] = "SUCCESS"
        task["updated_at"] = now
        task["message"] = "최적화가 완료되었습니다."
        task["result_model_id"] = task["source_model_id"] + 1000

    return ImprovementStatusResponse(**{k: v for k, v in task.items() if k != "task_type"})


@router.get("/task-types", response_model=list[TaskTypeResponse])
async def get_task_types(
    category: str | None = Query(None, description="optimization 또는 lightweight"),
    current_user: UserSchema = Depends(get_current_user),
):
    if category is not None and category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"유효하지 않은 category입니다: {category}. 가능한 값: {', '.join(sorted(_VALID_CATEGORIES))}",
        )

    if category is None:
        return _TASK_TYPES
    return [t for t in _TASK_TYPES if t["category"] == category]
