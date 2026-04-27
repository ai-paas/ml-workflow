import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from schemas.model_improvement import (
    CreateImprovementRequest,
    CreateImprovementResponse,
    ImprovementStatusResponse,
    TaskTypeResponse,
)
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


@router.post("", response_model=CreateImprovementResponse, status_code=202)
async def create_improvement_task(
    body: CreateImprovementRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    최적화/경량화 task 생성(비동기)

    `opt_enable_yn=true` 인 소스 모델에 대해 task를 큐에 올리고 `task_id`로 진행을 조회합니다(§6.1).

    ## Request (`CreateImprovementRequest`)
    - **source_model_id** (int): 대상 모델 ID
    - **task_type** (str): `GET /model-improvements/task-types`가 돌려주는 `name`과 동일(예: `tensorrt`, `pruning`)

    ## Response 202 (`CreateImprovementResponse`)
    - **task_id** (str, UUID): 추적 ID — `GET /model-improvements/status?task_id=` 에 전달
    - **status** (str): 초기값 **`PENDING`**
    - **source_model_id** (int)
    - **created_at** (str): ISO-8601

    ## Task **status** 수명(§6.2, 조회 API 기준)
    - `PENDING` → `RUNNING` → `SUCCEEDED` | `FAILED` (`FAILED` 시 응답 **error** 필드 가능)
    """
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
    """
    최적화/경량화 task 상태 조회(§6.2)

    ## Query
    - **task_id** (str, UUID): `POST /model-improvements` 응답의 `task_id`

    ## Response 200 (`ImprovementStatusResponse`)
    - **status** (str): `PENDING` | `RUNNING` | `SUCCEEDED` | `FAILED`
    - **result_model_id** (int, optional): `SUCCEEDED`일 때만(파생 모델 ID)
    - **message** (str, optional): 단계 설명
    - **error** (str, optional): `status` = `FAILED`일 때 사유(문서 §6.2)
    - **created_at** / **updated_at**: ISO-8601
    """
    task = _mock_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}'를 찾을 수 없습니다.")

    now = datetime.now(timezone.utc).isoformat()

    if task["status"] == "PENDING":
        task["status"] = "RUNNING"
        task["updated_at"] = now
        task["message"] = "최적화 작업이 진행 중입니다."
    elif task["status"] == "RUNNING":
        task["status"] = "SUCCEEDED"
        task["updated_at"] = now
        task["message"] = "최적화가 완료되었습니다."
        task["result_model_id"] = task["source_model_id"] + 1000

    return ImprovementStatusResponse(**{k: v for k, v in task.items() if k != "task_type"})


@router.get("/task-types", response_model=list[TaskTypeResponse])
async def get_task_types(
    category: str | None = Query(None, description="optimization 또는 lightweight"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    사용 가능한 `task_type` 목록(§6.3)

    ## Query
    - **category** (str, optional): `optimization` | `lightweight` — 생략 시 전체

    ## Response 200
    - 배열. 원소: **name**(§6.1 `task_type`에 그대로 사용), **category**, **description**(표시/가속기 힌트)
    """
    if category is not None and category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"유효하지 않은 category입니다: {category}. 가능한 값: {', '.join(sorted(_VALID_CATEGORIES))}",
        )

    if category is None:
        return _TASK_TYPES
    return [t for t in _TASK_TYPES if t["category"] == category]
