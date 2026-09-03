from typing import Annotated

from config.db.connect import SessionDepends
from core.optimization.optimization_client import OptimizationClient, get_optimization_client
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from schemas.model_improvement import (
    CreateImprovementRequest,
    CreateImprovementResponse,
    ImprovementStatusResponse,
    TaskTypeResponse,
)
from schemas.user import UserSchema
from services.model_improvement import ModelImprovementService, poll_improvement_task
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/model-improvements", tags=["Model Improvement"])

_VALID_CATEGORIES = frozenset({"optimization", "lightweight"})


def get_improvement_service(
    client: Annotated[OptimizationClient, Depends(get_optimization_client)],
) -> ModelImprovementService:
    return ModelImprovementService(client)


@router.post("", response_model=CreateImprovementResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_improvement_task(
    body: CreateImprovementRequest,
    background_tasks: BackgroundTasks,
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
    service: ModelImprovementService = Depends(get_improvement_service),
):
    """
    `opt_enable_yn=true` 인 소스 모델에 대해 최적화/경량화 task를 큐에 올린다.
    """
    created = await service.create_task(
        db,
        body,
        created_by_username=current_user.username,
    )
    # 상태 갱신과 결과 모델 등록을 폴링이 맡는다. 조회가 없어도 완료가 반영된다.
    background_tasks.add_task(poll_improvement_task, created.task_id)
    return created


@router.get("/status", response_model=ImprovementStatusResponse)
def get_improvement_status(
    task_id: str = Query(..., description="최적화 요청 응답의 task_id"),
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
    service: ModelImprovementService = Depends(get_improvement_service),
):
    """task_id로 진행·성공·실패 및 result_model_id를 조회한다. DB에 기록된 상태만 읽는다."""
    return service.get_task_status(db, task_id, current_username=current_user.username)


@router.get("/task-types", response_model=list[TaskTypeResponse])
async def get_task_types(
    category: str | None = Query(None, description="optimization 또는 lightweight"),
    source_model_id: int | None = Query(
        None,
        description="지정 시 해당 모델 repo_id에 맞는 허용 기법만 반환(예: DETR은 tensorrt/openvino/pruning)",
    ),
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
    service: ModelImprovementService = Depends(get_improvement_service),
):
    """최적화 서버에서 조회한 기법 목록."""
    if category is not None and category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"유효하지 않은 category입니다: {category}. 가능한 값: {', '.join(sorted(_VALID_CATEGORIES))}",
        )
    return await service.get_task_types(db, category, source_model_id=source_model_id)
