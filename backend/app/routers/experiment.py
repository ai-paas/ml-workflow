from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, HTTPException, status
from schemas.experiment import (
    ExperimentBaseSchema,
    ExperimentCreateRequest,
    ExperimentReadSchema,
    ExperimentUpdateRequest,
)
from schemas.user import UserSchema
from services.experiment import ExperimentService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/experiments", tags=["Experiment"])


@router.patch("/{experiment_id}", response_model=ExperimentReadSchema)
async def update_experiment(
    db: Session = SessionDepends,
    *,
    experiment_id: int,
    experiment_update_request: ExperimentUpdateRequest,
    current_user: UserSchema = Depends(get_current_user)
):
    try:
        return ExperimentService().update(db, experiment_id=experiment_id, obj_in=experiment_update_request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentReadSchema)
async def get_experiment(
    db: Session = SessionDepends, *, experiment_id: int, current_user: UserSchema = Depends(get_current_user)
):
    try:
        return ExperimentService().get(db, experiment_id=experiment_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
