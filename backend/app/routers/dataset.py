import logging
import traceback
from typing import Annotated

from config.db.connect import SessionDepends
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema, DatasetRegistryBaseSchema, DatasetValidationResponse
from schemas.user import UserSchema
from services.dataset import DatasetRegistryService, DatasetService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/datasets", tags=["Datasets"])

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/validate", response_model=DatasetValidationResponse)
def validate_dataset_file(
    *,
    file: UploadFile = File(...),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    데이터셋 파일 유효성 검증

    파일 형식, 구조 등을 검증합니다.
    """
    validation_result = DatasetService.validate_dataset_file(file)
    if not validation_result.get("is_valid"):
        logger.warning(f"데이터셋 검증 실패: {validation_result.get('message')}")
    return DatasetValidationResponse(**validation_result)


# TODO: 책임 분리 필요.
@router.post("", response_model=DatasetReadSchema)
def create_dataset(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    file: UploadFile = File(...),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Dataset Registry에 데이터셋을 등록하는 API

    파일 검증은 /datasets/validate API를 먼저 호출하여 수행하세요.
    """
    try:
        # 데이터셋 정보 저장
        dataset_data = DatasetBaseSchema(
            name=name,
            version=1,
            subversion=1,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )

        # DatasetService를 통해 DB에 저장
        db_dataset = DatasetService.create(db, obj_in=dataset_data, file=file)

        return db_dataset

    except ValueError as e:
        # 검증 오류 발생 시
        logger.error(f"데이터셋 생성 오류: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 기타 오류 발생 시
        traceback.print_exc()
        logger.error(f"데이터셋 등록 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터셋 등록 중 오류가 발생했습니다: {str(e)}")


@router.get("/{dataset_id}", response_model=DatasetReadSchema)
def read_dataset(dataset_id: int, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)):
    db_model = DatasetService().get(db, dataset_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model


@router.get("", response_model=list[DatasetReadSchema])
def read_datasets(
    skip: int = 0, limit: int = 10, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)
):
    datasets = DatasetService().get_multi(db)
    return datasets
