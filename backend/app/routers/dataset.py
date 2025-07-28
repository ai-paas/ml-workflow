import io
import logging
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Annotated

import yaml
from config.db.connect import SessionDepends
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema, DatasetRegistryBaseSchema
from schemas.user import UserSchema
from services.dataset import DatasetRegistryService, DatasetService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/datasets", tags=["Datasets"])

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    Dataset Registry에 데이터셋을 검증 및 등록하는 API
    """
    # TODO: COCO 2017 dataset 형식으로 검증 로직 구현 필요
    # - annotations/*.json 파일 검증
    # - train2017/ 디렉토리의 이미지 파일 검증
    # - val2017/ 디렉토리의 이미지 파일 검증
    # - test2017/ 디렉토리의 이미지 파일 검증

    try:
        # 업로드된 파일 내용 읽기
        file_content = file.file.read()

        # ZIP 파일 형식 검증 및 압축 해제
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with zipfile.ZipFile(io.BytesIO(file_content)) as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            raise ValueError("파일이 유효한 ZIP 형식이 아닙니다.")

        # 업로드된 파일명 추출
        file_name = Path(file.filename).name
        dataset_name = Path(file_name).stem

        # 압축 해제 후 ZIP 파일명과 동일한 루트 디렉토리 찾기
        root_dir = temp_dir / dataset_name
        if not root_dir.is_dir():
            root_dir = temp_dir  # 동일 이름의 디렉토리가 없으면 temp_dir 자체를 루트로 사용

        logger.info(f"데이터셋 루트 디렉토리: {root_dir}")

        # TODO : 데이터셋 구조 검증 로직 추가 필요

        # 파일 포인터 초기화 (업로드를 위해)
        file.file.seek(0)

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
        logger.error(f"데이터셋 검증 오류: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 기타 오류 발생 시
        traceback.print_exc()
        logger.error(f"데이터셋 등록 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터셋 등록 중 오류가 발생했습니다: {str(e)}")
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)


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
