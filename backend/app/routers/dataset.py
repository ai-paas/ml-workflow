import io
import os
import tempfile
from datetime import datetime
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from config.settings import get_settings
from datasets import load_dataset
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.security import APIKeyHeader
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema
from schemas.user import UserSchema
from services.dataset import DatasetService
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile as StarletteUploadFile
from utils.authentication import get_current_user

router = APIRouter(prefix="/datasets", tags=["Datasets"])

settings = get_settings()
# API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


# TODO: 책임 분리 필요.
@router.post("", response_model=DatasetReadSchema)
def create_dataset(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    dataset_format_id: Annotated[int, Form()],
    train_file: UploadFile | None = None,
    test_file: UploadFile | None = None,
    current_user: UserSchema = Depends(get_current_user)
):
    """
    Dataset Registry에 Model을 등록하는 API

    * params
        - model_format
            1. csv
    """

    # TODO: mocking data. 이후 외부에서 파일 받아와서 mlflow 등록가능하도록 호환 작업 필요.
    dataset = load_dataset("anindya64/hardhat")

    csv_train_buffer = io.BytesIO()
    csv_test_buffer = io.BytesIO()
    dataset["train"].to_csv(csv_train_buffer)
    dataset["test"].to_csv(csv_test_buffer)
    csv_train_buffer.seek(0)
    csv_test_buffer.seek(0)
    train_file = UploadFile(file=csv_train_buffer, filename="train.csv")
    test_file = UploadFile(file=csv_test_buffer, filename="test.csv")

    # with tempfile.TemporaryDirectory() as temp_dir:
    #     # 1. 임시 디렉토리 생성 및 데이터프레임을 CSV로 저장
    #     temp_train_path = os.path.join(temp_dir, "train.csv")
    #     temp_test_path = os.path.join(temp_dir, "test.csv")
    #     dataset['train'].to_csv(temp_train_path)
    #     dataset['test'].to_csv(temp_test_path)

    #     # 2. 저장한 CSV 파일을 UploadFile 객체로 로드
    #     with open(temp_train_path, "rb") as f:
    #         file_data = io.BytesIO(f.read())
    #         train_file = StarletteUploadFile(file=file_data, filename="train.csv")
    #     with open(temp_test_path, "rb") as f:
    #         file_data = io.BytesIO(f.read())
    #         test_file = StarletteUploadFile(file=file_data, filename="test.csv")

    dataset_schema = DatasetBaseSchema(
        name=name,
        description=description,
        dataset_format_id=dataset_format_id,
    )

    try:
        result = DatasetService().create(db, dataset_schema=dataset_schema, file=[train_file, test_file])
        return result
    except Exception as e:
        db.rollback()
        raise e


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
