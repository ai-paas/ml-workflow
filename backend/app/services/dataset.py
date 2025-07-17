import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import mlflow.data
import pandas as pd
from core.kubeflow.s3.s3_manager import S3Manager
from fastapi import UploadFile
from repos.dataset import dataset_registry_repository, dataset_repository
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema, DatasetRegistryBaseSchema, DatasetRegistryReadSchema
from sqlalchemy.orm import Session
from utils.dataset_registry import DatasetRegistry


class DatasetService:
    @staticmethod
    def get(db: Session, pk: int) -> DatasetReadSchema:
        return dataset_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[DatasetReadSchema]:
        return dataset_repository.get_multi(db, skip=skip, limit=limit)

    @staticmethod
    def get_all(db: Session) -> list[DatasetReadSchema]:
        return dataset_repository.get_all(db)

    @staticmethod
    def update(db: Session, db_obj, obj_in):
        return dataset_repository.update(db, db_obj=db_obj, obj_in=obj_in)

    @staticmethod
    def validate(self, data_format_id: int, dataset_uri: str) -> str:
        # TODO : validate pipeline 추가 필요
        # if data_format_id == 1:
        #     pipeline = DatasetLoader.load_pyfunc(dataset_uri)
        #     messages = [
        #         {"role": "user", "content": "Who are you?"},
        #     ]
        #     result = pipeline(messages, max_length=1024)
        #     retsult = ""
        # else:
        #     result = ""
        return ""

    @staticmethod
    def create(db: Session, *, obj_in: DatasetBaseSchema, file: UploadFile):
        # UploadFile의 file 속성은 SpooledTemporaryFile 객체이므로
        # 직접 임시 파일 경로를 사용할 수 있습니다

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file.filename
            temp_file_path.write_bytes(file.file.read())
            run_id, dataset_version, artifact_uri, dataset_uri = DatasetRegistry().log_dataset(temp_dir, obj_in.name)

        # 데이터셋 객체 생성
        dataset_obj = dataset_repository.create(db, obj_in=obj_in)
        dataset_id = dataset_obj.id

        # 데이터셋 레지스트리 정보 생성
        dataset_registry_repository.create(
            db,
            obj_in=DatasetRegistryBaseSchema(
                artifact_path=artifact_uri,
                uri=dataset_uri,
                dataset_id=dataset_id,
            ),
        )

        db.commit()
        return dataset_repository.get(db, dataset_id)


class DatasetRegistryService:
    @staticmethod
    def create(db: Session, *, obj_in: DatasetRegistryBaseSchema):
        return dataset_registry_repository.create(db, obj_in=obj_in)

    @staticmethod
    def get(db: Session, pk: int) -> DatasetRegistryReadSchema:
        return dataset_registry_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[DatasetRegistryReadSchema]:
        return dataset_registry_repository.get_multi(db, skip=skip, limit=limit)
