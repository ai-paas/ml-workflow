import os
import tempfile
from enum import Enum
from typing import Any

import pandas as pd
from core.kubeflow.s3.s3_manager import S3Manager
from fastapi import UploadFile
from repos.dataset import dataset_registry_repository, dataset_repository
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema, DatasetRegistryBaseSchema, DatasetRegistryReadSchema
from sqlalchemy.orm import Session
from utils.dataset_registry import DatasetRegistry


class DatasetService:
    def get(self, db: Session, pk: int) -> DatasetReadSchema:
        return dataset_repository.get(db, pk)

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> list[DatasetReadSchema]:
        return dataset_repository.get_multi(db, skip=skip, limit=limit)

    def get_all(self, db: Session) -> list[DatasetReadSchema]:
        return dataset_repository.get_all(db)

    def update(self, db: Session, db_obj, obj_in):
        return dataset_repository.update(db, db_obj=db_obj, obj_in=obj_in)

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

    def create(self, db: Session, *, obj_in: DatasetBaseSchema, file: UploadFile):
        # S3 매니저 인스턴스 가져오기
        s3_manager = S3Manager.get_instance()

        # 파일을 객체 스토리지에 업로드
        s3_file_url = s3_manager.upload_file(file)

        # 데이터셋 객체 생성
        dataset_obj = dataset_repository.create(db, obj_in=obj_in)
        dataset_id = dataset_obj.id

        # 데이터셋 레지스트리 정보 생성
        dataset_registry_repository.create(
            db,
            obj_in=DatasetRegistryBaseSchema(
                artifact_path=s3_file_url,
                uri=s3_file_url,
                dataset_id=dataset_id,
            ),
        )

        db.commit()
        return dataset_repository.get(db, dataset_id)
