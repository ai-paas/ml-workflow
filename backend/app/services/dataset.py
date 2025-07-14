import os
import tempfile
from enum import Enum
from typing import Any

import pandas as pd
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

    def create(self, db: Session, *, dataset_schema: DatasetBaseSchema, file: list[UploadFile]):
        # TODO: csv나 기타 여러 문서 타입을 지원해야 할것
        # with tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as temp_file:
        #     temp_file.write(file.file.read())
        #     temp_file_path = temp_file.name
        #     run_id, artifact_uri, dataset_version, dataset_uri = DatasetRegistry().log_dataset(
        #         dataset_dir=temp_file_path, dataset_name=dataset_schema.name)

        # 1. 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # 임시 파일 경로 지정
            temp_train_path = os.path.join(temp_dir, file[0].filename)
            temp_test_path = os.path.join(temp_dir, file[1].filename)
            # 2. UploadFile 내용을 임시 파일로 저장
            with open(temp_train_path, "wb") as temp_file:
                content = file[0].file.read()
                temp_file.write(content)
            with open(temp_test_path, "wb") as temp_file:
                content = file[1].file.read()
                temp_file.write(content)
            run_id, version, artifact_uri, dataset_uri = DatasetRegistry().log_dataset(
                dataset_dir=temp_dir,
                dataset_name=dataset_schema.name,
                # dataset={"train": pd.read_csv(file[0].file),
                #          "test": pd.read_csv(file[1].file)}
            )

        dataset_obj = dataset_repository.create(db, obj_in=dataset_schema)
        dataset_id = dataset_obj.id
        dataset_registry_repository.create(
            db,
            obj_in=DatasetRegistryBaseSchema(
                run_id=run_id,
                artifact_path=artifact_uri,
                dataset_uri=dataset_uri,
                dataset_id=dataset_id,
                version=version,
            ),
        )
        db.commit()
        return dataset_repository.get(db, dataset_id)
