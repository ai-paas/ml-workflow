import json
import os
import pickle
import shutil
import tempfile
from enum import Enum
from typing import Any, Optional

import torch
import torchvision
from fastapi import UploadFile
from huggingface_hub import snapshot_download
from repos.model import (
    model_format_repository,
    model_provider_repository,
    model_registry_repository,
    model_repository,
    model_type_repository,
)
from schemas.model import (
    ModelBaseSchema,
    ModelFormatReadSchema,
    ModelProviderReadSchema,
    ModelReadSchema,
    ModelRegistryBaseSchema,
    ModelRegistryReadSchema,
    ModelRegistryRequestSchema,
    ModelTypeReadSchema,
)
from sqlalchemy.orm import Session
from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
    AutoProcessor,
    AutoTokenizer,
    CLIPTokenizer,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    Owlv2Processor,
    Owlv2TextModel,
)
from utils.model_registry import ModelLoader, ModelRegistry


class MODEL_NAME(Enum):
    OWLV2 = "google/owlv2"
    YOLOX = "yolox"

    def __str__(self):
        return self.value


def is_yolox_remote_model(repo_id: str) -> bool:
    """
    repository ID가 YOLOX 모델인지 확인하는 함수

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        YOLOX 모델인지 여부
    """
    yolox_patterns = [
        "kadirnar/yolox_s-v0.1.1",
        "kadirnar/yolox_tiny-v0.1.1",
        "kadirnar/yolox_nano-v0.1.1",
        "kadirnar/yolox_m-v0.1.1",
        "kadirnar/yolox_l-v0.1.1",
        "kadirnar/yolox_x-v0.1.1",
    ]

    return repo_id in yolox_patterns or repo_id.startswith("kadirnar/yolox_")


def is_yolox_local_model(model_name: str) -> bool:
    return "yolox" in model_name.lower()


class ModelService:
    def get(self, db: Session, pk: int) -> ModelReadSchema:
        return model_repository.get(db, pk)

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> list[ModelReadSchema]:
        return model_repository.get_multi(db, skip=skip, limit=limit)

    def get_all(self, db: Session) -> list[ModelReadSchema]:
        return model_repository.get_all(db)

    def update(self, db: Session, db_obj, obj_in):
        return model_repository.update(db, db_obj=db_obj, obj_in=obj_in)

    def validate(self, model_format_id: int, model_uri: str) -> str:
        # TODO: model_format_id로부터 get 하도록 변경
        if model_format_id == 1:
            pipeline = ModelLoader.load_transformers(model_uri)
            messages = [
                {"role": "user", "content": "Who are you?"},
            ]
            result = pipeline(messages, max_length=1024)
        else:
            result = ""
        return result

    @staticmethod
    def load_transformers(model_uri: str):
        loaded_pipe = ModelLoader.load_transformers(model_uri)
        return loaded_pipe


class HuggingFaceModelService:
    def create(self, db: Session, *, model_schema: ModelBaseSchema):
        # model_format_id = model_schema.format_id
        repo_id = model_schema.name
        # transformers_db_obj = model_format_repository.get_by_name(db, "transformers")
        # if model_format_id == transformers_db_obj.id:  # transformers
        save_dir = self.load_and_save_transformers(repo_id)
        run_id, artifact_uri = ModelRegistry().log_artifact(model_name=repo_id, save_dir=save_dir)
        model_uri = ""
        # else:
        #     print("Error!!!")

        model_obj = model_repository.create(db, obj_in=model_schema)
        model_id = model_obj.id
        model_registry_repository.create(
            db,
            obj_in=ModelRegistryBaseSchema(
                artifact_path=artifact_uri,
                uri=model_uri,
                run_id=run_id,
                reference_model_id=model_id,
            ),
        )
        db.commit()
        return model_repository.get(db, model_id)

    @staticmethod
    def load_and_save_transformers(repo_id: str) -> str:
        """
        transformers 계열 Model을 Load하는 method

        * params
            * repo_id: str
                - from HuggingFace: model_name (e.g. 'openbmb/MiniCPM-V-2_6-gguf')
                - from User: Model과 Tokenizer가 함께 저장된 directory 경로

        * return
            - transformer_model used in mlflow.transformers.log_model
                ```python
                components = {
                    "model": model,
                    "tokenizer": tokenizer,
                }
                ```
        """
        # TODO: MS 제공 Model의 경우, trust_remote_code=True 옵션을 추가해야하는 경우 발견됨

        if repo_id.startswith(str(MODEL_NAME.OWLV2)):
            processor = Owlv2Processor.from_pretrained(repo_id)
            model = Owlv2ForObjectDetection.from_pretrained(repo_id)
            tokenizer = CLIPTokenizer.from_pretrained(repo_id)
        else:
            processor = AutoProcessor.from_pretrained(repo_id)
            model = AutoModelForObjectDetection.from_pretrained(repo_id)

            # 모델 설정을 확인하여 토크나이저 필요 여부 판단
            config = AutoConfig.from_pretrained(repo_id)
            if hasattr(config, "model_type") and config.model_type == "detr":
                tokenizer = None
            else:
                tokenizer = AutoTokenizer.from_pretrained(repo_id)

        components = {
            "model": model,
            "image_processor": processor,
        }

        if tokenizer is not None:
            components["tokenizer"] = tokenizer

        # 임시 디렉토리를 수동으로 생성하여 자동 삭제되지 않도록 함
        temp_dir = tempfile.mkdtemp()
        model.save_pretrained(temp_dir)
        processor.save_pretrained(temp_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(temp_dir)

        return temp_dir


class CustomModelService:
    """
    Llama.cpp 계열 gguf Model을 등록하는 method

    * params
        * model_path: gguf file path (e.g. "your/model/file/path.gguf")
    """

    def create(
        self,
        db: Session,
        *,
        model_schema: ModelBaseSchema,
        model_registry_schema: ModelRegistryRequestSchema = None,
        file: UploadFile = None,
    ):
        # 이미 pipeline에서 등록한 mlflow model_registry가 있다면 mlflow에 등록하지 말것
        if not model_registry_schema:
            contents = file.file.read()
            model_name = model_schema.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as temp_file:
                temp_file.write(contents)
                # temp_file_path = temp_file.name
                # model = torch.load(temp_file_path, map_location="cpu")
                run_id, artifact_uri = ModelRegistry().log_artifact(file=file, model_name=model_name)
                model_uri = ""
        else:
            # run_id = model_registry_schema.run_id
            artifact_uri = model_registry_schema.artifact_path
            # model_version = model_registry_schema.versions
            model_uri = model_registry_schema.uri
            run_id = model_registry_schema.run_id

        model_obj = model_repository.create(db, obj_in=model_schema)
        model_id = model_obj.id
        model_registry_repository.create(
            db,
            obj_in=ModelRegistryBaseSchema(
                artifact_path=artifact_uri,
                uri=model_uri,
                reference_model_id=model_id,
                run_id=run_id,
            ),
        )
        db.commit()
        return model_repository.get(db, model_id)


class ModelProviderService:
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ModelProviderReadSchema]:
        return model_provider_repository.get_by_name(db, name)


class ModelTypeService:
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ModelTypeReadSchema]:
        return model_type_repository.get_by_name(db, name)


class ModelFormatService:
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ModelFormatReadSchema]:
        return model_format_repository.get_by_name(db, name)
