import json
import os
import pickle
import tempfile
from enum import Enum
from typing import Any

import torch
import torchvision
from fastapi import UploadFile
from huggingface_hub import snapshot_download
from repos.model import model_provider_repository, model_registry_repository, model_repository
from schemas.model import (
    ModelBaseSchema,
    ModelProviderReadSchema,
    ModelReadSchema,
    ModelRegistryBaseSchema,
    ModelRegistryReadSchema,
    ModelRegistryRequestSchema,
)
from sqlalchemy.orm import Session
from transformers import (
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


def is_yolox_model(repo_id: str) -> bool:
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

    staticmethod

    def load_transformers(model_uri: str):
        loaded_pipe = ModelLoader.load_transformers(model_uri)
        return loaded_pipe


class HuggingFaceModelService:
    def create(self, db: Session, *, model_schema: ModelBaseSchema):
        model_format_id = model_schema.format_id
        repo_id = model_schema.name
        transformers_db_obj = model_repository.get_by_name(db, "transformers")
        # YOLOX 모델인지 먼저 확인 (transformers가 아닌 별도 처리)
        if is_yolox_model(repo_id):
            model = self.load_yolox(repo_id)
            run_id, artifact_uri, model_version, model_uri = ModelRegistry().log_yolox(model, repo_id)
        # TODO: model_format_id로부터 get 하도록 변경
        elif model_format_id == transformers_db_obj.id:  # transformers
            model = self.load_transformers(repo_id)
            run_id, artifact_uri, model_version, model_uri = ModelRegistry().log_transformers(model, repo_id)
        else:
            print("Error!!!")

        model_obj = model_repository.create(db, obj_in=model_schema)
        model_id = model_obj.id
        model_registry_repository.create(
            db,
            obj_in=ModelRegistryBaseSchema(
                artifact_path=artifact_uri,
                uri=model_uri,
                reference_model_id=model_id,
            ),
        )
        db.commit()
        return model_repository.get(db, model_id)

    @staticmethod
    def load_yolox(repo_id: str, device: str = "cpu") -> dict[str, Any]:
        """
        YOLOX 모델을 로드하는 메서드 (huggingface-hub 사용)

        * params
            * repo_id: str
                - HuggingFace repository ID (e.g., "kadirnar/yolox_s-v0.1.1")
            * device: str
                - 사용할 디바이스 ("cpu", "cuda:0" 등)

        * return
            - YOLOX 모델 딕셔너리
        """
        try:
            # 임시 디렉토리에 모델 다운로드
            import tempfile

            temp_dir = tempfile.mkdtemp()
            local_dir = os.path.join(temp_dir, "yolox_model")

            print(f"HuggingFace에서 YOLOX 모델 다운로드 중: {repo_id}")

            # 전체 모델 다운로드
            downloaded_path = snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

            print(f"모델이 다운로드되었습니다: {downloaded_path}")

            # 다운로드된 파일 확인
            model_files = os.listdir(downloaded_path)
            print(f"다운로드된 파일들: {model_files}")

            # 모델 관련 정보
            model_info = {
                "repo_id": repo_id,
                "local_path": downloaded_path,
                "model_files": model_files,
                "device": device,
            }

            # config.json 파일이 있다면 로드
            config_path = os.path.join(downloaded_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    model_info["config"] = config
                    print(f"Config 로드됨: {config}")

            # PyTorch 모델 파일 찾기 (.pth, .pt, .bin 등)
            model_file = None
            for file in model_files:
                if file.endswith((".pth", ".pt", ".bin")):
                    model_file = os.path.join(downloaded_path, file)
                    break

            if model_file and os.path.exists(model_file):
                print(f"모델 파일 발견: {model_file}")
                # PyTorch 모델 로드
                model_state_dict = torch.load(model_file, map_location=device)
                model_info["model_state_dict"] = model_state_dict
                model_info["model_file"] = model_file

            return model_info

        except Exception as e:
            print(f"YOLOX 모델 로드 중 오류 발생: {e}")
            raise Exception(f"YOLOX 모델 로드 중 오류 발생: {e}")

    @staticmethod
    def load_transformers(repo_id: str) -> dict[str, Any]:
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
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
        return {
            "model": model,
            "image_processor": processor,
            "tokenizer": tokenizer,
        }


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
            # TODO: gguf나 transformers 등의 여러 타입을 지원해야할것
            contents = file.file.read()
            model_name = model_schema.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as temp_file:
                temp_file.write(contents)
                temp_file_path = temp_file.name
                model = Llama(model_path=temp_file_path)
                run_id, artifact_uri, model_version, model_uri = ModelRegistry().log_llamacpp(model, model_name)
        else:
            run_id = model_registry_schema.run_id
            artifact_uri = model_registry_schema.artifact_path
            model_version = model_registry_schema.versions
            model_uri = model_registry_schema.model_uri

        model_obj = model_repository.create(db, obj_in=model_schema)
        model_id = model_obj.id
        model_registry_repository.create(
            db,
            obj_in=ModelRegistryBaseSchema(
                run_id=run_id, version=model_version, artifact_path=artifact_uri, model_uri=model_uri, model_id=model_id
            ),
        )
        db.commit()
        return model_repository.get(db, model_id)


class ModelProviderService:
    @staticmethod
    def get_by_name(db: Session, name: str) -> ModelProviderReadSchema:
        return model_provider_repository.get_by_name(db, name)
