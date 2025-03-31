import tempfile
from enum import Enum
from typing import Any

from fastapi import UploadFile
from repos.model import model_registry_repository, model_repository
from schemas.model import (
    ModelBaseSchema,
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

    def __str__(self):
        return self.value


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
        model_format_id = model_schema.model_format_id
        repo_id = model_schema.name
        # TODO: model_format_id로부터 get 하도록 변경
        if model_format_id == 1:  # transformers
            model = self.load_transformers(repo_id)
            run_id, artifact_uri, model_version, model_uri = ModelRegistry().log_transformers(model, repo_id)
        else:
            print("Error!!!")

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
        file: UploadFile = None
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
