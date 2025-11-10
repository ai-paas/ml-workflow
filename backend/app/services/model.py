import logging
import tempfile
import traceback
from enum import Enum
from typing import Any, Optional

from core.kubeflow.s3.mlflow_s3_manager import MLFlowS3Manager
from db.models.experiment import ExperimentModel
from db.models.model import Model
from db.models.service import Service, WorkflowComponent
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
from utils.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


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

    @staticmethod
    def check_model_references(db: Session, model_id: int) -> dict:
        """
        모델이 참조되고 있는지 확인

        Returns:
            dict: 참조 정보 {'has_references': bool, 'references': list}
        """

        references = []

        # 1. Experiment에서 참조 확인
        experiments = db.query(ExperimentModel).filter(ExperimentModel.reference_model_id == model_id).all()
        if experiments:
            references.append(
                {
                    "type": "experiment",
                    "count": len(experiments),
                    "items": [{"id": exp.id, "name": exp.name} for exp in experiments[:5]],  # 최대 5개만
                }
            )

        # 2. WorkflowComponent에서 참조 확인
        workflow_components = db.query(WorkflowComponent).filter(WorkflowComponent.model_id == model_id).all()
        if workflow_components:
            references.append(
                {
                    "type": "workflow_component",
                    "count": len(workflow_components),
                    "items": [
                        {"id": wc.id, "name": wc.name, "workflow_id": wc.workflow_id} for wc in workflow_components[:5]
                    ],
                }
            )

        # 3. 다른 모델의 parent_model_id로 참조 확인 (자식 모델 확인)
        child_models = db.query(Model).filter(Model.parent_model_id == model_id).all()
        if child_models:
            references.append(
                {
                    "type": "child_model",
                    "count": len(child_models),
                    "items": [{"id": m.id, "name": m.name} for m in child_models[:5]],
                }
            )

        # 4. Service에서 참조 확인
        services = db.query(Service).filter(Service.model_id == model_id).all()
        if services:
            references.append(
                {
                    "type": "service",
                    "count": len(services),
                    "items": [{"id": s.id, "name": s.name} for s in services[:5]],
                }
            )

        return {"has_references": len(references) > 0, "references": references}

    @staticmethod
    def delete(db: Session, model_id: int):
        """
        모델 삭제 - 참조 확인 후 안전하게 삭제
        """
        try:
            # 1. 모델 객체 가져오기
            model_obj = model_repository.get(db, model_id)
            if not model_obj:
                raise ValueError(f"모델 ID {model_id}를 찾을 수 없습니다.")

            # 2. 참조 관계 확인
            ref_check = ModelService.check_model_references(db, model_id)
            if ref_check["has_references"]:
                ref_details = []
                for ref in ref_check["references"]:
                    ref_details.append(f"- {ref['type']}: {ref['count']}개")

                raise RuntimeError(
                    "모델을 삭제할 수 없습니다. 다음 항목에서 참조되고 있습니다:\n"
                    + "\n".join(ref_details)
                    + "\n\n참조하는 항목을 먼저 삭제하거나 수정해주세요."
                )

            # 3. MLflow 정보 미리 저장 (DB 삭제 전에 필요)
            run_id = model_obj.registry.run_id if model_obj.registry else None
            artifact_path = model_obj.registry.artifact_path if model_obj.registry else None
            s3_artifact_path = artifact_path.replace("mlflow-artifacts:/", "") if artifact_path else None

            # 4. 트랜잭션 시작 - MLflow/S3 삭제 후 DB 커밋
            try:
                # 4-1. DB 삭제 준비 (아직 커밋하지 않음)
                # ModelRegistry는 CASCADE 설정으로 자동 삭제되지만 명시적으로 삭제
                if model_obj.registry:
                    model_registry_repository.delete(db, pk=model_obj.registry.id)

                # Model 삭제 (아직 커밋 안됨)
                model_repository.delete(db, pk=model_id)

                # 4-2. MLflow/S3 삭제 시도
                mlflow_deleted = False

                # MLflow artifacts 삭제
                if run_id:
                    try:
                        ModelRegistry().delete_run_artifacts(run_id)
                        mlflow_deleted = True
                    except Exception as mlflow_error:
                        # MLflow 삭제 실패시 DB 롤백
                        db.rollback()
                        raise RuntimeError(f"MLflow 아티팩트 삭제 실패 (DB 변경사항 롤백됨): {str(mlflow_error)}")

                # S3 폴더 삭제
                if s3_artifact_path:
                    try:
                        MLFlowS3Manager.get_instance().delete_folder(s3_artifact_path)
                    except Exception as s3_error:
                        # S3 삭제 실패 처리
                        # MLflow가 이미 삭제되었다면 복구 불가능하므로 경고만 하고 진행
                        if mlflow_deleted:
                            import warnings

                            warnings.warn(f"S3 폴더 삭제 실패 (MLflow는 이미 삭제됨): {str(s3_error)}")
                            # S3만 실패한 경우 DB는 커밋 (MLflow는 이미 삭제되었으므로)
                        else:
                            # MLflow도 삭제 안됐고 S3도 실패면 롤백
                            db.rollback()
                            raise RuntimeError(f"S3 폴더 삭제 실패 (DB 변경사항 롤백됨): {str(s3_error)}")

                # 4-3. 모든 삭제가 성공하면 DB 커밋
                db.commit()

            except Exception as e:
                # 이미 처리된 RuntimeError는 그대로 전달
                if isinstance(e, RuntimeError):
                    raise
                # 예상치 못한 에러는 롤백 후 전달
                db.rollback()
                raise RuntimeError(f"모델 삭제 중 예상치 못한 오류 발생: {str(e)}")

            return True

        except Exception as e:
            # 이미 처리된 에러는 그대로 전달
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            # 예상치 못한 에러
            db.rollback()
            raise RuntimeError(f"모델 삭제 중 오류 발생: {str(e)}")


class HuggingFaceModelService:
    def create(self, db: Session, *, model_schema: ModelBaseSchema):
        # model_format_id = model_schema.format_id
        repo_id = model_schema.repo_id
        # transformers_db_obj = model_format_repository.get_by_name(db, "transformers")
        # if model_format_id == transformers_db_obj.id:  # transformers
        save_dir = self.load_and_save_transformers(repo_id)
        model_name = repo_id.replace("/", "-")
        run_id, artifact_uri = ModelRegistry().log_artifact(model_name=model_name, save_dir=save_dir)
        model_uri = model_name
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
            if hasattr(config, "model_type") and (config.model_type == "detr" or config.model_type == "yolos"):
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
            model_name = model_schema.name
            model_name = model_name.replace("/", "-")
            # log_artifact 내부에서 file.file.read()를 호출하므로, 파일 포인터를 처음으로 되돌릴 필요 없음
            # 파일을 그대로 전달하면 됨
            run_id, artifact_uri = ModelRegistry().log_artifact(file=file, model_name=model_name)
            model_uri = model_name
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

    @staticmethod
    def get_all(db: Session) -> list[ModelProviderReadSchema]:
        return model_provider_repository.get_multi(db, skip=0, limit=10000)


class ModelTypeService:
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ModelTypeReadSchema]:
        return model_type_repository.get_by_name(db, name)

    @staticmethod
    def get_all(db: Session) -> list[ModelTypeReadSchema]:
        return model_type_repository.get_multi(db, skip=0, limit=10000)


class ModelFormatService:
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ModelFormatReadSchema]:
        return model_format_repository.get_by_name(db, name)

    @staticmethod
    def get_all(db: Session) -> list[ModelFormatReadSchema]:
        return model_format_repository.get_multi(db, skip=0, limit=10000)
