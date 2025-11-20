import logging
import os
import re
import tempfile
import traceback
import uuid
from enum import Enum
from typing import Any, Optional

from config.db.enums import ModelProviderEnum, ModelTypeEnum
from config.settings import get_settings
from core.kubeflow.kubeflow_manager import KubeflowManager
from core.kubeflow.s3.mlflow_s3_manager import MLFlowS3Manager
from db.models.model import Model
from fastapi import UploadFile
from huggingface_hub import snapshot_download
from kfp import dsl
from kfp.compiler import Compiler
from repos.experiment import experiment_repository
from repos.knowledge_base import knowledge_base_repository
from repos.model import (
    model_format_repository,
    model_provider_repository,
    model_registry_repository,
    model_repository,
    model_type_repository,
)
from repos.model_base_deployment import model_base_deployment_repository
from repos.workflow import workflow_component_repository
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
from services.model_base_deployment import ModelBaseDeploymentService
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
settings = get_settings()


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
    @staticmethod
    def get(db: Session, pk: int) -> Optional[Model]:
        """ID로 Model 객체 조회"""
        return model_repository.get(db, pk)

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> list[ModelReadSchema]:
        return model_repository.get_multi(db, skip=skip, limit=limit)

    def get_all(self, db: Session) -> list[ModelReadSchema]:
        return model_repository.get_all(db)

    def update(self, db: Session, db_obj, obj_in):
        return model_repository.update(db, db_obj=db_obj, obj_in=obj_in)

    def filter(
        self,
        db: Session,
        filters: dict[str, Any],
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelReadSchema]:
        """
        필터 조건에 따라 모델 목록을 조회합니다.

        Args:
            db: 데이터베이스 세션
            filters: 필터 조건 딕셔너리
                - type_id: 모델 타입 ID
                - provider_id: 모델 제공자 ID
                - format_id: 모델 포맷 ID
            skip: 건너뛸 레코드 수
            limit: 반환할 최대 레코드 수

        Returns:
            필터링된 모델 목록 (ModelReadSchema)
        """
        models = model_repository.filter(db, filters)
        # 페이지네이션 적용
        paginated_models = models[skip : skip + limit]
        return [self.get(db, model.id) for model in paginated_models]

    def filter_all(self, db: Session, filters: dict[str, Any], max_limit: int = 10000) -> list[ModelReadSchema]:
        """
        필터 조건에 따라 모든 모델 목록을 조회합니다 (페이지네이션 없음).

        Args:
            db: 데이터베이스 세션
            filters: 필터 조건 딕셔너리
                - type_id: 모델 타입 ID
                - provider_id: 모델 제공자 ID
                - format_id: 모델 포맷 ID
            max_limit: 최대 반환 레코드 수 (기본값: 10000)

        Returns:
            필터링된 모델 목록 (ModelReadSchema)
        """
        models = model_repository.filter(db, filters)
        limited_models = models[:max_limit]
        return [self.get(db, model.id) for model in limited_models]

    @staticmethod
    def check_model_references(db: Session, model_id: int) -> dict:
        """
        모델이 참조되고 있는지 확인

        Returns:
            dict: 참조 정보 {'has_references': bool, 'references': list}
        """

        references = []

        # 1. Experiment에서 참조 확인
        experiments = experiment_repository.get_by_reference_model_id(db, model_id)
        if experiments:
            references.append(
                {
                    "type": "experiment",
                    "count": len(experiments),
                    "items": [{"id": exp.id, "name": exp.name} for exp in experiments[:5]],  # 최대 5개만
                }
            )

        # 2. WorkflowComponent에서 참조 확인
        workflow_components = workflow_component_repository.get_by_model_id(db, model_id)
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
        child_models = model_repository.get_by_parent_model_id(db, model_id)
        if child_models:
            references.append(
                {
                    "type": "child_model",
                    "count": len(child_models),
                    "items": [{"id": m.id, "name": m.name} for m in child_models[:5]],
                }
            )

        # 4. KnowledgeBase에서 참조 확인 (embedding_model_id)
        knowledge_bases = knowledge_base_repository.get_by_embedding_model_id(db, model_id)
        if knowledge_bases:
            references.append(
                {
                    "type": "knowledge_base",
                    "count": len(knowledge_bases),
                    "items": [{"id": kb.id, "name": kb.name} for kb in knowledge_bases[:5]],  # 최대 5개만
                }
            )

        # Note: Service는 모델을 직접 참조하지 않습니다.
        # 모델은 WorkflowComponent를 통해 Workflow에 연결되고,
        # Workflow가 Service에 연결되는 구조입니다.
        # 따라서 Service에서의 참조 확인은 WorkflowComponent 확인으로 충분합니다.

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

            # 3. 모델 타입과 provider 확인 (명시적으로 조회)
            model_type = model_type_repository.get(db, model_obj.type_id)
            model_provider = model_provider_repository.get(db, model_obj.provider_id)

            model_type_name = model_type.name if model_type else None
            model_provider_name = model_provider.name if model_provider else None

            is_ollama_embedding = (
                model_type_name == ModelTypeEnum.EMBEDDING.value
                and model_provider_name == ModelProviderEnum.OLLAMA.value
            )
            is_ollama = model_provider_name == ModelProviderEnum.OLLAMA.value

            # 3-1. model_base_deployments 레코드 삭제 (외래키 제약 조건 때문에 먼저 삭제 필요)
            # cleanup_model_deployment가 파이프라인 완료 후 DB 레코드를 삭제함
            if is_ollama_embedding:
                try:
                    cleanup_result = ModelBaseDeploymentService.cleanup_model_deployment(db, model_id)
                    logger.info(f"Kubernetes resources cleanup result for Ollama Embedding model: {cleanup_result}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup Kubernetes resources for model {model_id}: {cleanup_error}")
                    # cleanup 실패 시 모델 삭제 중단 (외래키 제약 조건 위반 방지)
                    raise RuntimeError(f"모델 기본 배포 정보 정리 실패: {str(cleanup_error)}")

            # 3-2. Ollama 모델의 PVC 삭제 (모든 Ollama 모델에 대해 수행)
            # 배포 리소스 정리 후 PVC 삭제 (PVC가 사용 중일 수 있으므로)
            if is_ollama:
                try:
                    pvc_cleanup_result = OllamaModelService.delete_ollama_model_pvc(db, model_id)
                    logger.info(f"PVC cleanup result for Ollama model: {pvc_cleanup_result}")
                except Exception as pvc_error:
                    logger.error(f"Failed to cleanup PVC for Ollama model {model_id}: {pvc_error}")
                    # PVC 삭제 실패 시 모델 삭제 중단
                    raise RuntimeError(f"Ollama 모델 PVC 삭제 실패: {str(pvc_error)}")

            # 3-3. MLflow 정보 미리 저장 (Ollama provider가 아닌 경우에만 필요)
            run_id = None
            artifact_path = None
            s3_artifact_path = None

            if not is_ollama:
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

                # 4-2. MLflow/S3 삭제 시도 (Ollama provider가 아닌 경우에만)
                if not is_ollama:
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
        if not repo_id:
            raise ValueError("repo_id is required for HuggingFace models")
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


class OllamaModelService:
    """Ollama 모델 관련 서비스"""

    @staticmethod
    def delete_ollama_model_pvc(db: Session, model_id: int) -> dict:
        """
        Ollama 모델의 PVC를 삭제하는 공통 함수 (Kubeflow 파이프라인 사용)
        파이프라인을 시작만 하고 상태 체크하지 않음 (백그라운드 실행)

        Args:
            db: DB 세션
            model_id: 모델 ID

        Returns:
            dict: 삭제 결과 정보
        """
        try:
            # Kubeflow 파이프라인 생성 및 실행
            kf_manager = KubeflowManager()

            # 파이프라인 함수 생성
            pipeline_func = OllamaModelService._create_pvc_cleanup_pipeline_function(model_id=model_id)

            # 파이프라인 컴파일
            pipeline_name = f"cleanup-pvc-{model_id}"
            pipeline_filename = f"/tmp/{pipeline_name}-{uuid.uuid4().hex[:8]}.yaml"

            try:
                Compiler().compile(pipeline_func, pipeline_filename)

                # Kubeflow에 파이프라인 실행
                experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
                experiment = kf_manager.get_experiment_by_name(experiment_name=experiment_name)
                if not experiment:
                    experiment = kf_manager.create_experiment(experiment_name)

                exp_id = experiment.experiment_id if hasattr(experiment, "experiment_id") else experiment.id

                run = kf_manager.kfp_client.run_pipeline(
                    experiment_id=exp_id,
                    job_name=f"{pipeline_name}-run-{uuid.uuid4().hex[:8]}",
                    pipeline_package_path=pipeline_filename,
                    enable_caching=False,
                )

                # run_id 추출
                if hasattr(run, "run_id"):
                    run_id = run.run_id
                elif hasattr(run, "id"):
                    run_id = run.id
                else:
                    run_dict = run.to_dict() if hasattr(run, "to_dict") else {}
                    run_id = run_dict.get("run_id") or run_dict.get("id") or f"unknown-{uuid.uuid4().hex[:8]}"

                logger.info(f"PVC cleanup pipeline started with run_id: {run_id}")
                # 파이프라인은 백그라운드에서 실행되며, 상태 체크하지 않음

            finally:
                # 임시 파일 삭제
                if os.path.exists(pipeline_filename):
                    os.remove(pipeline_filename)

            return {
                "model_id": model_id,
                "cleanup_run_id": run_id,
                "status": "started",
                "message": "PVC cleanup pipeline started (running in background)",
            }

        except Exception as e:
            logger.error(f"Failed to start PVC cleanup pipeline for model_id {model_id}: {e}")
            raise RuntimeError(f"Ollama 모델 PVC 삭제 파이프라인 시작 실패: {str(e)}")

    @staticmethod
    def _create_pvc_cleanup_pipeline_function(model_id: int) -> callable:
        """
        Ollama 모델 PVC 정리를 위한 Kubeflow 파이프라인 함수 생성

        Args:
            model_id: 모델 ID

        Returns:
            파이프라인 함수
        """

        @dsl.pipeline(
            name=f"cleanup-pvc-{model_id}",
            description=f"Cleanup PVC for Ollama model {model_id}",
        )
        def cleanup_pvc_pipeline():
            OllamaModelService._create_pvc_cleanup_component_task(model_id=model_id)

        return cleanup_pvc_pipeline

    @staticmethod
    def _create_pvc_cleanup_component_task(model_id: int) -> Any:
        """
        Ollama 모델 PVC 정리 컴포넌트 태스크 생성

        Args:
            model_id: 모델 ID

        Returns:
            Kubeflow 태스크
        """

        @dsl.component(
            base_image="python:3.10",
            packages_to_install=[
                "kubernetes==28.1.0",
            ],
        )
        def cleanup_pvc_resources(model_id: int, namespace: str) -> str:
            import json  # noqa: F811
            import logging

            from kubernetes import client
            from kubernetes import config as k8s_config

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                # Kubernetes 설정
                k8s_config.load_incluster_config()

                label_selector = f"model-id={model_id}"

                deleted_count = 0
                failed_count = 0

                core_v1 = client.CoreV1Api()

                # PVC 삭제
                try:
                    logger.info(f"Querying PVCs for model_id: {model_id}")
                    pvcs = core_v1.list_namespaced_persistent_volume_claim(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    pvc_items = pvcs.items
                    logger.info(f"Found {len(pvc_items)} PVCs for model_id {model_id}")

                    for pvc in pvc_items:
                        pvc_name = pvc.metadata.name
                        try:
                            logger.info(f"Deleting PVC: {pvc_name} in namespace: {namespace}")
                            core_v1.delete_namespaced_persistent_volume_claim(
                                name=pvc_name,
                                namespace=namespace,
                            )
                            deleted_count += 1
                            logger.info(f"Successfully deleted PVC: {pvc_name}")
                        except client.exceptions.ApiException as e:
                            if e.status == 404:
                                logger.warning(f"PVC not found: {pvc_name}")
                            else:
                                logger.error(f"Failed to delete PVC {pvc_name}: {e.status} - {e.reason}")
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Unexpected error deleting PVC {pvc_name}: {e}")
                            failed_count += 1

                except Exception as e:
                    logger.warning(f"Error cleaning up PVCs: {e}")
                    # PVC가 없는 경우는 정상일 수 있으므로 에러로 처리하지 않음

                result_data = {
                    "model_id": model_id,
                    "deleted": deleted_count,
                    "failed": failed_count,
                    "status": "completed" if failed_count == 0 else "partial",
                }

                logger.info(f"PVC cleanup completed: {json.dumps(result_data)}")
                return json.dumps(result_data)

            except Exception as e:
                logger.error(f"PVC cleanup failed: {str(e)}")
                error_result = {
                    "model_id": model_id,
                    "deleted": 0,
                    "failed": 0,
                    "status": "failed",
                    "error": str(e),
                }
                return json.dumps(error_result)

        return cleanup_pvc_resources(
            model_id=model_id,
            namespace=settings.KUBEFLOW_NAMESPACE,
        )

    @staticmethod
    def _get_storage_size_for_model(ollama_model_name: str) -> str:
        """
        Ollama 모델 이름에 따라 필요한 PVC storage 용량을 반환합니다.

        Args:
            ollama_model_name: Ollama 모델 이름 (repo_id)

        Returns:
            str: Storage 용량 (예: "2Gi", "21Gi")
        """
        storage_map = {
            "bge-m3": "2Gi",
            "qwq:32b": "22Gi",
            "qwen3:32b": "22Gi",
            "qwen3:30b": "21Gi",
            "gpt-oss:20b": "16Gi",
            "ahmgam/medllama3-v20:latest": "7Gi",
            "taozhiyuai/openbiollm-llama-3:70b_q4_k_m": "45Gi",
            "gemma3:1b": "2Gi",
        }

        # 정확한 매칭 우선
        if ollama_model_name in storage_map:
            return storage_map[ollama_model_name]

        # 부분 매칭 (예: "qwq:32b"가 포함된 경우)
        for model_key, storage_size in storage_map.items():
            if model_key in ollama_model_name or ollama_model_name in model_key:
                return storage_size

        # 기본값 (매칭되지 않는 경우)
        logger.warning(f"Unknown model {ollama_model_name}, using default storage size 30Gi")
        return "30Gi"

    @staticmethod
    def download_ollama_model_to_pvc(
        db: Session,
        model_id: int,
        model_name: str,
        repo_id: str,
    ) -> str:
        """
        Ollama 모델을 PVC에 다운로드하는 Kubeflow 파이프라인 실행

        Args:
            db: DB 세션
            model_id: 모델 ID
            model_name: 모델 이름
            repo_id: Ollama 모델 repo_id

        Returns:
            str: PVC 이름
        """
        try:
            ollama_model_name = repo_id

            # PVC 이름 생성 (model_id 기반)
            sanitized_model_name = model_name.replace("/", "-").lower()
            pvc_name = f"ollama-model-{model_id}-{sanitized_model_name[:20]}"
            # DNS 1035 규칙에 맞게 정규화
            pvc_name = re.sub(r"[^a-z0-9-]", "-", pvc_name)
            pvc_name = re.sub(r"-+", "-", pvc_name)
            pvc_name = pvc_name.strip("-")
            if len(pvc_name) > 63:
                pvc_name = pvc_name[:63].rstrip("-")

            logger.info(f"Downloading Ollama model {ollama_model_name} to PVC: {pvc_name}")

            # 모델에 맞는 storage 용량 계산
            storage_size = OllamaModelService._get_storage_size_for_model(ollama_model_name)
            logger.info(f"Using storage size {storage_size} for model {ollama_model_name}")

            # Kubeflow 파이프라인 생성 및 실행
            kf_manager = KubeflowManager()

            # 파이프라인 함수 생성
            pipeline_func = OllamaModelService._create_download_pipeline_function(
                model_id=model_id,
                model_name=model_name,
                repo_id=repo_id,
                pvc_name=pvc_name,
                storage_size=storage_size,
            )

            # 파이프라인 컴파일
            unique_id = uuid.uuid4().hex[:6]
            pipeline_name = f"ollama-download-{model_id}-{unique_id}"
            pipeline_filename = f"/tmp/{pipeline_name}-{uuid.uuid4().hex[:8]}.yaml"

            try:
                Compiler().compile(pipeline_func, pipeline_filename)

                # Kubeflow에 파이프라인 실행
                experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
                experiment = kf_manager.get_experiment_by_name(experiment_name=experiment_name)
                if not experiment:
                    experiment = kf_manager.create_experiment(experiment_name)

                exp_id = experiment.experiment_id if hasattr(experiment, "experiment_id") else experiment.id

                kf_manager.kfp_client.run_pipeline(
                    experiment_id=exp_id,
                    job_name=f"{pipeline_name}-run-{uuid.uuid4().hex[:8]}",
                    pipeline_package_path=pipeline_filename,
                    enable_caching=False,
                )

                logger.info(f"Ollama model download pipeline started for model_id: {model_id}, PVC: {pvc_name}")

            finally:
                # 임시 파일 삭제
                if os.path.exists(pipeline_filename):
                    os.remove(pipeline_filename)

            return pvc_name

        except Exception as e:
            logger.error(f"Failed to download Ollama model to PVC: {e}")
            raise

    @staticmethod
    def _create_download_pipeline_function(
        model_id: int,
        model_name: str,
        repo_id: str,
        pvc_name: str,
        storage_size: str,
    ) -> callable:
        """
        Ollama 모델 다운로드를 위한 Kubeflow 파이프라인 함수 생성

        Args:
            model_id: 모델 ID
            model_name: 모델 이름
            repo_id: Ollama 모델 repo_id
            pvc_name: PVC 이름
            storage_size: PVC storage 용량 (예: "2Gi", "21Gi")

        Returns:
            파이프라인 함수
        """

        @dsl.pipeline(
            name=f"ollama-download-{model_id}",
            description=f"Download Ollama model {repo_id} to PVC",
        )
        def download_pipeline():
            OllamaModelService._create_download_component_task(
                model_id=model_id,
                model_name=model_name,
                repo_id=repo_id,
                pvc_name=pvc_name,
                storage_size=storage_size,
            )

        return download_pipeline

    @staticmethod
    def _create_download_component_task(
        model_id: int,
        model_name: str,
        repo_id: str,
        pvc_name: str,
        storage_size: str,
    ) -> Any:
        """
        Ollama 모델 다운로드 컴포넌트 태스크 생성

        Args:
            model_id: 모델 ID
            model_name: 모델 이름
            repo_id: Ollama 모델 repo_id
            pvc_name: PVC 이름
            storage_size: PVC storage 용량 (예: "2Gi", "21Gi")

        Returns:
            Kubeflow 태스크
        """

        @dsl.component(
            base_image="python:3.10",
            packages_to_install=[
                "kubernetes==28.1.0",
            ],
        )
        def download_ollama_model(
            model_id: int,
            model_name: str,
            repo_id: str,
            pvc_name: str,
            namespace: str,
            storage_size: str,
        ) -> str:
            import json
            import logging
            import time

            from kubernetes import client
            from kubernetes import config as k8s_config

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                # Kubernetes 설정
                k8s_config.load_incluster_config()

                ollama_model_name = repo_id
                core_v1 = client.CoreV1Api()

                logger.info(
                    f"Downloading Ollama model: {ollama_model_name} to PVC: {pvc_name} "
                    f"with storage size: {storage_size}"
                )

                # 1. PVC 생성
                pvc = client.V1PersistentVolumeClaim(
                    metadata=client.V1ObjectMeta(
                        name=pvc_name,
                        namespace=namespace,
                        labels={
                            "model-id": str(model_id),
                            "model-name": model_name.replace("/", "-"),
                            "app": "ollama-download",
                        },
                    ),
                    spec=client.V1PersistentVolumeClaimSpec(
                        access_modes=["ReadWriteOnce"],
                        resources=client.V1ResourceRequirements(requests={"storage": storage_size}),
                    ),
                )

                try:
                    core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
                    logger.info(f"Created PVC: {pvc_name}")
                except Exception as e:
                    # PVC가 이미 존재하는 경우 무시
                    if "already exists" not in str(e).lower():
                        logger.warning(f"PVC creation failed (may already exist): {e}")

                # 2. Job 생성하여 모델 다운로드
                job_name = f"ollama-download-{model_id}-{int(time.time())}"

                job = client.V1Job(
                    metadata=client.V1ObjectMeta(
                        name=job_name,
                        namespace=namespace,
                        labels={
                            "model-id": str(model_id),
                            "app": "ollama-download",
                        },
                    ),
                    spec=client.V1JobSpec(
                        template=client.V1PodTemplateSpec(
                            metadata=client.V1ObjectMeta(
                                labels={
                                    "model-id": str(model_id),
                                    "app": "ollama-download",
                                },
                                # annotations={
                                #     "sidecar.istio.io/inject": "false",
                                # },
                            ),
                            spec=client.V1PodSpec(
                                restart_policy="Never",
                                containers=[
                                    client.V1Container(
                                        name="ollama-download",
                                        image="ollama/ollama:latest",
                                        command=["/bin/sh", "-c"],
                                        args=[
                                            (
                                                "set -e && "
                                                "ollama serve & SERVE_PID=$! && "
                                                "sleep 10 && "
                                                f"ollama pull {ollama_model_name} 2>&1 | tee /tmp/pull.log && "
                                                "grep -q 'success' /tmp/pull.log && "
                                                "echo 'Model pull completed, stopping server...' && "
                                                "kill $SERVE_PID 2>/dev/null || true && "
                                                "sleep 2 && "
                                                "pkill -9 -f 'ollama serve' 2>/dev/null || true && "
                                                "pkill -9 ollama 2>/dev/null || true && "
                                                "sleep 1 && "
                                                "echo 'Model download completed' && "
                                                "exit 0"
                                            )
                                        ],
                                        resources=client.V1ResourceRequirements(
                                            requests={
                                                "memory": "4Gi",
                                                "cpu": "500m",
                                            },
                                            limits={
                                                "memory": "8Gi",
                                                "cpu": "2000m",
                                            },
                                        ),
                                        volume_mounts=[
                                            client.V1VolumeMount(name="model-data", mount_path="/root/.ollama")
                                        ],
                                    )
                                ],
                                volumes=[
                                    client.V1Volume(
                                        name="model-data",
                                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                            claim_name=pvc_name
                                        ),
                                    )
                                ],
                            ),
                        ),
                        backoff_limit=3,
                        ttl_seconds_after_finished=300,  # Job 완료 후 5분 뒤 자동 삭제
                    ),
                )

                # Job 생성
                batch_v1 = client.BatchV1Api()
                batch_v1.create_namespaced_job(namespace=namespace, body=job)
                logger.info(f"Created Job: {job_name}")

                # Job 완료 대기 (최대 30분)
                max_wait = 7200  # 120분
                wait_interval = 15  # 15초 간격
                elapsed = 0
                job_completed = False

                while elapsed < max_wait:
                    try:
                        job_status = batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)

                        if job_status.status.succeeded:
                            logger.info(f"Job {job_name} completed successfully")
                            job_completed = True
                            break
                        elif job_status.status.failed:
                            logger.error(f"Job {job_name} failed")
                            raise RuntimeError(f"Job {job_name} failed to download model")

                        time.sleep(wait_interval)
                        elapsed += wait_interval

                    except client.exceptions.ApiException as e:
                        # Job이 삭제된 경우 (404) - ttlSecondsAfterFinished에 의해 자동 삭제된 것으로 간주
                        if e.status == 404:
                            logger.info(
                                f"Job {job_name} not found (likely deleted after completion by ttlSecondsAfterFinished)"
                            )
                            job_completed = True
                            break
                        else:
                            logger.warning(f"Error checking job status: {e.status} - {e.reason}")
                            time.sleep(wait_interval)
                            elapsed += wait_interval
                    except Exception as e:
                        logger.warning(f"Error checking job status: {e}")
                        time.sleep(wait_interval)
                        elapsed += wait_interval

                if not job_completed:
                    raise RuntimeError(f"Job {job_name} did not complete within {max_wait} seconds")

                # # Job 삭제 (Job과 함께 pod도 함께 삭제됨)
                # # ttlSecondsAfterFinished가 설정되어 있어 자동 정리되지만, 즉시 정리하기 위해 명시적으로 삭제
                # try:
                #     batch_v1.delete_namespaced_job(
                #         name=job_name,
                #         namespace=namespace,
                #         propagation_policy="Foreground",  # Job과 pod를 함께 삭제
                #     )
                #     logger.info(f"Deleted Job {job_name} and associated pods")
                # except Exception as e:
                #     logger.warning(f"Failed to delete job: {e}. Job will be auto-cleaned by ttlSecondsAfterFinished")

                return json.dumps({"pvc_name": pvc_name, "status": "completed"})

            except Exception as e:
                logger.error(f"Failed to download Ollama model: {str(e)}")
                return json.dumps({"pvc_name": pvc_name, "status": "failed", "error": str(e)})

        return download_ollama_model(
            model_id=model_id,
            model_name=model_name,
            repo_id=repo_id,
            pvc_name=pvc_name,
            namespace=settings.KUBEFLOW_NAMESPACE,
            storage_size=storage_size,
        )
