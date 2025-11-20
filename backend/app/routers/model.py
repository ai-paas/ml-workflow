import json
import logging
from datetime import datetime
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from config.settings import get_settings
from db.models.model import ModelTaskType
from db.models.model_base_deployment import BaseDeploymentStatus
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.security import APIKeyHeader
from repos.model_base_deployment import model_base_deployment_repository
from schemas.model import (
    ModelBaseSchema,
    ModelBriefReadSchema,
    ModelFormatReadSchema,
    ModelProviderReadSchema,
    ModelReadSchema,
    ModelRegistryRequestSchema,
    ModelTypeReadSchema,
)
from schemas.user import UserSchema
from services.model import (
    CustomModelService,
    HuggingFaceModelService,
    ModelFormatService,
    ModelProviderService,
    ModelService,
    ModelTypeService,
    OllamaModelService,
    is_yolox_local_model,
    is_yolox_remote_model,
)
from services.model_base_deployment import ModelBaseDeploymentService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])

settings = get_settings()
# API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


# TODO: 책임 분리 필요.
@router.post("", response_model=ModelBriefReadSchema)
def create_model(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = None,
    repo_id: Annotated[str, Form()] = None,
    provider_id: Annotated[int, Form()],
    type_id: Annotated[int, Form()],
    format_id: Annotated[int, Form()],
    parent_model_id: Annotated[int, Form()] = None,
    task: Annotated[str, Form()] = None,
    parameter: Annotated[str, Form()] = None,
    sample_code: Annotated[str, Form()] = None,
    model_registry_schema: Annotated[str, Form()] = None,
    file: Annotated[UploadFile, File()] = None,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    모델 등록

    Model Registry에 모델을 등록합니다.
    제공자(provider)에 따라 HuggingFace 모델 또는 커스텀 모델로 등록됩니다.

    ## Request Body (multipart/form-data)
    - **name** (str, required): 모델 이름
        - 모델을 식별하기 위한 이름
        - YOLOX 모델인 경우 학습 가능 모델로 자동 설정됨
    - **description** (str, optional): 모델 설명
        - 모델에 대한 상세 설명
        - 생략 가능
    - **repo_id** (str, optional): 모델 저장소 ID
        - HuggingFace 모델인 경우: repository ID (예: "google/owlv2-base-patch16")
        - 커스텀 모델인 경우: 모델 식별자
        - 생략 가능 (null 허용)
    - **provider_id** (int, required): 모델 제공자 ID
        - **중요**: `GET /api/v1/models/providers` API로 제공자 목록을 조회한 후, 해당 제공자의 `id` 값을 사용해야 합니다
        - 예: `GET /api/v1/models/providers?provider_name=huggingface`로 조회하여 반환된 `id` 값 사용
        - 하드코딩된 숫자 값(1, 2, 3 등)을 사용하지 마세요
    - **type_id** (int, required): 모델 타입 ID
        - **중요**: `GET /api/v1/models/types` API로 타입 목록을 조회한 후, 해당 타입의 `id` 값을 사용해야 합니다
        - 예: `GET /api/v1/models/types?type_name=ODM`로 조회하여 반환된 `id` 값 사용
        - 하드코딩된 숫자 값을 사용하지 마세요
    - **format_id** (int, required): 모델 포맷 ID
        - **중요**: `GET /api/v1/models/formats` API로 포맷 목록을 조회한 후, 해당 포맷의 `id` 값을 사용해야 합니다
        - 예: `GET /api/v1/models/formats?format_name=transformers`로 조회하여 반환된 `id` 값 사용
        - 하드코딩된 숫자 값을 사용하지 마세요
    - **parent_model_id** (int, optional): 부모 모델 ID
        - 내부 시스템에서만 사용하는 파라미터
        - 프론트엔드에서는 전달하지 않아야 함
    - **task** (str, optional): 모델 태스크
        - 모델이 수행하는 작업 유형
        - 허용 값: "embedding", "text-generation", "object-detection" 중 하나만 가능
        - 생략 가능 (null 허용)
        - 다른 값 입력 시 422 에러 발생
    - **parameter** (str, optional): 모델 파라미터
        - 모델 관련 파라미터 정보
        - 최대 길이: 100자
        - 생략 가능
    - **sample_code** (str, optional): 샘플 코드
        - 모델 사용 예제 코드
        - 생략 가능
    - **file** (UploadFile, optional): 모델 파일
        - 커스텀 모델인 경우 업로드할 모델 파일
        - 생략 가능
    - **model_registry_schema** (str, optional): 모델 레지스트리 스키마 (JSON 문자열)
        - 내부 시스템에서만 사용하는 파라미터
        - 프론트엔드에서는 전달하지 않아야 함

    ## Response (ModelBriefReadSchema)
    - **id** (int): 모델 고유 ID
    - **name** (str): 모델 이름
    - **description** (str, optional): 모델 설명
    - **repo_id** (str, optional): 모델 저장소 ID
    - **provider_info** (ModelProviderReadSchema): 모델 제공자 정보
        - id (int): 제공자 ID
        - name (str): 제공자 이름
        - description (str): 제공자 설명
    - **type_info** (ModelTypeReadSchema): 모델 타입 정보
        - id (int): 타입 ID
        - name (str): 타입 이름
        - description (str): 타입 설명
    - **format_info** (ModelFormatReadSchema): 모델 포맷 정보
        - id (int): 포맷 ID
        - name (str): 포맷 이름
        - description (str): 포맷 설명
    - **parent_model_id** (int, optional): 부모 모델 ID
    - **task** (str, optional): 모델 태스크
    - **parameter** (str, optional): 모델 파라미터
    - **sample_code** (str, optional): 샘플 코드
    - **registry** (ModelRegistryReadSchema): 모델 레지스트리 정보
        - id (int): 레지스트리 ID
        - artifact_path (str): 아티팩트 경로
        - uri (str): 모델 URI
        - run_id (str, optional): MLflow 실행 ID
        - reference_model_id (int): 참조 모델 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각
    - **created_at** (datetime): 모델 생성 시각
    - **updated_at** (datetime): 모델 수정 시각

    ## Notes
    - **ID 값 조회**: `provider_id`, `type_id`, `format_id`는 각각 해당하는 조회 API(`/providers`, `/types`, `/formats`)
      를 먼저 호출하여 ID 값을 확인한 후 사용해야 합니다
    - HuggingFace 모델인 경우 provider_id가 huggingface의 ID와 일치해야 합니다
    - 커스텀 모델인 경우 provider_id가 custom의 ID와 일치해야 하며, file이 필요합니다
    - YOLOX 모델인 경우에만 자동으로 학습 가능 모델(learning_enable_yn=True)로 설정됩니다
    - **중요**: `parent_model_id`와 `model_registry_schema`는 내부 시스템에서만 사용하는 파라미터입니다. 프론트엔드에서는 이 파라미터들을 전달하지 않아야 합니다.

    ## Errors
    - 400: 유효하지 않은 요청 또는 필수 파라미터 누락
    - 401: 인증되지 않은 사용자
    - 500: 모델 등록 중 서버 내부 오류
    """

    # task 값 검증 (Enum 사용)
    if task is not None:
        try:
            # 문자열을 Enum으로 변환
            task_enum = ModelTaskType(task)
            task = task_enum.value  # Enum 값을 문자열로 변환하여 저장
        except ValueError:
            valid_tasks = [e.value for e in ModelTaskType]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"task는 다음 값 중 하나여야 합니다: {', '.join(valid_tasks)}. 입력된 값: {task}",
            )

    try:
        if model_registry_schema:
            model_registry_schema_json = json.loads(model_registry_schema)
            logger.info(model_registry_schema_json)
            model_registry_schema = ModelRegistryRequestSchema(**model_registry_schema_json)
            logger.info(model_registry_schema)
    except Exception as e:
        logger.error(f"model_registry schema error : {e}")
        logger.error(f"model_registry_schema = {model_registry_schema}")
        logger.error(f"model_registry_schema_json = {model_registry_schema_json}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="model_registry_schema is unprocessable entity"
        )
    # YOLOX 모델인지 확인 (repo_id 또는 name으로 확인)
    is_yolox = False
    if repo_id:
        is_yolox = is_yolox_remote_model(repo_id)
    if not is_yolox:
        is_yolox = is_yolox_local_model(name)

    # YOLOX 모델인 경우에만 learning_enable_yn을 true로 설정
    learning_enable_yn = is_yolox

    model = ModelBaseSchema(
        name=name,
        description=description,
        repo_id=repo_id,
        provider_id=provider_id,
        type_id=type_id,
        format_id=format_id,
        parent_model_id=parent_model_id,
        learning_enable_yn=learning_enable_yn,
        version=1,
        subversion=1,
        task=task,
        parameter=parameter,
        sample_code=sample_code,
    )
    custom_model_provider = ModelProviderService.get_by_name(db, "custom")
    huggingface_model_provider = ModelProviderService.get_by_name(db, "huggingface")
    ollama_model_provider = ModelProviderService.get_by_name(db, "ollama")
    gguf_format = ModelFormatService.get_by_name(db, "gguf")
    embedding_type = ModelTypeService.get_by_name(db, "Embedding")

    try:
        # Ollama + GGUF인 경우: 단순히 meta 정보만 DB에 등록
        if (
            ollama_model_provider
            and gguf_format
            and provider_id == ollama_model_provider.id
            and format_id == gguf_format.id
        ):
            if not repo_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="repo_id is required for Ollama models"
                )

            # 모델과 레지스트리 생성 (MLflow 없이)
            from repos.model import model_registry_repository, model_repository
            from schemas.model import ModelRegistryBaseSchema

            model_obj = model_repository.create(db, obj_in=model)
            model_id = model_obj.id

            # Ollama 모델을 PVC에 다운로드하는 파이프라인 실행
            pvc_name = None
            try:
                sanitized_model_name = name.replace("/", "-")
                pvc_name = OllamaModelService.download_ollama_model_to_pvc(
                    db=db,
                    model_id=model_id,
                    model_name=sanitized_model_name,
                    repo_id=repo_id,
                )
                logger.info(f"Started Ollama model download pipeline for model_id: {model_id}, PVC: {pvc_name}")
            except Exception as download_error:
                logger.error(f"Failed to start Ollama model download pipeline: {download_error}")
                # 다운로드 실패해도 모델 등록은 진행 (비동기이므로)

            # ModelRegistry 생성 (uri는 repo_id 사용, pvc는 다운로드 파이프라인에서 생성된 PVC 이름)
            model_registry_repository.create(
                db,
                obj_in=ModelRegistryBaseSchema(
                    artifact_path="",  # Ollama는 artifact_path 불필요
                    uri=repo_id,  # repo_id를 uri로 사용 (예: ahmgam/medllama3-v20)
                    reference_model_id=model_id,
                    run_id=None,  # MLflow run_id 없음
                    pvc=pvc_name,  # PVC 이름 저장
                ),
            )
            db.commit()

            # Embedding 타입이고 Ollama인 경우 자동 배포
            if embedding_type and type_id == embedding_type.id:
                logger.info(f"Auto-deploying Ollama embedding model: {model_id} (repo_id: {repo_id})")
                try:
                    sanitized_model_name = name.replace("/", "-")
                    ModelBaseDeploymentService.deploy_ollama_embedding_model(
                        db=db,
                        model_id=model_id,
                        model_name=sanitized_model_name,
                        repo_id=repo_id,
                        gpu_enabled=False,  # 기본값, 필요시 파라미터로 받을 수 있음
                    )
                    logger.info(f"Successfully initiated deployment for embedding model: {model_id}")
                except Exception as deploy_error:
                    logger.error(f"Failed to deploy embedding model {model_id}: {deploy_error}")
                    # 배포 실패해도 모델 등록은 성공으로 처리 (비동기 배포이므로)

            return model_repository.get(db, model_id)
        elif provider_id == huggingface_model_provider.id:  # HuggingFace
            if not repo_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="repo_id is required for HuggingFace models"
                )
            return HuggingFaceModelService().create(db, model_schema=model)
        elif provider_id == custom_model_provider.id:  # Custom
            return CustomModelService().create(
                db, model_schema=model, model_registry_schema=model_registry_schema, file=file
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported provider_id: {provider_id}"
            )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise e


@router.get("/types", response_model=list[ModelTypeReadSchema] | ModelTypeReadSchema)
def get_model_type(
    db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user), type_name: str = None
):
    """
    모델 타입 조회

    이름으로 특정 모델 타입을 조회하거나, 전체 모델 타입 목록을 조회합니다.

    ## Query Parameters
    - **type_name** (str, optional): 조회할 모델 타입 이름
        - 제공 시: 해당 이름의 모델 타입만 반환
        - 생략 시: 전체 모델 타입 목록 반환
        - 예: "Object Detection Model", "Fine-tuned Model" 등

    ## Response
    - **type_name 제공 시** (ModelTypeReadSchema): 단일 모델 타입 정보
        - id (int): 모델 타입 ID
        - name (str): 모델 타입 이름
        - description (str): 모델 타입 설명
    - **type_name 생략 시** (List[ModelTypeReadSchema]): 전체 모델 타입 목록
        - 각 항목은 ModelTypeReadSchema 형식

    ## Notes
    - type_name을 생략하면 전체 모델 타입 목록을 조회할 수 있습니다
    - type_name을 제공하면 해당 이름의 모델 타입만 조회합니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 해당 이름의 모델 타입을 찾을 수 없음 (type_name 제공 시)
    - 500: 서버 내부 오류
    """
    if type_name:
        result = ModelTypeService.get_by_name(db, type_name)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Model type '{type_name}' not found")
        return result
    else:
        return ModelTypeService.get_all(db)


@router.get("/formats", response_model=list[ModelFormatReadSchema] | ModelFormatReadSchema)
def get_model_format(
    db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user), format_name: str = None
):
    """
    모델 포맷 조회

    이름으로 특정 모델 포맷을 조회하거나, 전체 모델 포맷 목록을 조회합니다.

    ## Query Parameters
    - **format_name** (str, optional): 조회할 모델 포맷 이름
        - 제공 시: 해당 이름의 모델 포맷만 반환
        - 생략 시: 전체 모델 포맷 목록 반환
        - 예: "transformers", "sentence-transformers", "gguf", "bge-m3" 등

    ## Response
    - **format_name 제공 시** (ModelFormatReadSchema): 단일 모델 포맷 정보
        - id (int): 모델 포맷 ID
        - name (str): 모델 포맷 이름
        - description (str): 모델 포맷 설명
    - **format_name 생략 시** (List[ModelFormatReadSchema]): 전체 모델 포맷 목록
        - 각 항목은 ModelFormatReadSchema 형식

    ## Notes
    - format_name을 생략하면 전체 모델 포맷 목록을 조회할 수 있습니다
    - format_name을 제공하면 해당 이름의 모델 포맷만 조회합니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 해당 이름의 모델 포맷을 찾을 수 없음 (format_name 제공 시)
    - 500: 서버 내부 오류
    """
    if format_name:
        result = ModelFormatService.get_by_name(db, format_name)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Model format '{format_name}' not found")
        return result
    else:
        return ModelFormatService.get_all(db)


@router.get("/providers", response_model=list[ModelProviderReadSchema] | ModelProviderReadSchema)
def get_model_provider(
    db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user), provider_name: str = None
):
    """
    모델 제공자 조회

    이름으로 특정 모델 제공자를 조회하거나, 전체 모델 제공자 목록을 조회합니다.

    ## Query Parameters
    - **provider_name** (str, optional): 조회할 모델 제공자 이름
        - 제공 시: 해당 이름의 모델 제공자만 반환
        - 생략 시: 전체 모델 제공자 목록 반환
        - 예: "huggingface", "ollama", "custom" 등

    ## Response
    - **provider_name 제공 시** (ModelProviderReadSchema): 단일 모델 제공자 정보
        - id (int): 모델 제공자 ID
        - name (str): 모델 제공자 이름
        - description (str): 모델 제공자 설명
    - **provider_name 생략 시** (List[ModelProviderReadSchema]): 전체 모델 제공자 목록
        - 각 항목은 ModelProviderReadSchema 형식

    ## Notes
    - provider_name을 생략하면 전체 모델 제공자 목록을 조회할 수 있습니다
    - provider_name을 제공하면 해당 이름의 모델 제공자만 조회합니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 해당 이름의 모델 제공자를 찾을 수 없음 (provider_name 제공 시)
    - 500: 서버 내부 오류
    """
    if provider_name:
        result = ModelProviderService.get_by_name(db, provider_name)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Model provider '{provider_name}' not found")
        return result
    else:
        return ModelProviderService.get_all(db)


@router.get("/{model_id}", response_model=ModelReadSchema)
def read_model(model_id: int, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)):
    """
    모델 상세정보 조회

    특정 모델의 상세 정보를 조회합니다.
    제공자, 타입, 포맷 정보와 부모/자식 모델 관계를 포함하여 반환합니다.

    ## Path Parameters
    - **model_id** (int): 조회할 모델 ID

    ## Response (ModelReadSchema)
    - **id** (int): 모델 고유 ID
    - **name** (str): 모델 이름
    - **description** (str, optional): 모델 설명
    - **repo_id** (str, optional): 모델 저장소 ID
    - **provider_info** (ModelProviderReadSchema): 모델 제공자 정보
        - id (int): 제공자 ID
        - name (str): 제공자 이름
        - description (str): 제공자 설명
    - **type_info** (ModelTypeReadSchema): 모델 타입 정보
        - id (int): 타입 ID
        - name (str): 타입 이름
        - description (str): 타입 설명
    - **format_info** (ModelFormatReadSchema): 모델 포맷 정보
        - id (int): 포맷 ID
        - name (str): 포맷 이름
        - description (str): 포맷 설명
    - **parent_model_id** (int, optional): 부모 모델 ID
        - 파인튜닝된 모델인 경우 원본 모델 ID
    - **task** (str, optional): 모델 태스크
    - **parameter** (str, optional): 모델 파라미터
    - **sample_code** (str, optional): 샘플 코드
    - **registry** (ModelRegistryReadSchema): 모델 레지스트리 정보
        - id (int): 레지스트리 ID
        - artifact_path (str): 아티팩트 경로
        - uri (str): 모델 URI
        - run_id (str, optional): MLflow 실행 ID
        - reference_model_id (int): 참조 모델 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각
    - **parent_model** (ModelReadParentSchema, optional): 부모 모델 정보
        - id (int): 부모 모델 ID
        - name (str): 부모 모델 이름
        - description (str): 부모 모델 설명
        - parent_model (ModelReadParentSchema, optional): 상위 부모 모델 (재귀적)
    - **child_models** (List[ModelReadChildSchema], optional): 자식 모델 목록
        - id (int): 자식 모델 ID
        - name (str): 자식 모델 이름
        - description (str): 자식 모델 설명
        - child_models (List[ModelReadChildSchema], optional): 하위 자식 모델 (재귀적)
    - **created_at** (datetime): 모델 생성 시각
    - **updated_at** (datetime): 모델 수정 시각

    ## Notes
    - 모델의 모든 관련 정보(제공자, 타입, 포맷, 레지스트리)를 포함하여 반환합니다
    - 부모/자식 모델 관계는 재귀적으로 조회됩니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 모델을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    db_model = ModelService().get(db, model_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model


@router.get("", response_model=list[ModelBriefReadSchema])
def read_models(
    *,
    db: Session = SessionDepends,
    page_size: Optional[int] = Query(
        default=None,
        description="페이지 사이즈",
        examples=[10, 20, 30],
        ge=1,
        le=1000,
    ),
    page: Optional[int] = Query(
        default=None,
        description="페이지 번호",
        examples=[1, 2, 3],
        ge=1,
    ),
    model_type_id: Optional[int] = Query(
        default=None,
        description="모델 타입 ID로 필터링",
    ),
    model_provider_id: Optional[int] = Query(
        default=None,
        description="모델 제공자 ID로 필터링",
    ),
    model_format_id: Optional[int] = Query(
        default=None,
        description="모델 포맷 ID로 필터링",
    ),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    모델 목록 조회

    등록된 모델들의 목록을 페이지네이션하여 조회합니다.
    model_type_id, model_provider_id, model_format_id를 사용하여 필터링할 수 있습니다.
    특히 embedding 모델만 조회하려면 model_type_id에 Embedding 타입의 ID를 사용하세요.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
        - 생략 시: 전체 데이터 조회
        - 최소값: 1
    - **page_size** (int, optional): 페이지당 항목 수
        - 생략 시: 전체 데이터 조회
        - 범위: 1-1000
    - **model_type_id** (int, optional): 모델 타입 ID로 필터링
        - 예: Embedding 모델만 조회하려면 Embedding 타입의 ID 사용
        - `GET /api/v1/models/types` API로 타입 목록 조회 가능
    - **model_provider_id** (int, optional): 모델 제공자 ID로 필터링
        - `GET /api/v1/models/providers` API로 제공자 목록 조회 가능
    - **model_format_id** (int, optional): 모델 포맷 ID로 필터링
        - `GET /api/v1/models/formats` API로 포맷 목록 조회 가능

    ## Response (List[ModelBriefReadSchema])
    - **items** (List[ModelBriefReadSchema]): 모델 목록
        각 항목은 다음 정보를 포함:
        - id (int): 모델 고유 ID
        - name (str): 모델 이름
        - description (str, optional): 모델 설명
        - repo_id (str): 모델 저장소 ID
        - provider_info (ModelProviderReadSchema): 모델 제공자 정보
            - id (int): 제공자 ID
            - name (str): 제공자 이름
            - description (str): 제공자 설명
        - type_info (ModelTypeReadSchema): 모델 타입 정보
            - id (int): 타입 ID
            - name (str): 타입 이름
            - description (str): 타입 설명
        - format_info (ModelFormatReadSchema): 모델 포맷 정보
            - id (int): 포맷 ID
            - name (str): 포맷 이름
            - description (str): 포맷 설명
        - parent_model_id (int, optional): 부모 모델 ID
        - task (str, optional): 모델 태스크
        - parameter (str, optional): 모델 파라미터
        - sample_code (str, optional): 샘플 코드
        - registry (ModelRegistryReadSchema): 모델 레지스트리 정보
            - id (int): 레지스트리 ID
            - artifact_path (str): 아티팩트 경로
            - uri (str): 모델 URI
            - run_id (str, optional): MLflow 실행 ID
            - reference_model_id (int): 참조 모델 ID
            - created_at (datetime): 생성 시각
            - updated_at (datetime): 수정 시각
        - created_at (datetime): 모델 생성 시각
        - updated_at (datetime): 모델 수정 시각

    ## Notes
    - page와 page_size를 모두 생략하면 전체 데이터를 조회 (최대 10000개)
    - page와 page_size 중 하나라도 생략하면 전체 데이터를 조회합니다
    - 페이지네이션 사용 시 page와 page_size를 모두 제공해야 합니다
    - 필터링 파라미터(model_type_id, model_provider_id, model_format_id)는 함께 사용할 수 있습니다
    - Embedding 모델만 조회하려면 model_type_id에 Embedding 타입의 ID를 사용하세요

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    # 필터링 조건 구성
    filters = {}
    if model_type_id is not None:
        filters["type_id"] = model_type_id
    if model_provider_id is not None:
        filters["provider_id"] = model_provider_id
    if model_format_id is not None:
        filters["format_id"] = model_format_id

    # 필터가 있는 경우 filter 메서드 사용, 없는 경우 get_multi 사용
    if filters:
        # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
        if page is None or page_size is None:
            return ModelService().filter_all(db, filters=filters)
        # 페이지네이션 적용
        skip = page_size * (page - 1)
        return ModelService().filter(db, filters=filters, skip=skip, limit=page_size)
    else:
        # 필터가 없는 경우 기존 로직 사용
        if page is None or page_size is None:
            return ModelService().get_multi(db, skip=0, limit=10000)
        skip = page_size * (page - 1)
        return ModelService().get_multi(db, skip=skip, limit=page_size)


@router.delete("/{model_id}")
def delete_model(model_id: int, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)):
    """
    모델 삭제

    모델을 삭제합니다. 참조 관계를 확인하여 안전하게 삭제합니다.
    다른 엔티티에서 참조되고 있는 경우 삭제할 수 없습니다.

    ## Path Parameters
    - **model_id** (int): 삭제할 모델 ID

    ## Response
    - **success** (bool): 삭제 성공 여부
    - **message** (str): 삭제 결과 메시지

    ## Notes
    - 모델이 다음 엔티티에서 참조되고 있으면 삭제할 수 없습니다:
        - Experiment (실험)
        - WorkflowComponent (워크플로우 컴포넌트)
        - 다른 모델의 parent_model (자식 모델)
    - 참조 관계가 있는 경우 400 에러를 반환합니다

    ## Errors
    - 400: 모델이 다른 엔티티에서 참조되고 있어 삭제할 수 없음
    - 401: 인증되지 않은 사용자
    - 404: 모델을 찾을 수 없음
    - 500: 모델 삭제 중 서버 내부 오류
    """
    try:
        result = ModelService().delete(db, model_id)
        return {"success": result, "message": "모델이 성공적으로 삭제되었습니다."}
    except RuntimeError as e:
        # 참조 관계 때문에 삭제할 수 없는 경우
        if "참조되고 있습니다" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        # 기타 런타임 에러
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        # 모델을 찾을 수 없는 경우
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # 예상치 못한 에러
        raise HTTPException(status_code=500, detail=f"모델 삭제 중 오류가 발생했습니다: {str(e)}")


@router.put("/base-deployments/{model_id}/status")
def update_model_base_deployment_status(
    *,
    db: Session = SessionDepends,
    model_id: int,
    service_name: str = Body(...),
    service_hostname: str = Body(...),
    internal_url: Optional[str] = Body(None),
    status: str = Body(...),
    error_message: Optional[str] = Body(None),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    모델 기본 배포 상태 업데이트 (백엔드 서버 내부 전용 API)

    **⚠️ 경고: 이 API는 백엔드 서버 내부에서만 사용하는 내부 API입니다.**
    **프론트엔드나 외부 클라이언트에서 직접 호출해서는 안 됩니다.**

    Kubeflow 파이프라인 컴포넌트에서 배포 상태를 업데이트하기 위한 엔드포인트입니다.
    파이프라인 컴포넌트 내부에서 인증 토큰을 발급받아 사용합니다.

    ## 사용 목적
    - Kubeflow 파이프라인 컴포넌트에서 모델 배포 상태를 DB에 업데이트하기 위해 사용
    - 백엔드 서버 내부 시스템 간 통신용으로만 사용

    ## Path Parameters
    - **model_id** (int): 모델 ID

    ## Request Body
    - **service_name** (str): 서비스 이름
    - **service_hostname** (str): 서비스 호스트명
    - **internal_url** (str, optional): 내부 접근 URL
    - **status** (str): 배포 상태 ("deployed", "deploying", "failed")
    - **error_message** (str, optional): 오류 메시지 (실패 시)

    ## Response
    - **success** (bool): 업데이트 성공 여부
    - **message** (str): 결과 메시지

    ## Notes
    - **내부 API**: 백엔드 서버 내부에서만 사용하는 API입니다
    - **호출 주체**: Kubeflow 파이프라인 컴포넌트에서만 호출됩니다
    - **인증**: 인증 토큰이 필요하며, 파이프라인 컴포넌트에서 자동으로 발급받아 사용합니다
    - **프론트엔드 사용 금지**: 프론트엔드나 외부 클라이언트에서 직접 호출하지 마세요

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 배포 정보를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    logger.info(f"Updating deployment status for model_id: {model_id}, status: {status}")
    try:
        # 배포 정보 조회
        deployment = model_base_deployment_repository.get_by_model_id(db, model_id)
        if not deployment:
            raise HTTPException(status_code=404, detail=f"Deployment not found for model_id: {model_id}")

        # 상태 문자열을 BaseDeploymentStatus로 변환
        status_map = {
            "deployed": BaseDeploymentStatus.DEPLOYED,
            "deploying": BaseDeploymentStatus.DEPLOYING,
            "failed": BaseDeploymentStatus.FAILED,
        }

        deployment_status = status_map.get(status.lower())
        if not deployment_status:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        # 배포 정보 업데이트
        deployment.service_name = service_name
        deployment.service_hostname = service_hostname
        if internal_url:
            deployment.internal_url = internal_url

        # 상태 업데이트
        model_base_deployment_repository.update_status(db, deployment, deployment_status, error_message=error_message)

        return {"success": True, "message": "Deployment status updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update deployment status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update deployment status: {str(e)}")
