import json
import logging
from datetime import datetime
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.security import APIKeyHeader
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
)
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
    repo_id: Annotated[str, Form()],
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
        - "yolo"가 포함된 경우 학습 가능 모델로 자동 설정됨
    - **description** (str, optional): 모델 설명
        - 모델에 대한 상세 설명
        - 생략 가능
    - **repo_id** (str, required): 모델 저장소 ID
        - HuggingFace 모델인 경우: repository ID (예: "google/owlv2-base-patch16")
        - 커스텀 모델인 경우: 모델 식별자
    - **provider_id** (int, required): 모델 제공자 ID
        - 1: huggingface
        - 2: ollama
        - 3: custom
    - **type_id** (int, required): 모델 타입 ID
        - 예: Object Detection Model, Fine-tuned Model 등
    - **format_id** (int, required): 모델 포맷 ID
        - 예: transformers, sentence-transformers, gguf, bge-m3 등
    - **parent_model_id** (int, optional): 부모 모델 ID
        - 내부 시스템에서만 사용하는 파라미터
        - 프론트엔드에서는 전달하지 않아야 함
    - **task** (str, optional): 모델 태스크
        - 모델이 수행하는 작업 유형 (예: "object-detection", "text-classification")
        - 최대 길이: 500자
        - 생략 가능
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
    - **repo_id** (str): 모델 저장소 ID
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
    - HuggingFace 모델인 경우 provider_id가 huggingface의 ID와 일치해야 합니다
    - 커스텀 모델인 경우 provider_id가 custom의 ID와 일치해야 하며, file이 필요합니다
    - 모델 이름에 "yolo"가 포함되면 자동으로 학습 가능 모델로 설정됩니다
    - **중요**: `parent_model_id`와 `model_registry_schema`는 내부 시스템에서만 사용하는 파라미터입니다. 프론트엔드에서는 이 파라미터들을 전달하지 않아야 합니다.

    ## Errors
    - 400: 유효하지 않은 요청 또는 필수 파라미터 누락
    - 401: 인증되지 않은 사용자
    - 500: 모델 등록 중 서버 내부 오류
    """

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
    model = ModelBaseSchema(
        name=name,
        description=description,
        repo_id=repo_id,
        provider_id=provider_id,
        type_id=type_id,
        format_id=format_id,
        parent_model_id=parent_model_id,
        # TODO : 추후 학습 가능 모델에 대한 구분 기준을 모델 이름에서 고정기준으로 변경 필요. ex) provider 또는 type에 yolo 추가.
        learning_enable_yn="yolo" in name.lower(),
        version=1,
        subversion=1,
        task=task,
        parameter=parameter,
        sample_code=sample_code,
    )
    custom_model_provider = ModelProviderService.get_by_name(db, "custom")
    huggingface_model_provider = ModelProviderService.get_by_name(db, "huggingface")
    try:
        if provider_id == huggingface_model_provider.id:  # HuggingFace
            result = HuggingFaceModelService().create(db, model_schema=model)
        elif provider_id == custom_model_provider.id:  # Custom
            result = CustomModelService().create(
                db, model_schema=model, model_registry_schema=model_registry_schema, file=file
            )
        else:
            print("Error has been occured!")
        return result
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
    - **repo_id** (str): 모델 저장소 ID
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
    current_user: UserSchema = Depends(get_current_user),
):
    """
    모델 목록 조회

    등록된 모델들의 목록을 페이지네이션하여 조회합니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
        - 생략 시: 전체 데이터 조회
        - 최소값: 1
    - **page_size** (int, optional): 페이지당 항목 수
        - 생략 시: 전체 데이터 조회
        - 범위: 1-1000

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

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        models = ModelService().get_multi(db, skip=0, limit=10000)
        return models

    # 페이지네이션 적용
    skip = page_size * (page - 1)

    models = ModelService().get_multi(db, skip=skip, limit=page_size)
    return models


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
