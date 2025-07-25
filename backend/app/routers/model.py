import json
import logging
from datetime import datetime
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import APIKeyHeader
from schemas.model import (
    ModelBaseSchema,
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
@router.post("", response_model=ModelReadSchema)
def create_model(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    provider_id: Annotated[int, Form()],
    type_id: Annotated[int, Form()],
    format_id: Annotated[int, Form()],
    parent_model_id: Annotated[int, Form()] = None,
    model_registry_schema: Annotated[str, Form()] = None,
    file: Annotated[UploadFile, File()] = None,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Model Registry에 Model을 등록하는 API

    * params
        - model_provider
            1. huggingface
            2. ollama
            3. custom
        - model_type
            1. Object Detection Model
            4. Fine-tuned Model
        - model_format
            1. transformers
            2. sentence-transformers
            3. gguf
            4. bge-m3
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
        provider_id=provider_id,
        type_id=type_id,
        format_id=format_id,
        parent_model_id=parent_model_id,
        # TODO : 추후 학습 가능 모델에 대한 구분 기준을 모델 이름에서 고정기준으로 변경 필요. ex) provider 또는 type에 yolo 추가.
        learning_enable_yn="yolo" in name.lower(),
        version=1,
        subversion=1,
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


@router.get("/types", response_model=Optional[ModelTypeReadSchema])
def get_model_type(
    db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user), type_name: str = None
):
    if type_name:
        return ModelTypeService.get_by_name(db, type_name)
    else:
        raise HTTPException(status_code=400, detail="type_name is required")


@router.get("/formats", response_model=Optional[ModelFormatReadSchema])
def get_model_format(
    db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user), format_name: str = None
):
    if format_name:
        return ModelFormatService.get_by_name(db, format_name)
    else:
        raise HTTPException(status_code=400, detail="format_name is required")


@router.get("/providers", response_model=Optional[ModelProviderReadSchema])
def get_model_provider(
    db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user), provider_name: str = None
):
    if provider_name:
        return ModelProviderService.get_by_name(db, provider_name)
    else:
        raise HTTPException(status_code=400, detail="provider_name is required")


@router.get("/{model_id}", response_model=ModelReadSchema)
def read_model(model_id: int, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)):
    db_model = ModelService().get(db, model_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model


@router.get("", response_model=list[ModelReadSchema])
def read_models(
    skip: int = 0, limit: int = 10, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)
):
    models = ModelService().get_multi(db)
    return models


@router.get("/{model_id}/load")
def load_model(db: Session = SessionDepends, *, model_id: int, current_user: UserSchema = Depends(get_current_user)):
    db_model = ModelService().get(db, model_id)
    model_uri = db_model.model_registry.model_uri
    loaded_pipeline = ModelService.load_transformers(model_uri)
    model = loaded_pipeline.model
    # tokenizer = loaded_pipeline.tokenizer
    processor = loaded_pipeline.image_processor
    # TODO: User Login 실제로 해서 Model Load하기
    user_id = "default"
    value = {"model": model, "processor": processor}
    settings.add_user_model(user_id, value)
    return {
        "model": f"{db_model.name}",
        "user_id": user_id,
    }
