import json
import logging
from datetime import datetime
from typing import Annotated, Optional

from albumentations import Any
from config.db.connect import SessionDepends
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import APIKeyHeader
from schemas.model import ModelBaseSchema, ModelReadSchema, ModelRegistryRequestSchema
from schemas.user import UserSchema
from services.model import CustomModelService, HuggingFaceModelService, ModelService
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
    model_provider_id: Annotated[int, Form()],
    model_type_id: Annotated[int, Form()],
    model_format_id: Annotated[int, Form()],
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
        model_provider_id=model_provider_id,
        model_type_id=model_type_id,
        model_format_id=model_format_id,
    )

    try:
        if model_provider_id == 1:  # HuggingFace
            result = HuggingFaceModelService().create(db, model_schema=model)
        elif model_provider_id == 3:  # Custom
            result = CustomModelService().create(
                db, model_schema=model, model_registry_schema=model_registry_schema, file=file
            )
        else:
            print("Error has been occured!")
        return result
    except Exception as e:
        db.rollback()
        raise e


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
