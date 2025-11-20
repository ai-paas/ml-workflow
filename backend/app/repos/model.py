from typing import Optional

from db.models import Model, ModelRegistry
from db.models.model import ModelFormat, ModelProvider, ModelType
from repos.base import CRUDBase
from schemas.model import (
    ModelBaseSchema,
    ModelFormatCreateUpdateSchema,
    ModelProviderCreateUpdateSchema,
    ModelRegistryBaseSchema,
    ModelTypeCreateUpdateSchema,
)
from sqlalchemy.orm import Session


class ModelRepository(CRUDBase[Model, ModelBaseSchema, ModelBaseSchema]):
    def get_by_parent_model_id(self, db: Session, parent_model_id: int) -> list[Model]:
        """parent_model_id로 자식 모델 목록 조회"""
        return db.query(self.model).filter(self.model.parent_model_id == parent_model_id).all()


class ModelRegistryRepository(CRUDBase[ModelRegistry, ModelRegistryBaseSchema, ModelRegistryBaseSchema]):
    pass


class ModelProviderRepository(
    CRUDBase[ModelProvider, ModelProviderCreateUpdateSchema, ModelProviderCreateUpdateSchema]
):
    def get_by_name(self, db: Session, name: str) -> Optional[ModelProvider]:
        return db.query(self.model).filter(self.model.name == name).first()


class ModelFormatRepository(CRUDBase[ModelFormat, ModelFormatCreateUpdateSchema, ModelFormatCreateUpdateSchema]):
    def get_by_name(self, db: Session, name: str) -> Optional[ModelFormat]:
        return db.query(self.model).filter(self.model.name == name).first()


class ModelTypeRepository(CRUDBase[ModelType, ModelTypeCreateUpdateSchema, ModelTypeCreateUpdateSchema]):
    def get_by_name(self, db: Session, name: str) -> Optional[ModelType]:
        return db.query(self.model).filter(self.model.name == name).first()


model_repository = ModelRepository(Model)
model_registry_repository = ModelRegistryRepository(ModelRegistry)
model_provider_repository = ModelProviderRepository(ModelProvider)
model_format_repository = ModelFormatRepository(ModelFormat)
model_type_repository = ModelTypeRepository(ModelType)
