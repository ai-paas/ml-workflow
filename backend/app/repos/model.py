from db.models import Model, ModelRegistry
from db.models.model import ModelFormat, ModelProvider
from repos.base import CRUDBase
from schemas.model import (
    ModelBaseSchema,
    ModelFormatCreateUpdateSchema,
    ModelProviderCreateUpdateSchema,
    ModelRegistryBaseSchema,
)
from sqlalchemy.orm import Session


class ModelRepository(CRUDBase[Model, ModelBaseSchema, ModelBaseSchema]):
    pass


class ModelRegistryRepository(CRUDBase[ModelRegistry, ModelRegistryBaseSchema, ModelRegistryBaseSchema]):
    pass


class ModelProviderRepository(
    CRUDBase[ModelProvider, ModelProviderCreateUpdateSchema, ModelProviderCreateUpdateSchema]
):
    def get_by_name(self, db: Session, name: str) -> ModelProvider:
        return db.query(self.model).filter(self.model.name == name).first()


class ModelFormatRepository(CRUDBase[ModelFormat, ModelFormatCreateUpdateSchema, ModelFormatCreateUpdateSchema]):
    def get_by_name(self, db: Session, name: str) -> ModelFormat:
        return db.query(self.model).filter(self.model.name == name).first()


model_repository = ModelRepository(Model)
model_registry_repository = ModelRegistryRepository(ModelRegistry)
model_provider_repository = ModelProviderRepository(ModelProvider)
model_format_repository = ModelFormatRepository(ModelFormat)
