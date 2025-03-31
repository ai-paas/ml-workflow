from db.models import Model, ModelRegistry
from repos.base import CRUDBase
from schemas.model import ModelBaseSchema, ModelRegistryBaseSchema
from sqlalchemy.orm import Session


class ModelRepository(CRUDBase[Model, ModelBaseSchema, ModelBaseSchema]):
    pass


class ModelRegistryRepository(CRUDBase[ModelRegistry, ModelRegistryBaseSchema, ModelRegistryBaseSchema]):
    pass


model_repository = ModelRepository(Model)
model_registry_repository = ModelRegistryRepository(ModelRegistry)
