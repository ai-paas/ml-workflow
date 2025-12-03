from db.models import Dataset, DatasetRegistry
from repos.base import CRUDBase
from schemas.dataset import DatasetBaseSchema, DatasetRegistryBaseSchema, DatasetUpdateSchema
from sqlalchemy.orm import Session


class DatasetRepository(CRUDBase[Dataset, DatasetBaseSchema, DatasetUpdateSchema]):
    pass


class DatasetRegistryRepository(CRUDBase[DatasetRegistry, DatasetRegistryBaseSchema, DatasetRegistryBaseSchema]):
    pass


dataset_repository = DatasetRepository(Dataset)
dataset_registry_repository = DatasetRegistryRepository(DatasetRegistry)
