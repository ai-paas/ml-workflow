from typing import Any, Optional

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
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session


class ModelRepository(CRUDBase[Model, ModelBaseSchema, ModelBaseSchema]):
    def get_by_parent_model_id(self, db: Session, parent_model_id: int) -> list[Model]:
        """parent_model_id로 자식 모델 목록 조회"""
        return db.query(self.model).filter(self.model.parent_model_id == parent_model_id).all()

    def filter_with_visibility(
        self,
        db: Session,
        filters: dict[str, Any],
        visibility: str | None = None,
    ) -> list[Model]:
        """
        기존 컬럼 필터 + visibility 조건을 결합하여 모델을 조회한다.

        visibility SQL 변환:
            CATALOG → parent_model_id IS NULL AND opt_enable_yn = false
            CUSTOM  → parent_model_id IS NOT NULL OR opt_enable_yn = true
        """
        query = db.query(self.model)

        for attr, value in filters.items():
            if isinstance(value, list):
                query = query.filter(getattr(self.model, attr).in_(value))
            else:
                query = query.filter(getattr(self.model, attr) == value)

        if visibility == "CATALOG":
            query = query.filter(
                and_(
                    self.model.parent_model_id.is_(None),
                    self.model.opt_enable_yn == False,  # noqa: E712
                )
            )
        elif visibility == "CUSTOM":
            query = query.filter(
                or_(
                    self.model.parent_model_id.isnot(None),
                    self.model.opt_enable_yn == True,  # noqa: E712
                )
            )

        return query.all()


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
