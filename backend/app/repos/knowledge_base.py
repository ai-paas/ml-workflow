from typing import Optional

from db.models.knowledge_base import (
    ChunkType,
    KnowledgeBase,
    KnowledgeBaseFile,
    KnowledgeBaseSearchRecord,
    Language,
    SearchMethod,
)
from repos.base import CRUDBase
from schemas.knowledge_base import (
    KnowledgeBaseBaseSchema,
    KnowledgeBaseCreateSchema,
    KnowledgeBaseFileCreateSchema,
    KnowledgeBaseSearchRecordCreateSchema,
    KnowledgeBaseUpdateSchema,
)
from sqlalchemy.orm import Session


class KnowledgeBaseRepository(CRUDBase[KnowledgeBase, KnowledgeBaseBaseSchema, KnowledgeBaseUpdateSchema]):
    def get_by_collection_name(self, db: Session, collection_name: str) -> Optional[KnowledgeBase]:
        return db.query(self.model).filter(self.model.collection_name == collection_name).first()


class KnowledgeBaseFileRepository(
    CRUDBase[KnowledgeBaseFile, KnowledgeBaseFileCreateSchema, KnowledgeBaseFileCreateSchema]
):
    def get_by_knowledge_base_id(self, db: Session, knowledge_base_id: int) -> list[KnowledgeBaseFile]:
        return db.query(self.model).filter(self.model.knowledge_base_id == knowledge_base_id).all()

    def get_by_partition_name(
        self, db: Session, knowledge_base_id: int, partition_name: str
    ) -> Optional[KnowledgeBaseFile]:
        return (
            db.query(self.model)
            .filter(self.model.knowledge_base_id == knowledge_base_id, self.model.partition_name == partition_name)
            .first()
        )


class ChunkTypeRepository(CRUDBase[ChunkType, None, None]):
    def get_by_name(self, db: Session, name: str) -> Optional[ChunkType]:
        return db.query(self.model).filter(self.model.name == name).first()

    def get_all(self, db: Session) -> list[ChunkType]:
        return db.query(self.model).all()


class LanguageRepository(CRUDBase[Language, None, None]):
    def get_by_name(self, db: Session, name: str) -> Optional[Language]:
        return db.query(self.model).filter(self.model.name == name).first()

    def get_all(self, db: Session) -> list[Language]:
        return db.query(self.model).all()


class SearchMethodRepository(CRUDBase[SearchMethod, None, None]):
    def get_by_name(self, db: Session, name: str) -> Optional[SearchMethod]:
        return db.query(self.model).filter(self.model.name == name).first()

    def get_all(self, db: Session) -> list[SearchMethod]:
        return db.query(self.model).all()


class KnowledgeBaseSearchRecordRepository(
    CRUDBase[KnowledgeBaseSearchRecord, KnowledgeBaseSearchRecordCreateSchema, None]
):
    def get_by_knowledge_base_id(self, db: Session, knowledge_base_id: int) -> list[KnowledgeBaseSearchRecord]:
        return db.query(self.model).filter(self.model.knowledge_base_id == knowledge_base_id).all()


knowledge_base_repository = KnowledgeBaseRepository(KnowledgeBase)
knowledge_base_file_repository = KnowledgeBaseFileRepository(KnowledgeBaseFile)
knowledge_base_search_record_repository = KnowledgeBaseSearchRecordRepository(KnowledgeBaseSearchRecord)
chunk_type_repository = ChunkTypeRepository(ChunkType)
language_repository = LanguageRepository(Language)
search_method_repository = SearchMethodRepository(SearchMethod)
