from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from schemas.base import TimeStampSchemaMixin


class ChunkTypeReadSchema(BaseModel):
    id: int
    name: str
    description: str | None = None

    class Config:
        from_attributes = True


class LanguageReadSchema(BaseModel):
    id: int
    name: str
    description: str | None = None

    class Config:
        from_attributes = True


class SearchMethodReadSchema(BaseModel):
    id: int
    name: str
    description: str | None = None

    class Config:
        from_attributes = True


class KnowledgeBaseFileReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    object_storage_uri: str | None = None
    knowledge_base_id: int
    chunk_number: int
    partition_name: str

    class Config:
        from_attributes = True


class KnowledgeBaseBaseSchema(TimeStampSchemaMixin):
    name: str
    description: str | None = None
    embedding_model_id: int
    language_id: int
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    chunk_type_id: int
    search_method_id: int
    top_k: int
    threshold: float = Field(ge=0.0, le=1.0)


class KnowledgeBaseCreateSchema(BaseModel):
    name: str
    description: str | None = None
    language_id: int
    embedding_model_id: int
    chunk_size: int
    chunk_overlap: int
    chunk_type_id: int
    search_method_id: int
    top_k: int
    threshold: float = Field(ge=0.0, le=1.0)


class KnowledgeBaseUpdateSchema(BaseModel):
    name: str | None = None
    description: str | None = None


class KnowledgeBaseReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str | None = None
    embedding_model_id: int
    language_id: int
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    chunk_type_id: int
    search_method_id: int
    top_k: int
    threshold: float
    files: list[KnowledgeBaseFileReadSchema] = []

    class Config:
        from_attributes = True


class KnowledgeBaseBriefReadSchema(TimeStampSchemaMixin):
    id: int
    name: str
    description: str | None = None
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    threshold: float

    class Config:
        from_attributes = True


class KnowledgeBaseFileCreateSchema(BaseModel):
    name: str
    object_storage_uri: str | None = None
    knowledge_base_id: int
    chunk_number: int = 0
    partition_name: str


class KnowledgeBaseSearchRecordCreateSchema(BaseModel):
    knowledge_base_id: int
    source: str  # collection_name
    text: str  # query 내용


class KnowledgeBaseSearchRecordReadSchema(TimeStampSchemaMixin):
    id: int
    knowledge_base_id: int
    source: str
    text: str

    class Config:
        from_attributes = True


class KnowledgeBaseSearchRequestSchema(BaseModel):
    text: str  # 검색할 쿼리 텍스트


class SearchResultItemSchema(BaseModel):
    text: str
    score: float
    chunk_id: str  # Milvus의 id 필드 (큰 정수 정밀도 보존을 위해 문자열로 처리)
    partition_name: str  # 해당 데이터가 속한 파티션 이름
    file_name: str  # partition_name에 대응되는 knowledge_base_file의 파일명


class KnowledgeBaseSearchResponseSchema(BaseModel):
    results: list[SearchResultItemSchema]
    total: int
    search_method: str
