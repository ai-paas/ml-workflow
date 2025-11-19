from typing import Optional

import sqlalchemy as sa
from db.models.base import BaseModel, TimestampMixin
from db.models.model import Model
from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class ChunkType(BaseModel):
    __tablename__ = "chunk_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)


class Language(BaseModel):
    __tablename__ = "language"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(10), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)


class SearchMethod(BaseModel):
    __tablename__ = "search_method"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)


class KnowledgeBase(BaseModel, TimestampMixin):
    __tablename__ = "knowledge_base"
    __table_args__ = (
        Index("ix_knowledge_base_collection_name", "collection_name"),
        Index("ix_knowledge_base_embedding_model_id", "embedding_model_id"),
        Index("ix_knowledge_base_language_id", "language_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_model_id: Mapped[int] = mapped_column(ForeignKey("model.id"), nullable=False)
    language_id: Mapped[int] = mapped_column(ForeignKey("language.id"), nullable=False)
    collection_name: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type_id: Mapped[int] = mapped_column(ForeignKey("chunk_type.id"), nullable=False)
    search_method_id: Mapped[int] = mapped_column(ForeignKey("search_method.id"), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)

    embedding_model: Mapped["Model"] = relationship("Model")
    language: Mapped["Language"] = relationship("Language")
    chunk_type: Mapped["ChunkType"] = relationship("ChunkType")
    search_method: Mapped["SearchMethod"] = relationship("SearchMethod")
    files: Mapped[list["KnowledgeBaseFile"]] = relationship(
        "KnowledgeBaseFile", back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class KnowledgeBaseFile(BaseModel, TimestampMixin):
    __tablename__ = "knowledge_base_file"
    __table_args__ = (Index("ix_knowledge_base_file_knowledge_base_id", "knowledge_base_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    object_storage_uri: Mapped[Optional[str]] = mapped_column(String(4000), nullable=True)
    knowledge_base_id: Mapped[int] = mapped_column(ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False)
    chunk_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    partition_name: Mapped[str] = mapped_column(String(500), nullable=False)

    knowledge_base: Mapped["KnowledgeBase"] = relationship("KnowledgeBase", back_populates="files")


class KnowledgeBaseSearchRecord(BaseModel):
    __tablename__ = "knowledge_base_search_test_record"
    __table_args__ = (Index("ix_knowledge_base_search_test_record_knowledge_base_id", "knowledge_base_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_base_id: Mapped[int] = mapped_column(ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)  # collection_name
    text: Mapped[str] = mapped_column(Text, nullable=False)  # query 내용
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
    )

    knowledge_base: Mapped["KnowledgeBase"] = relationship("KnowledgeBase")
