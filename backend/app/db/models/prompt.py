from enum import Enum as PyEnum
from typing import Optional

from db.models.base import BaseModel, TimestampMixin
from sqlalchemy import Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class PromptVariableType(PyEnum):
    """프롬프트 변수 타입 열거형"""

    CONTEXT = "context"  # 컨텍스트 변수 (Knowledge Base 검색 결과 등)


class Prompt(BaseModel, TimestampMixin):
    """프롬프트 테이블"""

    __tablename__ = "prompt"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    prompt_variables: Mapped[list["PromptVariable"]] = relationship(
        "PromptVariable", back_populates="prompt", cascade="all, delete-orphan"
    )


class PromptVariable(BaseModel):
    """프롬프트 변수 테이블"""

    __tablename__ = "prompt_variable"
    __table_args__ = (Index("ix_prompt_variable_prompt_id", "prompt_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompt.id", ondelete="CASCADE"), nullable=False)

    # Relationships
    prompt: Mapped["Prompt"] = relationship("Prompt", back_populates="prompt_variables", passive_deletes=True)
