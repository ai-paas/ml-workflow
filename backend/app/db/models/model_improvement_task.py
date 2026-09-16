from datetime import datetime

from db.models.base import BaseModel
from sqlalchemy import TIMESTAMP, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column


class ModelImprovementTask(BaseModel):
    """최적화/경량화 비동기 task 추적(소스 모델·소유자·상태·결과 모델)."""

    __tablename__ = "model_improvement_task"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_by_username: Mapped[str] = mapped_column(String(100), nullable=False)
    last_known_status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    # 상태 조회가 최적화 서버를 거치지 않고 DB 만 읽으려면 실패 사유도 함께 남아 있어야 한다.
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    result_model_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model.id", ondelete="CASCADE"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
