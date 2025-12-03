import re
from datetime import datetime

from sqlalchemy import TIMESTAMP, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column


def pascal_to_snake_case(name: str):
    """alembic & SQLAlchemy migration 작업에서
    자동으로 TableName을 ClassName에서 Model을 제외하고
    snake case로 변환하는 메서드
    """
    if name.endswith("Model"):
        name = name[:-5]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class Base(DeclarativeBase):
    @declared_attr
    def __tablename__(cls) -> str:
        return pascal_to_snake_case(cls.__name__)


class BaseModel(Base):
    __abstract__ = True
    __mapper_args__ = {"polymorphic_identity": ""}

    def __repr__(self):
        items = [f"{column.key}={getattr(self, column.key)!r}" for column in self.__table__.columns]

        return f"<{self.__class__.__name__}({', '.join(items)})>"


class TimestampCreateMixin:
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())
    created_by: Mapped[str | None] = mapped_column(String(40), default="")


class TimestampUpdateMixin:
    updated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, onupdate=func.now(), default=func.now())
    updated_by: Mapped[str | None] = mapped_column(String(40), default="")


class TimestampDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    deleted_by: Mapped[str | None] = mapped_column(String(40), default="")


class TimestampMixin(TimestampCreateMixin, TimestampUpdateMixin, TimestampDeleteMixin):
    pass
