from db.models.base import BaseModel, TimestampCreateMixin, TimestampUpdateMixin
from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship


class UserModel(BaseModel, TimestampCreateMixin, TimestampUpdateMixin):
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    password: Mapped[str] = mapped_column(String(64), nullable=False)  # TODO: sha-256 hash 처리 필요.

    # Relationships for Service and Workflow
    services = relationship("Service", back_populates="creator")
    workflows = relationship("Workflow", back_populates="creator")
