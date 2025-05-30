from db.models.base import BaseModel, TimestampCreateMixin, TimestampMixin, TimestampUpdateMixin
from db.models.model import Model
from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class ServiceEndpoint(BaseModel, TimestampMixin):
    __tablename__ = "service_endpoint"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(String(4000), nullable=False)
    service_param_type_id: Mapped[int] = mapped_column(ForeignKey("service_param_type.id"))
    service_param_value: Mapped[str] = mapped_column(String(500), nullable=False)
    reference_model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))

    reference_model: Mapped["Model"] = relationship("Model")
    service_param_type: Mapped["ServiceParamType"] = relationship("ServiceParamType")


class ServiceParamType(BaseModel, TimestampMixin):
    __tablename__ = "service_param_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    param_name: Mapped[str] = mapped_column(String(500), nullable=False)
    param_type: Mapped[str] = mapped_column(String(100), nullable=False)
    default_value: Mapped[str] = mapped_column(String(500), nullable=False)
