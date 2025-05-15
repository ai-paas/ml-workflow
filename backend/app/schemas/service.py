from pydantic import BaseModel
from schemas.model import ModelReadSchema


class ServiceEndpointBaseSchema(BaseModel):
    url: str
    service_param_type_id: int
    service_param_value: str
    reference_model_id: int


class ServiceEndpointReadSchema(BaseModel):
    id: int
    url: str
    service_param_type_id: int
    service_param_value: str
    reference_model_id: int
    reference_model: "ModelReadSchema"
    service_param_type: "ServiceParamTypeReadSchema"

    class Config:
        from_attributes = True


class ServiceParamTypeBaseSchema(BaseModel):
    param_name: str
    param_type: str
    default_value: str


class ServiceParamTypeReadSchema(BaseModel):
    id: int
    param_name: str
    param_type: str
    default_value: str

    class Config:
        from_attributes = True
