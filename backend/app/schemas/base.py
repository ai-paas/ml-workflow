from datetime import datetime

from pydantic import BaseModel, Field


class TimeStampSchemaMixin(BaseModel):
    created_at: datetime | None = Field(default_factory=datetime.now)
    updated_at: datetime | None = Field(default_factory=datetime.now)
    deleted_at: datetime | None = Field(None)
    created_by: str | None = Field(None, max_length=40)
    updated_by: str | None = Field(None, max_length=40)
    deleted_by: str | None = Field(None, max_length=40)


class TimeStampUpdateSchema(BaseModel):
    updated_at: datetime | None = Field(default_factory=datetime.now)
    updated_by: str | None = Field(None, max_length=40)


class TimeStampCreateUpdateSchema(BaseModel):
    created_at: datetime | None = Field(default_factory=datetime.now)
    updated_at: datetime | None = Field(default_factory=datetime.now)
    created_by: str | None = Field(None, max_length=40)
    updated_by: str | None = Field(None, max_length=40)
