from schemas.base import TimeStampCreateUpdateSchema, TimeStampUpdateSchema


class UserSchema(TimeStampCreateUpdateSchema):
    id: int
    username: str
    name: str
    password: str

    class Config:
        from_attributes = True


class UserCreateSchema(TimeStampCreateUpdateSchema):
    username: str
    name: str
    password: str


class UserUpdateSchema(TimeStampUpdateSchema):
    username: str
    name: str
    password: str
