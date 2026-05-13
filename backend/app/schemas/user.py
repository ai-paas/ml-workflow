from schemas.base import TimeStampCreateUpdateSchema, TimeStampUpdateSchema


class UserSchema(TimeStampCreateUpdateSchema):
    id: int
    username: str
    name: str
    password: str

    class Config:
        from_attributes = True


class UserBriefSchema(TimeStampCreateUpdateSchema):
    """응답 노출용 사용자 정보 (password 미포함)."""

    id: int
    username: str
    name: str

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
