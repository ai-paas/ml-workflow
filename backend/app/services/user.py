from typing import Optional

from db.models.user import UserModel
from repos.user import user_repository
from schemas.user import UserCreateSchema, UserUpdateSchema
from sqlalchemy.orm import Session


class UserService:
    @staticmethod
    def get(db: Session, pk: int) -> Optional[UserModel]:
        return user_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[UserModel]:
        return user_repository.get_multi(db, skip=skip, limit=limit)

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[UserModel]:
        results = user_repository.filter(db, filters={"username": username})
        return results[0] if results else None

    @staticmethod
    def get_all(db: Session) -> list[UserModel]:
        return user_repository.get_all(db)

    @staticmethod
    def create(db: Session, obj_in: UserCreateSchema) -> UserModel:
        result = user_repository.create(db, obj_in=obj_in)
        db.commit()
        return result

    @staticmethod
    def update(db: Session, *, db_obj: UserModel, obj_in: UserUpdateSchema) -> UserModel:
        result = user_repository.update(db, db_obj=db_obj, obj_in=obj_in)
        db.commit()
        return result

    @staticmethod
    def delete(db: Session, *, pk: int) -> UserModel:
        result = user_repository.delete(db, pk=pk)
        db.commit()
        return result
