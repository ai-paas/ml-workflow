import json

from db.models.user import UserModel
from repos.base import CRUDBase
from requests import Session
from schemas.user import UserCreateSchema, UserUpdateSchema


class UserRepository(CRUDBase[UserModel, UserCreateSchema, UserUpdateSchema]):
    # TODO : 현재는 자동으로 속성이 전환되지 않아 chat_config_repository에 custom update, custom create 기능 생성. 개선되면 제거필요.
    @staticmethod
    def update(db: Session, *, db_obj: UserModel, obj_in: UserUpdateSchema) -> UserModel:
        """
        Applies changes to an existing object.

        :param db: The SQLAlchemy database session.
        :param db_obj: The existing SQLAlchemy model object to be updated.
        :param obj_in: The Pydantic model containing the data for the update.
        :return: The updated object.
        """
        obj_data = obj_in.model_dump(exclude_unset=True)
        for field in obj_data:
            setattr(db_obj, field, obj_data[field])
        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        return db_obj

    # TODO : 현재는 자동으로 속성이 전환되지 않아 chat_config_repository에 custom update, custom create 기능 생성. 개선되면 제거필요.

    def create(self, db: Session, *, obj_in: UserCreateSchema) -> UserModel:
        """
        Creates a new object in the database.

        :param db: The SQLAlchemy database session.
        :param obj_in: The Pydantic model containing the data for the object to be created.
        :return: The created object.
        """
        obj_data = obj_in.model_dump()
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        return db_obj


user_repository = UserRepository(UserModel)
