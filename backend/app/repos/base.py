from typing import Any, Generic, List, Type, TypeVar, Union

from db.models.base import Base
from pydantic import BaseModel
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        """
        Constructor for a generic CRUD object.

        This class can be used with FastAPI's dependency injection system. By defining an instance
        of this class as a dependency using FastAPI's Depends, you can easily perform CRUD operations
        in your route handlers.

        Example:
        ```
        crud_items = CRUDBase(Item)

        @app.get("/items/{item_id}")
        def read_item(item_id: int, db: Session = Depends(get_db)):
            return crud_items.get(db=db, pk=item_id)
        ```

        :param model: The SQLAlchemy model class.
        """
        self.model = model

    def get(self, db: Session, pk: Union[int, Any]) -> ModelType:
        """
        Retrieves an object from the database based on the given primary key.

        :param db: The SQLAlchemy database session.
        :param pk: The primary key of the object to be retrieved.
        :return: The retrieved object or None.
        """
        return db.query(self.model).get(pk)

    # TODO: Extend 'get_multi' method to include filtering capabilities using 'where' clauses.
    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        Retrieves multiple objects from the database with pagination.

        :param db: The SQLAlchemy database session.
        :param skip: Number of objects to skip.
        :param limit: Maximum number of objects to return.
        :return: List of objects.
        """
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        Creates a new object in the database.

        :param db: The SQLAlchemy database session.
        :param obj_in: The Pydantic model containing the data for the object to be created.
        :return: The created object.
        """
        db_obj = self.model(**obj_in.model_dump())
        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def update(db: Session, *, db_obj: ModelType, obj_in: UpdateSchemaType) -> ModelType:
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

    def delete(self, db: Session, *, pk: Union[int, Any]) -> ModelType:
        """
        Deletes an object from the database.

        :param db: The SQLAlchemy database session.
        :param pk: The primary key of the object to be deleted.
        :return: The deleted object.
        """
        obj = db.query(self.model).get(pk)
        db.delete(obj)
        db.flush()
        return obj

    def filter(self, db: Session, filters: dict[str, Any]) -> List[ModelType]:
        """
        Filters the records based on the provided filter criteria.

        :param db: The SQLAlchemy database session.
        :param filters: A dictionary of filter criteria.
        :return: List of filtered objects.
        """
        query = db.query(self.model)
        for attr, value in filters.items():
            if isinstance(value, list):
                query = query.filter(getattr(self.model, attr).in_(value))
            else:
                query = query.filter(getattr(self.model, attr) == value)
        return query

    def is_active(self, db: Session) -> bool:
        return db.is_active()
