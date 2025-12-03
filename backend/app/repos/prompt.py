from typing import Optional

from db.models.prompt import Prompt, PromptVariable
from repos.base import CRUDBase
from schemas.prompt import PromptBaseSchema, PromptCreateSchema, PromptUpdateSchema, PromptVariableBaseSchema
from sqlalchemy.orm import Session, joinedload


class PromptRepository(CRUDBase[Prompt, PromptBaseSchema, PromptUpdateSchema]):
    def get_with_variables(self, db: Session, pk: int) -> Optional[Prompt]:
        """변수 정보를 포함하여 프롬프트 조회"""
        return db.query(self.model).options(joinedload(Prompt.prompt_variables)).filter(self.model.id == pk).first()

    def get_multi_with_variables(self, db: Session, skip: int = 0, limit: int = 100) -> list[Prompt]:
        """변수 정보를 포함하여 프롬프트 목록 조회"""
        return db.query(self.model).options(joinedload(Prompt.prompt_variables)).offset(skip).limit(limit).all()


class PromptVariableRepository(CRUDBase[PromptVariable, PromptVariableBaseSchema, PromptVariableBaseSchema]):
    def get_by_prompt_id(self, db: Session, prompt_id: int) -> list[PromptVariable]:
        """프롬프트 ID로 변수 목록 조회"""
        return db.query(self.model).filter(self.model.prompt_id == prompt_id).all()

    def delete_by_prompt_id(self, db: Session, prompt_id: int) -> None:
        """프롬프트 ID로 모든 변수 삭제"""
        db.query(self.model).filter(self.model.prompt_id == prompt_id).delete()
        db.flush()


prompt_repository = PromptRepository(Prompt)
prompt_variable_repository = PromptVariableRepository(PromptVariable)
