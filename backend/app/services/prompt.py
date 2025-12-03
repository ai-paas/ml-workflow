import logging
from typing import Optional

from db.models.prompt import Prompt, PromptVariableType
from repos.prompt import prompt_repository, prompt_variable_repository
from schemas.prompt import (
    PromptCreateSchema,
    PromptReadSchema,
    PromptUpdateSchema,
    PromptVariableBaseSchema,
    PromptVariableReadSchema,
    PromptVariableTypeListSchema,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PromptService:
    """프롬프트 관련 비즈니스 로직"""

    @staticmethod
    def create(db: Session, obj_in: PromptCreateSchema) -> PromptReadSchema:
        """프롬프트 생성"""
        try:
            # 1. 프롬프트 생성
            prompt_obj = prompt_repository.create(db, obj_in=obj_in.prompt)
            prompt_id = prompt_obj.id

            # 2. 프롬프트 변수 생성
            if obj_in.prompt_variable:
                for var_type in obj_in.prompt_variable:
                    # Enum을 String으로 변환하여 저장
                    prompt_variable_repository.create(
                        db,
                        obj_in=PromptVariableBaseSchema(name=var_type.value, prompt_id=prompt_id),
                    )

            db.commit()
            db.refresh(prompt_obj)

            logger.info(f"프롬프트 생성 성공: {prompt_obj.name} (ID: {prompt_id})")
            return PromptService.get(db, prompt_id)

        except Exception as e:
            db.rollback()
            logger.error(f"프롬프트 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def get(db: Session, pk: int) -> Optional[PromptReadSchema]:
        """프롬프트 조회"""
        prompt_obj = prompt_repository.get_with_variables(db, pk)
        if not prompt_obj:
            return None

        return PromptReadSchema(
            id=prompt_obj.id,
            name=prompt_obj.name,
            description=prompt_obj.description,
            content=prompt_obj.content,
            prompt_variable=(
                [
                    PromptVariableReadSchema(
                        id=var.id,
                        name=var.name,
                        prompt_id=var.prompt_id,
                    )
                    for var in prompt_obj.prompt_variables
                ]
                if prompt_obj.prompt_variables
                else None
            ),
        )

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[PromptReadSchema]:
        """프롬프트 목록 조회"""
        prompts = prompt_repository.get_multi_with_variables(db, skip=skip, limit=limit)
        return [
            PromptReadSchema(
                id=prompt.id,
                name=prompt.name,
                description=prompt.description,
                content=prompt.content,
                prompt_variable=(
                    [
                        PromptVariableReadSchema(
                            id=var.id,
                            name=var.name,
                            prompt_id=var.prompt_id,
                        )
                        for var in prompt.prompt_variables
                    ]
                    if prompt.prompt_variables
                    else None
                ),
            )
            for prompt in prompts
        ]

    @staticmethod
    def update(db: Session, pk: int, obj_in: PromptUpdateSchema) -> Optional[PromptReadSchema]:
        """프롬프트 수정"""
        try:
            prompt_obj = prompt_repository.get(db, pk)
            if not prompt_obj:
                return None

            # 1. 프롬프트 기본 정보 업데이트
            update_data = {}
            if obj_in.name is not None:
                update_data["name"] = obj_in.name
            if obj_in.description is not None:
                update_data["description"] = obj_in.description
            if obj_in.content is not None:
                update_data["content"] = obj_in.content

            if update_data:
                prompt_repository.update(
                    db,
                    db_obj=prompt_obj,
                    obj_in=PromptUpdateSchema(**update_data),
                )

            # 2. 프롬프트 변수 업데이트
            if obj_in.prompt_variable is not None:
                # 기존 변수 삭제
                prompt_variable_repository.delete_by_prompt_id(db, pk)

                # 새 변수 생성
                for var_type in obj_in.prompt_variable:
                    # Enum을 String으로 변환하여 저장
                    prompt_variable_repository.create(
                        db,
                        obj_in=PromptVariableBaseSchema(name=var_type.value, prompt_id=pk),
                    )

            db.commit()
            db.refresh(prompt_obj)

            logger.info(f"프롬프트 수정 성공: {prompt_obj.name} (ID: {pk})")
            return PromptService.get(db, pk)

        except Exception as e:
            db.rollback()
            logger.error(f"프롬프트 수정 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def delete(db: Session, pk: int) -> bool:
        """프롬프트 삭제"""
        try:
            prompt_obj = prompt_repository.get(db, pk)
            if not prompt_obj:
                return False

            # CASCADE 설정으로 prompt_variable도 자동 삭제됨
            prompt_repository.delete(db, pk=pk)
            db.commit()

            logger.info(f"프롬프트 삭제 성공: (ID: {pk})")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"프롬프트 삭제 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def get_available_variable_types() -> PromptVariableTypeListSchema:
        """프롬프트 변수 가능한 타입 목록 조회"""
        available_types = [var_type.value for var_type in PromptVariableType]
        return PromptVariableTypeListSchema(available_types=available_types)
