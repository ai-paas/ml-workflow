"""Prompt API 라우터"""

import logging
from typing import Optional

from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, HTTPException, Query, status
from schemas.prompt import PromptCreateSchema, PromptReadSchema, PromptUpdateSchema, PromptVariableTypeListSchema
from schemas.user import UserSchema
from services.prompt import PromptService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompts", tags=["Prompts"])


@router.get("/variable-types", response_model=PromptVariableTypeListSchema)
def get_prompt_variable_types(
    current_user: UserSchema = Depends(get_current_user),
):
    """
    프롬프트 변수 가능한 타입 목록 조회

    프롬프트에서 사용할 수 있는 변수 타입 목록을 조회합니다.

    ## Response (PromptVariableTypeListSchema)
    - **available_types** (List[str]): 사용 가능한 변수 타입 목록
        - 현재는 "context"만 사용 가능

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    return PromptService.get_available_variable_types()


@router.post("", response_model=PromptReadSchema, status_code=status.HTTP_201_CREATED)
def create_prompt(
    *,
    db: Session = SessionDepends,
    prompt_data: PromptCreateSchema,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    프롬프트 생성

    새로운 프롬프트와 프롬프트 변수를 생성합니다.

    ## Request Body (PromptCreateSchema)
    - **prompt** (PromptBaseSchema, required): 프롬프트 기본 정보
        - **name** (str, required): 프롬프트 이름
        - **description** (str, optional): 프롬프트 설명
        - **content** (str, required): 프롬프트 내용
    - **prompt_variable** (List[str], optional): 프롬프트 변수 이름 목록

    ## Response (PromptReadSchema)
    - **id** (int): 프롬프트 ID
    - **name** (str): 프롬프트 이름
    - **description** (str, optional): 프롬프트 설명
    - **content** (str): 프롬프트 내용
    - **prompt_variable** (List[PromptVariableReadSchema], optional): 프롬프트 변수 목록
        - **id** (int): 변수 ID
        - **name** (str): 변수 이름
        - **prompt_id** (int): 프롬프트 ID

    ## Errors
    - 400: 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    try:
        return PromptService.create(db, prompt_data)
    except Exception as e:
        logger.error(f"프롬프트 생성 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"프롬프트 생성 실패: {str(e)}")


@router.get("/{prompt_id}", response_model=PromptReadSchema)
def read_prompt(
    *,
    db: Session = SessionDepends,
    prompt_id: int,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    프롬프트 조회

    특정 프롬프트의 상세 정보를 조회합니다.

    ## Path Parameters
    - **prompt_id** (int): 조회할 프롬프트 ID

    ## Response (PromptReadSchema)
    - **id** (int): 프롬프트 ID
    - **name** (str): 프롬프트 이름
    - **description** (str, optional): 프롬프트 설명
    - **content** (str): 프롬프트 내용
    - **prompt_variable** (List[PromptVariableReadSchema], optional): 프롬프트 변수 목록
        - **id** (int): 변수 ID
        - **name** (str): 변수 이름
        - **prompt_id** (int): 프롬프트 ID

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 프롬프트를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    prompt = PromptService.get(db, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="프롬프트를 찾을 수 없습니다")
    return prompt


@router.get("", response_model=list[PromptReadSchema])
def read_prompts(
    *,
    db: Session = SessionDepends,
    page_size: Optional[int] = Query(
        default=None,
        description="페이지 사이즈",
        examples=[10, 20, 30],
        ge=1,
        le=1000,
    ),
    page: Optional[int] = Query(
        default=None,
        description="페이지 번호",
        examples=[1, 2, 3],
        ge=1,
    ),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    프롬프트 목록 조회

    등록된 프롬프트 목록을 페이지네이션하여 조회합니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
        - 생략 시: 전체 데이터 조회
        - 최소값: 1
    - **page_size** (int, optional): 페이지당 항목 수
        - 생략 시: 전체 데이터 조회
        - 범위: 1-1000

    ## Response (List[PromptReadSchema])
    - 프롬프트 목록

    ## Notes
    - page와 page_size를 모두 생략하면 전체 데이터를 조회 (최대 10000개)
    - page와 page_size 중 하나라도 생략하면 전체 데이터를 조회합니다
    - 페이지네이션 사용 시 page와 page_size를 모두 제공해야 합니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        return PromptService.get_multi(db, skip=0, limit=10000)

    # 페이지네이션 적용
    skip = page_size * (page - 1)
    return PromptService.get_multi(db, skip=skip, limit=page_size)


@router.put("/{prompt_id}", response_model=PromptReadSchema)
def update_prompt(
    *,
    db: Session = SessionDepends,
    prompt_id: int,
    prompt_data: PromptUpdateSchema,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    프롬프트 수정

    기존 프롬프트의 정보를 수정합니다.

    ## Path Parameters
    - **prompt_id** (int): 수정할 프롬프트 ID

    ## Request Body (PromptUpdateSchema)
    - **name** (str, optional): 프롬프트 이름
    - **description** (str, optional): 프롬프트 설명
    - **content** (str, optional): 프롬프트 내용
    - **prompt_variable** (List[str], optional): 프롬프트 변수 이름 목록

    ## Response (PromptReadSchema)
    - **id** (int): 프롬프트 ID
    - **name** (str): 프롬프트 이름
    - **description** (str, optional): 프롬프트 설명
    - **content** (str): 프롬프트 내용
    - **prompt_variable** (List[PromptVariableReadSchema], optional): 프롬프트 변수 목록
        - **id** (int): 변수 ID
        - **name** (str): 변수 이름
        - **prompt_id** (int): 프롬프트 ID

    ## Errors
    - 400: 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 404: 프롬프트를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        prompt = PromptService.update(db, prompt_id, prompt_data)
        if prompt is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="프롬프트를 찾을 수 없습니다")
        return prompt
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"프롬프트 수정 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"프롬프트 수정 실패: {str(e)}")


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prompt(
    *,
    db: Session = SessionDepends,
    prompt_id: int,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    프롬프트 삭제

    프롬프트와 관련된 모든 변수를 삭제합니다.

    ## Path Parameters
    - **prompt_id** (int): 삭제할 프롬프트 ID

    ## Response
    - 204: 삭제 성공 (응답 본문 없음)

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 프롬프트를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        success = PromptService.delete(db, prompt_id)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="프롬프트를 찾을 수 없습니다")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"프롬프트 삭제 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"프롬프트 삭제 실패: {str(e)}")
