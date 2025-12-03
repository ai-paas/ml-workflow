import logging
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from schemas.knowledge_base import (
    ChunkTypeReadSchema,
    KnowledgeBaseBriefReadSchema,
    KnowledgeBaseCreateSchema,
    KnowledgeBaseReadSchema,
    KnowledgeBaseSearchRecordReadSchema,
    KnowledgeBaseSearchRequestSchema,
    KnowledgeBaseSearchResponseSchema,
    KnowledgeBaseUpdateSchema,
    LanguageReadSchema,
    SearchMethodReadSchema,
)
from schemas.user import UserSchema
from services.knowledge_base import ChunkTypeService, KnowledgeBaseService, LanguageService, SearchMethodService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])

logger = logging.getLogger(__name__)


@router.get("/chunk-types", response_model=list[ChunkTypeReadSchema])
def get_chunk_types(
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    청크 타입 목록 조회

    사용 가능한 모든 청크 타입 목록을 조회합니다.

    ## Response
    - **List[ChunkTypeReadSchema]**: 청크 타입 목록
        - id (int): 청크 타입 ID
        - name (str): 청크 타입 이름 (예: "RecursiveTextSplitter", "RecursiveCharacterSplitter")
        - description (str, optional): 청크 타입 설명

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    return ChunkTypeService.get_all(db)


@router.get("/languages", response_model=list[LanguageReadSchema])
def get_languages(
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    언어 목록 조회

    사용 가능한 모든 언어 목록을 조회합니다.

    ## Response
    - **List[LanguageReadSchema]**: 언어 목록
        - id (int): 언어 ID
        - name (str): 언어 코드 (예: "KO", "EN")
        - description (str, optional): 언어 설명 (예: "한국어", "영어")

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    return LanguageService.get_all(db)


@router.get("/search-methods", response_model=list[SearchMethodReadSchema])
def get_search_methods(
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    검색 방법 목록 조회

    사용 가능한 모든 검색 방법 목록을 조회합니다.

    ## Response
    - **List[SearchMethodReadSchema]**: 검색 방법 목록
        - id (int): 검색 방법 ID
        - name (str): 검색 방법 이름 (예: "vector")
        - description (str, optional): 검색 방법 설명

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    return SearchMethodService.get_all(db)


@router.post("", response_model=KnowledgeBaseReadSchema)
def create_knowledge_base(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = None,
    language_id: Annotated[int, Form()],
    embedding_model_id: Annotated[int, Form()],
    chunk_size: Annotated[int, Form()],
    chunk_overlap: Annotated[int, Form()],
    chunk_type_id: Annotated[int, Form()],
    search_method_id: Annotated[int, Form()],
    top_k: Annotated[int, Form()],
    threshold: Annotated[float, Form()],
    file: Annotated[UploadFile, File()],
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base 생성

    문서 파일을 업로드하여 Knowledge Base를 생성합니다.
    파일은 청크로 분할되고 임베딩되어 Milvus에 저장됩니다.

    ## Request Body (multipart/form-data)
    - **name** (str, required): Knowledge Base 이름
    - **description** (str, optional): Knowledge Base 설명
    - **language_id** (int, required): 언어 ID
        - `GET /api/v1/knowledge-bases/languages` API로 조회 가능
    - **embedding_model_id** (int, required): 임베딩 모델 ID
        - `GET /api/v1/models?model_type_id={embedding_type_id}` API로 조회 가능
    - **chunk_size** (int, required): 청크 크기
    - **chunk_overlap** (int, required): 청크 오버랩 크기
    - **chunk_type_id** (int, required): 청크 타입 ID
        - `GET /api/v1/knowledge-bases/chunk-types` API로 조회 가능
    - **search_method_id** (int, required): 검색 방법 ID
        - `GET /api/v1/knowledge-bases/search-methods` API로 조회 가능
    - **top_k** (int, required): 검색 시 반환할 상위 k개 결과 수
    - **threshold** (float, required): 검색 임계값 (0.0 ~ 1.0)
    - **file** (UploadFile, required): 업로드할 문서 파일
        - **지원 파일 타입**:
          - PDF: `.pdf`
          - Word: `.doc`, `.docx`
          - Excel: `.xls`, `.xlsx`
          - PowerPoint: `.ppt`, `.pptx`
          - CSV: `.csv`
        - 지원되지 않는 파일 타입 업로드 시 400 오류 발생

    ## Response (KnowledgeBaseReadSchema)
    - **id** (int): Knowledge Base ID
    - **name** (str): Knowledge Base 이름
    - **description** (str, optional): Knowledge Base 설명
    - **collection_name** (str): Milvus Collection 이름
    - **files** (List[KnowledgeBaseFileReadSchema]): 파일 목록
    - 기타 필드들...

    ## Errors
    - 400: 유효하지 않은 요청 또는 필수 파라미터 누락
    - 401: 인증되지 않은 사용자
    - 500: Knowledge Base 생성 중 서버 내부 오류
    """
    try:
        kb_schema = KnowledgeBaseCreateSchema(
            name=name,
            description=description,
            language_id=language_id,
            embedding_model_id=embedding_model_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_type_id=chunk_type_id,
            search_method_id=search_method_id,
            top_k=top_k,
            threshold=threshold,
        )
        return KnowledgeBaseService.create(db, obj_in=kb_schema, file=file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge Base 생성 중 오류가 발생했습니다: {str(e)}",
        )


@router.post("/{knowledge_base_id}/files", response_model=KnowledgeBaseReadSchema)
def add_file_to_knowledge_base(
    *,
    db: Session = SessionDepends,
    knowledge_base_id: int,
    file: Annotated[UploadFile, File()],
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base에 파일 추가

    기존 Knowledge Base에 문서 파일을 추가합니다.
    파일은 청크로 분할되고 임베딩되어 Milvus의 동일한 Collection에 Partition으로 추가됩니다.

    ## Path Parameters
    - **knowledge_base_id** (int): Knowledge Base ID

    ## Request Body (multipart/form-data)
    - **file** (UploadFile, required): 추가할 문서 파일

    ## Response (KnowledgeBaseReadSchema)
    - 업데이트된 Knowledge Base 정보

    ## Errors
    - 400: 유효하지 않은 요청 또는 파일 처리 실패
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base를 찾을 수 없음
    - 500: 파일 추가 중 서버 내부 오류
    """
    try:
        return KnowledgeBaseService.add_file(db, knowledge_base_id=knowledge_base_id, file=file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 파일 추가 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"파일 추가 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/{knowledge_base_id}/files/{file_id}", response_model=KnowledgeBaseReadSchema)
def delete_file_from_knowledge_base(
    *,
    db: Session = SessionDepends,
    knowledge_base_id: int,
    file_id: int,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base에서 파일 삭제

    Knowledge Base에서 특정 파일을 삭제합니다.
    DB에서 파일 정보를 삭제하고, Milvus에서 해당 Partition을 삭제합니다.

    ## Path Parameters
    - **knowledge_base_id** (int): Knowledge Base ID
    - **file_id** (int): 삭제할 파일 ID

    ## Response (KnowledgeBaseReadSchema)
    - 업데이트된 Knowledge Base 정보

    ## Errors
    - 400: 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base 또는 파일을 찾을 수 없음
    - 500: 파일 삭제 중 서버 내부 오류
    """
    try:
        return KnowledgeBaseService.delete_file(db, knowledge_base_id=knowledge_base_id, file_id=file_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 파일 삭제 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"파일 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("", response_model=list[KnowledgeBaseBriefReadSchema])
def get_knowledge_bases(
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
    Knowledge Base 목록 조회

    등록된 Knowledge Base 목록을 페이지네이션하여 조회합니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
        - 생략 시: 전체 데이터 조회
        - 최소값: 1
    - **page_size** (int, optional): 페이지당 항목 수
        - 생략 시: 전체 데이터 조회
        - 범위: 1-1000

    ## Response (List[KnowledgeBaseBriefReadSchema])
    - Knowledge Base 목록

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    if page is None or page_size is None:
        kb_list = KnowledgeBaseService.get_multi(db, skip=0, limit=10000)
        return kb_list

    skip = page_size * (page - 1)
    kb_list = KnowledgeBaseService.get_multi(db, skip=skip, limit=page_size)
    return kb_list


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseReadSchema)
def get_knowledge_base(
    knowledge_base_id: int,
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base 상세 조회

    특정 Knowledge Base의 상세 정보를 조회합니다.

    ## Path Parameters
    - **knowledge_base_id** (int): 조회할 Knowledge Base ID

    ## Response (KnowledgeBaseReadSchema)
    - Knowledge Base 상세 정보 및 파일 목록

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    kb = KnowledgeBaseService.get(db, knowledge_base_id)
    if not kb:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge Base를 찾을 수 없습니다.")
    return kb


@router.delete("/{knowledge_base_id}")
def delete_knowledge_base(
    knowledge_base_id: int,
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base 삭제

    Knowledge Base를 삭제합니다.
    DB에서 Knowledge Base 정보를 삭제하고, Milvus에서 Collection을 삭제합니다.

    ## Path Parameters
    - **knowledge_base_id** (int): 삭제할 Knowledge Base ID

    ## Response
    - **success** (bool): 삭제 성공 여부
    - **message** (str): 삭제 결과 메시지

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base를 찾을 수 없음
    - 500: 삭제 중 서버 내부 오류
    """
    try:
        result = KnowledgeBaseService.delete(db, knowledge_base_id)
        return {"success": result, "message": "Knowledge Base가 성공적으로 삭제되었습니다."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 삭제 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.put("/{knowledge_base_id}", response_model=KnowledgeBaseReadSchema)
def update_knowledge_base(
    *,
    db: Session = SessionDepends,
    knowledge_base_id: int,
    obj_in: KnowledgeBaseUpdateSchema,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base 수정

    Knowledge Base의 이름과 설명만 수정할 수 있습니다.

    ## Path Parameters
    - **knowledge_base_id** (int): 수정할 Knowledge Base ID

    ## Request Body
    - **name** (str, optional): 수정할 이름
    - **description** (str, optional): 수정할 설명

    ## Response (KnowledgeBaseReadSchema)
    - 수정된 Knowledge Base 정보

    ## Errors
    - 400: 유효하지 않은 요청 또는 수정할 데이터 없음
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base를 찾을 수 없음
    - 500: 수정 중 서버 내부 오류
    """
    try:
        return KnowledgeBaseService.update(db, knowledge_base_id=knowledge_base_id, obj_in=obj_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 수정 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"수정 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/{knowledge_base_id}/search", response_model=KnowledgeBaseSearchResponseSchema)
def search_knowledge_base(
    *,
    db: Session = SessionDepends,
    knowledge_base_id: int,
    obj_in: KnowledgeBaseSearchRequestSchema,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base 검색 테스트

    Knowledge Base에 저장된 문서를 검색합니다.
    Knowledge Base의 설정된 검색 방법(search_method), top_k, threshold를 사용하여 검색을 수행합니다.

    ## Path Parameters
    - **knowledge_base_id** (int): 검색할 Knowledge Base ID

    ## Request Body
    - **text** (str, required): 검색할 쿼리 텍스트

    ## Response (KnowledgeBaseSearchResponseSchema)
    - **results** (List[SearchResultItemSchema]): 검색 결과 목록
        - **text** (str): 검색된 문서 텍스트
        - **score** (float): 검색 점수 (유사도)
        - **distance** (float, optional): 거리 값
    - **total** (int): 검색 결과 총 개수
    - **search_method** (str): 사용된 검색 방법 (dense/sparse/hybrid)

    ## Errors
    - 400: 유효하지 않은 요청 또는 Knowledge Base를 찾을 수 없음
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base를 찾을 수 없음
    - 500: 검색 중 서버 내부 오류
    """
    try:
        return KnowledgeBaseService.search(
            db,
            knowledge_base_id=knowledge_base_id,
            query=obj_in.text,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 검색 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{knowledge_base_id}/search-records", response_model=list[KnowledgeBaseSearchRecordReadSchema])
def get_knowledge_base_search_records(
    knowledge_base_id: int,
    db: Session = SessionDepends,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Knowledge Base 검색 기록 조회

    특정 Knowledge Base에 대한 검색 기록을 조회합니다.

    ## Path Parameters
    - **knowledge_base_id** (int): 조회할 Knowledge Base ID

    ## Response (List[KnowledgeBaseSearchRecordReadSchema])
    - **id** (int): 검색 기록 ID
    - **knowledge_base_id** (int): Knowledge Base ID
    - **source** (str): Collection 이름
    - **text** (str): 검색 쿼리 텍스트
    - **created_at** (datetime): 검색 기록 생성 시간

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: Knowledge Base를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        return KnowledgeBaseService.get_search_records(db, knowledge_base_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Knowledge Base 검색 기록 조회 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 기록 조회 중 오류가 발생했습니다: {str(e)}",
        )
