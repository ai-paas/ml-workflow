import logging
import traceback
from typing import Annotated, Optional

from config.db.connect import SessionDepends
from config.db.enums import DatasetKindEnum
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from schemas.dataset import (
    DatasetBaseSchema,
    DatasetKindReadSchema,
    DatasetReadSchema,
    DatasetRegistryBaseSchema,
    DatasetUpdateSchema,
    DatasetValidationResponse,
)
from schemas.user import UserSchema
from services.dataset import DatasetRegistryService, DatasetService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/datasets", tags=["Datasets"])

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get("/kinds", response_model=list[DatasetKindReadSchema])
def get_dataset_kinds(current_user: UserSchema = Depends(get_current_user)):
    """
    데이터셋 분류(kind) 카탈로그 조회

    UI 의 데이터셋 유형 드롭다운을 채우기 위한 정적 카탈로그를 반환한다.
    각 분류는 `dataset_kind` 와 동일한 enum 값(name)과, 허용 형식·지원 모델 정보를 갖는다.

    ## Response (List[DatasetKindReadSchema])
    - **name** (str): 분류 식별자 (`object-detection` / `protein-classification`)
    - **description** (str): 분류 설명
    - **accepted_formats** (List[str]): 허용 데이터셋 형식 (구체 ZIP 사양은 별도 가이드에서 확정)
    - **supported_models** (List[str]): 해당 분류로 학습 가능한 모델 키 목록
    """
    return [
        DatasetKindReadSchema(
            name=DatasetKindEnum.OBJECT_DETECTION.value,
            description="객체 감지(YOLOX) 데이터셋",
            accepted_formats=["coco"],
            supported_models=["yolox_s", "yolox_m"],
        ),
        DatasetKindReadSchema(
            name=DatasetKindEnum.PROTEIN_CLASSIFICATION.value,
            description="단백질 서열 분류(ESM2, TCR-Epitope) 데이터셋",
            accepted_formats=["csv"],
            supported_models=["facebook/esm2_t6_8M_UR50D"],
        ),
    ]


@router.post("/validate", response_model=DatasetValidationResponse)
def validate_dataset(
    *,
    file: UploadFile = File(...),
    dataset_kind: DatasetKindEnum = Form(...),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    데이터셋 파일 유효성 검증

    업로드된 데이터셋 ZIP 이 요청한 `dataset_kind` 분류의 형식 요구사항을 충족하는지 검증한다.

    ## Request Body (multipart/form-data)
    - **file** (UploadFile, required): 검증할 데이터셋 ZIP 파일
    - **dataset_kind** (str enum, required): 검증 대상 분류
        - `object-detection`: COCO128 구조(annotations/instances_{train,val}2017.json + train2017/val2017)
        - `protein-classification`: TCR-Epitope CSV(필수 컬럼 epitope/cdr3b/label, label∈{0,1}, 행≥16)

    ## Response (DatasetValidationResponse)
    - **is_valid** (bool): 검증 성공 여부
    - **message** (str): 검증 결과 메시지
    - **details** (dict, optional): 검증 실패 시 errors 목록

    ## Notes
    - 출력에는 `dataset_kind` 가 포함되지 않는다(입력으로 이미 분류가 정해져 있음).
    - 형식 불충족은 200 + `is_valid=false` 로 응답한다.

    ## Errors
    - 401: 인증되지 않은 사용자
    - 422: 알 수 없는 `dataset_kind` 값
    - 500: 서버 내부 오류
    """
    result = DatasetService.validate(file=file, dataset_kind=dataset_kind)
    if not result.is_valid:
        logger.warning(f"데이터셋 검증 실패: {result.message}")
    return result


# TODO: 책임 분리 필요.
@router.post("", response_model=DatasetReadSchema)
def create_dataset(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[Optional[str], Form()] = None,
    dataset_kind: Annotated[DatasetKindEnum, Form()],
    file: UploadFile = File(...),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    데이터셋 등록

    Dataset Registry에 데이터셋을 등록합니다.
    업로드된 파일을 검증하고 MLflow에 등록한 후 데이터베이스에 메타데이터를 저장합니다.

    ## Request Body (multipart/form-data)
    - **name** (str, required): 데이터셋 이름
        - 데이터셋을 식별하기 위한 이름
        - Form 필드로 전달
    - **description** (str, optional): 데이터셋 설명
        - 데이터셋에 대한 상세 설명
        - Form 필드로 전달
        - 생략 가능 (기본값: None)
    - **file** (UploadFile, required): 데이터셋 ZIP 파일
        - COCO128 형식의 데이터셋이 ZIP으로 압축된 파일
        - 파일 검증은 /datasets/validate API를 먼저 호출하여 수행하는 것을 권장합니다

    ## Response (DatasetReadSchema)
    - **id** (int): 데이터셋 고유 ID
    - **name** (str): 데이터셋 이름
    - **description** (str, optional): 데이터셋 설명
        - 데이터셋에 대한 상세 설명 (없을 수 있음)
    - **dataset_registry** (DatasetRegistryReadSchema): 데이터셋 레지스트리 정보
        - id (int): 레지스트리 ID
        - artifact_path (str): 아티팩트 경로
            - MLflow에 저장된 데이터셋의 아티팩트 경로
        - uri (str): 데이터셋 URI
            - MLflow에서 접근 가능한 데이터셋 URI
        - dataset_id (int): 연결된 데이터셋 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각
    - **created_at** (datetime): 데이터셋 생성 시각
    - **updated_at** (datetime): 데이터셋 수정 시각

    ## Notes
    - 파일 검증은 /datasets/validate API를 먼저 호출하여 수행하는 것을 권장합니다
    - 데이터셋은 MLflow에 자동으로 등록되며, artifact_path와 uri가 생성됩니다
    - 등록된 데이터셋은 실험(Experiment) 생성 시 사용할 수 있습니다

    ## Errors
    - 400: 데이터셋 검증 실패 또는 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 500: 데이터셋 등록 중 서버 내부 오류
    """
    max_size_bytes = settings.DATASET_MAX_UPLOAD_SIZE_MB * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"업로드 파일 크기({file_size / (1024 * 1024):.1f}MB)가 "
            f"최대 허용 크기({settings.DATASET_MAX_UPLOAD_SIZE_MB}MB)를 초과했습니다.",
        )

    try:
        dataset_data = DatasetBaseSchema(
            name=name,
            description=description if description else None,
            version=1,
            subversion=1,
            kind=dataset_kind,
        )

        # DatasetService를 통해 DB에 저장 (ZIP 재검증은 수행하지 않음 — 사전 /validate 통과 전제)
        db_dataset = DatasetService.create(db, obj_in=dataset_data, file=file)

        return db_dataset

    except ValueError as e:
        # 검증 오류 발생 시
        logger.error(f"데이터셋 생성 오류: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 기타 오류 발생 시
        traceback.print_exc()
        logger.error(f"데이터셋 등록 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터셋 등록 중 오류가 발생했습니다: {str(e)}")


@router.get("/{dataset_id}", response_model=DatasetReadSchema)
def read_dataset(dataset_id: int, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)):
    """
    데이터셋 상세정보 조회

    특정 데이터셋의 상세 정보를 조회합니다.
    데이터셋 레지스트리 정보를 포함하여 반환합니다.

    ## Path Parameters
    - **dataset_id** (int): 조회할 데이터셋 ID

    ## Response (DatasetReadSchema)
    - **id** (int): 데이터셋 고유 ID
    - **name** (str): 데이터셋 이름
    - **description** (str, optional): 데이터셋 설명
        - 데이터셋에 대한 상세 설명 (없을 수 있음)
    - **dataset_registry** (DatasetRegistryReadSchema): 데이터셋 레지스트리 정보
        - id (int): 레지스트리 ID
        - artifact_path (str): 아티팩트 경로
            - MLflow에 저장된 데이터셋의 아티팩트 경로
        - uri (str): 데이터셋 URI
            - MLflow에서 접근 가능한 데이터셋 URI
        - dataset_id (int): 연결된 데이터셋 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각
    - **created_at** (datetime): 데이터셋 생성 시각
    - **updated_at** (datetime): 데이터셋 수정 시각

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 데이터셋을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    db_model = DatasetService().get(db, dataset_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_model


@router.get("", response_model=list[DatasetReadSchema])
def read_datasets(
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
    데이터셋 목록 조회

    등록된 데이터셋들의 목록을 페이지네이션하여 조회합니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
        - 생략 시: 전체 데이터 조회
        - 최소값: 1
    - **page_size** (int, optional): 페이지당 항목 수
        - 생략 시: 전체 데이터 조회
        - 범위: 1-1000

    ## Response (List[DatasetReadSchema])
    - **items** (List[DatasetReadSchema]): 데이터셋 목록
        각 항목은 다음 정보를 포함:
        - id (int): 데이터셋 고유 ID
        - name (str): 데이터셋 이름
        - description (str, optional): 데이터셋 설명
            - 데이터셋에 대한 상세 설명 (없을 수 있음)
        - dataset_registry (DatasetRegistryReadSchema): 데이터셋 레지스트리 정보
            - id (int): 레지스트리 ID
            - artifact_path (str): 아티팩트 경로
            - uri (str): 데이터셋 URI
            - dataset_id (int): 연결된 데이터셋 ID
            - created_at (datetime): 생성 시각
            - updated_at (datetime): 수정 시각
        - created_at (datetime): 데이터셋 생성 시각
        - updated_at (datetime): 데이터셋 수정 시각

    ## Notes
    - page와 page_size를 모두 생략하면 전체 데이터를 조회 (최대 10000개)
    - 페이지네이션 사용 시 page와 page_size를 모두 제공해야 합니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        datasets = DatasetService().get_multi(db, skip=0, limit=10000)
        return datasets

    # 페이지네이션 적용
    skip = page_size * (page - 1)

    datasets = DatasetService().get_multi(db, skip=skip, limit=page_size)
    return datasets


@router.put("/{dataset_id}", response_model=DatasetReadSchema)
def update_dataset(
    *,
    db: Session = SessionDepends,
    dataset_id: int,
    obj_in: DatasetUpdateSchema,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    데이터셋 정보 수정

    데이터셋의 이름(name)과 설명(description)을 수정합니다.

    ## Path Parameters
    - **dataset_id** (int): 수정할 데이터셋 ID

    ## Request Body (DatasetUpdateSchema)
    - **name** (str, optional): 새로운 데이터셋 이름
        - 생략 가능 (수정하지 않으려면 전달하지 않음)
    - **description** (str, optional): 새로운 데이터셋 설명
        - 생략 가능 (수정하지 않으려면 전달하지 않음)

    ## Response (DatasetReadSchema)
    - **id** (int): 데이터셋 고유 ID
    - **name** (str): 수정된 데이터셋 이름
    - **description** (str, optional): 수정된 데이터셋 설명
        - 데이터셋에 대한 상세 설명 (없을 수 있음)
    - **dataset_registry** (DatasetRegistryReadSchema): 데이터셋 레지스트리 정보
        - id (int): 레지스트리 ID
        - artifact_path (str): 아티팩트 경로
        - uri (str): 데이터셋 URI
        - dataset_id (int): 연결된 데이터셋 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각
    - **created_at** (datetime): 데이터셋 생성 시각
    - **updated_at** (datetime): 데이터셋 수정 시각

    ## Notes
    - name과 description 중 하나만 수정하거나 둘 다 수정할 수 있습니다
    - 수정하지 않을 필드는 요청에서 생략하면 됩니다

    ## Errors
    - 400: 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 404: 데이터셋을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        updated_dataset = DatasetService.update(db, dataset_id=dataset_id, obj_in=obj_in)
        return updated_dataset
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"데이터셋 수정 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터셋 수정 중 오류가 발생했습니다: {str(e)}")


@router.delete("/{dataset_id}")
def delete_dataset(
    *,
    db: Session = SessionDepends,
    dataset_id: int,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    데이터셋 삭제

    데이터셋을 삭제합니다. MLflow에 저장된 정보와 S3에 저장된 파일도 함께 삭제됩니다.

    ## Path Parameters
    - **dataset_id** (int): 삭제할 데이터셋 ID

    ## Response
    - **success** (bool): 삭제 성공 여부
    - **message** (str): 삭제 결과 메시지

    ## Notes
    - 데이터셋 삭제 시 다음 항목들이 함께 삭제됩니다:
        - 데이터베이스의 데이터셋 레코드
        - 데이터베이스의 데이터셋 레지스트리 레코드
        - MLflow에 저장된 run 및 artifacts
        - S3에 저장된 데이터셋 파일들
    - 삭제 작업은 원자적으로 수행되며, 중간에 실패하면 모든 변경사항이 롤백됩니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 데이터셋을 찾을 수 없음
    - 500: 데이터셋 삭제 중 서버 내부 오류
    """
    try:
        result = DatasetService.delete(db, dataset_id)
        return {"success": result, "message": "데이터셋이 성공적으로 삭제되었습니다."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"데이터셋 삭제 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터셋 삭제 중 오류가 발생했습니다: {str(e)}")
