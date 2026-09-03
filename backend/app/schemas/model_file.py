"""모델 저장 파일 조회 스키마."""

from enum import Enum as PyEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelStorageType(str, PyEnum):
    """모델 파일이 어디에 있는지.

    - MLFLOW: MLflow 아티팩트(S3). 목록·다운로드를 제공한다.
    - OLLAMA: Ollama 가 자체 형식으로 관리하는 볼륨. 목록을 제공하지 않는다.
    - NONE:   플랫폼이 보관하는 파일이 없다(원격 서빙 전용).
    """

    MLFLOW = "MLFLOW"
    OLLAMA = "OLLAMA"
    NONE = "NONE"

    def __str__(self) -> str:
        return self.value


class ModelFileEntry(BaseModel):
    """파일 한 건. `storage_type=MLFLOW` 일 때만 채워진다."""

    name: str = Field(description="location 기준 상대 경로 (예: data/model.safetensors)")
    size_bytes: int = Field(description="파일 크기(바이트)")
    last_modified: str = Field(description="마지막 수정 시각 (ISO8601, UTC 'Z')")
    download_url: str = Field(description="다운로드 URL 을 발급받을 API 경로. 파일을 바로 주는 링크가 아니다")


class ModelFileListResponse(BaseModel):
    """모델 파일 목록."""

    model_id: int
    model_name: str
    storage_type: ModelStorageType
    location: Optional[str] = Field(default=None, description="MLFLOW: S3 prefix, OLLAMA: 볼륨 이름, NONE: null")
    files: List[ModelFileEntry] = Field(default_factory=list, description="없으면 빈 배열. null 이 되지 않는다")
    next_cursor: Optional[str] = Field(default=None, description="다음 쪽 토큰. null 이면 목록이 끝난 것")
    message: Optional[str] = Field(default=None, description="files 가 빈 사유. 목록이 있으면 null")

    # `model_` 로 시작하는 필드명을 pydantic 보호 접두사에서 제외한다.
    model_config = {"protected_namespaces": ()}


class ModelFileDownloadUrlResponse(BaseModel):
    """다운로드 URL 발급 결과."""

    model_id: int
    name: str
    size_bytes: int
    download_url: str = Field(description="스토리지 서명 URL. 인증 없이 접근 가능하므로 노출에 주의")
    expires_at: str = Field(description="서명 URL 만료 시각 (ISO8601, UTC 'Z')")

    model_config = {"protected_namespaces": ()}
