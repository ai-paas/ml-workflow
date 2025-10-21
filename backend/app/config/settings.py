from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

current_directory = Path(__file__).parent
dotenv_path = current_directory / ".env"
load_dotenv(dotenv_path=dotenv_path)


class Settings(BaseSettings):
    """애플리케이션 설정 클래스

    환경 변수를 통해 설정값을 로드하며, 필수 설정값들의 유효성을 검사합니다.
    """

    model_config = SettingsConfigDict(
        case_sensitive=True,  # 대소문자 구분 허용
        env_file=".env",  # settings env file name
        env_file_encoding="utf-8",  # setting env file encoding
    )

    # Kubeflow 설정
    KUBEFLOW_ENDPOINT: str = Field(..., description="Kubeflow API 엔드포인트 URL")
    KUBEFLOW_USERNAME: str = Field(..., description="Kubeflow 사용자명")
    KUBEFLOW_PASSWORD: str = Field(..., description="Kubeflow 비밀번호")
    KUBEFLOW_NAMESPACE: str = Field(..., description="Kubeflow 네임스페이스")
    KUBEFLOW_EXPERIMENT_NAME: str = Field(..., description="Kubeflow 실험명")

    # 데이터베이스 설정
    DB_TYPE: str = Field(..., description="데이터베이스 타입 (예: postgresql)")
    DB_NAME: str = Field(..., description="데이터베이스 이름")
    DB_USER: str = Field(..., description="데이터베이스 사용자명")
    DB_PASSWORD: str = Field(..., description="데이터베이스 비밀번호")
    DB_HOST: str = Field(..., description="데이터베이스 호스트")
    DB_PORT: str = Field(..., description="데이터베이스 포트")

    # MLflow 설정
    MLFLOW_TRACKING_URI: str = Field(..., description="MLflow 추적 서버 URI")
    MLFLOW_TRACKING_USERNAME: str = Field(..., description="MLflow 추적 서버 사용자명")
    MLFLOW_TRACKING_PASSWORD: str = Field(..., description="MLflow 추적 서버 비밀번호")
    MLFLOW_EXPERIMENT_NAME: str = Field(..., description="MLflow 실험명")

    # MLflow S3 설정
    MLFLOW_S3_ENDPOINT_URL: str = Field(..., description="MLflow S3 엔드포인트 URL")
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS 액세스 키 ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS 시크릿 액세스 키")
    MLFLOW_S3_BUCKET: str = Field(
        ...,
        description="MLflow 아티팩트 저장용 S3 버킷 이름. \
        모델 파일과 실험 결과를 저장하는 데 사용됩니다.",
    )

    # Innogrid Object Storage 설정
    INNOGRID_OBJECT_STORAGE_ENDPOINT: str = Field(..., description="Innogrid Object Storage 엔드포인트")
    INNOGRID_OBJECT_STORAGE_ACCESS_KEY: str = Field(..., description="Innogrid Object Storage 액세스 키")
    INNOGRID_OBJECT_STORAGE_SECRET_KEY: str = Field(..., description="Innogrid Object Storage 시크릿 키")
    INNOGRID_OBJECT_STORAGE_BUCKET: str = Field(..., description="Innogrid Object Storage 버킷 이름")

    # API 및 인증 설정
    REST_API_URL: str = Field(..., description="REST API 기본 URL")
    DEMO_PASSWORD: str = Field(..., description="데모용 비밀번호")
    LOGIN_SECRET_KEY: str = Field(..., description="로그인 세션 암호화 키")

    # KServe 설정
    KSERVE_GPU: bool = Field(default=False, description="KServe GPU 사용 여부")
    KSERVE_GATEWAY_URL: str = Field(
        default="http://10.10.30.154:80",
        description="KServe Istio Gateway URL \
(외부 접근용)",
    )

    # 기타 설정
    USER_MODELS: dict[str, dict] = Field(default_factory=dict, description="사용자 정의 모델 설정")

    @field_validator("MLFLOW_S3_BUCKET")
    @classmethod
    def validate_mlflow_s3_bucket(cls, v: str) -> str:
        """MLflow S3 버킷 이름 유효성 검사"""
        if not v or not v.strip():
            raise ValueError(
                "MLFLOW_S3_BUCKET은 비어있을 수 없습니다. \
                    MLflow 아티팩트 저장을 위한 S3 버킷 이름을 설정해주세요."
            )

        return v.strip()

    @field_validator("DB_PORT")
    @classmethod
    def validate_db_port(cls, v: str) -> str:
        """데이터베이스 포트 유효성 검사"""
        if not v or not v.strip():
            raise ValueError("DB_PORT는 비어있을 수 없습니다.")

        return v.strip()

    @property
    def get_db_uri(self) -> str:
        """Environment variables로부터 DB 정보를 받아와 URI를 반환."""
        return f"{self.DB_TYPE}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def user_models(self) -> dict[str, dict]:
        return self.USER_MODELS

    def add_user_model(self, key: str, value: Any):
        self.USER_MODELS[key] = value


@lru_cache
def get_settings():
    return Settings()
