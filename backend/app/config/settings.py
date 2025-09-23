from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

current_directory = Path(__file__).parent
dotenv_path = current_directory / ".env"
load_dotenv(dotenv_path=dotenv_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,  # 대소문자 구분 허용
        env_file=".env",  # settings env file name
        env_file_encoding="utf-8",  # setting env file encoding
    )

    KUBEFLOW_ENDPOINT: str
    KUBEFLOW_USERNAME: str
    KUBEFLOW_PASSWORD: str
    KUBEFLOW_NAMESPACE: str
    KUBEFLOW_EXPERIMENT_NAME: str

    DB_TYPE: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str

    MLFLOW_TRACKING_URI: str
    MLFLOW_TRACKING_USERNAME: str
    MLFLOW_TRACKING_PASSWORD: str
    MLFLOW_EXPERIMENT_NAME: str

    MLFLOW_S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    MLFLOW_S3_BUCKET: str

    INNOGRID_OBJECT_STORAGE_ENDPOINT: str
    INNOGRID_OBJECT_STORAGE_ACCESS_KEY: str
    INNOGRID_OBJECT_STORAGE_SECRET_KEY: str
    INNOGRID_OBJECT_STORAGE_BUCKET: str

    REST_API_URL: str
    DEMO_PASSWORD: str
    LOGIN_SECRET_KEY: str

    KSERVE_GPU: bool = False

    USER_MODELS: dict[str, dict] = {}

    @property
    def get_db_uri(self) -> str:
        """Environment variables로부터 DB 정보를 받아와 URI를 반환"""
        return f"{self.DB_TYPE}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def user_models(self) -> dict[str, dict]:
        return self.USER_MODELS

    def add_user_model(self, key: str, value: Any):
        self.USER_MODELS[key] = value


@lru_cache
def get_settings():
    return Settings()
