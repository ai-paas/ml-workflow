from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
current_directory = Path(__file__).parent
dotenv_path = current_directory / ".env"
load_dotenv(dotenv_path=dotenv_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,  # 대소문자 구분 허용
        env_file=".env",  # settings env file name
        env_file_encoding="utf-8",  # setting env file encoding
    )

    REST_API_URL: str


@lru_cache
def get_settings():
    return Settings()
