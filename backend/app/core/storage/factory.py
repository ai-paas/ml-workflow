from typing import Optional

from config.settings import get_settings
from core.storage.base import StorageClient

settings = get_settings()

_instance: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """환경변수 DATASET_STORAGE_TYPE에 따라 적절한 StorageClient 구현체를 반환 (싱글턴)"""
    global _instance
    if _instance is not None:
        return _instance

    storage_type = settings.DATASET_STORAGE_TYPE

    if storage_type == "s3":
        from core.storage.s3_client import S3StorageClient

        _instance = S3StorageClient(
            endpoint_url=settings.S3_ENDPOINT,
            access_key=settings.S3_ACCESS_KEY,
            secret_key=settings.S3_SECRET_KEY,
            bucket=settings.S3_BUCKET,
        )
    elif storage_type == "hubconnect":
        from core.storage.hubconnect_client import HubConnectStorageClient

        _instance = HubConnectStorageClient(
            base_url=settings.DATALAKE_API_URL,
            username=settings.DATALAKE_API_USERNAME,
            password=settings.DATALAKE_API_PASSWORD,
            bucket_name=settings.DATALAKE_BUCKET_NAME,
        )
    else:
        raise ValueError(f"지원하지 않는 DATASET_STORAGE_TYPE: {storage_type}")

    return _instance
