import logging
import os
from pathlib import Path

import boto3
from core.storage.base import StorageClient

logger = logging.getLogger(__name__)


class S3StorageClient(StorageClient):
    """S3 호환 오브젝트 스토리지 클라이언트 (MinIO, AWS S3, Ceph S3 등)"""

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket: str):
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """버킷 존재 확인, 없으면 생성"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except self.s3_client.exceptions.ClientError:
            logger.info(f"버킷 생성: {self.bucket}")
            self.s3_client.create_bucket(Bucket=self.bucket)

    def upload_file(self, local_path: str, object_key: str) -> str:
        with open(local_path, "rb") as f:
            self.s3_client.upload_fileobj(f, self.bucket, object_key)
        logger.info(f"S3 업로드 완료: s3://{self.bucket}/{object_key}")
        return object_key

    def download_file(self, object_key: str, local_dir: str) -> str:
        filename = Path(object_key).name
        local_path = os.path.join(local_dir, filename)
        self.s3_client.download_file(self.bucket, object_key, local_path)
        logger.info(f"S3 다운로드 완료: {local_path}")
        return local_path

    def delete_object(self, object_key: str) -> bool:
        self.s3_client.delete_object(Bucket=self.bucket, Key=object_key)
        logger.info(f"S3 오브젝트 삭제 완료: {object_key}")
        return True

    def delete_folder(self, prefix: str) -> bool:
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        delete_keys = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    delete_keys.append({"Key": obj["Key"]})

        for key_obj in delete_keys:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key_obj["Key"])

        logger.info(f"S3 폴더 삭제 완료: {prefix} ({len(delete_keys)}개 오브젝트)")
        return True
