import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from core.storage.base import StorageClient

logger = logging.getLogger(__name__)


class HubConnectStorageClient(StorageClient):
    """KISTI Hub-Connect API 기반 데이터레이크 스토리지 클라이언트"""

    def __init__(self, base_url: str, username: str, password: str, bucket_name: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.bucket_name = bucket_name
        self._token: Optional[str] = None

    def _authenticate(self) -> str:
        """POST /api/v1/auth/login 으로 Bearer 토큰 발급"""
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/api/v1/auth/login",
                data={
                    "username": self.username,
                    "password": self.password,
                    "grant_type": "password",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            self._token = response.json()["access_token"]
            return self._token

    def _get_auth_header(self) -> dict[str, str]:
        if not self._token:
            self._authenticate()
        return {"Authorization": f"Bearer {self._token}"}

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """401 수신 시 토큰 재발급 후 1회 재시도"""
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_header())
        kwargs["headers"] = headers

        with httpx.Client(timeout=kwargs.pop("timeout", 60.0)) as client:
            response = getattr(client, method)(url, **kwargs)
            if response.status_code == 401:
                logger.info("Hub-Connect 토큰 만료, 재발급 시도")
                self._authenticate()
                kwargs["headers"].update(self._get_auth_header())
                response = getattr(client, method)(url, **kwargs)
            response.raise_for_status()
            return response

    def _ensure_bucket(self) -> None:
        """버킷 존재 확인, 없으면 생성"""
        try:
            self._request_with_retry("get", f"{self.base_url}/api/v1/buckets/{self.bucket_name}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"버킷 생성: {self.bucket_name}")
                self._request_with_retry(
                    "post",
                    f"{self.base_url}/api/v1/buckets",
                    json={"name": self.bucket_name},
                )
            else:
                raise

    def upload_file(self, local_path: str, object_key: str) -> str:
        self._ensure_bucket()
        prefix = str(Path(object_key).parent)

        if not self._token:
            self._authenticate()

        with open(local_path, "rb") as f:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(
                    f"{self.base_url}/api/v1/buckets/{self.bucket_name}/objects",
                    params={"prefix": prefix},
                    files={"file": (Path(local_path).name, f)},
                    headers=self._get_auth_header(),
                )
                if response.status_code == 401:
                    self._authenticate()
                    f.seek(0)
                    response = client.post(
                        f"{self.base_url}/api/v1/buckets/{self.bucket_name}/objects",
                        params={"prefix": prefix},
                        files={"file": (Path(local_path).name, f)},
                        headers=self._get_auth_header(),
                    )
                response.raise_for_status()

        logger.info(f"Hub-Connect 업로드 완료: {self.bucket_name}/{object_key}")
        return object_key

    def download_file(self, object_key: str, local_dir: str) -> str:
        filename = Path(object_key).name
        local_path = os.path.join(local_dir, filename)

        if not self._token:
            self._authenticate()

        with httpx.Client(timeout=600.0) as client:
            with client.stream(
                "GET",
                f"{self.base_url}/api/v1/buckets/{self.bucket_name}/objects/{object_key}",
                headers=self._get_auth_header(),
            ) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)

        logger.info(f"Hub-Connect 다운로드 완료: {local_path}")
        return local_path

    def delete_object(self, object_key: str) -> bool:
        try:
            self._request_with_retry(
                "delete",
                f"{self.base_url}/api/v1/buckets/{self.bucket_name}/objects/{object_key}",
            )
            logger.info(f"Hub-Connect 오브젝트 삭제 완료: {object_key}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"삭제 대상 미존재 (무시): {object_key}")
                return True
            raise

    def delete_folder(self, prefix: str) -> bool:
        try:
            self._request_with_retry(
                "delete",
                f"{self.base_url}/api/v1/buckets/{self.bucket_name}/folders/{prefix}",
            )
            logger.info(f"Hub-Connect 폴더 삭제 완료: {prefix}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"삭제 대상 폴더 미존재 (무시): {prefix}")
                return True
            raise
