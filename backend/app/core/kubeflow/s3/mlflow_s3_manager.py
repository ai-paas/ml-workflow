import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

import boto3
from config.settings import get_settings
from fastapi import UploadFile

settings = get_settings()


class MLFlowS3Manager:
    __instance: Optional["MLFlowS3Manager"] = None

    def __init__(self):
        """
        __instance가 None일 때만 초기화를 수행합니다.
        """
        if not hasattr(self, "initialized"):
            self.endpoint = settings.MLFLOW_S3_ENDPOINT_URL
            self.access_key = settings.MLFLOW_S3_ACCESS_KEY_ID
            self.secret_key = settings.MLFLOW_S3_SECRET_ACCESS_KEY
            self.bucket = settings.MLFLOW_S3_BUCKET
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                # verify=False,
            )
            self.initialized = True

    @classmethod
    def get_instance(cls) -> "MLFlowS3Manager":
        """
        S3Manager의 인스턴스를 반환합니다.
        """
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def download_file(self, file_url: str):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.s3_client.download_file(
                self.bucket,
                file_url,
                temp_file.name,
            )
            # file_url에서 파일 이름 추출
            original_filename = Path(file_url).name

            upload_file = UploadFile(
                filename=original_filename,
                file=open(temp_file.name, "rb"),
                headers={"content-type": "application/octet-stream"},
            )

            return upload_file
        except Exception as e:
            raise RuntimeError(f"파일 다운로드 중 오류 발생: {str(e)}")

    def upload_file(self, file: UploadFile):
        try:
            # FastAPI의 UploadFile에서 파일 객체 가져오기
            file_obj = file.file
            filename = file.filename
            file_url = f"{filename}"
            # S3 클라이언트를 사용하여 파일 업로드
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket,
                file_url,
            )

            # 업로드된 파일의 URL 생성
            # TODO: 사용자, 팀 저장소 정보에 따라 구분하도록 수정 필요.
            return file_url

        except Exception as e:
            raise Exception(f"파일 업로드 중 오류 발생: {str(e)}")

    def delete_object(self, file_url: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=file_url)
            return True
        except Exception as e:
            raise Exception(f"파일 삭제 중 오류 발생: {str(e)}")

    @staticmethod
    def s3_path_from_artifact_uri(artifact_uri: Optional[str]) -> Optional[str]:
        """MLflow artifact_uri 에서 S3 오브젝트 경로(버킷명 제외)를 추출한다. 파싱 불가 시 None.

        delete_folder 에 넘길 prefix 를 얻는 용도. 두 형식을 지원한다:
          mlflow-artifacts:/0/abc/artifacts                  -> 0/abc/artifacts
          s3://mlflow/8/abc/artifacts/google-owlv2-base-...  -> 8/abc/artifacts/google-owlv2-base-...
        """
        if not artifact_uri:
            return None
        if artifact_uri.startswith("mlflow-artifacts:/"):
            return artifact_uri.replace("mlflow-artifacts:/", "")
        if artifact_uri.startswith("s3://"):
            rest = artifact_uri.replace("s3://", "")  # bucket/key...
            if "/" in rest:
                return rest.split("/", 1)[1]  # 첫 '/' 이후(버킷명 제거)
        return None

    def delete_folder(self, folder_path: str) -> bool:
        """
        지정된 폴더 경로와 그 안의 모든 파일을 삭제합니다.

        Args:
            folder_path: 삭제할 폴더 경로 (예: "environment/reference/123/")

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 폴더 경로가 '/'로 끝나지 않으면 추가
            if not folder_path.endswith("/"):
                folder_path = f"{folder_path}/"

            # 해당 폴더 내의 모든 객체 리스트 가져오기
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=folder_path)

            # 삭제할 객체 목록 생성
            delete_keys = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        delete_keys.append({"Key": obj["Key"]})

            # 객체가 있는 경우에만 삭제 실행
            # 일부 S3 호환 스토리지의 Content-MD5 헤더 요구사항을 피하기 위해
            # delete_objects 대신 개별 delete_object 호출 사용
            if delete_keys:
                for key_obj in delete_keys:
                    try:
                        self.s3_client.delete_object(Bucket=self.bucket, Key=key_obj["Key"])
                    except Exception as delete_error:
                        # 개별 객체 삭제 실패는 로깅만 하고 계속 진행
                        # (이미 삭제된 객체일 수 있음)
                        warnings.warn(f"S3 객체 삭제 실패 (Key: {key_obj['Key']}): {str(delete_error)}")

            return True
        except Exception as e:
            raise Exception(f"폴더 삭제 중 오류 발생: {str(e)}")

    def list_objects(
        self, prefix: str, limit: int = 1000, cursor: Optional[str] = None
    ) -> Tuple[List[dict], Optional[str]]:
        """prefix 아래 오브젝트를 한 쪽 나열한다. 반환 (목록, next_cursor).

        목록 항목: {"key", "size", "last_modified"}. last_modified 는 boto3 가 주는 tz-aware UTC 그대로다.
        키는 사전순으로 오므로 쪽을 넘어가도 순서가 유지되고 중복·누락이 없다.
        """
        kwargs: dict = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": max(1, limit)}
        if cursor:
            kwargs["ContinuationToken"] = cursor
        resp = self.s3_client.list_objects_v2(**kwargs)
        items = [
            {"key": o["Key"], "size": o["Size"], "last_modified": o["LastModified"]}
            for o in (resp.get("Contents") or [])
            # 디렉터리 표시용 키가 섞여 들어오면 파일이 아니므로 뺀다.
            if not o["Key"].endswith("/")
        ]
        return items, resp.get("NextContinuationToken")

    def object_exists(self, key: str) -> Optional[dict]:
        """오브젝트가 있으면 {"size", "last_modified"}, 없으면 None."""
        try:
            head = self.s3_client.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        return {"size": head["ContentLength"], "last_modified": head["LastModified"]}

    def presigned_get_url(self, key: str, expires_in: int = 300) -> str:
        """다운로드용 서명 URL. 인증 없이 접근 가능하므로 로그에 남기지 않는다."""
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=max(1, int(expires_in)),
        )

    def get_full_url(self, file_url: str):
        url = f"{self.endpoint}/{self.bucket}/{file_url}"
        encoded_url = quote(url, safe=":/?=")
        return encoded_url
