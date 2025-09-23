import tempfile
from pathlib import Path
from typing import Optional
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
            self.access_key = settings.AWS_ACCESS_KEY_ID
            self.secret_key = settings.AWS_SECRET_ACCESS_KEY
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
            raise Exception(f"파일 다운로드 중 오류 발생: {str(e)}")

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
            if delete_keys:
                self.s3_client.delete_objects(Bucket=self.bucket, Delete={"Objects": delete_keys})

            return True
        except Exception as e:
            raise Exception(f"폴더 삭제 중 오류 발생: {str(e)}")

    def get_full_url(self, file_url: str):
        url = f"{self.endpoint}/{self.bucket}/{file_url}"
        encoded_url = quote(url, safe=":/?=")
        return encoded_url
