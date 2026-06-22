#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_eval 공통 헬퍼.

YOLOX(`CustomTrainModel`)와 ESM2(`EsmFineTuner`)가 공유하는 MLflow 설정,
데이터셋 다운로드, REST API 연동을 모아둔다. 무거운 학습 의존성(torch/transformers/YOLOX)을
import 하지 않으므로 entrypoint 에서 가볍게 가져올 수 있다.
"""

import os
import tempfile
from pathlib import Path

import mlflow
import requests
from loguru import logger


def setup_mlflow(args) -> None:
    """MLflow tracking URI / S3 자격 환경변수 설정."""
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.mlflow_s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_access_key
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)


def _download_from_s3(args) -> str:
    """boto3 로 S3 호환 스토리지에서 데이터셋 다운로드. 다운로드 디렉토리 경로를 반환."""
    import boto3

    object_key = args.dataset_download_ref
    filename = Path(object_key).name
    download_dir = tempfile.mkdtemp()
    local_path = os.path.join(download_dir, filename)

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.dataset_s3_endpoint_url,
        aws_access_key_id=args.dataset_s3_access_key,
        aws_secret_access_key=args.dataset_s3_secret_key,
    )
    logger.info(f"S3에서 다운로드: s3://{args.dataset_s3_bucket}/{object_key} → {local_path}")
    s3_client.download_file(args.dataset_s3_bucket, object_key, local_path)
    logger.info(f"S3 다운로드 완료: {local_path}")
    return download_dir


def _download_from_hubconnect(args) -> str:
    """Hub-Connect REST API 로 데이터레이크에서 데이터셋 다운로드. 다운로드 디렉토리 경로를 반환."""
    import httpx

    object_key = args.dataset_download_ref
    filename = Path(object_key).name
    download_dir = tempfile.mkdtemp()
    local_path = os.path.join(download_dir, filename)

    with httpx.Client(timeout=600.0) as client:
        token_resp = client.post(
            f"{args.datalake_api_url}/api/v1/auth/login",
            data={
                "username": args.datalake_api_username,
                "password": args.datalake_api_password,
                "grant_type": "password",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_resp.raise_for_status()
        token = token_resp.json()["access_token"]

        logger.info(f"Hub-Connect에서 다운로드: {args.datalake_bucket_name}/{object_key} → {local_path}")
        with client.stream(
            "GET",
            f"{args.datalake_api_url}/api/v1/buckets/{args.datalake_bucket_name}/objects/{object_key}",
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    logger.info(f"Hub-Connect 다운로드 완료: {local_path}")
    return download_dir


def download_dataset_dir(args) -> str:
    """dataset_storage_type 에 따라 데이터셋을 다운로드하고, zip 이 들어있는 디렉토리를 반환."""
    if args.dataset_storage_type == "s3":
        return _download_from_s3(args)
    if args.dataset_storage_type == "hubconnect":
        return _download_from_hubconnect(args)
    raise ValueError(f"지원하지 않는 storage_type: {args.dataset_storage_type}")


def get_token_from_restapi(url: str, username: str, password: str) -> str:
    """REST API 인증 토큰 획득."""
    try:
        response = requests.post(
            f"{url}/api/v1/authentications/token",
            data={"username": username, "password": password},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        logger.error(f"REST API 로그인 실패: {response.status_code}")
        return ""
    except requests.exceptions.ConnectionError:
        logger.warning(f"REST API 서버에 연결할 수 없습니다: {url}")
        return ""
    except Exception as e:
        logger.error(f"REST API 토큰 획득 중 오류 발생: {e}")
        return ""


def update_experiment(
    restapi_url: str,
    restapi_token: str,
    experiment_id: int,
    status: str = None,
    mlflow_run_id: str = None,
    kubeflow_run_id: str = None,
):
    """실험 정보 업데이트 (internal-access 전용 API). status/mlflow_run_id 등을 PATCH."""
    try:
        data = {}
        if status:
            data["status"] = status
        if mlflow_run_id:
            data["mlflow_run_id"] = mlflow_run_id
        if kubeflow_run_id:
            data["kubeflow_run_id"] = kubeflow_run_id
        response = requests.patch(
            f"{restapi_url}/api/v1/experiments/{experiment_id}/internal-access",
            json=data,
            headers={"Authorization": f"Bearer {restapi_token}"},
        )
        if response.status_code == 200:
            logger.info("실험 업데이트 성공")
            return response.json()
        logger.error(f"실험 업데이트 실패: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"REST API 서버에 연결할 수 없습니다: {restapi_url}")
        return None
