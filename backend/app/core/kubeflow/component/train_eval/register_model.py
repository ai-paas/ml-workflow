import uuid

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "mlflow==2.17.0",
        "requests==*",
        "loguru==0.7.3",
    ],
)
def register_model_component(
    experiment_id: int,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    parent_model_id: int,
    train_model_name: str,
    description: str,
    restapi_url: str,
    restapi_username: str,
    restapi_password: str,
):
    import glob
    import json
    import logging
    import os
    import re

    import mlflow
    import requests
    from mlflow import MlflowClient

    logger = logging.getLogger(__name__)

    def get_experiment_run_id(experiment_id: int, restapi_url: str, restapi_token: str):
        response = requests.get(
            f"{restapi_url}/api/v1/experiments/{experiment_id}",
            headers={"Authorization": f"Bearer {restapi_token}"},
        )
        return response.json()["mlflow_run_id"]

    def insert_metadata(
        run_id: str,
        artifact_uri: str,
        parent_model_id: int,
        model_version: str,
        model_uri: str,
        train_model_name: str,
        description: str,
        restapi_url: str,
        restapi_token: str,
    ):
        """메타데이터 삽입"""
        try:
            if not restapi_token:
                logger.warning("REST API 토큰이 없어 메타데이터 삽입을 건너뜁니다.")
                return None

            # API 토큰 헤더 설정
            headers = {"Authorization": f"Bearer {restapi_token}"}

            # provider, type, format ID 조회 (타임아웃 추가)
            provider_response = requests.get(
                f"{restapi_url}/api/v1/models/providers",
                headers=headers,
                params={"provider_name": "custom"},
                timeout=10,
            )
            if provider_response.status_code != 200:
                raise Exception(f"Provider 조회 실패: {provider_response.text}")
            provider_id = provider_response.json().get("id")

            type_response = requests.get(
                f"{restapi_url}/api/v1/models/types", headers=headers, params={"type_name": "Fine-Tuned"}
            )
            if type_response.status_code != 200:
                raise Exception(f"Type 조회 실패: {type_response.text}")
            type_id = type_response.json().get("id")

            format_response = requests.get(
                f"{restapi_url}/api/v1/models/formats", headers=headers, params={"format_name": "pytorch"}
            )
            if format_response.status_code != 200:
                raise Exception(f"Format 조회 실패: {format_response.text}")
            format_id = format_response.json().get("id")

            data = {
                "name": train_model_name,
                "description": description,
                "provider_id": provider_id,
                "type_id": type_id,
                "format_id": format_id,
                "parent_model_id": parent_model_id,
                "model_registry_schema": json.dumps(
                    {
                        "artifact_path": artifact_uri,
                        "uri": model_uri,
                        "run_id": run_id,
                    }
                ),
            }

            api_endpoint = f"{restapi_url}/api/v1/models"
            response = requests.post(api_endpoint, headers=headers, data=data)

            if response.status_code == 200:
                logger.info("메타데이터 삽입 성공")
                return response.json()
            else:
                logger.error(f"메타데이터 삽입 실패: {response.status_code}")
                logger.error(f"메타데이터 삽입 실패: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            logger.warning(f"REST API 서버에 연결할 수 없습니다: {restapi_url}")
            return None
        except Exception as e:
            logger.error(f"메타데이터 삽입 중 오류 발생: {e}")
            return None

    def get_token_from_restapi(url: str, username: str, password: str) -> str:
        """REST API 토큰 획득"""
        try:
            response = requests.post(
                f"{url}/api/v1/authentications/token",
                data={"username": username, "password": password},
                timeout=10,  # 타임아웃 추가
            )

            if response.status_code == 200:
                return response.json()["access_token"]
            else:
                logger.error(f"REST API 로그인 실패: {response.status_code}")
                return ""

        except requests.exceptions.ConnectionError:
            logger.warning(f"REST API 서버에 연결할 수 없습니다: {url}")
            return ""
        except Exception as e:
            logger.error(f"REST API 토큰 획득 중 오류 발생: {e}")
            return ""

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    run_id = get_experiment_run_id(
        experiment_id, restapi_url, get_token_from_restapi(restapi_url, restapi_username, restapi_password)
    )
    download_uuid_path = uuid.uuid4()
    local_artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=download_uuid_path)

    # 체크포인트 파일 찾기
    best_ckpt = glob.glob(os.path.join(local_artifact_path, "**/best_ckpt.pth"), recursive=True)

    if best_ckpt:
        model_path = best_ckpt[0]
    else:
        # best_ckpt가 없으면 가장 마지막 epoch 체크포인트 찾기
        epoch_ckpts = glob.glob(os.path.join(local_artifact_path, "**/epoch_*_ckpt.pth"), recursive=True)

        if not epoch_ckpts:
            raise Exception("체크포인트 파일을 찾을 수 없습니다.")

        # epoch 번호 추출해서 가장 큰 것 선택
        latest_epoch = -1
        model_path = None

        for ckpt in epoch_ckpts:
            epoch_num = int(re.search(r"epoch_(\d+)_ckpt.pth", os.path.basename(ckpt)).group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                model_path = ckpt

    with mlflow.start_run(run_name=f"{uuid.uuid4()}-model") as run:
        mlflow.log_artifact(local_path=model_path, artifact_path=download_uuid_path, run_id=run.info.run_id)

        insert_metadata(
            run_id=run_id,
            artifact_uri=run.info.artifact_uri,
            parent_model_id=parent_model_id,
            model_version="1",
            model_uri=run.info.artifact_uri,
            train_model_name=train_model_name,
            description=description,
            restapi_url=restapi_url,
            restapi_token=get_token_from_restapi(restapi_url, restapi_username, restapi_password),
        )
