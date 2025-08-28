import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional

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

    class RESTAPIClient:
        """REST API 클라이언트 클래스"""

        def __init__(self, base_url: str, username: str, password: str):
            self.base_url = base_url.rstrip("/")
            self.username = username
            self.password = password
            self._token: Optional[str] = None
            self._session = requests.Session()
            self._session.timeout = 10

        def _get_headers(self) -> Dict[str, str]:
            """인증 헤더 반환"""
            if not self._token:
                self._token = self._get_token()
            return {"Authorization": f"Bearer {self._token}"} if self._token else {}

        def _get_token(self) -> Optional[str]:
            """REST API 토큰 획득"""
            try:
                response = self._session.post(
                    f"{self.base_url}/api/v1/authentications/token",
                    data={"username": self.username, "password": self.password},
                )

                if response.status_code == 200:
                    return response.json().get("access_token")
                else:
                    logger.error(f"REST API 로그인 실패: {response.status_code} - {response.text}")
                    return None

            except requests.exceptions.ConnectionError:
                logger.warning(f"REST API 서버에 연결할 수 없습니다: {self.base_url}")
                return None
            except Exception as e:
                logger.error(f"REST API 토큰 획득 중 오류 발생: {e}")
                return None

        def get_experiment_run_id(self, experiment_id: int) -> Optional[str]:
            """실험 ID로부터 MLflow run ID 조회"""
            try:
                response = self._session.get(
                    f"{self.base_url}/api/v1/experiments/{experiment_id}",
                    headers=self._get_headers(),
                )

                if response.status_code == 200:
                    return response.json().get("mlflow_run_id")
                else:
                    logger.error(f"실험 조회 실패: {response.status_code} - {response.text}")
                    return None

            except Exception as e:
                logger.error(f"실험 run ID 조회 중 오류 발생: {e}")
                return None

        def get_model_metadata(self) -> Dict[str, int]:
            """모델 메타데이터 조회 (provider, type, format ID)"""
            metadata = {}

            try:
                # Provider 조회
                provider_response = self._session.get(
                    f"{self.base_url}/api/v1/models/providers",
                    headers=self._get_headers(),
                    params={"provider_name": "custom"},
                )
                if provider_response.status_code == 200:
                    metadata["provider_id"] = provider_response.json().get("id")
                else:
                    raise Exception(f"Provider 조회 실패: {provider_response.text}")

                # Type 조회
                type_response = self._session.get(
                    f"{self.base_url}/api/v1/models/types",
                    headers=self._get_headers(),
                    params={"type_name": "Fine-Tuned"},
                )
                if type_response.status_code == 200:
                    metadata["type_id"] = type_response.json().get("id")
                else:
                    raise Exception(f"Type 조회 실패: {type_response.text}")

                # Format 조회
                format_response = self._session.get(
                    f"{self.base_url}/api/v1/models/formats",
                    headers=self._get_headers(),
                    params={"format_name": "pytorch"},
                )
                if format_response.status_code == 200:
                    metadata["format_id"] = format_response.json().get("id")
                else:
                    raise Exception(f"Format 조회 실패: {format_response.text}")

                return metadata

            except Exception as e:
                logger.error(f"모델 메타데이터 조회 중 오류 발생: {e}")
                raise

        def insert_model_metadata(self, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """모델 메타데이터 삽입"""
            try:
                if not self._token:
                    logger.warning("REST API 토큰이 없어 메타데이터 삽입을 건너뜁니다.")
                    return None

                response = self._session.post(
                    f"{self.base_url}/api/v1/models",
                    headers=self._get_headers(),
                    data=model_data,
                )

                if response.status_code == 200:
                    logger.info("메타데이터 삽입 성공")
                    return response.json()
                else:
                    logger.error(f"메타데이터 삽입 실패: {response.status_code} - {response.text}")
                    return None

            except requests.exceptions.ConnectionError:
                logger.warning(f"REST API 서버에 연결할 수 없습니다: {self.base_url}")
                return None
            except Exception as e:
                logger.error(f"메타데이터 삽입 중 오류 발생: {e}")
                return None

    def find_checkpoint_file(artifact_path: str) -> str:
        """체크포인트 파일 찾기"""
        # best_ckpt 파일 먼저 찾기
        best_ckpt = glob.glob(os.path.join(artifact_path, "**/best_ckpt.pth"), recursive=True)

        if best_ckpt:
            return best_ckpt[0]

        # best_ckpt가 없으면 가장 마지막 epoch 체크포인트 찾기
        epoch_ckpts = glob.glob(os.path.join(artifact_path, "**/epoch_*_ckpt.pth"), recursive=True)

        if not epoch_ckpts:
            raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다.")

        # epoch 번호 추출해서 가장 큰 것 선택
        latest_epoch = -1
        model_path = None

        for ckpt in epoch_ckpts:
            match = re.search(r"epoch_(\d+)_ckpt.pth", os.path.basename(ckpt))
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    model_path = ckpt

        if not model_path:
            raise FileNotFoundError("유효한 체크포인트 파일을 찾을 수 없습니다.")

        return model_path

    # 메인 로직
    try:
        # REST API 클라이언트 초기화
        api_client = RESTAPIClient(restapi_url, restapi_username, restapi_password)

        # MLflow 설정
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

        # 실험 run ID 조회
        run_id = api_client.get_experiment_run_id(experiment_id)
        if not run_id:
            raise Exception("실험 run ID를 조회할 수 없습니다.")

        # 아티팩트 다운로드
        download_uuid_path = str(uuid.uuid4())
        local_artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=download_uuid_path)

        try:
            # 체크포인트 파일 찾기
            model_path = find_checkpoint_file(local_artifact_path)

            # MLflow에 모델 등록
            with mlflow.start_run(run_name=f"{uuid.uuid4()}-model") as run:
                mlflow.log_artifact(local_path=model_path, artifact_path=download_uuid_path, run_id=run.info.run_id)

                # 모델 메타데이터 조회
                metadata = api_client.get_model_metadata()

                # 모델 데이터 준비
                model_data = {
                    "name": train_model_name,
                    "description": description,
                    "provider_id": metadata["provider_id"],
                    "type_id": metadata["type_id"],
                    "format_id": metadata["format_id"],
                    "parent_model_id": parent_model_id,
                    "model_registry_schema": json.dumps(
                        {
                            "artifact_path": run.info.artifact_uri,
                            "uri": run.info.artifact_uri,
                            "run_id": run_id,
                        }
                    ),
                }

                # 메타데이터 삽입
                api_client.insert_model_metadata(model_data)

        finally:
            # 임시 파일 정리
            if os.path.exists(local_artifact_path):
                try:
                    shutil.rmtree(local_artifact_path)
                    logger.info("임시 아티팩트 파일 정리 완료")
                except Exception as e:
                    logger.warning(f"임시 파일 정리 중 오류: {e}")

    except Exception as e:
        logger.error(f"모델 등록 중 오류 발생: {e}")
        raise
