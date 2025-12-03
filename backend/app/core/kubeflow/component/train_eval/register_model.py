from typing import Any, Dict, Optional

from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["mlflow==2.17.0", "requests==2.32.5", "loguru==0.7.3", "boto3==1.41.1"],
)
def register_model_component(
    experiment_id: int,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_s3_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    parent_model_id: int,
    train_model_name: str,
    description: str,
    restapi_url: str,
    restapi_username: str,
    restapi_password: str,
    provider_name: str,  # Enum 값 전달
    type_name: str,  # Enum 값 전달
    yolox_format_name: str,  # Enum 값 전달
    pytorch_format_name: str,  # Enum 값 전달
):
    import glob
    import json
    import logging
    import os
    import re
    import shutil
    import uuid

    import mlflow
    import requests

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

        def get_experiment_info(self, experiment_id: int) -> Optional[Dict[str, Any]]:
            """실험 ID로부터 실험 정보 조회 (mlflow_run_id, reference_model_id)"""
            try:
                response = self._session.get(
                    f"{self.base_url}/api/v1/experiments/{experiment_id}",
                    headers=self._get_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "mlflow_run_id": data.get("mlflow_run_id"),
                        "reference_model_id": data.get("reference_model_id"),
                    }
                else:
                    logger.error(f"실험 조회 실패: {response.status_code} - {response.text}")
                    return None

            except Exception as e:
                logger.error(f"실험 정보 조회 중 오류 발생: {e}")
                return None

        def get_model_info(self, model_id: int) -> Optional[Dict[str, Any]]:
            """모델 ID로부터 모델 정보 조회 (name, format_id)"""
            try:
                response = self._session.get(
                    f"{self.base_url}/api/v1/models/{model_id}",
                    headers=self._get_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    # format_info 객체에서 id 추출
                    format_id = None
                    if data.get("format_info"):
                        format_id = data["format_info"].get("id")

                    return {"name": data.get("name"), "format_id": format_id}
                else:
                    logger.error(f"모델 조회 실패: {response.status_code} - {response.text}")
                    return None

            except Exception as e:
                logger.error(f"모델 정보 조회 중 오류 발생: {e}")
                return None

        def get_model_metadata(self, provider_name: str, type_name: str) -> Dict[str, int]:
            """모델 메타데이터 조회 (provider, type ID)"""
            metadata = {}

            try:
                # Provider 조회
                provider_response = self._session.get(
                    f"{self.base_url}/api/v1/models/providers",
                    headers=self._get_headers(),
                    params={"provider_name": provider_name},
                )
                if provider_response.status_code == 200:
                    metadata["provider_id"] = provider_response.json().get("id")
                else:
                    raise Exception(f"Provider 조회 실패: {provider_response.text}")

                # Type 조회
                type_response = self._session.get(
                    f"{self.base_url}/api/v1/models/types",
                    headers=self._get_headers(),
                    params={"type_name": type_name},
                )
                if type_response.status_code == 200:
                    metadata["type_id"] = type_response.json().get("id")
                else:
                    raise Exception(f"Type 조회 실패: {type_response.text}")

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
        # AWS 자격 증명 환경 변수 설정 (MLflow S3 접근용)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

        # REST API 클라이언트 초기화
        api_client = RESTAPIClient(restapi_url, restapi_username, restapi_password)

        # MLflow 설정
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

        # 실험 정보 조회 (mlflow_run_id, reference_model_id)
        experiment_info = api_client.get_experiment_info(experiment_id)
        if not experiment_info:
            raise Exception("실험 정보를 조회할 수 없습니다.")

        run_id = experiment_info.get("mlflow_run_id")
        reference_model_id = experiment_info.get("reference_model_id")

        if not run_id:
            raise Exception("실험 run ID를 조회할 수 없습니다.")
        if not reference_model_id:
            raise Exception("실험의 reference_model_id를 조회할 수 없습니다.")

        # 아티팩트 다운로드
        local_artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        try:
            # 체크포인트 파일 찾기
            original_model_path = find_checkpoint_file(local_artifact_path)

            # reference_model의 정보 조회 (name, format_id)
            reference_model_info = api_client.get_model_info(reference_model_id)
            if not reference_model_info:
                raise Exception(f"모델 ID {reference_model_id}의 정보를 조회할 수 없습니다.")

            reference_model_name = reference_model_info.get("name")
            reference_format_id = reference_model_info.get("format_id")

            if not reference_model_name:
                raise Exception(f"모델 ID {reference_model_id}의 이름을 조회할 수 없습니다.")

            # format_id가 없는 경우 기본값 설정 (pytorch 또는 yolox 등)
            if not reference_format_id:
                logger.warning(
                    f"모델 ID {reference_model_id}의 format_id를 조회할 수 없습니다. 기본값으로 pytorch 사용"
                )
                # 모델 이름으로 format 유추
                if yolox_format_name.lower() in reference_model_name.lower():
                    # YOLOX 모델인 경우
                    format_response = api_client._session.get(
                        f"{api_client.base_url}/api/v1/models/formats",
                        headers=api_client._get_headers(),
                        params={"format_name": yolox_format_name},
                    )
                    if format_response.status_code == 200:
                        reference_format_id = format_response.json().get("id")
                else:
                    # 기본값으로 pytorch 사용
                    format_response = api_client._session.get(
                        f"{api_client.base_url}/api/v1/models/formats",
                        headers=api_client._get_headers(),
                        params={"format_name": pytorch_format_name},
                    )
                    if format_response.status_code == 200:
                        reference_format_id = format_response.json().get("id")

                if not reference_format_id:
                    raise Exception("기본 format_id도 조회할 수 없습니다.")

            reference_model_name = reference_model_name.replace("/", "-")
            # 파일명을 reference_model_name.pth로 변경
            model_dir = os.path.dirname(original_model_path)
            new_model_filename = f"{reference_model_name}.pth"
            new_model_path = os.path.join(model_dir, new_model_filename)

            # 파일명 변경
            shutil.copy2(original_model_path, new_model_path)
            logger.info(f"체크포인트 파일명 변경: {os.path.basename(original_model_path)} -> {new_model_filename}")

            # MLflow에 모델 등록
            with mlflow.start_run(run_name=f"{uuid.uuid4()}-model") as run:
                mlflow.log_artifact(
                    local_path=new_model_path, artifact_path=reference_model_name, run_id=run.info.run_id
                )

                # 모델 메타데이터 조회 (provider, type만 조회)
                metadata = api_client.get_model_metadata(provider_name, type_name)

                # 모델 데이터 준비 (format_id는 reference_model의 것을 사용)
                model_data = {
                    "name": train_model_name,
                    "description": description,
                    "provider_id": metadata["provider_id"],
                    "type_id": metadata["type_id"],
                    "format_id": reference_format_id,  # reference_model의 format_id 사용
                    "parent_model_id": parent_model_id,
                    "model_registry_schema": json.dumps(
                        {
                            "artifact_path": f"{run.info.artifact_uri}/{reference_model_name}",
                            "uri": reference_model_name,
                            "run_id": run.info.run_id,
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
