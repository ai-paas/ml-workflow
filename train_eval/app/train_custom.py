#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import requests
from mlflow import MlflowClient

# 현재 파일의 절대 경로 얻기
current_path = Path(__file__).absolute().parent

logger = logging.getLogger(__name__)


class CustomTrainModel:
    """커스텀 모델 학습 클래스"""

    def __init__(
        self,
        train_name: str,
        model_name: str,
        model_uri: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        mlflow_experiment_name: str,
        dataset_artifact_uri: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
        **kwargs,
    ):
        """
        커스텀 모델 학습 클래스 초기화

        Args:
            train_name: 학습 실행명
            model_name: 모델명
            model_uri: 모델 URI
            mlflow_tracking_uri: MLflow 추적 URI
            mlflow_s3_endpoint_url: MLflow S3 엔드포인트 URL
            aws_access_key_id: AWS 액세스 키 ID
            aws_secret_access_key: AWS 시크릿 액세스 키
            mlflow_experiment_name: MLflow 실험명
            dataset_artifact_uri: 데이터셋 아티팩트 URI
            restapi_url: REST API URL
            restapi_username: REST API 사용자명
            restapi_password: REST API 비밀번호
        """
        self.train_name = train_name
        self.model_name = model_name
        self.model_uri = model_uri
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_s3_endpoint_url = mlflow_s3_endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.mlflow_experiment_name = mlflow_experiment_name
        self.dataset_artifact_uri = dataset_artifact_uri
        self.restapi_url = restapi_url
        self.restapi_username = restapi_username
        self.restapi_password = restapi_password

        # 기본 설정
        self.output_dir = current_path / "outputs"
        self.mlflow_run_id = None
        self.client = None

        # MLflow 설정
        self._setup_mlflow()

    def _setup_mlflow(self):
        """MLflow 설정"""
        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.client = MlflowClient()

        # 실험 설정 또는 생성
        try:
            experiment = self.client.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(
                    name=self.mlflow_experiment_name,
                    artifact_location=f"s3://mlflow-artifacts/{self.mlflow_experiment_name}",
                )
                logger.info(f"실험 생성됨: {self.mlflow_experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"기존 실험 사용: {self.mlflow_experiment_name}")
        except Exception as e:
            logger.error(f"MLflow 실험 설정 중 오류: {e}")
            raise

    def preprocess(self):
        """데이터 전처리"""
        try:
            # 데이터 전처리 로직 구현
            logger.info("데이터 전처리 완료")
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {e}")
            raise

    def train(self):
        """모델 학습"""
        try:
            # MLflow 실행 시작
            with mlflow.start_run(
                experiment_id=self.client.get_experiment_by_name(self.mlflow_experiment_name).experiment_id
            ) as run:
                self.mlflow_run_id = run.info.run_id

                # 하이퍼파라미터 로깅
                mlflow.log_params(
                    {
                        "model_name": self.model_name,
                        "model_uri": self.model_uri,
                    }
                )

                # 출력 디렉토리 생성
                self.output_dir.mkdir(parents=True, exist_ok=True)

                # MLflow에서 모델 불러오기
                logger.info(f"MLflow에서 모델 불러오기: {self.model_uri}")
                try:
                    model = mlflow.pyfunc.load_model(self.model_uri)
                    logger.debug("모델정보:", model)
                    logger.info("모델 로드 완료")
                except Exception as e:
                    logger.error(f"모델 로드 중 오류: {e}")
                    raise

                # 여기에 실제 학습 로직 구현
                logger.info("학습 완료")

        except Exception as e:
            logger.error(f"학습 중 오류: {e}")
            raise

    def postprocess(self):
        """학습 후 처리"""
        try:
            # 모델 저장 및 등록
            train_model_name = f"{self.model_name}-custom-fine-tuned"

            with mlflow.start_run(run_name=train_model_name) as run:
                # 모델 아티팩트 로깅
                # 여기에 모델 저장 로직 구현
                mlflow.log_artifact(str(self.output_dir), "model")

                # 모델 등록
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, train_model_name)

                # TODO: metadata 저장 로직 주석 해제되면 같이 해제
                # run_id = run.info.run_id
                # artifact_uri = mlflow.get_artifact_uri()

                # 최신 모델 버전 가져오기
                model_version = self.client.get_latest_versions(name=train_model_name, stages=["None"])[0].version
                logger.info(f"모델 버전: {model_version}")

                # TODO: metadata 저장 로직 주석 해제되면 같이 해제
                # train_model_uri = f"models:/{train_model_name}/{model_version}"

                logger.info(f"모델 등록 완료: {train_model_name} (버전: {model_version})")

                # TODO : model_학습 제대로 완료되면 같이 테스트
                # 메타데이터 저장
                # self.insert_metadata(
                #     run_id=run_id,
                #     artifact_uri=artifact_uri,
                #     model_version=model_version,
                #     model_uri=train_model_uri,
                #     train_model_name=train_model_name,
                #     restapi_url=self.restapi_url,
                #     restapi_token=self.get_token_from_restapi(
                #         url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                #     ),
                # )

        except Exception as e:
            logger.error(f"후처리 중 오류 발생: {e}")
            traceback.print_exc()
            raise

    def insert_metadata(
        self,
        run_id: str,
        artifact_uri: str,
        model_version: str,
        model_uri: str,
        train_model_name: str,
        restapi_url: str,
        restapi_token: str,
    ):
        """메타데이터 삽입"""
        try:
            data = {
                "name": train_model_name,
                "description": f"커스텀 파인튜닝 모델: {train_model_name}",
                "model_provider_id": 3,
                "model_type_id": 4,
                "model_format_id": 1,
                "model_registry_schema": json.dumps(
                    {
                        "run_id": run_id,
                        "artifact_path": artifact_uri,
                        "versions": model_version,
                        "model_uri": model_uri,
                        "framework": "custom",
                        "model_type": "custom",
                    }
                ),
            }

            api_endpoint = f"{restapi_url}/api/v1/models"
            headers = {"Authorization": f"Bearer {restapi_token}"}
            response = requests.post(api_endpoint, headers=headers, data=data)

            if response.status_code == 200:
                logger.info("메타데이터 삽입 성공")
                return response.json()
            else:
                logger.error(f"메타데이터 삽입 실패: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"메타데이터 삽입 중 오류 발생: {e}")
            return None

    def get_token_from_restapi(self, url: str, username: str, password: str) -> str:
        """REST API 토큰 획득"""
        try:
            response = requests.post(
                f"{url}/api/v1/authentications/token", data={"username": username, "password": password}
            )

            if response.status_code == 200:
                return response.json()["access_token"]
            else:
                logger.error(f"REST API 로그인 실패: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"REST API 토큰 획득 중 오류 발생: {e}")
            return ""


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="커스텀 모델 학습")

    # 기본 설정
    parser.add_argument("--train_name", type=str, required=True, help="학습 실행명")
    parser.add_argument("--model_name", type=str, required=True, help="모델명")
    parser.add_argument("--model_uri", type=str, required=True, help="모델 URI")
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True, help="MLflow 추적 URI")
    parser.add_argument("--mlflow_experiment_name", type=str, required=True, help="MLflow 실험명")
    parser.add_argument("--mlflow_s3_endpoint_url", type=str, required=True, help="MLflow S3 엔드포인트 URL")
    parser.add_argument("--aws_access_key_id", type=str, required=True, help="AWS 액세스 키 ID")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help="AWS 시크릿 액세스 키")
    parser.add_argument("--dataset_artifact_uri", type=str, required=True, help="데이터셋 아티팩트 URI")
    parser.add_argument("--restapi_url", type=str, required=True, help="REST API URL")
    parser.add_argument("--restapi_username", type=str, required=True, help="REST API 사용자명")
    parser.add_argument("--restapi_password", type=str, required=True, help="REST API 비밀번호")

    # 인자 파싱
    args = parser.parse_args()

    # 모델 초기화
    model = CustomTrainModel(
        train_name=args.train_name,
        model_name=args.model_name,
        model_uri=args.model_uri,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_s3_endpoint_url=args.mlflow_s3_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        dataset_artifact_uri=args.dataset_artifact_uri,
        restapi_url=args.restapi_url,
        restapi_username=args.restapi_username,
        restapi_password=args.restapi_password,
    )

    try:
        # 데이터 전처리
        model.preprocess()

        # 학습 실행
        model.train()

        # 후처리
        model.postprocess()

        logger.info("학습 완료!")

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
