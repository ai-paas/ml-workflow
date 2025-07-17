#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

import mlflow
import requests
import torch
from loguru import logger
from mlflow import MlflowClient
from YOLOX.exps.default import yolox_l, yolox_m, yolox_nano, yolox_s, yolox_tiny, yolox_x

# 현재 파일의 절대 경로 얻기
current_path = Path(__file__).absolute().parent


class CustomTrainModel:
    """커스텀 모델 학습 클래스"""

    def __init__(
        self,
        train_name: str,
        model_name: str,
        model_artifact_path: str,
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
        self.model_artifact_path = model_artifact_path
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MLflow 설정
        self._setup_mlflow()

    def _setup_mlflow(self):
        """MLflow 설정"""
        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

        self.client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)

    def preprocess(self):
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)

            # 모델 아티팩트 다운로드
            self.model_artifacts = mlflow.artifacts.download_artifacts(artifact_uri=self.model_artifact_path)
            logger.info(f"모델 아티팩트: {self.model_artifacts}")

            # 데이터셋 아티팩트 다운로드 및 압축 해제
            self.dataset_artifacts = mlflow.artifacts.download_artifacts(artifact_uri=self.dataset_artifact_uri)
            logger.info(f"데이터셋 아티팩트 다운로드: {self.dataset_artifacts}")

            # 데이터셋 zip 파일 찾기
            dataset_zip = list(Path(self.dataset_artifacts).glob("*.zip"))[0]

            # 압축 해제할 디렉토리 생성 (dataset_artifacts 경로에 _extracted 추가)
            extract_dir = Path(self.dataset_artifacts + "_extracted")
            extract_dir.mkdir(parents=True, exist_ok=True)

            # zip 파일 압축 해제
            import zipfile

            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                logger.info(f"데이터셋 압축 해제 시작: {dataset_zip} -> {extract_dir}")
                zip_ref.extractall(extract_dir)
                logger.info("데이터셋 압축 해제 완료")

            # 압축 해제된 경로를 dataset_artifacts로 업데이트
            self.dataset_artifacts = str(extract_dir)
            logger.info(f"최종 데이터셋 경로: {self.dataset_artifacts}")

        except Exception as e:
            logger.error(f"전처리 중 오류 발생: {e}")
            raise

    def train(self):
        """모델 학습"""
        try:
            # model_artifacts에서 파일명 추출
            model_path = Path(self.model_artifacts)
            model_file = list(model_path.glob("*.pth"))[0]  # .pth 파일 찾기
            model_name = model_file.stem  # 확장자를 제외한 파일명

            # YOLOX exp 매핑 딕셔너리
            exp_mapping = {
                "yolox_l": yolox_l,
                "yolox_s": yolox_s,
                "yolox_m": yolox_m,
                "yolox_x": yolox_x,
                "yolox_tiny": yolox_tiny,
                "yolox_nano": yolox_nano,
            }

            # 파일명과 일치하는 exp 찾기
            matched_exp = None
            matched_exp_path = None
            for exp_name, exp_module in exp_mapping.items():
                if exp_name in model_name.lower():
                    matched_exp = exp_module.Exp()
                    # exp 모듈의 파일 경로 찾기
                    matched_exp_path = Path(exp_module.__file__)
                    logger.info(f"매칭된 YOLOX exp: {exp_name}")
                    logger.info(f"exp 파일 경로: {matched_exp_path}")
                    break

            if matched_exp is None:
                raise ValueError(f"모델 파일명 '{model_name}'과 일치하는 YOLOX exp를 찾을 수 없습니다.")

            # 매칭된 exp로 모델 초기화
            # self.exp = matched_exp
            # logger.info(f"YOLOX exp 초기화 완료: {type(self.exp).__name__}")

            # YOLOX train.py 실행을 위한 명령어 구성
            cmd = [
                sys.executable,
                "-m",
                "YOLOX.tools.train",
                "-f",
                str(matched_exp_path),  # exp 파일 경로
                "-c",
                str(model_file),  # 체크포인트 경로
                "-b",
                "64",  # 기본 batch size
                "-d",
                "1",  # 기본 device 수
                "--fp16",  # fp16 사용
                # "-o",        # GPU 점유
            ]

            logger.info(f"실행 명령: {' '.join(cmd)}")

            try:
                # 환경 변수 설정
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0"  # 단일 GPU 사용
                env["YOLOX_DATADIR"] = str(self.dataset_artifacts)  # 데이터셋 경로 설정

                # YOLOX 학습 실행
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    cwd=str(current_path),
                )

                # 실시간 로그 출력
                for line in iter(process.stdout.readline, ""):
                    if line:
                        line = line.rstrip()
                        logger.info(line)

                process.wait()

                if process.returncode == 0:
                    logger.info("YOLOX 학습이 성공적으로 완료되었습니다!")
                else:
                    raise RuntimeError(f"YOLOX 학습이 실패했습니다. 종료 코드: {process.returncode}")

            except Exception as e:
                logger.error(f"YOLOX 학습 실행 중 오류: {e}")
                raise

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
    parser.add_argument("--model_artifact_path", type=str, required=True, help="모델 아티팩트 경로")
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
        model_artifact_path=args.model_artifact_path,
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
