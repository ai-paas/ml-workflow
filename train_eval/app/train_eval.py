#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
import uuid
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
        """모델 학습을 위한 전처리"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)

            # 모델 아티팩트 다운로드
            self.model_artifacts_dir = mlflow.artifacts.download_artifacts(artifact_uri=self.model_artifact_path)
            logger.info(f"모델 아티팩트: {self.model_artifacts_dir}")

            # 데이터셋 아티팩트 다운로드 및 압축 해제
            self.dataset_artifacts_dir = mlflow.artifacts.download_artifacts(artifact_uri=self.dataset_artifact_uri)
            logger.info(f"데이터셋 아티팩트 다운로드: {self.dataset_artifacts_dir}")

            # 데이터셋 zip 파일 찾기
            dataset_zip = list(Path(self.dataset_artifacts_dir).glob("*.zip"))[0]

            # 압축 해제할 디렉토리 생성 (dataset_artifacts 경로에 _extracted 추가)
            extract_dir = Path(self.dataset_artifacts_dir) / "COCO"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # zip 파일 압축 해제
            import zipfile

            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                # zip 파일 내의 모든 파일 경로 가져오기
                all_files = zip_ref.namelist()
                # 최상위 폴더 찾기
                top_level_dirs = set()
                for file_path in all_files:
                    parts = file_path.split("/")
                    if len(parts) > 1:  # 폴더가 있는 경우
                        top_level_dirs.add(parts[0])

                # 최상위 폴더가 하나인지 확인
                if len(top_level_dirs) == 1:
                    top_dir = top_level_dirs.pop()
                    # 최상위 폴더가 annotations, train, val로 시작하는지 확인
                    if not (
                        top_dir.startswith("annotations") or top_dir.startswith("train") or top_dir.startswith("val")
                    ):
                        logger.info(f"최상위 폴더 '{top_dir}' 제거 후 압축 해제")
                        # 파일 압축 해제
                        for file_info in zip_ref.filelist:
                            file_path = file_info.filename
                            if file_path.startswith(f"{top_dir}/"):  # 최상위 폴더로 시작하는 경우
                                # 최상위 폴더명 제거
                                extracted_path = file_path.split("/", 1)[1]
                                if extracted_path and not file_info.filename.endswith("/"):  # 폴더가 아닌 파일만 처리
                                    # 파일 추출
                                    source = zip_ref.read(file_info)
                                    target = extract_dir / extracted_path
                                    # 필요한 경우 부모 디렉토리 생성
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    # 파일 저장
                                    target.write_bytes(source)
                    else:
                        logger.info(f"최상위 폴더 '{top_dir}'가 예약된 이름으로 시작하므로 그대로 압축 해제")
                        zip_ref.extractall(extract_dir)
                else:
                    logger.info("최상위 폴더가 여러 개이거나 없으므로 그대로 압축 해제")
                    zip_ref.extractall(extract_dir)

            logger.info("데이터셋 압축 해제 완료")

            # 압축 해제된 경로를 dataset_artifacts로 업데이트
            logger.info(f"최종 데이터셋 경로: {self.dataset_artifacts_dir}")

        except Exception as e:
            logger.error(f"전처리 중 오류 발생: {e}")
            raise

    def train(self):
        """모델 학습"""
        try:
            # model_artifacts에서 파일명 추출
            model_path = Path(self.model_artifacts_dir)
            model_file = list(model_path.glob("*.pth"))[0]  # .pth 파일 찾기
            model_file = model_file.absolute()  # 전체 경로로 변환
            model_name = Path(model_file).stem  # 확장자를 제외한 파일명

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
            matched_exp_name = None  # exp 이름 저장 추가

            for exp_name, exp_module in exp_mapping.items():
                if exp_name in model_name.lower():
                    matched_exp = exp_module.Exp()
                    matched_exp_name = exp_name
                    # exp 모듈의 파일 경로 찾기
                    file_path = exp_module.__file__
                    if file_path is not None:
                        matched_exp_path = Path(file_path)
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
                "yolox.tools.train",
                "-f",
                str(matched_exp_path),  # exp 파일 경로
                "-c",
                str(model_file),  # 체크포인트 경로
                "-b",
                "64",  # 배치 사이즈 증가
                "-d",
                "1",  # 기본 device 수
                "--fp16",  # fp16 사용
                # "-o",        # GPU 점유
            ]

            logger.info(f"실행 명령: {' '.join(cmd)}")

            try:
                with mlflow.start_run(run_name=self.train_name):
                    # YOLOX 데이터셋 경로 환경변수 설정
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = "0"  # 단일 GPU 사용
                    env["YOLOX_DATADIR"] = str(self.dataset_artifacts_dir)

                    # YOLOX_outputs 디렉토리 모니터링
                    output_dir = current_path / "YOLOX_outputs" / matched_exp_name
                    last_logged_files = set()
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
                    if process.stdout is not None:
                        for line in iter(process.stdout.readline, ""):
                            if line:
                                line = line.rstrip()
                                logger.info(line)

                                # 체크포인트 저장 확인
                                if "Save weights to" in line:
                                    # 새로운 체크포인트 파일 찾기
                                    current_files = set()
                                    if output_dir.exists():
                                        for f in output_dir.glob("epoch_*_ckpt.pth"):
                                            current_files.add(f)

                                    # 새로 생성된 파일 찾기
                                    new_files = current_files - last_logged_files

                                    # 새 파일들을 MLflow에 로깅
                                    for f in new_files:
                                        logger.info(f"새로운 체크포인트 발견: {f}")
                                        mlflow.log_artifact(str(f), f"checkpoints_{matched_exp_name}_{uuid.uuid4()}")
                                        logger.info(f"체크포인트를 MLflow에 로깅했습니다: {f.name}")

                                    # 로깅된 파일 목록 업데이트
                                    last_logged_files = current_files
                    else:
                        logger.warning("프로세스의 stdout이 None입니다.")

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

            # with mlflow.start_run(run_name=train_model_name) as run:
            # 모델 아티팩트 로깅
            # 여기에 모델 저장 로직 구현

            logger.info(f"모델 등록 완료: {train_model_name}")

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
