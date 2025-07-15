#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger

import torch

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices

import argparse
import json
import logging
import mlflow
import os
import requests
import traceback
from mlflow import MlflowClient
from pathlib import Path
from typing import Any, Dict, Optional

# 현재 파일의 절대 경로 얻기
current_path = Path(__file__).absolute().parent


class YOLOXTrainModel:
    """YOLOX 모델 학습 클래스"""

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
        exp_file: str,  # 실험 파일 경로
        batch_size: int = 8,
        devices: int = 1,
        exp_name: str = "yolox_s",
        resume: bool = False,
        ckpt: Optional[str] = None,
        start_epoch: int = 0,
        num_machines: int = 1,
        machine_rank: int = 0,
        dist_url: str = "auto",
        dist_backend: str = "nccl",
        cache: bool = False,
        fp16: bool = False,
        occupy: bool = False,
        logger_type: str = "tensorboard",
        **kwargs,
    ):
        """
        YOLOX 모델 학습 클래스 초기화

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
            exp_file: YOLOX 실험 파일 경로
            batch_size: 배치 크기
            devices: 사용할 GPU 장치 수
            exp_name: 실험명
            resume: 재개 여부
            ckpt: 체크포인트 경로
            start_epoch: 시작 에폭
            num_machines: 머신 수
            machine_rank: 머신 순위
            dist_url: 분산 URL
            dist_backend: 분산 백엔드
            cache: 캐시 사용 여부
            fp16: FP16 사용 여부
            occupy: GPU 점유 여부
            logger_type: 로거 타입
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
        self.exp_file = exp_file
        self.batch_size = batch_size
        self.devices = devices
        self.exp_name = exp_name
        self.resume = resume
        self.ckpt = ckpt
        self.start_epoch = start_epoch
        self.num_machines = num_machines
        self.machine_rank = machine_rank
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.cache = cache
        self.fp16 = fp16
        self.occupy = occupy
        self.logger_type = logger_type

        # 기본 설정
        self.data_dir = current_path / "data"
        self.output_dir = current_path / "YOLOX_outputs"
        self.mlflow_run_id = None
        self.client = None

        # MLflow 설정
        self._setup_mlflow()

        # 실험 설정 로드
        self.exp = get_exp(self.exp_file, None)

        # 데이터 디렉토리 설정 (실험 파일에서 설정되지 않은 경우)
        if not hasattr(self.exp, "data_dir") or self.exp.data_dir is None:
            self.exp.data_dir = str(self.data_dir)

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
            # YOLOX는 내부적으로 데이터 전처리를 처리하므로 별도 처리 불필요
            logger.info("데이터 전처리 완료")

        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {e}")
            raise

    def train(self):
        """모델 학습"""
        logger.info("YOLOX 모델 학습을 시작합니다...")

        # 설정 업데이트
        self.exp.output_dir = str(self.output_dir)
        self.exp.exp_name = self.exp_name

        # 학습 인자 설정
        args = argparse.Namespace(
            experiment_name=self.exp_name,
            name=self.exp_name,
            dist_backend=self.dist_backend,
            dist_url=self.dist_url,
            batch_size=self.batch_size,
            devices=self.devices,
            resume=self.resume,
            ckpt=self.ckpt,
            start_epoch=self.start_epoch,
            num_machines=self.num_machines,
            machine_rank=self.machine_rank,
            cache=self.cache,
            fp16=self.fp16,
            occupy=self.occupy,
            logger=self.logger_type,
            opts=None,
        )

        # GPU 설정
        configure_nccl()
        configure_omp()

        # 분산 학습 설정
        if self.devices > 1:
            # 멀티 GPU 학습
            torch.cuda.set_device(0)
            torch.distributed.init_process_group(
                backend=self.dist_backend, init_method=self.dist_url, world_size=self.devices, rank=0
            )

            # 각 GPU에서 학습 실행
            torch.multiprocessing.spawn(self._train_worker, args=(args,), nprocs=self.devices, join=True)
        else:
            # 단일 GPU 학습
            self._train_worker(0, args)

    def _train_worker(self, gpu: int, args: argparse.Namespace):
        """각 GPU에서 실행되는 학습 워커"""

        # GPU 설정
        torch.cuda.set_device(gpu)

        # 분산 학습 설정 (멀티 GPU인 경우)
        if self.devices > 1:
            torch.distributed.init_process_group(
                backend=self.dist_backend, init_method=self.dist_url, world_size=self.devices, rank=gpu
            )

        # MLflow 실행 시작
        with mlflow.start_run(
            experiment_id=self.client.get_experiment_by_name(self.mlflow_experiment_name).experiment_id
        ) as run:
            self.mlflow_run_id = run.info.run_id

            # 하이퍼파라미터 로깅
            mlflow.log_params(
                {
                    "batch_size": self.batch_size,
                    "devices": self.devices,
                    "exp_name": self.exp_name,
                    "fp16": self.fp16,
                    "cache": self.cache,
                    "num_classes": self.exp.num_classes,
                    "data_dir": self.exp.data_dir,
                    "exp_file": self.exp_file,
                }
            )

            # 출력 디렉토리 생성
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # 학습 실행
            try:
                # YOLOX 학습 런처 실행
                from yolox.core import Trainer

                trainer = Trainer(self.exp, args)
                trainer.train()

                logger.info(f"GPU {gpu}에서 학습 완료")

            except Exception as e:
                logger.error(f"GPU {gpu}에서 학습 중 오류: {e}")
                raise

    def postprocess(self):
        """학습 후 처리"""
        try:
            # 최적 모델 찾기
            best_ckpt_path = self.output_dir / "best_ckpt.pth"
            if not best_ckpt_path.exists():
                # 최신 체크포인트 사용
                latest_ckpt_path = self.output_dir / "latest_ckpt.pth"
                if latest_ckpt_path.exists():
                    best_ckpt_path = latest_ckpt_path
                else:
                    raise FileNotFoundError("학습된 모델을 찾을 수 없습니다.")

            # 모델 저장 및 등록
            train_model_name = f"{self.model_name}-yolox-fine-tuned"

            with mlflow.start_run(run_name=train_model_name) as run:
                # 체크포인트 로깅
                mlflow.log_artifact(str(best_ckpt_path), "model")

                # 모델 등록
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, train_model_name)

                run_id = run.info.run_id
                artifact_uri = mlflow.get_artifact_uri()

                # 최신 모델 버전 가져오기
                model_version = self.client.get_latest_versions(name=train_model_name, stages=["None"])[0].version

                train_model_uri = f"models:/{train_model_name}/{model_version}"

                logger.info(f"모델 등록 완료: {train_model_name} (버전: {model_version})")

                # 메타데이터 저장
                self.insert_metadata(
                    run_id=run_id,
                    artifact_uri=artifact_uri,
                    model_version=model_version,
                    model_uri=train_model_uri,
                    train_model_name=train_model_name,
                    restapi_url=self.restapi_url,
                    restapi_token=self.get_token_from_restapi(
                        url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                    ),
                )

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
                "description": f"YOLOX 파인튜닝 모델: {train_model_name}",
                "model_provider_id": 3,
                "model_type_id": 4,
                "model_format_id": 1,
                "model_registry_schema": json.dumps(
                    {
                        "run_id": run_id,
                        "artifact_path": artifact_uri,
                        "versions": model_version,
                        "model_uri": model_uri,
                        "framework": "yolox",
                        "model_type": "object_detection",
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
    parser = argparse.ArgumentParser(description="YOLOX 모델 학습")

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

    # YOLOX 설정
    parser.add_argument("--exp_file", type=str, required=True, help="YOLOX 실험 파일 경로")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--devices", type=int, default=1, help="사용할 GPU 장치 수")
    parser.add_argument("--exp_name", type=str, default="yolox_s", help="실험명")
    parser.add_argument("--resume", action="store_true", help="재개 여부")
    parser.add_argument("--ckpt", type=str, help="체크포인트 경로")
    parser.add_argument("--start_epoch", type=int, default=0, help="시작 에폭")
    parser.add_argument("--num_machines", type=int, default=1, help="머신 수")
    parser.add_argument("--machine_rank", type=int, default=0, help="머신 순위")
    parser.add_argument("--dist_url", type=str, default="auto", help="분산 URL")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="분산 백엔드")
    parser.add_argument("--cache", action="store_true", help="캐시 사용 여부")
    parser.add_argument("--fp16", action="store_true", help="FP16 사용 여부")
    parser.add_argument("--occupy", action="store_true", help="GPU 점유 여부")
    parser.add_argument("--logger_type", type=str, default="tensorboard", help="로거 타입")

    # 인자 파싱
    args = parser.parse_args()

    # 모델 초기화
    model = YOLOXTrainModel(
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
        exp_file=args.exp_file,
        batch_size=args.batch_size,
        devices=args.devices,
        exp_name=args.exp_name,
        resume=args.resume,
        ckpt=args.ckpt,
        start_epoch=args.start_epoch,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        cache=args.cache,
        fp16=args.fp16,
        occupy=args.occupy,
        logger_type=args.logger_type,
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
