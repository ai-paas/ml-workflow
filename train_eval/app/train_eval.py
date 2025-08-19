#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import mlflow
import requests
import torch
from loguru import logger
from mlflow import MlflowClient
from YOLOX.exps.default import yolox_l, yolox_m, yolox_nano, yolox_s, yolox_tiny, yolox_x
from YOLOX.tools import train
from YOLOX.tools.train import make_parser
from YOLOX.yolox.core.launch import launch
from YOLOX.yolox.exp.build import get_exp
from YOLOX.yolox.exp.yolox_base import check_exp_value
from YOLOX.yolox.utils.dist import get_num_devices
from YOLOX.yolox.utils.setup_env import configure_module

# 현재 파일의 절대 경로 얻기
current_path = Path(__file__).absolute().parent


class CustomTrainModel:
    """커스텀 모델 학습 클래스"""

    def __init__(
        self,
        train_name: str,
        model_id: int,
        result_model_name: str,
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
        gpu_limit: str,
        batch_size: str,
        epochs: str,
        save_period: str,
        weight_decay: str,
        lr0: str,
        lrf: str,
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
        self.model_id = model_id
        self.result_model_name = result_model_name
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
        self.batch_size = batch_size
        self.gpu_limit = gpu_limit
        self.epochs = epochs
        self.save_period = save_period
        self.weight_decay = weight_decay
        self.lr0 = lr0
        self.lrf = lrf
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

    def get_num_classes_from_json(self, json_file_path):
        """COCO format JSON 파일에서 클래스 수를 읽어옵니다."""
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                num_classes = len(data["categories"])
                logger.info(f"데이터셋의 클래스 수: {num_classes}")
                return num_classes
        except Exception as e:
            logger.error(f"JSON 파일 읽기 실패: {e}")
            raise

    def create_modified_exp_file(self, matched_exp_path, matched_exp_name, modifications=None):
        """임시 파일을 생성하여 수정된 exp 내용을 저장합니다.

        Args:
            matched_exp_path (Path): 원본 exp 파일 경로
            matched_exp_name (str): exp 이름
            modifications (dict): 수정할 속성과 값의 딕셔너리 (예: {'num_classes': 71, 'max_epoch': 3})

        Returns:
            str: 수정된 exp 파일의 경로
        """
        try:
            # 원본 exp 파일 읽기
            with open(matched_exp_path, "r") as f:
                exp_content = f.read()

            # 기본 수정사항이 없으면 빈 딕셔너리 사용
            if modifications is None:
                modifications = {}

            # __init__ 함수 찾기
            init_pattern = r"(def __init__\(self\):.*?super\([^)]+\)\.__init__\(\))"

            # 모든 수정사항을 하나의 문자열로 만들기
            modifications_str = "\n        ".join([f"self.{attr} = {value}" for attr, value in modifications.items()])

            # __init__ 함수와 super().__init__() 호출을 찾아서 그 뒤에 수정사항 추가
            replacement = f"\\1\n        {modifications_str}"

            exp_content = re.sub(init_pattern, replacement, exp_content, flags=re.DOTALL)

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix=f"{matched_exp_name}_", delete=False
            ) as temp_file:
                temp_file.write(exp_content)
                logger.info(f"수정된 exp 파일 생성: {temp_file.name}")
                logger.info(f"적용된 수정사항: {modifications}")
                return temp_file.name

        except Exception as e:
            logger.error(f"exp 파일 수정 중 오류 발생: {e}")
            raise

    # 체크포인트 파일명에서 에포크 번호를 추출하는 함수 추가
    def _extract_epoch_num(self, checkpoint_path: Path) -> str:
        """체크포인트 파일명에서 에포크 번호를 추출합니다."""
        match = re.search(r"epoch_(\d+)_ckpt\.pth", checkpoint_path.name)
        if match:
            return match.group(1)
        return "unknown"

    def train(self):
        """모델 학습"""
        try:
            # model_artifacts에서 파일명 추출
            model_path = Path(self.model_artifacts_dir)
            model_file = list(model_path.glob("*.pth"))[0]  # .pth 파일 찾기
            model_file = model_file.absolute()  # 전체 경로로 변환
            model_filename = Path(model_file).stem  # 확장자를 제외한 파일명

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
                if exp_name in model_filename.lower():
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
                raise ValueError(f"모델 파일명 '{model_filename}'과 일치하는 YOLOX exp를 찾을 수 없습니다.")

            # COCO 데이터셋 경로에서 annotation 파일 찾기
            coco_path = Path(self.dataset_artifacts_dir) / "COCO"
            annotation_file = coco_path / "annotations" / "instances_val2017.json"

            if not annotation_file.exists():
                raise FileNotFoundError(f"Annotation 파일을 찾을 수 없습니다: {annotation_file}")

            # 클래스 수 가져오기
            num_classes = self.get_num_classes_from_json(annotation_file)
            # exp 파일 수정사항 정의
            modifications = {
                "num_classes": num_classes,
                "max_epoch": self.epochs,
                "save_history_ckpt": False if self.save_period == "-1" else True,
                "weight_decay": self.weight_decay,
                "min_lr_ratio": self.lrf,
                "basic_lr_per_img": float(self.lr0) / float(self.batch_size),
            }
            # 임시 파일 생성 및 경로 저장
            temp_exp_path = self.create_modified_exp_file(
                matched_exp_path=matched_exp_path, matched_exp_name=matched_exp_name, modifications=modifications
            )

            # YOLOX 환경변수 설정
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["YOLOX_DATADIR"] = str(self.dataset_artifacts_dir)

            # with mlflow.start_run(run_name=self.train_name) as run:
            os.environ["YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS"] = "TRUE"
            os.environ["YOLOX_MLFLOW_LOG_Nth_EPOCH_MODELS"] = "FALSE" if self.save_period == "-1" else "TRUE"
            os.environ["YOLOX_MLFLOW_RUN_NAME"] = self.train_name
            os.environ["MLFLOW_EXPERIMENT_NAME"] = self.mlflow_experiment_name

            os.environ["YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS"] = (
                "10000" if self.save_period == "-1" else self.save_period
            )

            # train.py의 실행 코드를 직접 구현
            configure_module()

            # 인자 파싱
            parser = make_parser()
            args = parser.parse_args(
                [
                    "-f",
                    str(temp_exp_path),
                    "-c",
                    str(model_file),
                    "-b",
                    self.batch_size,
                    "-d",
                    self.gpu_limit,
                    "--fp16",
                    "--logger",
                    "mlflow",
                ]
            )

            exp = get_exp(args.exp_file, args.name)
            exp.merge(args.opts)
            check_exp_value(exp)

            if not args.experiment_name:
                args.experiment_name = exp.exp_name

            num_gpu = get_num_devices() if args.devices is None else args.devices
            num_gpu = min(num_gpu, get_num_devices())

            if args.cache is not None:
                exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

            dist_url = "auto" if args.dist_url is None else args.dist_url
            with mlflow.start_run(run_name=self.train_name) as run:
                # train.py의 launch 실행
                os.environ["MLFLOW_NESTED_RUN"] = "TRUE"
                os.environ["MLFLOW_RUN_ID"] = run.info.run_id
                launch(
                    train.main,
                    num_gpu,
                    args.num_machines,
                    args.machine_rank,
                    backend=args.dist_backend,
                    dist_url=dist_url,
                    args=(exp, args),
                )
                self.insert_metadata(
                    run_id=run.info.run_id,
                    artifact_uri=run.info.artifact_uri,
                    model_id=self.model_id,
                    model_version="1",
                    model_uri="",
                    train_model_name=self.result_model_name,
                    restapi_url=self.restapi_url,
                    restapi_token=self.get_token_from_restapi(
                        url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                    ),
                )

        except Exception as e:
            logger.error(f"학습 중 오류: {e}")
            raise

    def postprocess(self):
        """학습 후 처리"""
        try:
            pass
        except Exception as e:
            logger.error(f"후처리 중 오류 발생: {e}")
            traceback.print_exc()
            raise

    def insert_metadata(
        self,
        run_id: str,
        artifact_uri: str,
        model_id: int,
        model_version: str,
        model_uri: str,
        train_model_name: str,
        restapi_url: str,
        restapi_token: str,
    ):
        """메타데이터 삽입"""
        try:
            # API 토큰 헤더 설정
            headers = {"Authorization": f"Bearer {restapi_token}"}

            # provider, type, format ID 조회
            provider_response = requests.get(
                f"{restapi_url}/api/v1/models/providers", headers=headers, params={"provider_name": "custom"}
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
                "description": f"fine-tuned model: {train_model_name}",
                "provider_id": provider_id,
                "type_id": type_id,
                "format_id": format_id,
                "parent_model_id": model_id,
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
    parser.add_argument("--model_id", type=int, required=True, help="모델 ID")
    parser.add_argument("--result_model_name", type=str, required=True, help="모델명")
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
    parser.add_argument("--gpu_limit", type=str, required=True, help="GPU 제한")
    parser.add_argument("--batch_size", type=str, required=True, help="배치 크기")
    parser.add_argument("--epochs", type=str, required=True, help="에포크 수")
    parser.add_argument("--save_period", type=str, required=True, help="저장 주기")
    parser.add_argument("--weight_decay", type=str, required=True, help="가중치 감소")
    parser.add_argument("--lr0", type=str, required=True, help="초기 학습률")
    parser.add_argument("--lrf", type=str, required=True, help="학습률 감소 비율")

    # 인자 파싱
    args = parser.parse_args()

    # 모델 초기화
    model = CustomTrainModel(
        train_name=args.train_name,
        model_id=args.model_id,
        result_model_name=args.result_model_name,
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
        gpu_limit=args.gpu_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_period=args.save_period,
        weight_decay=args.weight_decay,
        lr0=args.lr0,
        lrf=args.lrf,
    )

    try:
        # 데이터 전처리
        model.preprocess()

        # 학습 실행
        model.train()

        # 후처리
        # model.postprocess()

        logger.info("학습 완료!")

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
