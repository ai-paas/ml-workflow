#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Optional

import mlflow
import requests
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from mlflow import MlflowClient
from torch.nn.parallel import DistributedDataParallel as DDP
from YOLOX.exps.default import yolox_l, yolox_m, yolox_nano, yolox_s, yolox_tiny, yolox_x
from YOLOX.tools import train
from YOLOX.tools.train import make_parser
from YOLOX.yolox.core.launch import launch
from YOLOX.yolox.exp.build import get_exp
from YOLOX.yolox.exp.yolox_base import check_exp_value

# eval 관련 import 추가
from YOLOX.yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
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
        experiment_id: int,
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
        self.experiment_id = experiment_id
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
                self.update_experiment(
                    experiment_id=self.experiment_id,
                    status="RUNNING",
                    mlflow_run_id=run.info.run_id,
                    restapi_url=self.restapi_url,
                    restapi_token=self.get_token_from_restapi(
                        url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                    ),
                )
                launch(
                    train.main,
                    num_gpu,
                    args.num_machines,
                    args.machine_rank,
                    backend=args.dist_backend,
                    dist_url=dist_url,
                    args=(exp, args),
                )

                # 학습 완료 후 평가 실행
                logger.info("학습 완료! 평가를 시작합니다.")
                self.evaluate_after_training(run, temp_exp_path, matched_exp_name, modifications)

                # self.insert_metadata(
                #     run_id=run.info.run_id,
                #     artifact_uri=run.info.artifact_uri,
                #     model_id=self.model_id,
                #     model_version="1",
                #     model_uri="",
                #     train_model_name=self.result_model_name,
                #     restapi_url=self.restapi_url,
                #     restapi_token=self.get_token_from_restapi(
                #         url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                #     ),
                # )
            self.update_experiment(
                experiment_id=self.experiment_id,
                status="COMPLETED",
                restapi_url=self.restapi_url,
                restapi_token=self.get_token_from_restapi(
                    url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                ),
            )

        except Exception as e:
            logger.error(f"학습 중 오류: {e}")
            self.update_experiment(
                experiment_id=self.experiment_id,
                status="FAILED",
                restapi_url=self.restapi_url,
                restapi_token=self.get_token_from_restapi(
                    url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
                ),
            )
            raise

    def evaluate_after_training(self, run, temp_exp_path, matched_exp_name, modifications):
        """학습 후 평가 실행"""
        try:
            logger.info("학습된 모델로 평가를 시작합니다.")

            # MLflow 아티팩트 다운로드
            artifacts_dir = mlflow.artifacts.download_artifacts(artifact_uri=run.info.artifact_uri)
            logger.info(f"아티팩트 다운로드 완료: {artifacts_dir}")

            # 학습된 모델의 체크포인트 경로 찾기
            output_dir = Path(artifacts_dir)
            ckpt_files = list(output_dir.glob("**/*.pth"))

            if not ckpt_files:
                logger.warning("평가할 체크포인트 파일을 찾을 수 없습니다.")
                return

            # 가장 최근 체크포인트 또는 best_ckpt.pth 찾기
            best_ckpt = None
            for ckpt_file in ckpt_files:
                if "best_ckpt.pth" in ckpt_file.name:
                    best_ckpt = ckpt_file
                    break

            if not best_ckpt:
                # best_ckpt가 없으면 가장 최근 파일 사용
                best_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)

            logger.info(f"평가에 사용할 체크포인트: {best_ckpt}")

            # 평가 실행
            self._run_evaluation(temp_exp_path, str(best_ckpt), run)

        except Exception as e:
            logger.error(f"학습 후 평가 중 오류: {e}")
            raise

    def _run_evaluation(self, temp_exp_path: str, ckpt_path: str, run):
        """평가 실행 로직"""
        try:
            # YOLOX 환경변수 설정
            os.environ["YOLOX_DATADIR"] = str(self.dataset_artifacts_dir)

            # eval.py의 실행 코드를 직접 구현
            configure_module()

            # 인자 파싱 (eval용)
            parser = argparse.ArgumentParser("YOLOX Eval")
            parser.add_argument("-expn", "--experiment-name", type=str, default=None)
            parser.add_argument("-n", "--name", type=str, default=None, help="model name")
            parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
            parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
            parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
            parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
            parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
            parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
            parser.add_argument(
                "-f", "--exp_file", default=None, type=str, help="please input your experiment description file"
            )
            parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
            parser.add_argument("--conf", default=None, type=float, help="test conf")
            parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
            parser.add_argument("--tsize", default=None, type=int, help="test img size")
            parser.add_argument("--seed", default=None, type=int, help="eval seed")
            parser.add_argument(
                "--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating."
            )
            parser.add_argument(
                "--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing."
            )
            parser.add_argument(
                "--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing."
            )
            parser.add_argument(
                "--legacy",
                dest="legacy",
                default=False,
                action="store_true",
                help="To be compatible with older versions",
            )
            parser.add_argument(
                "--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set."
            )
            parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.")
            parser.add_argument(
                "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
            )

            # 기본 인자 설정
            eval_args = [
                "-f",
                temp_exp_path,
                "-c",
                ckpt_path,
                "-b",
                self.batch_size,
                "-d",
                self.gpu_limit,
            ]

            args = parser.parse_args(eval_args)

            exp = get_exp(args.exp_file, args.name)
            exp.merge(args.opts)

            if not args.experiment_name:
                args.experiment_name = exp.exp_name

            num_gpu = get_num_devices() if args.devices is None else args.devices
            num_gpu = min(num_gpu, get_num_devices())

            # dist_url = "auto" if args.dist_url is None else args.dist_url

            # 평가 실행
            os.environ["MLFLOW_NESTED_RUN"] = "TRUE"
            os.environ["MLFLOW_RUN_ID"] = run.info.run_id

            self._execute_evaluation(exp, args, num_gpu, run)

        except Exception as e:
            logger.error(f"평가 실행 중 오류: {e}")
            raise

    def _execute_evaluation(self, exp, args, num_gpu, run):
        """실제 평가 실행"""
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting, ")

        is_distributed = num_gpu > 1

        # set environment variables for distributed training
        configure_nccl()
        cudnn.benchmark = True

        rank = get_local_rank()

        file_name = os.path.join(exp.output_dir, args.experiment_name)

        if rank == 0:
            os.makedirs(file_name, exist_ok=True)

        setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
        logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        logger.info("Model Structure:\n{}".format(str(model)))

        evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
        evaluator.per_class_AP = True
        evaluator.per_class_AR = True

        torch.cuda.set_device(rank)
        model.cuda(rank)
        model.eval()

        if not args.speed and not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint from {}".format(ckpt_file))
            loc = "cuda:{}".format(rank)
            ckpt = torch.load(ckpt_file, map_location=loc, weights_only=False)
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if is_distributed:
            model = DDP(model, device_ids=[rank])

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert (
                not args.fuse and not is_distributed and args.batch_size == 1
            ), "TensorRT model is not support model fusing and distributed inferencing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(trt_file), "TensorRT model is not found!\n Run tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
        else:
            trt_file = None
            decoder = None

        # start evaluate
        *_, summary = evaluator.evaluate(model, is_distributed, args.fp16, trt_file, decoder, exp.test_size)
        logger.info("\n" + summary)

        # 평가 결과를 MLflow에 로깅
        self._log_evaluation_results(summary, run)

    def _log_evaluation_results(self, summary: str, run):
        """평가 결과를 MLflow에 로깅"""
        try:
            # 1. 기본 메트릭 추출 및 로깅
            metrics = self._extract_basic_metrics(summary)

            # 2. 클래스별 성능 분석
            class_analysis = self._analyze_per_class_performance(summary)

            # 3. MLflow에 메트릭 로깅
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)
                    logger.info(f"{metric_name}: {metric_value}")

            # 4. 분석 결과와 원본 평가 결과를 파일로 저장 후 MLflow에 로깅
            temp_file_path = os.path.join(tempfile.gettempdir(), "evaluation_results.txt")
            with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                # 분석 결과 추가
                temp_file.write("=" * 80 + "\n")
                temp_file.write("YOLO 모델 평가 결과 분석\n")
                temp_file.write("=" * 80 + "\n\n")

                # 기본 메트릭 요약
                temp_file.write("📊 성능 요약\n")
                temp_file.write("-" * 40 + "\n")
                metric_display_names = {
                    "mAP_0.5_0.95": "mAP@0.5:0.95",
                    "AP50": "AP50",
                    "AP75": "AP75",
                    "AR_0.5_0.95": "AR@0.5:0.95",
                    "AP_small": "AP_small",
                    "AP_medium": "AP_medium",
                    "AP_large": "AP_large",
                }

                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        display_name = metric_display_names.get(metric_name, metric_name)
                        if (
                            metric_name.startswith("mAP")
                            or metric_name.startswith("AP")
                            or metric_name.startswith("AR")
                        ):
                            temp_file.write(f"{display_name}: {metric_value:.3f} ({metric_value*100:.1f}%)\n")
                        else:
                            temp_file.write(f"{display_name}: {metric_value:.3f}\n")
                temp_file.write("\n")

                # 클래스별 분석
                temp_file.write("🎯 클래스별 성능 분석\n")
                temp_file.write("-" * 40 + "\n")
                temp_file.write(f"전체 클래스 수: {class_analysis.get('total_classes', 0)}\n")
                temp_file.write(f"검출된 클래스 수: {class_analysis.get('detected_classes', 0)}\n")
                temp_file.write(f"검출되지 않은 클래스 수 (nan): {class_analysis.get('nan_classes', 0)}\n")
                temp_file.write(f"AP=0인 클래스 수: {class_analysis.get('zero_ap_classes', 0)}\n")
                temp_file.write("\n")

                # 성능이 좋은 클래스
                best_classes = class_analysis.get("best_performing_classes", [])
                if best_classes:
                    temp_file.write("🏆 성능이 좋은 클래스 (상위 5개)\n")
                    for i, (class_id, ap) in enumerate(best_classes[:5], 1):
                        temp_file.write(f"  {i}. Class {class_id}: AP = {ap:.3f} ({ap*100:.1f}%)\n")
                    temp_file.write("\n")

                # 원본 평가 결과
                temp_file.write("원본 평가 결과\n")
                temp_file.write("-" * 40 + "\n")
                temp_file.write(summary)

            mlflow.log_artifact(temp_file_path, "evaluation")

            # 임시 파일 삭제
            os.unlink(temp_file_path)

            logger.info("평가 결과가 MLflow에 로깅되었습니다.")

        except Exception as e:
            logger.error(f"평가 결과 로깅 중 오류: {e}")
            raise

    def _extract_basic_metrics(self, summary: str) -> dict:
        """기본 메트릭 추출"""
        metrics = {}

        # mAP@0.5:0.95 추출
        map_pattern = r"Average Precision.*IoU=0\.50:0\.95.*area=\s*all.*maxDets=100.*=\s*([\d.-]+)"
        map_match = re.search(map_pattern, summary)
        if map_match:
            metrics["mAP_0.5_0.95"] = float(map_match.group(1))  # 콜론을 언더스코어로 변경

        # AP50 추출
        ap50_pattern = r"Average Precision.*IoU=0\.50.*area=\s*all.*maxDets=100.*=\s*([\d.-]+)"
        ap50_match = re.search(ap50_pattern, summary)
        if ap50_match:
            metrics["AP50"] = float(ap50_match.group(1))

        # AP75 추출
        ap75_pattern = r"Average Precision.*IoU=0\.75.*area=\s*all.*maxDets=100.*=\s*([\d.-]+)"
        ap75_match = re.search(ap75_pattern, summary)
        if ap75_match:
            metrics["AP75"] = float(ap75_match.group(1))

        # AR 추출
        ar_pattern = r"Average Recall.*IoU=0\.50:0\.95.*area=\s*all.*maxDets=100.*=\s*([\d.-]+)"
        ar_match = re.search(ar_pattern, summary)
        if ar_match:
            metrics["AR_0.5_0.95"] = float(ar_match.group(1))  # 콜론을 언더스코어로 변경

        # 객체 크기별 성능
        size_patterns = {
            "AP_small": r"Average Precision.*IoU=0\.50:0\.95.*area=\s*small.*maxDets=100.*=\s*([\d.-]+)",
            "AP_medium": r"Average Precision.*IoU=0\.50:0\.95.*area=\s*medium.*maxDets=100.*=\s*([\d.-]+)",
            "AP_large": r"Average Precision.*IoU=0\.50:0\.95.*area=\s*large.*maxDets=100.*=\s*([\d.-]+)",
        }

        for metric_name, pattern in size_patterns.items():
            match = re.search(pattern, summary)
            if match:
                metrics[metric_name] = float(match.group(1))

        return metrics

    def _analyze_per_class_performance(self, summary: str) -> dict:
        """클래스별 성능 분석"""
        analysis = {
            "total_classes": 0,
            "detected_classes": 0,
            "nan_classes": 0,
            "zero_ap_classes": 0,
            "best_performing_classes": [],
            "worst_performing_classes": [],
        }

        # per class AP 테이블에서 클래스별 성능 추출
        ap_table_pattern = r"per class AP:(.*?)per class AR:"
        ap_table_match = re.search(ap_table_pattern, summary, re.DOTALL)

        if ap_table_match:
            ap_table = ap_table_match.group(1)

            # 클래스별 AP 값 추출
            class_ap_pattern = r"\|\s*(\d+)\s*\|\s*([\d.-]+|nan)\s*\|"
            class_aps = re.findall(class_ap_pattern, ap_table)

            analysis["total_classes"] = len(class_aps)
            class_performances = []

            for class_id, ap_value in class_aps:
                class_id = int(class_id)

                if ap_value.lower() == "nan":
                    analysis["nan_classes"] += 1
                    class_performances.append((class_id, None))
                else:
                    ap_float = float(ap_value)
                    if ap_float > 0:
                        analysis["detected_classes"] += 1
                        class_performances.append((class_id, ap_float))

                        # 성능이 좋은 클래스 (AP > 10%)
                        if ap_float > 10.0:
                            analysis["best_performing_classes"].append((class_id, ap_float))
                    else:
                        analysis["zero_ap_classes"] += 1
                        class_performances.append((class_id, ap_float))

            # 성능 순으로 정렬
            valid_performances = [(cid, ap) for cid, ap in class_performances if ap is not None]
            if valid_performances:
                valid_performances.sort(key=lambda x: x[1], reverse=True)
                analysis["best_performing_classes"] = valid_performances[:5]  # 상위 5개
                analysis["worst_performing_classes"] = valid_performances[-5:]  # 하위 5개

        return analysis

    def postprocess(self):
        """학습 후 처리"""
        try:
            pass
        except Exception as e:
            logger.error(f"후처리 중 오류 발생: {e}")
            traceback.print_exc()
            raise

    def update_experiment(
        self,
        restapi_url: str,
        restapi_token: str,
        experiment_id: int,
        status: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        kubeflow_run_id: Optional[str] = None,
    ):
        try:
            data = {}
            if status:
                data["status"] = status
            if mlflow_run_id:
                data["mlflow_run_id"] = mlflow_run_id
            if kubeflow_run_id:
                data["kubeflow_run_id"] = kubeflow_run_id
            response = requests.patch(
                f"{restapi_url}/api/v1/experiments/{experiment_id}",
                json=data,
                headers={"Authorization": f"Bearer {restapi_token}"},
            )
            if response.status_code == 200:
                logger.info("실험 업데이트 성공")
                return response.json()
            else:
                logger.error(f"실험 업데이트 실패: {response.status_code}")
                logger.error(f"실험 업데이트 실패: {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"REST API 서버에 연결할 수 없습니다: {self.restapi_url}")
            return None

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

        except requests.exceptions.ConnectionError:
            logger.warning(f"REST API 서버에 연결할 수 없습니다: {restapi_url}")
            return None
        except Exception as e:
            logger.error(f"메타데이터 삽입 중 오류 발생: {e}")
            return None

    def get_token_from_restapi(self, url: str, username: str, password: str) -> str:
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


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="커스텀 모델 학습")

    # 기본 설정
    parser.add_argument("--train_name", type=str, required=True, help="학습 실행명")
    parser.add_argument("--model_id", type=int, required=True, help="모델 ID")
    parser.add_argument("--experiment_id", type=int, required=True, help="실험 ID")
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
        experiment_id=args.experiment_id,
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

        # 학습 실행 (자동으로 평가도 실행됨)
        model.train()

        logger.info("학습 및 평가 완료!")

    except Exception as e:
        logger.error(f"작업 중 오류 발생: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
