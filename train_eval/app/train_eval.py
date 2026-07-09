#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""학습 entrypoint. --model_kind 로 YOLOX / ESM2 분기.

무거운 의존성(YOLOX/torch, transformers/peft)은 각 분기에서 lazy import 한다.
"""

import argparse
import os
import traceback

from loguru import logger


def build_arg_parser() -> argparse.ArgumentParser:
    """학습 entrypoint argparse. YOLOX/ESM2 공통. (transformers/peft 를 import 하지 않는다.)"""
    parser = argparse.ArgumentParser(description="커스텀 모델 학습")

    # 모델군 분기 키 (yolox | esm2 | esmc)
    parser.add_argument("--model_kind", type=str, required=True, choices=["yolox", "esm2", "esmc"], help="모델군")

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
    parser.add_argument("--dataset_download_ref", type=str, required=True, help="다운로드할 데이터셋 오브젝트 키")
    parser.add_argument(
        "--dataset_storage_type", type=str, required=True, choices=["s3", "hubconnect"], help="데이터셋 저장소 유형"
    )
    parser.add_argument("--dataset_s3_endpoint_url", type=str, default="")
    parser.add_argument("--dataset_s3_access_key", type=str, default="")
    parser.add_argument("--dataset_s3_secret_key", type=str, default="")
    parser.add_argument("--dataset_s3_bucket", type=str, default="")
    parser.add_argument("--datalake_api_url", type=str, default="")
    parser.add_argument("--datalake_api_username", type=str, default="")
    parser.add_argument("--datalake_api_password", type=str, default="")
    parser.add_argument("--datalake_bucket_name", type=str, default="")
    parser.add_argument("--restapi_url", type=str, required=True, help="REST API URL")
    parser.add_argument("--restapi_username", type=str, required=True, help="REST API 사용자명")
    parser.add_argument("--restapi_password", type=str, required=True, help="REST API 비밀번호")
    parser.add_argument("--internal_api_key", type=str, default="", help="internal-access 콜백용 X-Internal-API-Key")
    parser.add_argument("--gpu_limit", type=str, required=True, help="GPU 제한")
    parser.add_argument("--batch_size", type=str, required=True, help="배치 크기")
    parser.add_argument("--epochs", type=str, required=True, help="에포크 수")
    parser.add_argument("--save_period", type=str, required=True, help="저장 주기")
    parser.add_argument("--weight_decay", type=str, required=True, help="가중치 감소")
    parser.add_argument("--learning_rate", type=str, required=True, help="학습률 (lr0/lrf 통합 단일 필드)")

    return parser


def main():
    """메인 함수 — --model_kind 로 YOLOX / ESM2 분기."""
    args = build_arg_parser().parse_args()

    # BFM 분기(ESM2/ESMC): transformers/peft 는 이 경로에서만 lazy import 된다.
    if args.model_kind in ("esm2", "esmc"):
        if args.model_kind == "esmc":
            # 대형 ESMC 는 MIG 슬라이스 등에서 PyTorch 기본 caching allocator 의 NVML 여유메모리 조회가
            # "NVML_SUCCESS INTERNAL ASSERT" 로 죽는 경우가 있어, NVML 을 안 쓰는 CUDA async allocator 로
            # 전환한다(torch CUDA 초기화 전이라 여기서 세팅해도 유효). esm2/yolox 경로는 건드리지 않는다.
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
            # allocator 가 실제로 무엇으로 켜졌는지 학습 로그에 남긴다('native' 면 폴백된 것).
            import torch as _torch
            from app.esmc_finetuner import EsmcFineTuner as FineTuner

            logger.info(
                f"[esmc] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r} "
                f"cuda_allocator_backend={_torch.cuda.get_allocator_backend()!r} "
                f"cuda_device_count={_torch.cuda.device_count()}"
            )
        else:
            from app.esm2_finetuner import EsmFineTuner as FineTuner

        runner = FineTuner.from_args(args)
        try:
            runner.preprocess()
            runner.train()
            logger.info(f"{args.model_kind} 학습 완료!")
        except Exception as e:
            logger.error(f"작업 중 오류 발생: {e}")
            traceback.print_exc()
            raise
        return

    # 기본 분기: YOLOX (CustomTrainModel/YOLOX 는 여기서 lazy import)
    from app.yolox_finetuner import CustomTrainModel

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
        dataset_download_ref=args.dataset_download_ref,
        dataset_storage_type=args.dataset_storage_type,
        dataset_s3_endpoint_url=args.dataset_s3_endpoint_url,
        dataset_s3_access_key=args.dataset_s3_access_key,
        dataset_s3_secret_key=args.dataset_s3_secret_key,
        dataset_s3_bucket=args.dataset_s3_bucket,
        datalake_api_url=args.datalake_api_url,
        datalake_api_username=args.datalake_api_username,
        datalake_api_password=args.datalake_api_password,
        datalake_bucket_name=args.datalake_bucket_name,
        restapi_url=args.restapi_url,
        restapi_username=args.restapi_username,
        restapi_password=args.restapi_password,
        internal_api_key=args.internal_api_key,
        gpu_limit=args.gpu_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_period=args.save_period,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
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
