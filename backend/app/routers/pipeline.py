import base64
import io
import json
import logging
import os
import uuid

import requests
from config.db.connect import SessionDepends
from config.settings import get_settings
from core.kubeflow.component.serve.serve import serving_component
from core.kubeflow.component.train_eval.train_eval import container_train_eval_component
from core.kubeflow.kubeflow_manager import KubeflowManager
from fastapi import APIRouter, Body, Depends
from kfp import dsl
from PIL import Image
from schemas.user import UserSchema
from services.dataset import DatasetService
from services.model import ModelService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
# TODO: 추후 전역 레벨 관리로 수정 필요
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@router.delete("/experiments/{experiment_name}")
def delete_experiment(*, experiment_name: str, current_user: UserSchema = Depends(get_current_user)) -> dict:
    kf = KubeflowManager()
    return kf.delete_experiment(experiment_name=experiment_name)


@router.post("/experiments/{experiment_name}")
def create_experiment(*, experiment_name: str, current_user: UserSchema = Depends(get_current_user)) -> dict:
    kf = KubeflowManager()
    experiment = kf.create_experiment(experiment_name=experiment_name)
    return experiment.experiment_id if experiment else -1


@router.post("/training/container", response_model=bool)
def container_train(
    *,
    db: Session = SessionDepends,
    model_id: int,
    dataset_id: int,
    current_user: UserSchema = Depends(get_current_user),
    train_name: str = Body(""),
    result_model_name: str = Body(""),
    gpus: str = Body("1"),
    batch_size: str = Body("64"),
    epochs: str = Body("5"),
    save_period: str = Body("1"),
    weight_decay: str = Body("5e-4"),
    lr0: str = Body("0.01"),
    lrf: str = Body("0.05"),
):
    @dsl.pipeline
    def train_pipeline(
        model_id: int,
        model_uri: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        model_artifact_path: str,
        model_name: str,
        dataset_artifact_uri: str,
        mlflow_experiment_name: str,
        train_name: str,
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
    ):
        container_train_eval_component(
            model_id=model_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_artifact_path=model_artifact_path,
            model_uri=model_uri,
            train_name=train_name,
            result_model_name=result_model_name,
            dataset_artifact_uri=dataset_artifact_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            restapi_url=restapi_url,
            restapi_username=restapi_username,
            restapi_password=restapi_password,
            gpu_limit=gpu_limit,
            batch_size=batch_size,
            epochs=epochs,
            save_period=save_period,
            weight_decay=weight_decay,
            lr0=lr0,
            lrf=lrf,
        )

    try:
        db_model = ModelService().get(db, model_id)
        model_uri = db_model.registry.uri
        model_artifact_path = db_model.registry.artifact_path
        dataset_model = DatasetService().get(db, dataset_id)
        dataset_artifact_uri = os.path.join(
            dataset_model.dataset_registry.artifact_path, dataset_model.dataset_registry.uri
        )
        kf = KubeflowManager()
        client = kf.get_kfp_client()
        # TODO: mocking data. 이후 수정 필요.
        kubeflow_experiment_name = "aipaas-ml-workflow"
        mlflow_experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        experiment = kf.get_experiment_by_name(experiment_name=kubeflow_experiment_name)
        # kf.create_pipeline(sample_pipeline, pipeline_name )

        client.create_run_from_pipeline_func(
            train_pipeline,
            # lightweight_component,
            enable_caching=False,  # overrides the above disabling of caching
            experiment_id=experiment.experiment_id,
            arguments={
                "model_id": model_id,
                "model_artifact_path": model_artifact_path,
                "model_uri": model_uri,
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "result_model_name": result_model_name,
                "mlflow_experiment_name": mlflow_experiment_name,
                "mlflow_s3_endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL,
                "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                "dataset_artifact_uri": dataset_artifact_uri,
                "train_name": train_name,
                # TODO: mocking data. 이후 수정 필요.
                "restapi_url": settings.REST_API_URL,
                "restapi_username": "surromind",
                "restapi_password": settings.DEMO_PASSWORD,
                "gpu_limit": gpus,
                "batch_size": batch_size,
                "epochs": epochs,
                "save_period": save_period,
                "weight_decay": weight_decay,
                "lr0": lr0,
                "lrf": lrf,
            },
        )
        return True
    except Exception as e:
        logger.error(f"error occured when register pipeline : {e}")
        return False


@router.post("/serving/{pk}", response_model=bool)
def serve(
    *,
    db: Session = SessionDepends,
    model_id: int,
    inference_service_name: str,
    request_gpu: str = "1",
    request_cpu: str = "200m",
    request_memory: str = "2Gi",
    limit_gpu: str = "1",
    limit_cpu: str = "500m",
    limit_memory: str = "4Gi",
    current_user: UserSchema = Depends(get_current_user),
):
    @dsl.pipeline
    def serve_pipeline(
        inference_service_name: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        mlflow_experiment_name: str,
        model_uri: str,
        model_name: str,
        s3_storage_uri: str,
        framework: str,  # 프레임워크 파라미터 추가
        run_id: str,  # run_id 파라미터 추가
        kserve_gpu_yn: bool,
        request_gpu: str,
        request_cpu: str,
        request_memory: str,
        limit_gpu: str,
        limit_cpu: str,
        limit_memory: str,
    ) -> str:
        task = serving_component(
            inference_service_name=inference_service_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            mlflow_experiment_name=mlflow_experiment_name,
            model_name=model_name,
            model_uri=model_uri,
            s3_storage_uri=s3_storage_uri,
            framework=framework,  # 프레임워크 전달
            run_id=run_id,  # run_id 전달
            kserve_gpu_yn=kserve_gpu_yn,
            request_gpu=request_gpu,
            request_cpu=request_cpu,
            request_memory=request_memory,
            limit_gpu=limit_gpu,
            limit_cpu=limit_cpu,
            limit_memory=limit_memory,
        )
        return task.output

    try:
        logger.info(f"KServe Use GPU = {settings.KSERVE_GPU}")
        db_model = ModelService().get(db, model_id)
        model_uri = db_model.registry.uri
        model_name = db_model.name

        # TODO : display_name 이나 serving_name 관리 필요.
        model_name = model_name.replace("/", "-")

        # 프레임워크 결정: model_format에서 가져오기
        framework = "pytorch"  # 기본값
        if db_model.format_info:
            format_name = db_model.format_info.name.lower()
            if format_name in ["pytorch", "keras", "onnx"]:
                framework = format_name
            elif format_name == "transformers":
                framework = "pytorch"  # transformers는 기본적으로 pytorch 기반

        run_id = db_model.registry.run_id if db_model.registry.run_id else None

        logger.info(f"Model framework: {framework}, run_id: {run_id}")

        # TODO: singleton instance로 변경필요.
        kf = KubeflowManager()
        client = kf.get_kfp_client()

        # TODO: mocking data. 이후 외부에서 인자로 받아야함.
        kubeflow_experiment_name = "aipaas-ml-workflow"
        mlflow_experiment_name = settings.MLFLOW_EXPERIMENT_NAME

        experiment = kf.get_experiment_by_name(experiment_name=kubeflow_experiment_name)

        # s3 model_stroage_uri mocking data. currently not use.
        s3_storage_uri = "s3://mlflow/mlflow-artifacts/..."

        client.create_run_from_pipeline_func(
            serve_pipeline,
            enable_caching=False,  # overrides the above disabling of caching
            experiment_id=experiment.experiment_id,
            arguments={
                # TODO: mocking data. 이후 외부에서 인자로 받아야함.
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "mlflow_experiment_name": mlflow_experiment_name,
                "mlflow_s3_endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL,
                "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                "inference_service_name": inference_service_name,
                "model_name": model_name,
                "model_uri": model_uri,
                "s3_storage_uri": s3_storage_uri,
                "framework": framework,  # 프레임워크 전달
                "run_id": run_id,  # run_id 전달
                "kserve_gpu_yn": settings.KSERVE_GPU,  # kserve_gpu -> kserve_gpu_yn으로 수정
                # TODO: 추후 외부에서 인자로 받아야함.
                "request_gpu": request_gpu,
                "request_cpu": request_cpu,
                "request_memory": request_memory,
                "limit_gpu": limit_gpu,
                "limit_cpu": limit_cpu,
                "limit_memory": limit_memory,
            },
        )
        return True
    except Exception as e:
        logger.error(f"error occured when register pipeline : {e}")
        return False
