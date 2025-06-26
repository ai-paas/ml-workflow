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
    train_name: str,
    current_user: UserSchema = Depends(get_current_user),
):
    @dsl.pipeline
    def train_pipeline(
        model_uri: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        model_name: str,
        dataset_artifact_uri: str,
        mlflow_experiment_name: str,
        train_name: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
    ):
        container_train_eval_component(
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_uri=model_uri,
            train_name=train_name,
            model_name=model_name,
            dataset_artifact_uri=dataset_artifact_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            restapi_url=restapi_url,
            restapi_username=restapi_username,
            restapi_password=restapi_password,
        )
        # TODO: singleton instance로 변경필요.

    try:
        db_model = ModelService().get(db, model_id)
        model_uri = db_model.model_registry.model_uri

        model_name = db_model.name
        dataset_model = DatasetService().get(db, dataset_id)
        dataset_artifact_uri = os.path.join(
            dataset_model.dataset_registry.artifact_path, dataset_model.dataset_registry.dataset_uri
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
                "model_uri": model_uri,
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "model_name": model_name,
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
        model_uri = db_model.model_registry.model_uri
        model_name = db_model.name

        # TODO : display_name 이나 serving_name 관리 필요.
        model_name = model_name.replace("/", "-")

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
                "kserve_gpu": settings.KSERVE_GPU,
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
