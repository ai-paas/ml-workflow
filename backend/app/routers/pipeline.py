import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
from config.db.connect import SessionDepends
from config.db.enums import ModelFormatEnum, ModelProviderEnum, ModelTypeEnum
from config.settings import get_settings
from core.kubeflow.component.train_eval.register_model import register_model_component
from core.kubeflow.component.train_eval.train_eval import container_train_eval_component
from core.kubeflow.kubeflow_manager import KubeflowManager
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from kfp import dsl
from schemas.experiment import (
    ExperimentBaseSchema,
    HyperparameterBaseSchema,
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    TrainingRequest,
    TrainingStatusResponse,
)
from schemas.user import UserSchema
from services.dataset import DatasetService
from services.experiment import ExperimentService, HyperparameterService, HyperparameterTypeService
from services.metrics_polling import poll_training_metrics
from services.model import ModelService
from services.registration_polling import poll_registration_status
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────
# POST /training — 전체 Body(JSON) 통일 + 백그라운드 메트릭 폴링
# ──────────────────────────────────────────────


@router.post("/training", response_model=dict)
def container_train(
    *,
    db: Session = SessionDepends,
    body: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    학습 파이프라인 생성 및 실행

    모델과 데이터셋을 사용하여 Kubeflow Pipeline 기반의 학습 파이프라인을 생성하고 실행합니다.
    요청은 전체 Body(JSON)로 통일되며, 학습 시작 후 백그라운드에서 MLflow 메트릭 폴링이 시작됩니다.
    """
    model_id = body.model_id
    dataset_id = body.dataset_id
    train_name = body.train_name
    description = body.description
    gpus = body.gpus
    batch_size = body.batch_size
    epochs = body.epochs
    save_period = body.save_period
    weight_decay = body.weight_decay
    lr0 = body.lr0
    lrf = body.lrf

    try:
        gpu_count = int(gpus)
        if gpu_count <= 0:
            raise HTTPException(
                status_code=400,
                detail="GPU 개수는 필수적으로 1개 이상으로 설정해야 합니다. 현재 설정된 값: " + str(gpu_count),
            )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"GPU 개수 값 '{gpus}'이 유효하지 않습니다. 숫자로 입력해주세요.",
        )

    @dsl.pipeline
    def train_pipeline(
        model_id: int,
        model_uri: str,
        experiment_id: int,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        model_artifact_path: str,
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
        namespace: str,
        train_image_url: str,
        image_pull_secret_name: str,
    ):
        container_train_eval_component(
            model_id=model_id,
            experiment_id=experiment_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_artifact_path=model_artifact_path,
            model_uri=model_uri,
            train_name=train_name,
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
            namespace=namespace,
            train_image_url=train_image_url,
            image_pull_secret_name=image_pull_secret_name,
        )

    try:
        db_model = ModelService().get(db, model_id)
        if db_model is None:
            raise HTTPException(status_code=404, detail=f"모델 ID '{model_id}'를 찾을 수 없습니다.")

        if not db_model.learning_enable_yn:
            raise HTTPException(
                status_code=400,
                detail=f"모델 ID '{model_id}'는 학습이 불가능한 모델입니다. 학습 가능한 모델만 사용할 수 있습니다.",
            )

        model_uri = db_model.registry.uri
        model_artifact_path = db_model.registry.artifact_path
        dataset_model = DatasetService().get(db, dataset_id)
        dataset_artifact_uri = os.path.join(
            dataset_model.dataset_registry.artifact_path, dataset_model.dataset_registry.uri
        )
        kf = KubeflowManager()
        client = kf.get_kfp_client()
        kubeflow_experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
        mlflow_experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        kubeflow_experiment = kf.get_experiment_by_name(experiment_name=kubeflow_experiment_name)
        if not kubeflow_experiment:
            kubeflow_experiment = kf.create_experiment(kubeflow_experiment_name)
        experiment_db_obj = ExperimentService().create(
            db,
            obj_in=ExperimentBaseSchema(
                name=train_name,
                description=description,
                reference_model_id=model_id,
                dataset_id=dataset_id,
                status="CREATED",
            ),
        )

        create_hyperparameter(db, experiment_db_obj.id, "epochs", epochs)
        create_hyperparameter(db, experiment_db_obj.id, "batch_size", batch_size)
        create_hyperparameter(db, experiment_db_obj.id, "weight_decay", weight_decay)
        create_hyperparameter(db, experiment_db_obj.id, "lr0", lr0)
        create_hyperparameter(db, experiment_db_obj.id, "lrf", lrf)
        create_hyperparameter(db, experiment_db_obj.id, "gpus", gpus)
        create_hyperparameter(db, experiment_db_obj.id, "save_period", save_period)

        client.create_run_from_pipeline_func(
            train_pipeline,
            enable_caching=False,
            experiment_id=kubeflow_experiment.experiment_id,
            arguments={
                "model_id": model_id,
                "experiment_id": experiment_db_obj.id,
                "model_artifact_path": model_artifact_path,
                "model_uri": model_uri,
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "mlflow_experiment_name": mlflow_experiment_name,
                "mlflow_s3_endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL,
                "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                "dataset_artifact_uri": dataset_artifact_uri,
                "train_name": train_name,
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
                "namespace": settings.KUBEFLOW_NAMESPACE,
                "train_image_url": settings.TRAIN_IMAGE_URL,
                "image_pull_secret_name": settings.KUBEFLOW_IMAGE_PULL_SECRET,
            },
        )

        background_tasks.add_task(
            poll_training_metrics,
            experiment_id=experiment_db_obj.id,
        )

        return {
            "experiment_id": experiment_db_obj.id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"error occured when register pipeline : {e}")
        return {
            "experiment_id": None,
        }


# ──────────────────────────────────────────────
# GET /training/{experiment_id}/status — @deprecated, GET /experiments/{id}로 대체
# 기존 코드 유지, 신규 사용 금지
# ──────────────────────────────────────────────


@router.get(
    "/training/{experiment_id}/status",
    response_model=TrainingStatusResponse,
    deprecated=True,
)
async def get_training_status(
    db: Session = SessionDepends,
    *,
    experiment_id: int,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    @deprecated — GET /api/v1/experiments/{experiment_id} 사용 권장.

    기존 Path Parameter 방식의 학습 상태 조회. 하위 호환을 위해 유지합니다.
    """
    try:
        monitor = PipelineTrainingMonitor(
            mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
            experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
        )

        experiment_db_model = ExperimentService.get(db, experiment_id)
        if experiment_db_model is None:
            raise HTTPException(status_code=404, detail=f"실험 ID '{experiment_id}'을 찾을 수 없습니다.")

        run_id = experiment_db_model.mlflow_run_id
        max_epoch = 0
        for hp in experiment_db_model.hyperparameters:
            if hp.hyperparameter_type.param_name == "epochs":
                max_epoch = int(hp.value)
                break

        status_data = monitor.get_training_status(run_id, max_epoch)

        if status_data is None:
            raise HTTPException(status_code=404, detail="학습 상태가 존재하지 않습니다.")

        return TrainingStatusResponse(**status_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"학습 상태 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# POST /model/registration — Body(JSON) + 응답 객체화 + run_id 저장 + 백그라운드 폴링
# ──────────────────────────────────────────────


@router.post("/model/registration", response_model=ModelRegistrationResponse)
def register_model(
    *,
    db: Session = SessionDepends,
    body: ModelRegistrationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    학습 완료된 모델 등록 파이프라인 실행

    Query Parameter → Body(JSON) 변경. 응답이 객체로 확장되었으며,
    KFP run_id를 DB에 저장하고 백그라운드에서 등록 상태 폴링이 시작됩니다.
    """
    model_name = body.model_name
    description = body.description
    experiment_id = body.experiment_id

    @dsl.pipeline
    def register_model_pipeline(
        parent_model_id: int,
        train_model_name: str,
        description: str,
        experiment_id: int,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
        provider_name: str,
        type_name: str,
        yolox_format_name: str,
        pytorch_format_name: str,
        mlflow_run_id: str = "",
    ):
        register_model_component(
            parent_model_id=parent_model_id,
            train_model_name=train_model_name,
            description=description,
            experiment_id=experiment_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            restapi_url=restapi_url,
            restapi_username=restapi_username,
            restapi_password=restapi_password,
            provider_name=provider_name,
            type_name=type_name,
            yolox_format_name=yolox_format_name,
            pytorch_format_name=pytorch_format_name,
            mlflow_run_id=mlflow_run_id,
        )

    try:
        kf = KubeflowManager()
        client = kf.get_kfp_client()

        kubeflow_experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
        mlflow_experiment_name = settings.MLFLOW_EXPERIMENT_NAME

        kubeflow_experiment = kf.get_experiment_by_name(experiment_name=kubeflow_experiment_name)
        if not kubeflow_experiment:
            kubeflow_experiment = kf.create_experiment(kubeflow_experiment_name)

        experiment_db_obj = ExperimentService().get(db, experiment_id)
        if experiment_db_obj is None:
            raise HTTPException(status_code=404, detail=f"실험 ID '{experiment_id}'을 찾을 수 없습니다.")

        parent_model_id = experiment_db_obj.reference_model_id

        provider_name = ModelProviderEnum.CUSTOM.value
        type_name = ModelTypeEnum.ODM.value
        yolox_format_name = ModelFormatEnum.YOLOX.value
        pytorch_format_name = ModelFormatEnum.PYTORCH.value

        run_result = client.create_run_from_pipeline_func(
            register_model_pipeline,
            enable_caching=False,
            experiment_id=kubeflow_experiment.experiment_id,
            arguments={
                "parent_model_id": parent_model_id,
                "train_model_name": model_name,
                "description": description,
                "experiment_id": experiment_id,
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "mlflow_experiment_name": mlflow_experiment_name,
                "mlflow_s3_endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL,
                "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                "restapi_url": settings.REST_API_URL,
                "restapi_username": "surromind",
                "restapi_password": settings.DEMO_PASSWORD,
                "provider_name": provider_name,
                "type_name": type_name,
                "yolox_format_name": yolox_format_name,
                "pytorch_format_name": pytorch_format_name,
                "mlflow_run_id": experiment_db_obj.mlflow_run_id or "",
            },
        )

        registration_run_id = run_result.run_id

        experiment_db_obj.registration_kubeflow_run_id = registration_run_id
        experiment_db_obj.registration_status = "PIPELINE_SUBMITTED"
        experiment_db_obj.model_register_msg = None
        experiment_db_obj.registered_model_id = None
        db.commit()

        background_tasks.add_task(
            poll_registration_status,
            experiment_id=experiment_id,
            registration_run_id=registration_run_id,
        )

        return ModelRegistrationResponse(
            accepted=True,
            experiment_id=experiment_id,
            message="모델 등록 파이프라인이 시작되었습니다.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"error occured when register pipeline : {e}")
        return ModelRegistrationResponse(
            accepted=False,
            experiment_id=experiment_id,
            message=f"모델 등록 파이프라인 시작에 실패했습니다: {str(e)}",
        )


# ──────────────────────────────────────────────
# PipelineTrainingMonitor (기존 유지)
# ──────────────────────────────────────────────


class PipelineTrainingMonitor:
    def __init__(self, mlflow_tracking_uri: str, experiment_name: str):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)

    def get_training_status(self, run_id: Optional[str], max_epoch: int):
        """학습 상태 정보를 가져옵니다."""
        try:
            if run_id is None:
                return None

            run = self.client.get_run(run_id)

            metrics_data = {}
            metric_names = ["train/total_loss", "train/epoch", "AP50", "AP75", "val/best_ap", "mAP_0.5_0.95"]

            for metric_name in metric_names:
                try:
                    history = self.client.get_metric_history(run_id, metric_name)
                    if history:
                        metric_list = [
                            {
                                "key": metric.key,
                                "value": metric.value,
                                "timestamp": metric.timestamp,
                                "step": metric.step,
                            }
                            for metric in history
                        ]
                        metric_list.sort(key=lambda x: (x["step"], x["timestamp"]))
                        metrics_data[metric_name] = metric_list
                    else:
                        metrics_data[metric_name] = []
                except Exception as e:
                    logger.warning(f"메트릭 '{metric_name}' 조회 실패: {e}")
                    metrics_data[metric_name] = []

            current_epoch = 0
            if "train/epoch" in metrics_data and metrics_data["train/epoch"]:
                current_epoch = int(metrics_data["train/epoch"][-1]["value"])

            status = "RUNNING"
            if run.info.status == "FINISHED":
                status = "FINISHED"
            elif run.info.status == "FAILED":
                status = "FAILED"

            end_time = None
            if run.info.end_time:
                end_time = run.info.end_time
            elif status == "RUNNING":
                end_time = int(datetime.now().timestamp() * 1000)

            elapsed_time = max(0, (end_time - run.info.start_time) // 1000) if end_time else 0

            return {
                "status": status,
                "start_time": run.info.start_time,
                "end_time": end_time,
                "elapsed_time": elapsed_time,
                "max_epoch": max_epoch,
                "current_epoch": current_epoch,
                "loss_history": metrics_data.get("train/total_loss", []),
                "epoch_history": metrics_data.get("train/epoch", []),
                "average_precision_50_history": metrics_data.get("AP50", []),
                "average_precision_75_history": metrics_data.get("AP75", []),
                "best_average_precision_history": metrics_data.get("val/best_ap", []),
                "average_precision_50_95_history": metrics_data.get("mAP_0.5_0.95", []),
            }

        except Exception as e:
            logger.error(f"학습 상태 가져오기 실패: {e}")
            return None


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────


def create_hyperparameter(db: Session, experiment_id: int, param_name: str, value: str):
    hp_type_obj = HyperparameterTypeService().get_by_param_name(db, param_name)
    return HyperparameterService().create(
        db,
        obj_in=HyperparameterBaseSchema(
            experiment_id=experiment_id,
            hyperparameter_type_id=hp_type_obj.id,
            value=value,
        ),
    )
