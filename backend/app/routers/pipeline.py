import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
from config.db.connect import SessionDepends
from config.settings import get_settings
from core.kubeflow.component.serve.serve import serving_component
from core.kubeflow.component.train_eval.register_model import register_model_component
from core.kubeflow.component.train_eval.train_eval import container_train_eval_component
from core.kubeflow.kubeflow_manager import KubeflowManager
from fastapi import APIRouter, Body, Depends, HTTPException
from kfp import dsl
from schemas.experiment import ExperimentBaseSchema, HyperparameterBaseSchema, TrainingStatusResponse
from schemas.user import UserSchema
from services.dataset import DatasetService
from services.experiment import ExperimentService, HyperparameterService, HyperparameterTypeService
from services.model import ModelService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
# TODO: 추후 전역 레벨 관리로 수정 필요
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@router.delete("/mlflow/experiments/{mlflow_experiment_name}")
def delete_mlflow_experiment(
    *, mlflow_experiment_name: str, current_user: UserSchema = Depends(get_current_user)
) -> dict:
    kf = KubeflowManager()
    return kf.delete_experiment(experiment_name=mlflow_experiment_name)


@router.post("/mlflow/experiments/{mlflow_experiment_name}")
def create_mlflow_experiment(
    *, mlflow_experiment_name: str, current_user: UserSchema = Depends(get_current_user)
) -> dict:
    kf = KubeflowManager()
    experiment = kf.create_experiment(experiment_name=mlflow_experiment_name)
    return experiment.experiment_id if experiment else -1


@router.post("/training", response_model=dict)
def container_train(
    *,
    db: Session = SessionDepends,
    model_id: int,
    dataset_id: int,
    current_user: UserSchema = Depends(get_current_user),
    train_name: str = Body(""),
    description: str = Body(""),
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
        train_image_url: str,
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
            train_image_url=train_image_url,
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
        kubeflow_experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
        mlflow_experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        kubeflow_experiment = kf.get_experiment_by_name(experiment_name=kubeflow_experiment_name)
        # kf.create_pipeline(sample_pipeline, pipeline_name )
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
            # lightweight_component,
            enable_caching=False,  # overrides the above disabling of caching
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
                # TODO: mocking data. 이후 수정 필요.
                "restapi_url": settings.REST_API_URL,
                "restapi_username": "surromind",
                "restapi_password": settings.DEMO_PASSWORD,
                "train_image_url": train_image_url,
                "gpu_limit": gpus,
                "batch_size": batch_size,
                "epochs": epochs,
                "save_period": save_period,
                "weight_decay": weight_decay,
                "lr0": lr0,
                "lrf": lrf,
            },
        )
        return {
            "experiment_id": experiment_db_obj.id,
        }
    except Exception as e:
        logger.error(f"error occured when register pipeline : {e}")
        return {
            "experiment_id": None,
        }


@router.post("/model/registration", response_model=bool)
def register_model(
    *,
    db: Session = SessionDepends,
    model_name: str,
    description: str,
    experiment_id: int,
    current_user: UserSchema = Depends(get_current_user),
):
    @dsl.pipeline
    def register_model_pipeline(
        parent_model_id: int,
        train_model_name: str,
        description: str,
        experiment_id: int,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
    ):
        register_model_component(
            parent_model_id=parent_model_id,
            train_model_name=train_model_name,
            description=description,
            experiment_id=experiment_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            restapi_url=restapi_url,
            restapi_username=restapi_username,
            restapi_password=restapi_password,
        )

    try:
        kf = KubeflowManager()
        client = kf.get_kfp_client()

        # TODO: mocking data. 이후 외부에서 인자로 받아야함.
        kubeflow_experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
        mlflow_experiment_name = settings.MLFLOW_EXPERIMENT_NAME

        kubeflow_experiment = kf.get_experiment_by_name(experiment_name=kubeflow_experiment_name)

        experiment_db_obj = ExperimentService().get(db, experiment_id)
        parent_model_id = experiment_db_obj.reference_model_id

        client.create_run_from_pipeline_func(
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
                "restapi_url": settings.REST_API_URL,
                "restapi_username": "surromind",
                "restapi_password": settings.DEMO_PASSWORD,
            },
        )
        return True
    except Exception as e:
        logger.error(f"error occured when register pipeline : {e}")
        return False


@router.post("/serving", response_model=bool)
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
        infer_image_url: str,
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
            infer_image_url=infer_image_url,
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
        kubeflow_experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
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
                "infer_image_url": settings.INFER_IMAGE_URL,
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


@router.get("/training/{experiment_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(
    db: Session = SessionDepends, *, experiment_id: int, current_user: UserSchema = Depends(get_current_user)
):
    """
    특정 experiment_id로 학습 파이프라인의 현재 상태를 가져옵니다.

    Args:
        experiment_id: 학습 PK (experiment_id)

    Returns:
        현재 학습 상태 정보 (epoch, loss, AP 등)
    """
    try:
        # 일회용 인스턴스 생성
        monitor = PipelineTrainingMonitor(
            mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI, experiment_name=settings.MLFLOW_EXPERIMENT_NAME
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
        # 학습 상태 가져오기
        status_data = monitor.get_training_status(run_id, int(max_epoch))

        if status_data is None:
            raise HTTPException(status_code=404, detail="학습 상태가 존재하지 않습니다.")

        return TrainingStatusResponse(**status_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"학습 상태 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training Monitor 클래스
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

            # 모든 메트릭 데이터를 딕셔너리 형태로 변환하여 가져오기
            metrics_data = {}
            metric_names = ["train/total_loss", "train/epoch", "AP50", "AP75", "val/best_ap", "mAP_0.5_0.95"]

            for metric_name in metric_names:
                try:
                    history = self.client.get_metric_history(run_id, metric_name)
                    # Metric 객체를 딕셔너리로 변환하고 정렬
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
                        # step 우선, timestamp 차선으로 정렬
                        metric_list.sort(key=lambda x: (x["step"], x["timestamp"]))
                        metrics_data[metric_name] = metric_list
                    else:
                        metrics_data[metric_name] = []
                except Exception as e:
                    logger.warning(f"메트릭 '{metric_name}' 조회 실패: {e}")
                    metrics_data[metric_name] = []

            # 현재 epoch 계산
            current_epoch = 0
            if "train/epoch" in metrics_data and metrics_data["train/epoch"]:
                current_epoch = int(metrics_data["train/epoch"][-1]["value"])

            # 상태 결정
            status = "RUNNING"
            if run.info.status == "FINISHED":
                status = "FINISHED"
            elif run.info.status == "FAILED":
                status = "FAILED"

            # 종료 시간 계산
            end_time = None
            if run.info.end_time:
                end_time = run.info.end_time
            elif status == "RUNNING":
                end_time = int(datetime.now().timestamp() * 1000)

            return {
                "status": status,
                "start_time": run.info.start_time,
                "end_time": end_time,
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
