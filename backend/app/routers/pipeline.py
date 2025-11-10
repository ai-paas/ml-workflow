import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
from config.db.connect import SessionDepends
from config.settings import get_settings
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
    """
    학습 파이프라인 생성 및 실행

    모델과 데이터셋을 사용하여 Kubeflow Pipeline 기반의 학습 파이프라인을 생성하고 실행합니다.
    학습 실험(Experiment)을 생성하고 하이퍼파라미터를 설정한 후, Kubeflow에서 학습 작업을 시작합니다.

    ## Request Body
    - **model_id** (int, required): 학습에 사용할 모델 ID
        - 모델 레지스트리에 등록된 모델의 고유 ID
    - **dataset_id** (int, required): 학습에 사용할 데이터셋 ID
        - 데이터셋 레지스트리에 등록된 데이터셋의 고유 ID
    - **train_name** (str, optional): 학습 실험 이름
        - 기본값: 빈 문자열
        - 실험을 식별하기 위한 이름
    - **description** (str, optional): 학습 실험 설명
        - 기본값: 빈 문자열
        - 실험에 대한 상세 설명
    - **gpus** (str, optional): 사용할 GPU 개수
        - 기본값: "1"
        - 학습에 할당할 GPU 리소스 수
    - **batch_size** (str, optional): 배치 크기
        - 기본값: "64"
        - 한 번에 처리할 샘플 수
    - **epochs** (str, optional): 학습 에포크 수
        - 기본값: "5"
        - 전체 데이터셋을 몇 번 반복 학습할지 설정
    - **save_period** (str, optional): 모델 저장 주기
        - 기본값: "1"
        - 몇 에포크마다 모델을 저장할지 설정
    - **weight_decay** (str, optional): 가중치 감쇠(정규화) 계수
        - 기본값: "5e-4"
        - 오버피팅 방지를 위한 L2 정규화 계수
    - **lr0** (str, optional): 초기 학습률
        - 기본값: "0.01"
        - 학습 시작 시 사용할 학습률
    - **lrf** (str, optional): 최종 학습률
        - 기본값: "0.05"
        - 학습 종료 시 사용할 학습률 (lr0의 비율)

    ## Response
    - **experiment_id** (int): 생성된 실험(Experiment)의 고유 ID
        - 학습 상태 조회 및 모델 등록 시 사용
        - 실패 시 null 반환

    ## Process Flow
    1. 모델 및 데이터셋 정보 조회
    2. MLflow 및 Kubeflow 설정 확인
    3. 실험(Experiment) 레코드 생성
    4. 하이퍼파라미터 저장 (epochs, batch_size, weight_decay, lr0, lrf, gpus, save_period)
    5. Kubeflow Pipeline 실행 시작

    ## Notes
    - 학습은 비동기로 실행되며, 즉시 완료되지 않습니다
    - 학습 상태는 `/training/{experiment_id}/status` API로 조회할 수 있습니다
    - 학습 완료 후 `/model/registration` API로 학습된 모델을 등록할 수 있습니다
    - 실패 시 experiment_id가 null로 반환되며, 에러는 로그에 기록됩니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 모델 또는 데이터셋을 찾을 수 없음
    - 500: 파이프라인 생성 또는 실행 중 서버 내부 오류
    """

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
    """
    학습 완료된 모델 등록 파이프라인 실행

    학습이 완료된 실험(Experiment)의 결과 모델을 모델 레지스트리에 등록하는 파이프라인을 실행합니다.
    MLflow에 저장된 학습된 모델을 조회하여 새로운 모델로 등록하며, 부모 모델과의 관계를 설정합니다.

    ## Request Body
    - **model_name** (str, required): 등록할 모델 이름
        - 새로 등록될 모델의 이름
        - 모델 레지스트리에서 식별하기 위한 이름
    - **description** (str, required): 모델 설명
        - 등록할 모델에 대한 상세 설명
        - 학습 조건, 성능 등에 대한 정보 포함 권장
    - **experiment_id** (int, required): 학습 실험 ID
        - 학습이 완료된 실험(Experiment)의 고유 ID
        - 해당 실험의 MLflow run에서 모델을 가져옴

    ## Response
    - **success** (bool): 파이프라인 실행 성공 여부
        - True: 파이프라인 실행 성공
        - False: 파이프라인 실행 실패 (에러는 로그에 기록됨)

    ## Process Flow
    1. 실험(Experiment) 정보 조회 및 부모 모델 ID 확인
    2. Kubeflow Pipeline 생성 및 실행
    3. MLflow에서 학습된 모델 조회
    4. 모델 레지스트리에 새 모델로 등록
    5. 부모 모델과의 관계 설정

    ## Notes
    - 학습이 완료된 실험에 대해서만 사용해야 합니다
    - 등록된 모델은 부모 모델의 자식 모델로 설정됩니다
    - 파이프라인 실행은 비동기로 진행되며, 즉시 완료되지 않습니다
    - 실패 시 False를 반환하며, 상세 에러는 로그에 기록됩니다
    - 모델 등록 후 모델 목록에서 조회할 수 있습니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 실험(Experiment)을 찾을 수 없음
    - 500: 파이프라인 실행 중 서버 내부 오류
    """

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


@router.get("/training/{experiment_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(
    db: Session = SessionDepends, *, experiment_id: int, current_user: UserSchema = Depends(get_current_user)
):
    """
    학습 파이프라인 상태 조회

    특정 실험(Experiment)의 학습 진행 상태와 메트릭을 조회합니다.
    MLflow에서 실시간 학습 메트릭을 가져와 현재 epoch, loss, 평균 정밀도(AP) 등의 정보를 제공합니다.

    ## Path Parameters
    - **experiment_id** (int): 조회할 실험(Experiment)의 고유 ID
        - 학습 파이프라인 생성 시 반환된 experiment_id 사용

    ## Response (TrainingStatusResponse)
    - **status** (str): 학습 상태
        - "RUNNING": 학습 진행 중
        - "FINISHED": 학습 완료
        - "FAILED": 학습 실패
    - **start_time** (int): 학습 시작 시각 (밀리초 단위 타임스탬프)
    - **end_time** (int): 학습 종료 시각 (밀리초 단위 타임스탬프)
        - 진행 중인 경우 현재 시각
    - **max_epoch** (int): 설정된 최대 에포크 수
    - **current_epoch** (int): 현재 진행 중인 에포크
    - **loss_history** (List[dict]): 손실(loss) 히스토리
        - 각 항목: key (str), value (float), timestamp (int), step (int)
    - **epoch_history** (List[dict]): 에포크 히스토리
        - 각 항목: key (str), value (float), timestamp (int), step (int)
    - **average_precision_50_history** (List[dict]): AP@50 히스토리
        - IoU 0.5 기준 평균 정밀도
    - **average_precision_75_history** (List[dict]): AP@75 히스토리
        - IoU 0.75 기준 평균 정밀도
    - **best_average_precision_history** (List[dict]): 최고 평균 정밀도 히스토리
        - 검증 세트에서의 최고 성능
    - **average_precision_50_95_history** (List[dict]): mAP@0.5:0.95 히스토리
        - IoU 0.5~0.95 범위의 평균 평균 정밀도

    ## Notes
    - 학습이 시작되지 않은 경우 일부 메트릭이 비어있을 수 있습니다
    - 메트릭은 step과 timestamp 기준으로 정렬되어 반환됩니다
    - 특정 메트릭 조회 실패 시 해당 메트릭은 빈 리스트로 반환됩니다
    - 학습이 완료되면 status가 "FINISHED"로 변경됩니다
    - 실시간으로 학습 진행 상황을 모니터링할 수 있습니다

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 실험을 찾을 수 없거나 학습 상태가 존재하지 않음
    - 500: MLflow 연결 또는 메트릭 조회 중 서버 내부 오류
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
