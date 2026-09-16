import json
import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
from config.db.connect import SessionDepends
from config.db.enums import ModelFormatEnum, ModelProviderEnum
from config.settings import get_settings
from core.kubeflow.component.train_eval.register_model import register_model_component
from core.kubeflow.component.train_eval.train_eval import container_train_eval_component
from core.kubeflow.kubeflow_manager import KubeflowManager
from core.training.families import resolve_family
from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, HTTPException, UploadFile
from kfp import dsl
from schemas.dataset import DatasetBaseSchema
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
from services.experiment import ExperimentService, HyperparameterService
from services.metrics_polling import poll_training_metrics
from services.model import ModelService, resolve_recommended_hparams
from services.registration_polling import poll_registration_status
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── 학습 하이퍼파라미터 백필 (user input → 모델별 recommended → 시스템 fallback) ──
SYSTEM_HPARAM_DEFAULTS: dict[str, str] = {
    "learning_rate": "0.001",
    "batch_size": "16",
    "epochs": "10",
    "weight_decay": "0.0005",
    "save_period": "1",
    "gpus": "1",
}
HPARAM_KEYS = tuple(SYSTEM_HPARAM_DEFAULTS.keys())


def backfill_hparams(body: TrainingRequest, recommended: dict[str, str]) -> dict[str, str]:
    """user input → 모델별 recommended_hparams → SYSTEM_HPARAM_DEFAULTS 순으로 채운다."""
    resolved: dict[str, str] = {}
    for key in HPARAM_KEYS:
        user_val = getattr(body, key, None)
        if user_val is not None and user_val != "":
            resolved[key] = user_val
        elif key in recommended:
            resolved[key] = recommended[key]
        else:
            resolved[key] = SYSTEM_HPARAM_DEFAULTS[key]
    return resolved


def _resolve_root_family(db: Session, db_model):
    """lineage root 를 1회 조회하고 학습 가능 모델군(family)을 함께 반환. (root, family|None).

    재학습 자식은 repo_id 가 비어 있을 수 있으므로 lineage root 의 (format, repo_id) 로 군을 판정한다.
    판별 기준은 core.training.families.TRAINABLE_FAMILIES 단일 레지스트리(yolox 는 format, esm2 는 repo_id).
    root 가 None(삭제 등)이면 (None, None).
    """
    root_id = ModelService.resolve_lineage_root_model_id(db, db_model.id)
    root = ModelService.get(db, root_id)
    if root is None:
        return None, None
    fmt = root.format_info.name if root.format_info else ""
    fam = resolve_family(format_name=fmt, repo_id=(root.repo_id or ""))
    return root, fam


# ──────────────────────────────────────────────
# POST /training — multipart/form-data (dataset_id XOR dataset_file) + 백그라운드 메트릭 폴링
# ──────────────────────────────────────────────


@router.post("/training", response_model=dict)
def container_train(
    *,
    db: Session = SessionDepends,
    body: TrainingRequest = Depends(TrainingRequest.as_form),
    dataset_file: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    학습 파이프라인 생성 및 실행

    모델과 데이터셋을 사용하여 Kubeflow Pipeline 기반의 학습 파이프라인을 생성하고 실행합니다.
    요청은 multipart/form-data 로 통일되며 (dataset_id XOR dataset_file),
    학습 시작 후 백그라운드에서 MLflow 메트릭 폴링이 시작됩니다.
    YOLOX/ESM2 양쪽을 지원하며, 모델군에 맞지 않는 데이터셋 분류는 400 으로 거부합니다.

    ## Response (200, dict)
    - **experiment_id** (int): 생성된 실험 ID. (내부 오류 시 null 로 격하하지 않고 500 으로 응답)

    ## Status (DB `experiment`와의 관계, 참고)
    - 성공 시 새 실험 행은 **status = `CREATED`** 로 생성된 뒤,
      MLflow·메트릭 폴링에 따라 `RUNNING` → 학습 정상 완료 시 `COMPLETED`(MLflow `FINISHED`에 대응), 실패 시 `FAILED` 등으로 갱신됨.
    - 리모델링 설계(문서)에서는 완료 상태를 `FINISHED`로 표기하기도 하며, API/DB는 위와 같이 `COMPLETED`를 사용하는 경우가 일반적임.

    ## Errors
    - **400**: GPU 0 이하, 유효하지 않은 인자
    - **401**: 미인증
    - **404**: 모델/데이터셋 없음
    - **500**: 내부 오류(파이프라인 생성/제출 실패 등). 이전엔 200 `{experiment_id: null}` 로 격하됐으나 이제 500 으로 명확히 응답.
    """
    model_id = body.model_id
    train_name = body.train_name
    description = body.description

    # 실험명 중복 거부 — 같은 이름이 이미 있으면 학습(실험) 생성을 막는다.
    # (요청 측에서 실험명 뒤에 uuid 등을 붙여 유니크하게 보내야 통과한다.)
    if train_name and ExperimentService.get_by_name(db, train_name) is not None:
        raise HTTPException(status_code=409, detail=f"이미 존재하는 실험명입니다: '{train_name}'")

    # 데이터셋 입력: dataset_id XOR dataset_file (정확히 하나)
    has_dataset_id = body.dataset_id is not None
    has_dataset_file = dataset_file is not None
    if has_dataset_id == has_dataset_file:
        raise HTTPException(
            status_code=400,
            detail="dataset_id 와 dataset_file 중 정확히 하나만 제공해야 합니다.",
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
        dataset_download_ref: str,
        dataset_storage_type: str,
        dataset_s3_endpoint_url: str,
        dataset_s3_access_key: str,
        dataset_s3_secret_key: str,
        dataset_s3_bucket: str,
        datalake_api_url: str,
        datalake_api_username: str,
        datalake_api_password: str,
        datalake_bucket_name: str,
        mlflow_experiment_name: str,
        train_name: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
        internal_api_key: str,
        gpu_limit: str,
        batch_size: str,
        epochs: str,
        save_period: str,
        weight_decay: str,
        learning_rate: str,
        model_kind: str,
        namespace: str,
        train_image_url: str,
        image_pull_secret_name: str,
        gpu_resource_key: str,
        node_selector_json: str,
        tolerations_json: str,
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
            dataset_download_ref=dataset_download_ref,
            dataset_storage_type=dataset_storage_type,
            dataset_s3_endpoint_url=dataset_s3_endpoint_url,
            dataset_s3_access_key=dataset_s3_access_key,
            dataset_s3_secret_key=dataset_s3_secret_key,
            dataset_s3_bucket=dataset_s3_bucket,
            datalake_api_url=datalake_api_url,
            datalake_api_username=datalake_api_username,
            datalake_api_password=datalake_api_password,
            datalake_bucket_name=datalake_bucket_name,
            mlflow_experiment_name=mlflow_experiment_name,
            restapi_url=restapi_url,
            restapi_username=restapi_username,
            restapi_password=restapi_password,
            internal_api_key=internal_api_key,
            gpu_limit=gpu_limit,
            batch_size=batch_size,
            epochs=epochs,
            save_period=save_period,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            model_kind=model_kind,
            namespace=namespace,
            train_image_url=train_image_url,
            image_pull_secret_name=image_pull_secret_name,
            gpu_resource_key=gpu_resource_key,
            node_selector_json=node_selector_json,
            tolerations_json=tolerations_json,
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

        # 모델군 ↔ 데이터셋 호환성 판정 (lineage root 1회 조회 → family)
        root, fam = _resolve_root_family(db, db_model)
        if fam is None:
            raise HTTPException(status_code=400, detail=f"학습 가능한 모델이 아닙니다: {db_model.name}")
        # 모델군 ↔ type_info 교차검증 — 손상된 타입(예: 자식 등록 ODM 오저장)을 학습 진입에서 차단(backstop).
        if db_model.type_info and db_model.type_info.name != fam.model_type.value:
            raise HTTPException(
                status_code=400,
                detail=f"모델 '{db_model.name}' 타입 정보가 계보와 불일치합니다 "
                f"(계보상 {fam.model_type.value}, 등록 타입 {db_model.type_info.name}). 모델 메타 재등록·정정이 필요합니다.",
            )
        expected_kind = fam.dataset_kind.value
        model_kind = fam.key

        # 데이터셋 확보: dataset_file 이면 즉시 등록, dataset_id 면 조회 후 kind 검사
        if has_dataset_file:
            # dataset_kind 는 '업로드한 데이터의 사실'이라 필수다. 모델에서 도출(자동부여)하면
            # 검사가 동어반복이 되어 호환성 검증이 무력화된다 → 클라이언트가 명시한 값을 모델 요구와 대조한다.
            if body.dataset_kind is None:
                raise HTTPException(status_code=400, detail="dataset_file 동반 시 dataset_kind 는 필수입니다.")
            if body.dataset_kind.value != expected_kind:
                raise HTTPException(
                    status_code=400,
                    detail=f"데이터셋 분류 '{body.dataset_kind.value}' 가 "
                    f"모델이 요구하는 분류 '{expected_kind}' 와 일치하지 않습니다.",
                )
            resolved_kind = body.dataset_kind
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            auto_name = f"auto-{train_name or 'dataset'}-{ts}"
            auto_desc = f"학습 요청 시 자동 등록된 데이터셋 kind={resolved_kind.value}"
            dataset_obj = DatasetService.create(
                db,
                obj_in=DatasetBaseSchema(
                    name=auto_name,
                    description=auto_desc,
                    version=1,
                    subversion=1,
                    kind=resolved_kind,
                ),
                file=dataset_file,
            )
            dataset_id = dataset_obj.id
        else:
            dataset_id = body.dataset_id
            dataset_obj = DatasetService().get(db, dataset_id)
            if dataset_obj is None:
                raise HTTPException(status_code=404, detail=f"데이터셋 ID '{dataset_id}'를 찾을 수 없습니다.")
            if (dataset_obj.kind or "") != expected_kind:
                raise HTTPException(
                    status_code=400,
                    detail=f"데이터셋 분류 '{dataset_obj.kind}' 가 "
                    f"모델이 요구하는 분류 '{expected_kind}' 와 일치하지 않습니다.",
                )

        # 하이퍼파라미터 백필 (user → recommended → system)
        recommended = resolve_recommended_hparams(db, db_model)
        hparams = backfill_hparams(body, recommended)
        try:
            if int(hparams["gpus"]) <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="GPU 개수는 1 이상으로 설정해야 합니다. 현재 값: " + hparams["gpus"],
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"GPU 개수 값 '{hparams['gpus']}'이 유효하지 않습니다. 숫자로 입력해주세요.",
            )

        model_uri = db_model.registry.uri
        model_artifact_path = db_model.registry.artifact_path
        # BFM(ESM2/ESMC)은 (최초/재학습 모두) 항상 lineage root 카탈로그의 base 를 LoRA 의 기반으로 쓴다.
        # 자식 모델의 registry 는 adapter 를 가리키므로, base 는 lineage root 의 registry 에서 가져온다.
        if model_kind in ("esm2", "esmc"):
            # lineage root 는 위 _resolve_root_family 에서 이미 1회 조회했으므로 재사용(M3 중복조회 제거).
            if root is not None and root.registry:
                model_uri = root.registry.uri
                model_artifact_path = root.registry.artifact_path
        dataset_download_ref = dataset_obj.dataset_registry.uri
        dataset_storage_type = settings.DATASET_STORAGE_TYPE
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

        # 실제 적용된 하이퍼파라미터만 평탄 KV 로 기록 (재현·감사 가능)
        for name, value in hparams.items():
            create_hyperparameter(db, experiment_db_obj.id, name, value)

        # 학습 Job 노드 배치.
        # 1) GPU 노드풀 프로파일(MIG taint 등)이 있으면 그 정적 배치를 채택(toleration+nodeSelector+자원키 주입).
        # 2) 프로파일이 없고 SERVING_NODE_NAMES 화이트리스트가 있으면, 서빙과 동일한 노드 인벤토리에서
        #    여유 GPU 가 있는 노드를 골라 hostname 으로 고정한다 → 서빙이 화이트리스트로 피하는 비호환 GPU
        #    노드(예: 학습 이미지의 PyTorch 가 지원하지 않는 구형 GPU)를 학습 Job 도 함께 피한다.
        # 3) 둘 다 없으면 기존 기본(무 nodeSelector).
        from core.serving.gpu_placement import parse_gpu_pool_profiles, static_training_placement
        from core.serving.serving_k8s_inventory import choose_training_node_placement

        _profiles = parse_gpu_pool_profiles(getattr(settings, "GPU_POOL_PROFILES_JSON", "[]") or "[]")
        _serving_nodes = (getattr(settings, "SERVING_NODE_NAMES", "") or "").strip()
        if _profiles:
            _train_placement = static_training_placement(_profiles)
        elif _serving_nodes:
            _train_placement = choose_training_node_placement(
                requested_gpus=int(hparams["gpus"]),
                default_vram_bytes=settings.SERVING_DEFAULT_GPU_VRAM_BYTES,
                vram_overrides_json=settings.SERVING_NODE_VRAM_OVERRIDES_JSON,
                serving_node_names_csv=_serving_nodes,
                exclude_control_plane_nodes=not settings.SERVING_INCLUDE_CONTROL_PLANE_NODES,
            ) or static_training_placement([])
        else:
            _train_placement = static_training_placement([])

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
                "aws_access_key_id": settings.MLFLOW_S3_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.MLFLOW_S3_SECRET_ACCESS_KEY,
                "dataset_download_ref": dataset_download_ref,
                "dataset_storage_type": dataset_storage_type,
                "dataset_s3_endpoint_url": settings.S3_ENDPOINT,
                "dataset_s3_access_key": settings.S3_ACCESS_KEY,
                "dataset_s3_secret_key": settings.S3_SECRET_KEY,
                "dataset_s3_bucket": settings.S3_BUCKET,
                "datalake_api_url": settings.DATALAKE_API_URL,
                "datalake_api_username": settings.DATALAKE_API_USERNAME,
                "datalake_api_password": settings.DATALAKE_API_PASSWORD,
                "datalake_bucket_name": settings.DATALAKE_BUCKET_NAME,
                "train_name": train_name,
                "restapi_url": settings.REST_API_URL,
                "restapi_username": "surromind",
                "restapi_password": settings.DEMO_PASSWORD,
                "internal_api_key": settings.INTERNAL_API_KEY,
                "gpu_limit": hparams["gpus"],
                "batch_size": hparams["batch_size"],
                "epochs": hparams["epochs"],
                "save_period": hparams["save_period"],
                "weight_decay": hparams["weight_decay"],
                "learning_rate": hparams["learning_rate"],
                "model_kind": model_kind,
                "namespace": settings.KUBEFLOW_NAMESPACE,
                "train_image_url": settings.TRAIN_IMAGE_URL,
                "image_pull_secret_name": settings.KUBEFLOW_IMAGE_PULL_SECRET,
                "gpu_resource_key": _train_placement["gpu_resource_key"],
                "node_selector_json": json.dumps(_train_placement["node_selector"]),
                "tolerations_json": json.dumps(_train_placement["tolerations"]),
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
        # 침묵 실패 방지: 내부 오류를 200 {experiment_id: null} 로 격하하지 않고 500 으로 명확히 알린다.
        logger.exception("학습 파이프라인 생성/제출 중 오류")
        raise HTTPException(
            status_code=500,
            detail=f"학습 파이프라인 생성 중 오류가 발생했습니다: {e}",
        ) from e


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
    @deprecated — `GET /api/v1/experiments/{experiment_id}`(실험 상세) 사용 권장.

    MLflow run 기반으로 에폭·손실·AP 히스토리 등 **메트릭**을 돌려줍니다. `experiment` DB의
    `status` / `train_msg`와는 별도 경로(조회 전용)입니다.

    ## Response (`TrainingStatusResponse`) — `status` 필드
    - **RUNNING**: MLflow run이 아직 끝나지 않은 경우(또는 메트릭이 진행 중으로 해석될 때)
    - **FINISHED**: MLflow `run.info.status` 가
      `FINISHED` 인 경우(조회 API 표기; DB `experiment.status` 는 동기화 시 `COMPLETED`로 저장될 수 있음)
    - **FAILED**: MLflow run이 실패로 종료된 경우

    ## Errors
    - **404**: 실험 없음, 또는 MLflow run을 찾을 수 없음
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
            if hp.param_name == "epochs":
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

    ## Response (`ModelRegistrationResponse`, 200)
    - **accepted** (bool): 제출이 수락되었는지
    - **experiment_id** (int): 대상 실험 ID(요청과 동일; 실패 응답에도 동일 ID가 올 수 있음)
    - **message** (str): 안내 문구

    ## `experiment.registration_status` (제출 이후, `GET /experiments/{id}` 로 확인)
    - 제출 직전·미요청: **`NOT_REQUESTED`**
    - 본 API로 KFP run이 잡힌 뒤: **`PIPELINE_SUBMITTED`**
    - 백그라운드 폴링에 따라: **`SUCCESS`**(등록 완료, `registered_model_id` 채움) / **`FAILED`**(실패 또는 타임아웃)
    - 동일한 등록·메시지는 응답 **model_register_msg** 등에 반영

    ## Errors
    - **400**: 실험 상태로 등록 불가, 바디 검증 실패
    - **401**: 미인증
    - **404**: 실험 없음
    - **500**: 제출/파이프라인 오류(구현에 따라 `accepted: false` 등)
    """
    model_name = body.model_name
    description = body.description
    experiment_id = body.experiment_id

    # 모델명 중복 사전 거부(조기 종료) — KFP 제출 전에 막는다. 단일 가드 메서드 재사용.
    # (재등록 요청 측에서 모델명 뒤에 uuid 등을 붙여 유니크하게 보내야 통과한다.)
    ModelService.assert_model_name_available(db, model_name)

    @dsl.pipeline
    def register_model_pipeline(
        reference_model_id: int,
        lineage_root_parent_model_id: int,
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
        output_task: str = "",
        mlflow_run_id: str = "",
    ):
        register_model_component(
            reference_model_id=reference_model_id,
            lineage_root_parent_model_id=lineage_root_parent_model_id,
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
            output_task=output_task,
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

        reference_model_id = experiment_db_obj.reference_model_id
        try:
            lineage_root_parent_model_id = ModelService.resolve_lineage_root_model_id(db, reference_model_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # 파인튜닝 자식은 MLflow registry(LoRA adapter/가중치) 기반이므로 provider 는 항상 CUSTOM 으로 등록한다.
        # (huggingface 로 상속하면 create_model 이 repo_id 필수 + HF Hub 다운로드 경로로 가서 실패한다.)
        # model_type(coarse)은 reference 에서 상속(ESM2 → BFM, YOLOX → ODM). task(fine)는 전이한다:
        #   base(fill-mask 등) → 자식은 family.output_task(protein-classification). 부모 task 상속이 아님.
        reference_model = ModelService().get(db, reference_model_id)
        provider_name = ModelProviderEnum.CUSTOM.value
        _ref_root, _ref_fam = _resolve_root_family(db, reference_model) if reference_model is not None else (None, None)
        # KFP 격리: families 를 백엔드에서 resolve 해 output_task 를 컴포넌트 파라미터로 전달(컴포넌트 body 에선 import 불가).
        output_task = _ref_fam.output_task if _ref_fam is not None else ""
        if reference_model is not None and reference_model.type_info:
            type_name = reference_model.type_info.name
        else:
            # type 미해결 시 ODM 조용한 오저장 대신 모델군 레지스트리로 재판정, 그래도 없으면 400.
            if _ref_fam is None:
                raise HTTPException(
                    status_code=400,
                    detail="reference 모델의 타입을 확정할 수 없어 자식 모델을 등록할 수 없습니다. 모델 메타를 정정해 주세요.",
                )
            type_name = _ref_fam.model_type.value
        yolox_format_name = ModelFormatEnum.YOLOX.value
        pytorch_format_name = ModelFormatEnum.PYTORCH.value

        run_result = client.create_run_from_pipeline_func(
            register_model_pipeline,
            enable_caching=False,
            experiment_id=kubeflow_experiment.experiment_id,
            arguments={
                "reference_model_id": reference_model_id,
                "lineage_root_parent_model_id": lineage_root_parent_model_id,
                "train_model_name": model_name,
                "description": description,
                "experiment_id": experiment_id,
                "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
                "mlflow_experiment_name": mlflow_experiment_name,
                "mlflow_s3_endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL,
                "aws_access_key_id": settings.MLFLOW_S3_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.MLFLOW_S3_SECRET_ACCESS_KEY,
                "restapi_url": settings.REST_API_URL,
                "restapi_username": "surromind",
                "restapi_password": settings.DEMO_PASSWORD,
                "provider_name": provider_name,
                "type_name": type_name,
                "yolox_format_name": yolox_format_name,
                "pytorch_format_name": pytorch_format_name,
                "output_task": output_task,
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
    return HyperparameterService().create(
        db,
        obj_in=HyperparameterBaseSchema(
            experiment_id=experiment_id,
            param_name=param_name,
            value=value,
        ),
    )
