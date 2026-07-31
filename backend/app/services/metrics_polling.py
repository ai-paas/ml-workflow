import json
import logging
import time
from datetime import datetime, timezone

import mlflow
from config.db.session import SessionLocal
from config.settings import get_settings
from db.models.experiment import ExperimentMetricsModel, ExperimentModel
from mlflow.exceptions import MlflowException
from services.experiment import ExperimentService
from sqlalchemy import select

logger = logging.getLogger(__name__)
settings = get_settings()

MLFLOW_TO_EXPERIMENT_STATUS = {
    "RUNNING": "RUNNING",
    "FINISHED": "COMPLETED",
    "FAILED": "FAILED",
}

POLL_INTERVAL = 3
RUN_ID_WAIT_INTERVAL = 5
# 학습 파드는 base 모델을 내려받아 로드한 뒤에야 MLflow run 을 만든다. 대형 모델(예: ESMC-6B 24GB)은
# 이 선작업(다운로드+bf16 로드)이 길어 300초로는 "run 미생성"으로 오판된다 → 30분으로 둔다.
RUN_ID_WAIT_MAX = 1800


def build_loss_history(loss_history: list, epoch_history: list) -> list[dict]:
    epoch_map = {m.step: int(m.value) for m in epoch_history}
    epoch_losses: dict[int, list[float]] = {}

    for m in loss_history:
        epoch = epoch_map.get(m.step)
        if epoch is not None:
            epoch_losses.setdefault(epoch, []).append(m.value)

    return [{"epoch": ep, "loss": round(sum(vals) / len(vals), 6)} for ep, vals in sorted(epoch_losses.items())]


def _extract_max_epoch(experiment) -> int:
    for hp in experiment.hyperparameters:
        if hp.param_name == "epochs":
            return int(hp.value)
    return 0


def _poll_deadline(experiment_id: int) -> float:
    """실험 생성 시각 + 상한을 epoch 초로. 재시작해도 늘어나지 않도록 폴링 시작 시각을 쓰지 않는다.

    생성 시각을 못 읽으면 지금부터 상한을 잰다. 상한 없음(None)으로 두면 폴링이 영원히 끝나지 않는다.
    """
    limit = int(settings.TRAINING_POLL_TIMEOUT_SEC or 86400)
    db = SessionLocal()
    try:
        experiment = ExperimentService.get(db, experiment_id)
        created = getattr(experiment, "created_at", None) if experiment else None
    except Exception as e:
        logger.warning(f"Experiment {experiment_id} 생성 시각 조회 실패 — 지금부터 상한 적용: {e}")
        return time.time() + limit
    finally:
        db.close()
    if created is None:
        logger.warning(f"Experiment {experiment_id} 생성 시각 없음 — 지금부터 상한 적용")
        return time.time() + limit
    base = created if created.tzinfo is not None else created.replace(tzinfo=timezone.utc)
    return base.timestamp() + limit


def _deadline_passed(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def _wait_for_run_id(experiment_id: int, deadline: float | None = None) -> tuple[str | None, int]:
    """DB에서 experiment의 mlflow_run_id가 세팅될 때까지 대기."""
    waited = 0
    while waited < RUN_ID_WAIT_MAX:
        if _deadline_passed(deadline):
            logger.warning(f"Experiment {experiment_id} 추적 상한 초과 (run_id 대기 중)")
            return None, 0
        db = SessionLocal()
        try:
            experiment = ExperimentService.get(db, experiment_id)
            if experiment and experiment.mlflow_run_id:
                max_epoch = _extract_max_epoch(experiment)
                return experiment.mlflow_run_id, max_epoch
        finally:
            db.close()
        time.sleep(RUN_ID_WAIT_INTERVAL)
        waited += RUN_ID_WAIT_INTERVAL

    logger.warning(f"mlflow_run_id not set within {RUN_ID_WAIT_MAX}s for experiment {experiment_id}")
    return None, 0


def _collect_metrics(client, run) -> dict:
    run_id = run.info.run_id

    start_time = run.info.start_time
    end_time_ms = run.info.end_time
    now_ms = int(time.time() * 1000)
    elapsed_time = ((end_time_ms or now_ms) - start_time) // 1000 if start_time else None
    end_time = datetime.fromtimestamp(end_time_ms / 1000) if end_time_ms else None

    epoch_history = client.get_metric_history(run_id, "train/epoch")

    loss_history_raw = client.get_metric_history(run_id, "train/total_loss")
    loss_history = build_loss_history(loss_history_raw, epoch_history)
    latest_loss = loss_history[-1]["loss"] if loss_history else None
    current_epoch = loss_history[-1]["epoch"] if loss_history else 0

    ap_history = client.get_metric_history(run_id, "val/best_ap")
    average_precision = ap_history[-1].value if ap_history else None

    # ESM2 평가 메트릭 슬롯. YOLOX 학습에는 해당 키가 없어 None → upsert 시 무시된다.
    # (ESM2 의 average_precision 은 매핑하지 않아 NULL 로 남는다 — AUC-ROC/PR 은 MLflow run 에만 기록)
    def _last(metric_key: str):
        history = client.get_metric_history(run_id, metric_key)
        return history[-1].value if history else None

    return {
        "elapsed_time": elapsed_time,
        "end_time": end_time,
        "current_epoch": current_epoch,
        "loss": latest_loss,
        "loss_history": json.dumps(loss_history),
        "average_precision": average_precision,
        "accuracy": _last("eval_accuracy"),
        "precision_value": _last("eval_precision"),
        "recall": _last("eval_recall"),
    }


def _resolve_train_msg(run, current_epoch: int) -> str:
    status = run.info.status
    if status == "FINISHED":
        return "학습이 완료되었습니다."
    elif status == "FAILED":
        return "학습이 실패하였습니다."
    elif current_epoch == 0:
        return "메트릭 기록 대기 중입니다."
    else:
        return f"학습 진행 중 (epoch {current_epoch})"


def _upsert_metrics(experiment_id: int, metrics: dict):
    db = SessionLocal()
    try:
        row = db.execute(
            select(ExperimentMetricsModel).where(ExperimentMetricsModel.experiment_id == experiment_id)
        ).scalar_one_or_none()

        if row is None:
            row = ExperimentMetricsModel(experiment_id=experiment_id, **metrics)
            db.add(row)
        else:
            for key, value in metrics.items():
                if value is not None:
                    setattr(row, key, value)

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to upsert metrics for experiment {experiment_id}: {e}")
    finally:
        db.close()


def _update_experiment(experiment_id: int, mlflow_status: str | None = None, train_msg: str | None = None):
    """experiment.status와 train_msg를 업데이트한다."""
    db = SessionLocal()
    try:
        experiment = db.execute(select(ExperimentModel).where(ExperimentModel.id == experiment_id)).scalar_one_or_none()
        if experiment is None:
            return

        changed = False
        if mlflow_status is not None:
            new_status = MLFLOW_TO_EXPERIMENT_STATUS.get(mlflow_status)
            if new_status and experiment.status != new_status:
                experiment.status = new_status
                changed = True

        if train_msg is not None and experiment.train_msg != train_msg:
            experiment.train_msg = train_msg
            changed = True

        if changed:
            db.commit()
            logger.info(f"Experiment {experiment_id} updated: status={experiment.status}, train_msg={train_msg}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update experiment for {experiment_id}: {e}")
    finally:
        db.close()


def _fail_experiment(experiment_id: int, train_msg: str):
    """experiment를 FAILED 상태로 설정하고 train_msg를 기록한다."""
    db = SessionLocal()
    try:
        experiment = db.execute(select(ExperimentModel).where(ExperimentModel.id == experiment_id)).scalar_one_or_none()
        if experiment and experiment.status not in ("COMPLETED", "FAILED"):
            experiment.status = "FAILED"
            experiment.train_msg = train_msg
            db.commit()
            logger.info(f"Experiment {experiment_id} marked as FAILED: {train_msg}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to mark experiment {experiment_id} as FAILED: {e}")
    finally:
        db.close()


FINAL_COLLECT_DELAY = 5


def _timeout_msg() -> str:
    hours = int(settings.TRAINING_POLL_TIMEOUT_SEC or 86400) // 3600
    return f"추적 시간({hours}시간)이 초과되었습니다. 학습은 계속 진행 중일 수 있습니다."


def poll_training_metrics(experiment_id: int):
    """학습 실험의 MLflow 메트릭을 주기적으로 조회하여 DB에 업데이트한다.

    어떤 경로로 끝나든 실험을 COMPLETED 또는 FAILED 로 확정한다. 활성 상태로 남겨두면 학습 파드가
    사라진 뒤에도 MLflow run 이 RUNNING 으로 남아 폴링이 끝나지 않고, 재기동할 때마다 되살아난다.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    deadline = _poll_deadline(experiment_id)

    mlflow_run_id, max_epoch = _wait_for_run_id(experiment_id, deadline)
    if mlflow_run_id is None:
        if _deadline_passed(deadline):
            _fail_experiment(experiment_id, _timeout_msg())
        else:
            _fail_experiment(
                experiment_id,
                f"학습 시작 전 오류가 발생했습니다. MLflow run이 {RUN_ID_WAIT_MAX}초 내에 생성되지 않았습니다.",
            )
        return

    while True:
        try:
            run = client.get_run(mlflow_run_id)
            metrics = _collect_metrics(client, run)
            metrics["max_epoch"] = max_epoch

            _upsert_metrics(experiment_id, metrics)

            status = run.info.status
            train_msg = _resolve_train_msg(run, metrics["current_epoch"])
            _update_experiment(experiment_id, mlflow_status=status, train_msg=train_msg)

            if status in ("FINISHED", "FAILED"):
                time.sleep(FINAL_COLLECT_DELAY)
                run = client.get_run(mlflow_run_id)
                final_metrics = _collect_metrics(client, run)
                final_metrics["max_epoch"] = max_epoch
                if status == "FINISHED" and final_metrics["current_epoch"] < max_epoch:
                    final_metrics["current_epoch"] = max_epoch
                _upsert_metrics(experiment_id, final_metrics)
                _update_experiment(experiment_id, train_msg=_resolve_train_msg(run, final_metrics["current_epoch"]))
                break

            # 상한을 넘겼는데 아직 진행 중이다. 여기서 끝내지 않으면 실험이 활성 상태로 방치된다.
            if _deadline_passed(deadline):
                _fail_experiment(experiment_id, _timeout_msg())
                logger.warning(f"Experiment {experiment_id} 추적 상한 초과 — FAILED 로 종결")
                break

        except MlflowException as e:
            # run 이나 experiment 가 MLflow 에서 지워진 경우. 다시 물어봐도 결과가 달라지지 않으므로
            # 상한까지 기다리지 않고 끝낸다.
            if getattr(e, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
                _fail_experiment(experiment_id, "MLflow 에서 이 학습 기록을 찾을 수 없습니다.")
                logger.warning(f"Experiment {experiment_id} MLflow run {mlflow_run_id} 없음 — FAILED 로 종결")
                break
            logger.error(f"Metrics polling error for experiment {experiment_id}: {e}")
            if _deadline_passed(deadline):
                _fail_experiment(experiment_id, "학습 상태를 확인하지 못한 채 추적 시간이 초과되었습니다.")
                break

        except Exception as e:
            # MLflow 조회, 메트릭 파싱, DB 쓰기 중 어디서든 날 수 있다. 대개 일시적이라 다시 시도하되,
            # 상한을 넘기면 활성 상태로 남기지 않고 종결한다.
            logger.error(f"Metrics polling error for experiment {experiment_id}: {e}")
            if _deadline_passed(deadline):
                _fail_experiment(experiment_id, "학습 상태를 확인하지 못한 채 추적 시간이 초과되었습니다.")
                break

        time.sleep(POLL_INTERVAL)
