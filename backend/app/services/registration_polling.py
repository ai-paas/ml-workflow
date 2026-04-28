import logging
import time

from config.db.session import SessionLocal
from core.kubeflow.kubeflow_manager import KubeflowManager
from services.experiment import ExperimentService
from services.model import ModelService

logger = logging.getLogger(__name__)

REG_POLL_INTERVAL = 1
REG_POLL_MAX_WAIT = 600


def _extract_kfp_run_status(run) -> str | None:
    """
    KFP run 객체에서 상태 문자열을 추출한다.
    kfp V2beta1Run 객체의 다양한 속성 경로를 시도한다.
    """
    if hasattr(run, "state") and run.state:
        return str(run.state)
    if hasattr(run, "status") and run.status:
        return str(run.status)
    if hasattr(run, "run"):
        if hasattr(run.run, "status") and run.run.status:
            return str(run.run.status)
        if hasattr(run.run, "state") and run.run.state:
            return str(run.run.state)
    if hasattr(run, "to_dict"):
        run_dict = run.to_dict()
        return run_dict.get("state") or run_dict.get("status")
    return None


def _finalize_registration(
    experiment_id: int,
    registration_status: str,
    message: str,
):
    db = SessionLocal()
    try:
        experiment = ExperimentService.get(db, experiment_id)
        if experiment is None:
            return

        experiment.registration_status = registration_status
        experiment.model_register_msg = message

        if registration_status == "SUCCESS":
            registered_model = ModelService.get_latest_child_model_for_registration(db, experiment.reference_model_id)
            if registered_model:
                experiment.registered_model_id = registered_model.id

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to finalize registration for experiment {experiment_id}: {e}")
    finally:
        db.close()


def poll_registration_status(
    experiment_id: int,
    registration_run_id: str,
):
    """모델 등록 파이프라인의 KFP 상태를 폴링하여 experiment DB를 업데이트한다."""
    waited = 0

    while waited < REG_POLL_MAX_WAIT:
        try:
            kf = KubeflowManager()
            run = kf.kfp_client.get_run(registration_run_id)
            kfp_status = _extract_kfp_run_status(run)

            if kfp_status is None:
                time.sleep(REG_POLL_INTERVAL)
                waited += REG_POLL_INTERVAL
                continue

            status_upper = kfp_status.upper()

            if status_upper in ("SUCCEEDED", "SUCCESS", "COMPLETED"):
                _finalize_registration(
                    experiment_id,
                    "SUCCESS",
                    "모델 등록이 완료되었습니다.",
                )
                return

            elif status_upper in ("FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"):
                _finalize_registration(
                    experiment_id,
                    "FAILED",
                    "모델 등록 파이프라인이 실패했습니다.",
                )
                return

        except Exception as e:
            logger.error(f"Registration polling error for experiment {experiment_id}: {e}")

        time.sleep(REG_POLL_INTERVAL)
        waited += REG_POLL_INTERVAL

    logger.warning(f"Registration polling timed out for experiment {experiment_id}")
    _finalize_registration(
        experiment_id,
        "FAILED",
        f"등록 파이프라인 상태 확인 시간 초과 ({REG_POLL_MAX_WAIT}초)",
    )
