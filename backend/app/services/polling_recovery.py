"""프로세스와 함께 사라진 백그라운드 폴링을 앱 기동 시 다시 건다.

최적화 진행 상태와 학습 메트릭은 요청 처리 중에 띄운 백그라운드 작업이 갱신한다. 그 작업은
프로세스 안에서만 살아 있으므로 재배포·재시작하면 사라지고, 원격에서는 끝났는데 DB 에는
진행 중으로 남는 작업이 생긴다. 여기서 활성 상태로 남은 것들을 찾아 폴링을 이어받는다.
"""

from __future__ import annotations

import logging
import threading

from config.db.session import SessionLocal
from db.models.experiment import ExperimentModel

logger = logging.getLogger(__name__)

# 학습이 아직 끝나지 않은 상태. CREATED 는 파이프라인은 떴지만 MLflow run 이 아직 없는 구간이다.
_ACTIVE_EXPERIMENT_STATUSES = ("CREATED", "RUNNING")


def resume_open_training_polls() -> int:
    """끝나지 않은 학습 실험의 메트릭 폴링을 다시 건다. 시작한 개수를 돌려준다.

    poll_training_metrics 는 학습이 끝날 때까지 도는 동기 루프라, FastAPI 의 공용 스레드풀에
    올리면 실험 수만큼 슬롯을 계속 점유해 다른 동기 엔드포인트를 굶긴다. 전용 데몬 스레드로 띄운다.
    """
    from services.metrics_polling import poll_training_metrics

    db = SessionLocal()
    try:
        rows = db.query(ExperimentModel).filter(ExperimentModel.status.in_(_ACTIVE_EXPERIMENT_STATUSES)).all()
        experiment_ids = [r.id for r in rows]
    except Exception as e:
        logger.error("학습 폴링 재개 대상 조회 실패: %s", e)
        return 0
    finally:
        db.close()

    for eid in experiment_ids:
        threading.Thread(
            target=poll_training_metrics,
            args=(eid,),
            name=f"resume-train-poll-{eid}",
            daemon=True,
        ).start()
    if experiment_ids:
        logger.info("학습 폴링 재개: %d건 %s", len(experiment_ids), experiment_ids)
    return len(experiment_ids)


def resume_background_polls() -> None:
    """기동 시 한 번 호출. 어느 한쪽이 실패해도 다른 쪽은 계속 재개한다."""
    from services.model_improvement import resume_open_improvement_polls

    try:
        resume_open_improvement_polls()
    except Exception as e:
        logger.error("최적화 폴링 재개 실패: %s", e)

    try:
        resume_open_training_polls()
    except Exception as e:
        logger.error("학습 폴링 재개 실패: %s", e)
