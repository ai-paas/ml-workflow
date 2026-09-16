import logging
from contextlib import asynccontextmanager

from config.db.session import SessionLocal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api_router
from services.polling_recovery import resume_background_polls
from utils.db_clock import refresh_db_utc_offset

logger = logging.getLogger(__name__)


def _measure_db_clock() -> None:
    """DB 시각 기준을 재어 캐시. 실패해도 기동은 막지 않는다(UTC 로 가정하고 진행)."""
    db = SessionLocal()
    try:
        refresh_db_utc_offset(db)
    except Exception as e:
        logger.warning("DB 시각 기준 확인 실패 — UTC 로 가정합니다: %s", e)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시각 컬럼은 DB 세션 타임존으로 저장된다. 응답에 UTC 로 내보내려면 그 오프셋을 알아야 하므로
    # 기동할 때 한 번 재어 둔다(설정으로 두면 DB 와 어긋나는 순간 조용히 틀어진다).
    _measure_db_clock()

    # 최적화·학습 진행 상태는 백그라운드 폴링이 갱신하는데, 그 폴링은 프로세스와 함께 사라진다.
    # 재배포를 건너뛴 작업이 진행 중인 채로 남지 않도록 기동할 때 이어받는다.
    resume_background_polls()
    yield


app = FastAPI(lifespan=lifespan)

# CORS 설정
origins = [
    "*",  # 모든 출처 허용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
