"""DB 서버 시각을 기준으로 삼는 시간 유틸.

시각 컬럼(`TIMESTAMP`)은 타임존을 담지 않고 `func.now()` 로 채워지므로, 그 값이 어느
타임존인지는 DB 세션 설정에 달려 있다. 애플리케이션이 따로 타임존을 설정해 두면 DB 와
어긋나는 순간 조용히 틀어지므로, 기준을 DB 하나로 둔다.

- 만료·경과 판정: `db_now()` 와 naive 끼리 비교한다. 타임존을 알 필요가 없다.
- 외부로 내보내는 절대시각: `db_utc_offset()` 으로 UTC 로 환산한다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_utc_offset: Optional[timedelta] = None


def db_now(db: Session) -> datetime:
    """DB 서버의 현재 시각(naive). 시각 컬럼과 같은 기준이라 그대로 비교할 수 있다."""
    return db.execute(select(func.now())).scalar_one()


def refresh_db_utc_offset(db: Session) -> timedelta:
    """DB 세션 타임존의 UTC 오프셋을 재어 캐시한다. 기동 시 한 번 호출."""
    global _utc_offset
    delta = db_now(db) - datetime.utcnow()
    # 왕복 지연으로 초 단위 오차가 섞이므로 분 단위로 정규화한다.
    _utc_offset = timedelta(minutes=round(delta.total_seconds() / 60))
    logger.info("DB 시각 기준 확인: UTC 대비 %+.1f시간", _utc_offset.total_seconds() / 3600)
    return _utc_offset


def db_utc_offset() -> timedelta:
    """캐시된 오프셋. 아직 재지 않았으면 0(=DB 가 UTC)으로 본다."""
    return _utc_offset if _utc_offset is not None else timedelta(0)


def to_aware_utc(dt: datetime) -> datetime:
    """DB 에서 읽은 naive 시각을 UTC aware 로 환산. 이미 aware 면 UTC 로 변환만 한다."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    return (dt - db_utc_offset()).replace(tzinfo=timezone.utc)


def isoformat_utc(dt: Optional[datetime]) -> str:
    """naive DB 시각을 ISO8601 Z 표기로. None 이면 현재 시각."""
    if dt is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return to_aware_utc(dt).isoformat().replace("+00:00", "Z")
