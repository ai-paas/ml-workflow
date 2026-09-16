"""모델별 서빙 경로 선호도 태그."""

from __future__ import annotations

from enum import Enum as PyEnum


class ServingMode(str, PyEnum):
    """`PREDEFINED_MODEL_CONFIGS` 항목의 `serving_mode` 키 값.

    키를 생략한 항목은 `LOCAL` 로 간주한다.
    """

    LOCAL = "local"  # 실물 가중치를 직접 로드해 서빙(Ollama/KServe). 기본값.
    REMOTE_PREFERRED = "remote_preferred"  # 원격 우선, 원격 미설정 시 실물 가중치로 폴백.
    REMOTE_ONLY = "remote_only"  # 원격 전용. 원격 미설정 시 배포 거부.

    def __str__(self) -> str:
        return self.value
