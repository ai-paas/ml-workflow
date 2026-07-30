"""
WorkflowServingVolumeLock: (model_id, serving_node) 키 단위 직렬화 계약.

기본 구현은 프로세스 내 threading.Lock(다중 워커·다중 Pod 간 직렬화는 별도 구현체 주입).
"""

from __future__ import annotations

import threading
from typing import Optional, Protocol, runtime_checkable


def serving_volume_lock_key(model_id: int, serving_node_name: str) -> str:
    """도메인 키 (M, N) → 단일 문자열."""
    n = (serving_node_name or "").strip()
    return f"serving_pvc:{int(model_id)}:{n}"


@runtime_checkable
class WorkflowServingVolumeLock(Protocol):
    """최소 계약."""

    def try_acquire_nonblocking(self, key: str) -> bool:
        """즉시 획득 시 True, 불가 시 False."""

    def acquire_blocking(self, key: str, timeout_sec: float) -> bool:
        """timeout_sec 동안 대기 후 획득 시 True."""

    def release(self, key: str) -> None:
        """try_acquire / acquire_blocking 성공 후 반드시 호출."""


class ThreadingWorkflowServingVolumeLock:
    """단일 API 프로세스·단일 워커 환경용 (표)."""

    def __init__(self) -> None:
        self._meta = threading.Lock()
        self._per_key: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._meta:
            if key not in self._per_key:
                self._per_key[key] = threading.Lock()
            return self._per_key[key]

    def try_acquire_nonblocking(self, key: str) -> bool:
        return self._lock_for(key).acquire(blocking=False)

    def acquire_blocking(self, key: str, timeout_sec: float) -> bool:
        return self._lock_for(key).acquire(timeout=timeout_sec)

    def release(self, key: str) -> None:
        with self._meta:
            lk = self._per_key.get(key)
        if lk is None:
            return
        try:
            lk.release()
        except RuntimeError:
            pass


_lock_instance: Optional[ThreadingWorkflowServingVolumeLock] = None
_lock_singleton_meta = threading.Lock()


def get_workflow_serving_volume_lock() -> WorkflowServingVolumeLock:
    """앱 전역 기본 락(프로세스 내). 분산 필요 시 동일 Protocol 구현으로 교체."""
    global _lock_instance
    with _lock_singleton_meta:
        if _lock_instance is None:
            _lock_instance = ThreadingWorkflowServingVolumeLock()
        return _lock_instance
