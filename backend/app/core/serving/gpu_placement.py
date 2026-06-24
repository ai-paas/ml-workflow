"""GPU 노드풀 프로파일 기반 배치(placement) 헬퍼.

`GPU_POOL_PROFILES_JSON` 의 프로파일로 노드/Job 에 주입할
GPU 자원키 · nodeSelector · tolerations 를 결정한다. (MIG taint 노드 등)

설계: docs/k8s-gpu-allocation/README.md
- 노드 라벨이 프로파일 match_labels 에 매칭되면 그 프로파일을 채택(우선).
- 매칭 프로파일이 없으면 기본값(dev) = nvidia.com/gpu, selector/toleration 없음.
- 운영(single/MIG)도 자원키는 nvidia.com/gpu 그대로 — 핵심은 toleration + nodeSelector 주입.

node_selector/tolerations 는 KFP 파라미터 직렬화·frozen dataclass 친화를 위해 **JSON 문자열**로 다룬다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_GPU_RESOURCE_KEY = "nvidia.com/gpu"


def parse_gpu_pool_profiles(raw: str) -> list[dict]:
    """GPU_POOL_PROFILES_JSON 파싱. 실패/비어있으면 []."""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("GPU_POOL_PROFILES_JSON 파싱 실패, 무시: %s", raw[:200])
        return []
    if not isinstance(data, list):
        logger.warning("GPU_POOL_PROFILES_JSON 은 JSON 배열이어야 합니다. 무시: %s", raw[:200])
        return []
    return [p for p in data if isinstance(p, dict) and isinstance(p.get("match_labels"), dict)]


def match_pool(labels: dict[str, str], profiles: list[dict]) -> Optional[dict]:
    """노드 라벨이 match_labels 를 모두 포함하는 첫 프로파일을 반환(없으면 None)."""
    for p in profiles:
        ml = p.get("match_labels") or {}
        if all(str(labels.get(k)) == str(v) for k, v in ml.items()):
            return p
    return None


def _tolerations_from_taints(taints: Optional[list[dict]]) -> list[dict]:
    """노드 taint 를 그대로 견디는 toleration 목록 생성(서빙: 노드 컨텍스트 있을 때)."""
    out: list[dict] = []
    for t in taints or []:
        key = t.get("key")
        if not key:
            continue
        tol: dict[str, Any] = {"key": key, "effect": t.get("effect") or "NoSchedule"}
        if t.get("value") not in (None, ""):
            tol["operator"] = "Equal"
            tol["value"] = t["value"]
        else:
            tol["operator"] = "Exists"
        out.append(tol)
    return out


def _placement_from_pool(pool: dict, *, default_gpu_key: str, taints: Optional[list[dict]]) -> dict:
    """프로파일(pool)에서 배치 정보 산출.

    필수: match_labels(어느 노드인지) + tolerations(taint 통과). 나머지는 자동/기본:
    - node_selector 미지정 → match_labels 를 그대로 사용(매칭 노드는 그 라벨을 가지므로 안전).
    - resource_key 미지정 → nvidia.com/gpu(single 기본).
    - tolerations 미지정 → (서빙만) 노드 taint 에서 생성. 학습은 노드를 미리 모르므로 생성 불가 → 명시 필요.
    """
    match_labels = pool.get("match_labels") or {}

    node_selector = pool.get("node_selector")
    if node_selector is None:
        node_selector = dict(match_labels)

    tolerations = pool.get("tolerations")
    if not tolerations:
        tolerations = _tolerations_from_taints(taints)  # 서빙: 선택 노드 taint / 학습(taints=None): []

    return {
        "gpu_resource_key": (pool.get("resource_key") or default_gpu_key),
        "node_selector": (node_selector or {}),
        "tolerations": (tolerations or []),
        "slice_vram_gib": pool.get("slice_vram_gib"),
    }


def _default_placement(default_gpu_key: str) -> dict:
    return {"gpu_resource_key": default_gpu_key, "node_selector": {}, "tolerations": [], "slice_vram_gib": None}


def resolve_placement(
    labels: dict[str, str],
    *,
    profiles: list[dict],
    taints: Optional[list[dict]] = None,
    default_gpu_key: str = DEFAULT_GPU_RESOURCE_KEY,
) -> dict:
    """노드 라벨 + 프로파일로 배치 정보를 결정(서빙).

    매칭 프로파일이 있으면 _placement_from_pool(자동 도출), 없으면 dev 기본.
    반환: {gpu_resource_key, node_selector(dict), tolerations(list[dict]), slice_vram_gib(int|None)}
    """
    pool = match_pool(labels, profiles)
    if pool is None:
        return _default_placement(default_gpu_key)
    return _placement_from_pool(pool, default_gpu_key=default_gpu_key, taints=taints)


def static_training_placement(profiles: list[dict], default_gpu_key: str = DEFAULT_GPU_RESOURCE_KEY) -> dict:
    """학습 Job 용 정적 배치 — 플래너/노드 인벤토리를 거치지 않으므로 첫 프로파일을 채택.

    노드 taint 를 미리 못 보므로 tolerations 미지정 시 match_labels 에서 생성한다.
    프로파일이 없으면(dev) 자원키만 기본 → 기존 동작 그대로.
    """
    if not profiles:
        return _default_placement(default_gpu_key)
    return _placement_from_pool(profiles[0], default_gpu_key=default_gpu_key, taints=None)
