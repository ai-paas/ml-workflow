"""
§7.9 / §7.9.1 사전정의 서빙 메타 조회·파생 및 워크플로우 배포용 리소스 계획.

GPU VRAM·호스트 메모리·CPU 밀리코어는 PREDEFINED_MODEL_CONFIGS에만 두고,
파생 모델은 부모 repo_id 맵 + 고정 계수로 산출한다.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from db.models.model import Model, ModelTaskType
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SERVING_META_KEYS = (
    "serving_vram_need_bytes",
    "serving_memory_request_gpu",
    "serving_gpu_pod_cpu_request_millicores",
    "serving_memory_request_cpu",
    "serving_cpu_request_millicores",
)

ServingMetaSource = Literal["direct", "parent_derived"]


@dataclass(frozen=True)
class ServingResourcePlan:
    """백엔드에서 확정해 KFP 컴포넌트에만 넘기는 동결 값."""

    device_type: Literal["GPU", "CPU"]
    gpu_count: int
    memory_request: str
    cpu_request: str
    memory_limit: str
    cpu_limit: str
    serving_vram_need_bytes: int
    serving_meta_source: ServingMetaSource
    parent_repo_id: Optional[str]
    fallback_reason: Optional[str]
    serving_node_name: Optional[str] = None
    planner_note: Optional[str] = None
    #: §7.3.2 가상 점유: 플래너가 잔여를 차감할 때 사용한 노드(핀 해제여도 내부 선정 노드).
    reservation_node_name: Optional[str] = None


def _has_complete_serving_keys(cfg: dict[str, Any]) -> bool:
    for k in SERVING_META_KEYS:
        if k not in cfg or cfg[k] is None or cfg[k] == "":
            return False
    return True


def _parse_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field}: 잘못된 값입니다.")
    if isinstance(value, int):
        n = value
    else:
        s = str(value).strip()
        if not s:
            raise ValueError(f"{field}: 비어 있습니다.")
        n = int(s, 10)
    if n < 0:
        raise ValueError(f"{field}: 음수일 수 없습니다.")
    return n


_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTPE]i|[KMGTPE])\s*$", re.IGNORECASE)


def k8s_memory_quantity_to_bytes(quantity: str) -> int:
    """Kubernetes memory quantity (Gi, Mi, …)를 바이트로 환산."""
    m = _MEMORY_RE.match(quantity or "")
    if not m:
        raise ValueError(f"지원하지 않는 memory quantity: {quantity!r}")
    num = float(m.group(1))
    suf = m.group(2).lower()
    binary = suf.endswith("i")
    base = 1024 if binary else 1000
    unit = suf.rstrip("i").lower()
    exp = {"k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}.get(unit)
    if exp is None:
        raise ValueError(f"지원하지 않는 memory 단위: {quantity!r}")
    return int(num * (base**exp))


def format_k8s_memory_from_bytes(num_bytes: int) -> str:
    """§7.9.1: Mi/Gi 형태로 반올림 (최소 1Mi)."""
    mi = 1024 * 1024
    n = max(1, round(num_bytes / mi))
    if n % 1024 == 0:
        return f"{n // 1024}Gi"
    return f"{n}Mi"


def _derive_meta_from_parent(
    parent_slice: dict[str, Any],
    *,
    learning_enable_yn: bool,
    opt_enable_yn: bool,
) -> dict[str, Any]:
    """§7.9.1 파생 규칙."""
    f_factor = 1.0
    if learning_enable_yn:
        f_factor *= 0.98
    if opt_enable_yn:
        f_factor *= 0.88

    vram = max(
        1, round(_parse_positive_int(parent_slice["serving_vram_need_bytes"], "serving_vram_need_bytes") * f_factor)
    )
    gpu_mc = max(
        100,
        round(
            _parse_positive_int(
                parent_slice["serving_gpu_pod_cpu_request_millicores"], "serving_gpu_pod_cpu_request_millicores"
            )
            * f_factor
        ),
    )
    cpu_mc = max(
        100,
        round(
            _parse_positive_int(parent_slice["serving_cpu_request_millicores"], "serving_cpu_request_millicores")
            * f_factor
        ),
    )

    mem_gpu_b = int(k8s_memory_quantity_to_bytes(str(parent_slice["serving_memory_request_gpu"])) * f_factor)
    mem_cpu_b = int(k8s_memory_quantity_to_bytes(str(parent_slice["serving_memory_request_cpu"])) * f_factor)

    return {
        "serving_vram_need_bytes": vram,
        "serving_memory_request_gpu": format_k8s_memory_from_bytes(mem_gpu_b),
        "serving_gpu_pod_cpu_request_millicores": gpu_mc,
        "serving_memory_request_cpu": format_k8s_memory_from_bytes(mem_cpu_b),
        "serving_cpu_request_millicores": cpu_mc,
    }


def resolve_normalized_serving_meta(
    db: Session,
    model: Model,
    *,
    predefined_configs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], ServingMetaSource, Optional[str]]:
    """
    직접/부모 파생으로 정규화된 서빙 메타(5키)를 반환.

    Returns:
        (normalized_dict, source, parent_repo_id or None)
    """
    if not model.repo_id:
        raise ValueError(f"model_id={model.id}: repo_id가 없어 서빙 메타를 조회할 수 없습니다.")

    child_cfg = predefined_configs.get(model.repo_id)
    if child_cfg and _has_complete_serving_keys(child_cfg):
        meta = {
            "serving_vram_need_bytes": _parse_positive_int(
                child_cfg["serving_vram_need_bytes"], "serving_vram_need_bytes"
            ),
            "serving_memory_request_gpu": str(child_cfg["serving_memory_request_gpu"]).strip(),
            "serving_gpu_pod_cpu_request_millicores": _parse_positive_int(
                child_cfg["serving_gpu_pod_cpu_request_millicores"], "serving_gpu_pod_cpu_request_millicores"
            ),
            "serving_memory_request_cpu": str(child_cfg["serving_memory_request_cpu"]).strip(),
            "serving_cpu_request_millicores": _parse_positive_int(
                child_cfg["serving_cpu_request_millicores"], "serving_cpu_request_millicores"
            ),
        }
        for mk in ("serving_memory_request_gpu", "serving_memory_request_cpu"):
            k8s_memory_quantity_to_bytes(meta[mk])
        return meta, "direct", None

    if model.parent_model_id is None:
        raise ValueError(
            f"model_id={model.id}, repo_id={model.repo_id!r}: PREDEFINED_MODEL_CONFIGS에 "
            f"필수 5키가 없고 parent_model_id도 없습니다. (§7.9)"
        )

    parent = db.get(Model, model.parent_model_id)
    if not parent or not parent.repo_id:
        raise ValueError(
            f"model_id={model.id}: parent_model_id={model.parent_model_id} 인 부모 모델을 찾을 수 없거나 repo_id가 없습니다."
        )

    parent_cfg = predefined_configs.get(parent.repo_id)
    if not parent_cfg or not _has_complete_serving_keys(parent_cfg):
        raise ValueError(
            f"model_id={model.id}: 부모 repo_id={parent.repo_id!r}에 대한 사전정의 서빙 메타(5키)가 없습니다. (§7.9.1)"
        )

    parent_slice = {k: parent_cfg[k] for k in SERVING_META_KEYS}
    derived = _derive_meta_from_parent(
        parent_slice,
        learning_enable_yn=bool(model.learning_enable_yn),
        opt_enable_yn=bool(model.opt_enable_yn),
    )
    return derived, "parent_derived", parent.repo_id


def decide_try_gpu_path(
    *,
    task: Optional[str],
    framework: str,
    execute_parameters: dict[str, Any],
    kserve_gpu_enabled: bool,
    default_try_gpu: bool,
) -> Tuple[bool, Optional[str]]:
    """
    GPU 경로를 시도할지(인벤토리·k 산정 전 단계). False면 CPU 경로.
    """
    force_cpu = bool(execute_parameters.get("force_cpu"))
    device_override = (execute_parameters.get("device_type") or "").strip().upper()

    task_l = (task or "").lower()
    embedding = task_l == ModelTaskType.EMBEDDING.value

    try_gpu = default_try_gpu and not force_cpu and device_override != "CPU" and not embedding

    if framework != "ollama" and not kserve_gpu_enabled:
        try_gpu = False

    fallback_reason: Optional[str] = None
    if not try_gpu:
        if embedding:
            fallback_reason = "task_embedding_cpu"
        elif force_cpu or device_override == "CPU":
            fallback_reason = "user_cpu"
        elif framework != "ollama" and not kserve_gpu_enabled:
            fallback_reason = "kserve_gpu_disabled"
        elif not default_try_gpu:
            fallback_reason = "default_try_gpu_false"

    return try_gpu, fallback_reason


def millicores_to_k8s_cpu(millicores: int) -> str:
    """밀리코어를 Kubernetes cpu quantity로 변환."""
    if millicores < 1:
        millicores = 1
    if millicores % 1000 == 0:
        return str(millicores // 1000)
    return f"{millicores}m"


def build_serving_resource_plan(
    *,
    normalized_meta: dict[str, Any],
    task: Optional[str],
    framework: str,
    execute_parameters: dict[str, Any],
    kserve_gpu_enabled: bool,
    default_try_gpu: bool,
    default_gpu_vram_bytes: int,
    serving_meta_source: ServingMetaSource,
    parent_repo_id: Optional[str],
    serving_node_name: Optional[str] = None,
    planner_note: Optional[str] = None,
    log_context: Optional[str] = None,
) -> ServingResourcePlan:
    """
    인벤토리 없이(또는 K8s 조회 실패 시) §7 단순 경로: decide_try_gpu_path + k는 default_gpu_vram_bytes 기준 ceil.
    """
    v_need = int(normalized_meta["serving_vram_need_bytes"])
    mem_gpu = str(normalized_meta["serving_memory_request_gpu"])
    mem_cpu = str(normalized_meta["serving_memory_request_cpu"])
    gpu_pod_mc = int(normalized_meta["serving_gpu_pod_cpu_request_millicores"])
    cpu_mc = int(normalized_meta["serving_cpu_request_millicores"])

    try_gpu, fallback_reason = decide_try_gpu_path(
        task=task,
        framework=framework,
        execute_parameters=execute_parameters,
        kserve_gpu_enabled=kserve_gpu_enabled,
        default_try_gpu=default_try_gpu,
    )

    if try_gpu and default_gpu_vram_bytes > 0:
        k = max(1, math.ceil(v_need / float(default_gpu_vram_bytes)))
    elif try_gpu:
        k = 1
    else:
        k = 0

    if try_gpu and k > 0:
        mem_req, cpu_req = mem_gpu, millicores_to_k8s_cpu(gpu_pod_mc)
        device: Literal["GPU", "CPU"] = "GPU"
    else:
        mem_req, cpu_req = mem_cpu, millicores_to_k8s_cpu(cpu_mc)
        device = "CPU"
        k = 0

    ctx = f" | {log_context}" if log_context else ""
    dev_ko = "GPU" if device == "GPU" else "CPU"
    try_ko = "예" if try_gpu else "아니오"
    logger.info(
        "[serving_planner]%s 노드 정보 없이 단순 계산으로 플랜함 — 최종 디바이스 %s, GPU %d장, "
        "모델 VRAM 안내 %s, Pod 메모리·CPU %s / %s, GPU 경로를 시도했는지 %s, "
        "참고사유·메모 %r / %r, 장당 VRAM 기본값 %s",
        ctx,
        dev_ko,
        k,
        format_k8s_memory_from_bytes(v_need),
        mem_req,
        cpu_req,
        try_ko,
        fallback_reason,
        planner_note,
        format_k8s_memory_from_bytes(int(default_gpu_vram_bytes)),
    )

    return ServingResourcePlan(
        device_type=device,
        gpu_count=k,
        memory_request=mem_req,
        cpu_request=cpu_req,
        memory_limit=mem_req,
        cpu_limit=cpu_req,
        serving_vram_need_bytes=v_need,
        serving_meta_source=serving_meta_source,
        parent_repo_id=parent_repo_id,
        fallback_reason=fallback_reason,
        serving_node_name=serving_node_name,
        planner_note=planner_note,
    )


def serving_meta_validation_error(
    db: Session,
    model: Model,
    *,
    predefined_configs: dict[str, dict[str, Any]],
) -> Optional[str]:
    try:
        resolve_normalized_serving_meta(db, model, predefined_configs=predefined_configs)
        return None
    except ValueError as e:
        return str(e)
