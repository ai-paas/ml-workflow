"""
§7.4~7.7 서빙 리소스 플래너: K8s 인벤토리 + GPU 슬랙 최소 노드·k, CPU 노드 선정.

인벤토리 조회 실패 시 serving_resource_meta.build_serving_resource_plan 으로 폴백.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

from config.settings import Settings
from core.serving.serving_k8s_inventory import NodeInventory, collect_node_inventory, whitelist_rank
from core.serving.serving_resource_meta import (
    ServingMetaSource,
    ServingResourcePlan,
    build_serving_resource_plan,
    decide_try_gpu_path,
    format_k8s_memory_from_bytes,
    k8s_memory_quantity_to_bytes,
    millicores_to_k8s_cpu,
)


def apply_chain_reservation_from_plan(
    frees: dict[str, tuple[int, int, int]],
    plan: ServingResourcePlan,
) -> None:
    """§7.3.2: 확정 플랜의 k·memory·cpu 요청을 선택 노드 잔여에서 차감(max(0,·))."""
    rn = plan.reservation_node_name
    if not rn:
        return
    try:
        mem_b = k8s_memory_quantity_to_bytes(plan.memory_request)
    except ValueError:
        mem_b = 0
    cpu_mc = _cpu_string_to_millicores(plan.cpu_request)
    k = max(0, int(plan.gpu_count))
    g, m, c = frees.get(rn, (0, 0, 0))
    frees[rn] = (max(0, g - k), max(0, m - mem_b), max(0, c - cpu_mc))


logger = logging.getLogger(__name__)


def _append_planner_note(base: str, extra: Optional[str]) -> str:
    if not extra:
        return base
    return f"{base};{extra}" if base else extra


def _cpu_string_to_millicores(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    if s.endswith("m"):
        return max(0, int(float(s[:-1])))
    try:
        return max(0, int(math.ceil(float(s) * 1000)))
    except ValueError:
        return 0


def _cap_memory_cpu_to_node(
    mem_req: str,
    cpu_req: str,
    node: NodeInventory,
) -> tuple[str, str, str, str]:
    """§7.6: 노드 allocatable 을 넘지 않도록 request/limit 캡."""
    req_mem = k8s_memory_quantity_to_bytes(mem_req)
    req_mc = _cpu_string_to_millicores(cpu_req)
    # §7.6: allocatable 상한 캡 (노드 총량 기준)
    cap_mem = max(1, min(req_mem, node.alloc_memory_bytes or req_mem))
    cap_mc = max(1, min(req_mc, node.alloc_cpu_millicores or req_mc))
    ms = format_k8s_memory_from_bytes(cap_mem)
    cs = millicores_to_k8s_cpu(cap_mc)
    return ms, cs, ms, cs


def _choose_gpu_node(
    nodes: list[NodeInventory],
    v_need: int,
    whitelist: list[str],
) -> Optional[tuple[str, int, int]]:
    """§7.5: (node_name, k, slack_bytes). 없으면 None."""
    best: Optional[tuple[str, int, int]] = None
    best_key: Optional[tuple] = None

    for n in nodes:
        if n.alloc_gpu <= 0 or not n.v_card_bytes or n.v_card_bytes <= 0:
            continue
        v_card = int(n.v_card_bytes)
        k = max(1, math.ceil(v_need / float(v_card)))
        g_free = n.g_free
        if k > g_free:
            continue
        slack = k * v_card - v_need
        slot_remain = g_free - k
        key = (slack, k, -slot_remain, whitelist_rank(n.name, whitelist), n.name)
        if best_key is None or key < best_key:
            best_key = key
            best = (n.name, k, slack)

    return best


def _choose_cpu_node(
    nodes: list[NodeInventory],
    mem_req_bytes: int,
    cpu_req_millicores: int,
    whitelist: list[str],
) -> Optional[str]:
    """§7.7: 노드 이름 하나 또는 None."""
    tier1: list[tuple[tuple, str]] = []
    tier2: list[tuple[tuple, str]] = []

    for n in nodes:
        if n.m_free < mem_req_bytes or n.c_free < cpu_req_millicores:
            continue
        margin_m = n.m_free - mem_req_bytes
        margin_c = n.c_free - cpu_req_millicores
        key = (-margin_m, -margin_c, whitelist_rank(n.name, whitelist), n.name)
        if n.alloc_gpu == 0:
            tier1.append((key, n.name))
        else:
            tier2.append((key, n.name))

    pool = tier1 if tier1 else tier2
    if not pool:
        return None
    pool.sort(key=lambda x: x[0])
    return pool[0][1]


def _plan_serving_resources_with_inventory_nodes(
    nodes: Sequence[NodeInventory],
    *,
    normalized_meta: dict[str, Any],
    task: Optional[str],
    framework: str,
    execute_parameters: dict[str, Any],
    settings: Settings,
    serving_meta_source: ServingMetaSource,
    parent_repo_id: Optional[str],
) -> ServingResourcePlan:
    """인벤토리 리스트가 비어 있지 않을 때 §7.4~7.7 플랜(§7.3.2는 호출부에서 잔여 조정)."""
    v_need = int(normalized_meta["serving_vram_need_bytes"])
    mem_gpu = str(normalized_meta["serving_memory_request_gpu"])
    mem_cpu = str(normalized_meta["serving_memory_request_cpu"])
    gpu_pod_mc = int(normalized_meta["serving_gpu_pod_cpu_request_millicores"])
    cpu_mc = int(normalized_meta["serving_cpu_request_millicores"])

    try_gpu, fb_try = decide_try_gpu_path(
        task=task,
        framework=framework,
        execute_parameters=execute_parameters,
        kserve_gpu_enabled=settings.KSERVE_GPU,
        default_try_gpu=settings.DEFAULT_TRY_GPU,
    )

    wl = [x.strip() for x in (settings.SERVING_NODE_NAMES or "").split(",") if x.strip()]

    pin = bool(getattr(settings, "SERVING_PIN_SELECTED_NODE", True))
    fallback_reason: Optional[str] = fb_try
    pod_inventory_note: Optional[str] = None
    node_list = list(nodes)
    if node_list and not node_list[0].pod_requests_applied:
        pod_inventory_note = "pod_requests_not_applied_allocatable_only"

    if try_gpu:
        picked = _choose_gpu_node(node_list, v_need, wl)
        if picked:
            node_name, k, slack = picked
            node = next(n for n in node_list if n.name == node_name)
            mr, cr, ml, cl = _cap_memory_cpu_to_node(mem_gpu, millicores_to_k8s_cpu(gpu_pod_mc), node)
            return ServingResourcePlan(
                device_type="GPU",
                gpu_count=k,
                memory_request=mr,
                cpu_request=cr,
                memory_limit=ml,
                cpu_limit=cl,
                serving_vram_need_bytes=v_need,
                serving_meta_source=serving_meta_source,
                parent_repo_id=parent_repo_id,
                fallback_reason=fallback_reason,
                serving_node_name=(node_name if pin else None),
                planner_note=_append_planner_note(f"k8s_gpu slack_bytes={slack}", pod_inventory_note),
                reservation_node_name=node_name,
            )
        extra = "no_feasible_gpu_node"
        fallback_reason = ", ".join(x for x in (fallback_reason, extra) if x)
        logger.info("GPU 후보 노드에서 실행 가능한 (k, G_free) 조합 없음 — CPU 경로")

    mem_req_b = k8s_memory_quantity_to_bytes(mem_cpu)
    cpu_node = _choose_cpu_node(node_list, mem_req_b, cpu_mc, wl)
    if cpu_node:
        node = next(n for n in node_list if n.name == cpu_node)
        mr, cr, ml, cl = _cap_memory_cpu_to_node(mem_cpu, millicores_to_k8s_cpu(cpu_mc), node)
        return ServingResourcePlan(
            device_type="CPU",
            gpu_count=0,
            memory_request=mr,
            cpu_request=cr,
            memory_limit=ml,
            cpu_limit=cl,
            serving_vram_need_bytes=v_need,
            serving_meta_source=serving_meta_source,
            parent_repo_id=parent_repo_id,
            fallback_reason=fallback_reason,
            serving_node_name=(cpu_node if pin else None),
            planner_note=_append_planner_note("k8s_cpu_path", pod_inventory_note),
            reservation_node_name=cpu_node,
        )

    logger.warning("CPU 배치 후보도 없음 — allocatable 미충족 가능, 기본 플랜 폴백")
    return build_serving_resource_plan(
        normalized_meta=normalized_meta,
        task=task,
        framework=framework,
        execute_parameters=execute_parameters,
        kserve_gpu_enabled=settings.KSERVE_GPU,
        default_try_gpu=settings.DEFAULT_TRY_GPU,
        default_gpu_vram_bytes=settings.SERVING_DEFAULT_GPU_VRAM_BYTES,
        serving_meta_source=serving_meta_source,
        parent_repo_id=parent_repo_id,
        fallback_reason=fallback_reason,
        planner_note=_append_planner_note("fallback_no_cpu_node_match", pod_inventory_note),
    )


def plan_serving_resources_with_k8s(
    *,
    normalized_meta: dict[str, Any],
    task: Optional[str],
    framework: str,
    execute_parameters: dict[str, Any],
    settings: Settings,
    serving_meta_source: ServingMetaSource,
    parent_repo_id: Optional[str],
    nodes_override: Optional[Sequence[NodeInventory]] = None,
) -> ServingResourcePlan:
    """
    nodes_override 가 있으면 그 스냅샷(연쇄 잔여 반영본)으로만 플랜한다.
    없으면 collect_node_inventory 로 1회 조회한다.
    """
    nodes = (
        list(nodes_override)
        if nodes_override is not None
        else collect_node_inventory(
            default_vram_bytes=settings.SERVING_DEFAULT_GPU_VRAM_BYTES,
            vram_overrides_json=settings.SERVING_NODE_VRAM_OVERRIDES_JSON,
            serving_node_names_csv=settings.SERVING_NODE_NAMES or "",
        )
    )

    if not nodes:
        logger.info("서빙 노드 인벤토리가 비어 있음 — 기본 VRAM ceil 플랜으로 폴백")
        return build_serving_resource_plan(
            normalized_meta=normalized_meta,
            task=task,
            framework=framework,
            execute_parameters=execute_parameters,
            kserve_gpu_enabled=settings.KSERVE_GPU,
            default_try_gpu=settings.DEFAULT_TRY_GPU,
            default_gpu_vram_bytes=settings.SERVING_DEFAULT_GPU_VRAM_BYTES,
            serving_meta_source=serving_meta_source,
            parent_repo_id=parent_repo_id,
            planner_note="fallback_no_k8s_inventory",
        )

    return _plan_serving_resources_with_inventory_nodes(
        nodes,
        normalized_meta=normalized_meta,
        task=task,
        framework=framework,
        execute_parameters=execute_parameters,
        settings=settings,
        serving_meta_source=serving_meta_source,
        parent_repo_id=parent_repo_id,
    )
