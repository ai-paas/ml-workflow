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
    log_context: Optional[str] = None,
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
    before = (g, m, c)
    frees[rn] = (
        max(0, g - k),
        max(0, m - mem_b),
        max(0, c - cpu_mc),
    )
    after = frees[rn]
    ctx = f" | {log_context}" if log_context else ""
    logger.info(
        "[serving_planner]%s 이전 모델 배정 반영: 노드 '%s'의 남은 자원에서 "
        "GPU %d장·메모리 %s·CPU %s 만큼 빼서 다음 모델이 쓸 잔여를 맞춤. "
        "차감 전 (남은 GPU장수, 메모리, CPU)=%s → 차감 후=%s",
        ctx,
        rn,
        k,
        format_k8s_memory_from_bytes(mem_b),
        millicores_to_k8s_cpu(cpu_mc),
        before,
        after,
    )


logger = logging.getLogger(__name__)

_LOG_TAG = "[serving_planner]"


def _log_ctx_suffix(log_context: Optional[str]) -> str:
    return f" | {log_context}" if log_context else ""


def _try_gpu_policy_korean(
    try_gpu: bool,
    fb_try: Optional[str],
    task: Optional[str],
    framework: str,
    settings: Settings,
    execute_parameters: dict[str, Any],
) -> str:
    """GPU 경로를 탈지 말지에 대한 사람이 읽기 쉬운 한 줄."""
    if not try_gpu:
        bits = []
        task_l = (task or "").lower()
        if task_l == "embedding":
            bits.append("임베딩 작업은 정책상 CPU만 사용")
        if bool(execute_parameters.get("force_cpu")):
            bits.append("실행 파라미터에서 CPU 강제")
        if (execute_parameters.get("device_type") or "").strip().upper() == "CPU":
            bits.append("실행 파라미터에서 CPU만 쓰라고 지정됨")
        if not settings.DEFAULT_TRY_GPU:
            bits.append("서버 설정에서 GPU 우선 시도가 꺼져 있음")
        if fb_try and not bits:
            bits.append(f"내부 사유 코드: {fb_try}")
        elif fb_try:
            bits.append(f"참고 코드: {fb_try}")
        detail = " ".join(bits) if bits else "정책 또는 설정에 따름"
        return "GPU는 아예 고려하지 않고 CPU 계획만 만듭니다. " + detail
    return "우선 GPU로 배치할 수 있는지 보고, 안 되면 CPU로 내립니다."


def _log_inventory_snapshot(log_context: Optional[str], nodes: Sequence[NodeInventory]) -> None:
    """§7.3.1: 노드별 잔여를 한국어 한 줄씩만 요약."""
    suf = _log_ctx_suffix(log_context)
    for n in nodes:
        v_card_s = format_k8s_memory_from_bytes(int(n.v_card_bytes)) if n.v_card_bytes else "알 수 없음"
        pod_note = (
            "클러스터 다른 Pod의 요청을 반영함"
            if n.pod_requests_applied
            else "Pod 합산 실패로 노드 할당량만 사용(보수적)"
        )
        logger.info(
            "%s%s 노드 '%s': %s. 전체 GPU %d장, 배치 가능한 남은 GPU %d장, "
            "한 장당 VRAM 추정 %s, 남은 메모리 %s, 남은 CPU %s (노드 메모리·CPU 상한 %s / %s)",
            _LOG_TAG,
            suf,
            n.name,
            pod_note,
            n.alloc_gpu,
            n.g_free,
            v_card_s,
            format_k8s_memory_from_bytes(n.m_free),
            millicores_to_k8s_cpu(n.c_free),
            format_k8s_memory_from_bytes(n.alloc_memory_bytes),
            millicores_to_k8s_cpu(n.alloc_cpu_millicores),
        )


def _log_try_gpu_gate(
    log_context: Optional[str],
    try_gpu: bool,
    fb_try: Optional[str],
    task: Optional[str],
    framework: str,
    settings: Settings,
    execute_parameters: dict[str, Any],
) -> None:
    suf = _log_ctx_suffix(log_context)
    logger.info(
        "%s%s %s (task=%s, 프레임워크=%s)",
        _LOG_TAG,
        suf,
        _try_gpu_policy_korean(try_gpu, fb_try, task, framework, settings, execute_parameters),
        task or "-",
        framework,
    )


def _log_gpu_placement_summary(
    log_context: Optional[str],
    nodes: list[NodeInventory],
    v_need: int,
    wl: list[str],
) -> None:
    """GPU 후보를 노드별로 한두 줄로 요약."""
    suf = _log_ctx_suffix(log_context)
    v_need_s = format_k8s_memory_from_bytes(v_need)
    eligible: list[str] = []
    skipped: list[str] = []
    for n in nodes:
        if n.alloc_gpu <= 0:
            skipped.append(f"{n.name}(GPU 없음·0장)")
            continue
        if not n.v_card_bytes or n.v_card_bytes <= 0:
            skipped.append(f"{n.name}(장당 VRAM 정보 없음)")
            continue
        v_card = int(n.v_card_bytes)
        k_need = max(1, math.ceil(v_need / float(v_card)))
        g_free = n.g_free
        if k_need > g_free:
            skipped.append(
                f"{n.name}(이 모델에 필요한 GPU {k_need}장 > 이 노드에 남은 GPU {g_free}장, "
                f"장당 VRAM {format_k8s_memory_from_bytes(v_card)})"
            )
            continue
        eligible.append(n.name)
    wl_note = f"노드 화이트리스트 적용 중: {', '.join(wl)}" if wl else "노드 화이트리스트 없음(Ready 노드 전부 검토)"
    logger.info(
        "%s%s GPU 배치 검토: 모델 VRAM 안내값 %s. %s. " "배치 가능해 보이는 노드: %s. 제외된 노드: %s",
        _LOG_TAG,
        suf,
        v_need_s,
        wl_note,
        ", ".join(eligible) if eligible else "(없음)",
        "; ".join(skipped) if skipped else "(없음)",
    )


def _log_cpu_placement_summary(
    log_context: Optional[str],
    nodes: list[NodeInventory],
    mem_req_bytes: int,
    cpu_req_millicores: int,
) -> None:
    suf = _log_ctx_suffix(log_context)
    mem_s = format_k8s_memory_from_bytes(mem_req_bytes)
    cpu_s = millicores_to_k8s_cpu(cpu_req_millicores)
    skipped: list[str] = []
    ok_count = 0
    for n in nodes:
        if n.m_free < mem_req_bytes and n.c_free < cpu_req_millicores:
            skipped.append(
                f"{n.name}(메모리·CPU 모두 부족: 남은 메모리 {format_k8s_memory_from_bytes(n.m_free)}, "
                f"남은 CPU {millicores_to_k8s_cpu(n.c_free)})"
            )
        elif n.m_free < mem_req_bytes:
            skipped.append(f"{n.name}(메모리 부족: 남음 {format_k8s_memory_from_bytes(n.m_free)} < 필요 {mem_s})")
        elif n.c_free < cpu_req_millicores:
            skipped.append(f"{n.name}(CPU 부족: 남음 {millicores_to_k8s_cpu(n.c_free)} < 필요 {cpu_s})")
        else:
            ok_count += 1
    logger.info(
        "%s%s CPU 배치 검토: Pod에 요청할 메모리 %s, CPU %s. " "조건을 만족하는 노드 %d개, 조건 미달로 제외: %s",
        _LOG_TAG,
        suf,
        mem_s,
        cpu_s,
        ok_count,
        "; ".join(skipped[:12]) + (" …(이하 생략)" if len(skipped) > 12 else "") if skipped else "(없음)",
    )


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
    log_context: Optional[str] = None,
) -> ServingResourcePlan:
    """인벤토리 리스트가 비어 있지 않을 때 §7.4~7.7 플랜(§7.3.2는 호출부에서 잔여 조정)."""
    v_need = int(normalized_meta["serving_vram_need_bytes"])
    mem_gpu = str(normalized_meta["serving_memory_request_gpu"])
    mem_cpu = str(normalized_meta["serving_memory_request_cpu"])
    gpu_pod_mc = int(normalized_meta["serving_gpu_pod_cpu_request_millicores"])
    cpu_mc = int(normalized_meta["serving_cpu_request_millicores"])

    try_gpu, fb_try = decide_try_gpu_path(
        task=task,
        execute_parameters=execute_parameters,
        default_try_gpu=settings.DEFAULT_TRY_GPU,
    )

    wl = [x.strip() for x in (settings.SERVING_NODE_NAMES or "").split(",") if x.strip()]

    pin = bool(getattr(settings, "SERVING_PIN_SELECTED_NODE", True))
    fallback_reason: Optional[str] = fb_try
    pod_inventory_note: Optional[str] = None
    node_list = list(nodes)
    if node_list and not node_list[0].pod_requests_applied:
        pod_inventory_note = "pod_requests_not_applied_allocatable_only"

    suf = _log_ctx_suffix(log_context)
    wl_human = ", ".join(wl) if wl else "지정 없음(클러스터에서 Ready인 노드 전부)"
    logger.info(
        "%s%s 자원 계획 시작 — 모델 VRAM 안내 %s, GPU Pod 메모리·CPU 요청 %s / %s, "
        "CPU 전용 경로일 때 메모리·CPU %s / %s, 노드 이름 제한: %s",
        _LOG_TAG,
        suf,
        format_k8s_memory_from_bytes(v_need),
        mem_gpu,
        millicores_to_k8s_cpu(gpu_pod_mc),
        mem_cpu,
        millicores_to_k8s_cpu(cpu_mc),
        wl_human,
    )
    _log_inventory_snapshot(log_context, node_list)
    _log_try_gpu_gate(log_context, try_gpu, fb_try, task, framework, settings, execute_parameters)

    if try_gpu:
        _log_gpu_placement_summary(log_context, node_list, v_need, wl)
        picked = _choose_gpu_node(node_list, v_need, wl)
        if picked:
            node_name, k, slack = picked
            node = next(n for n in node_list if n.name == node_name)
            mr, cr, ml, cl = _cap_memory_cpu_to_node(mem_gpu, millicores_to_k8s_cpu(gpu_pod_mc), node)
            v_card = int(node.v_card_bytes or 0)
            slot_remain = node.g_free - k
            slack_s = format_k8s_memory_from_bytes(slack)
            pin_msg = f"스케줄러에 노드 '{node_name}' 고정" if pin else "노드 고정 없이 요청만 전달"
            logger.info(
                "%s%s GPU 배정 확정 — 노드 '%s', 이 Pod에 쓸 GPU %d장(모델 VRAM %s 기준·장당 VRAM %s), "
                "남는 VRAM 여유 약 %s, 같은 노드에 남은 GPU 슬롯 %d장, Pod 메모리·CPU 요청 %s / %s, %s",
                _LOG_TAG,
                suf,
                node_name,
                k,
                format_k8s_memory_from_bytes(v_need),
                format_k8s_memory_from_bytes(v_card),
                slack_s,
                slot_remain,
                mr,
                cr,
                pin_msg,
            )
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
                gpu_resource_key=node.gpu_resource_key,
                node_selector_json=node.node_selector_json,
                tolerations_json=node.tolerations_json,
            )
        extra = "no_feasible_gpu_node"
        fallback_reason = ", ".join(x for x in (fallback_reason, extra) if x)
        logger.info(
            "%s%s GPU에 둘 곳이 없어 CPU 배치로 넘어갑니다. "
            "이유: 어느 노드에서도 '필요한 GPU 장 수'가 '남은 GPU 슬롯' 이하가 되지 않거나, "
            "GPU·장당 VRAM 정보가 없어 계산할 수 없었습니다. (위 로그의 '제외된 노드' 참고)",
            _LOG_TAG,
            suf,
        )

    mem_req_b = k8s_memory_quantity_to_bytes(mem_cpu)
    _log_cpu_placement_summary(log_context, node_list, mem_req_b, cpu_mc)
    cpu_node = _choose_cpu_node(node_list, mem_req_b, cpu_mc, wl)
    if cpu_node:
        node = next(n for n in node_list if n.name == cpu_node)
        mr, cr, ml, cl = _cap_memory_cpu_to_node(mem_cpu, millicores_to_k8s_cpu(cpu_mc), node)
        pool_ko = (
            "GPU가 달리지 않은 노드를 우선" if node.alloc_gpu == 0 else "GPU가 있는 노드이지만 CPU 요청만으로 배치"
        )
        margin_m = node.m_free - mem_req_b
        margin_c = node.c_free - cpu_mc
        pin_msg = f"스케줄러에 노드 '{cpu_node}' 고정" if pin else "노드 고정 없이 요청만 전달"
        came_from_gpu = try_gpu and ("no_feasible_gpu_node" in (fallback_reason or ""))
        transition = " (GPU 배치가 불가능해 CPU로 내려옴)" if came_from_gpu else ""
        logger.info(
            "%s%s CPU 배정 확정%s — 노드 '%s', 선정 방식: %s. "
            "요청 메모리·CPU %s / %s, 노드 잔여에서 남는 여유 약 메모리 %s·CPU %s, %s",
            _LOG_TAG,
            suf,
            transition,
            cpu_node,
            pool_ko,
            mr,
            cr,
            format_k8s_memory_from_bytes(margin_m),
            millicores_to_k8s_cpu(margin_c),
            pin_msg,
        )
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
            # CPU 가 taint 걸린 GPU 노드(tier2)로 떨어질 수 있으므로 선택 노드의 toleration/selector 를 전파
            gpu_resource_key=node.gpu_resource_key,
            node_selector_json=node.node_selector_json,
            tolerations_json=node.tolerations_json,
        )

    logger.warning(
        "%s%s 어떤 노드도 CPU 조건(메모리·CPU)을 만족하지 못해, 클러스터 정보 없이 기본 공식으로만 플랜합니다.",
        _LOG_TAG,
        suf,
    )
    return build_serving_resource_plan(
        normalized_meta=normalized_meta,
        task=task,
        framework=framework,
        execute_parameters=execute_parameters,
        default_try_gpu=settings.DEFAULT_TRY_GPU,
        default_gpu_vram_bytes=settings.SERVING_DEFAULT_GPU_VRAM_BYTES,
        serving_meta_source=serving_meta_source,
        parent_repo_id=parent_repo_id,
        fallback_reason=fallback_reason,
        planner_note=_append_planner_note("fallback_no_cpu_node_match", pod_inventory_note),
        log_context=log_context,
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
    log_context: Optional[str] = None,
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
            gpu_pool_profiles_json=getattr(settings, "GPU_POOL_PROFILES_JSON", "[]") or "[]",
            exclude_control_plane_nodes=not settings.SERVING_INCLUDE_CONTROL_PLANE_NODES,
        )
    )

    if not nodes:
        logger.info(
            "%s%s 클러스터 노드 목록을 가져오지 못했거나 대상 노드가 없어, "
            "노드 잔여 없이 기본 VRAM·요청값만으로 플랜합니다.",
            _LOG_TAG,
            _log_ctx_suffix(log_context),
        )
        return build_serving_resource_plan(
            normalized_meta=normalized_meta,
            task=task,
            framework=framework,
            execute_parameters=execute_parameters,
            default_try_gpu=settings.DEFAULT_TRY_GPU,
            default_gpu_vram_bytes=settings.SERVING_DEFAULT_GPU_VRAM_BYTES,
            serving_meta_source=serving_meta_source,
            parent_repo_id=parent_repo_id,
            planner_note="fallback_no_k8s_inventory",
            log_context=log_context,
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
        log_context=log_context,
    )
