"""
서빙 리소스 플래너: K8s 인벤토리 + GPU 슬랙 최소 노드·k, CPU 노드 선정.

인벤토리 조회 실패 시 serving_resource_meta.build_serving_resource_plan 으로 폴백.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

from config.settings import Settings
from core.serving.serving_k8s_inventory import NodeInventory, collect_node_inventory, whitelist_rank
from core.serving.serving_resource_meta import (
    SERVING_CPU_LIMIT_KEYS,
    ServingCapacityError,
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
    """확정 플랜의 k·memory·cpu 요청을 선택 노드 잔여에서 차감(max(0,·))."""
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
    """노드별 잔여를 한국어 한 줄씩만 요약."""
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
    *,
    cpu_req_millicores: int = 0,
    mem_req_bytes: int = 0,
    cpu_headroom_millicores: int = 0,
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
        if n.c_free - cpu_headroom_millicores < cpu_req_millicores:
            skipped.append(
                f"{n.name}(GPU는 있으나 CPU 부족: 쓸 수 있는 CPU "
                f"{millicores_to_k8s_cpu(max(0, n.c_free - cpu_headroom_millicores))} "
                f"< 필요 {millicores_to_k8s_cpu(cpu_req_millicores)})"
            )
            continue
        if n.m_free < mem_req_bytes:
            skipped.append(
                f"{n.name}(GPU는 있으나 메모리 부족: 남음 {format_k8s_memory_from_bytes(n.m_free)} "
                f"< 필요 {format_k8s_memory_from_bytes(mem_req_bytes)})"
            )
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
        "%s%s GPU 배치 검토: 모델 VRAM 안내값 %s. %s. 배치 가능해 보이는 노드: %s. 제외된 노드: %s",
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
    cpu_headroom_millicores: int = 0,
) -> None:
    suf = _log_ctx_suffix(log_context)
    mem_s = format_k8s_memory_from_bytes(mem_req_bytes)
    cpu_s = millicores_to_k8s_cpu(cpu_req_millicores)
    skipped: list[str] = []
    ok_count = 0
    for n in nodes:
        # 시스템 몫을 뺀, 서빙이 실제로 쓸 수 있는 CPU
        usable_c = max(0, n.c_free - cpu_headroom_millicores)
        if n.m_free < mem_req_bytes and usable_c < cpu_req_millicores:
            skipped.append(
                f"{n.name}(메모리·CPU 모두 부족: 남은 메모리 {format_k8s_memory_from_bytes(n.m_free)}, "
                f"쓸 수 있는 CPU {millicores_to_k8s_cpu(usable_c)})"
            )
        elif n.m_free < mem_req_bytes:
            skipped.append(f"{n.name}(메모리 부족: 남음 {format_k8s_memory_from_bytes(n.m_free)} < 필요 {mem_s})")
        elif usable_c < cpu_req_millicores:
            skipped.append(f"{n.name}(CPU 부족: 쓸 수 있음 {millicores_to_k8s_cpu(usable_c)} < 필요 {cpu_s})")
        else:
            ok_count += 1
    headroom_s = (
        f", 노드당 시스템 몫 {millicores_to_k8s_cpu(cpu_headroom_millicores)} 제외"
        if cpu_headroom_millicores > 0
        else ""
    )
    logger.info(
        "%s%s CPU 배치 검토: Pod에 요청할 메모리 %s, CPU %s%s. 조건을 만족하는 노드 %d개, 조건 미달로 제외: %s",
        _LOG_TAG,
        suf,
        mem_s,
        cpu_s,
        headroom_s,
        ok_count,
        "; ".join(skipped[:12]) + (" …(이하 생략)" if len(skipped) > 12 else "") if skipped else "(없음)",
    )


def _capacity_rejection_message(
    nodes: Sequence[NodeInventory],
    *,
    mem_req_bytes: int,
    cpu_req_millicores: int,
    cpu_headroom_millicores: int,
    log_context: Optional[str],
    gpu_path_cpu_millicores: Optional[int] = None,
    gpu_path_mem_bytes: Optional[int] = None,
) -> str:
    """배치 거부 사유를 사용자에게 그대로 보여줄 한국어 문장으로.

    CPU 는 밀리코어로 적는다. millicores_to_k8s_cpu 는 2000 을 '2' 로 줄이는데,
    자원 부족을 설명하는 문장에서는 '2000m' 쪽이 노드 잔여와 바로 비교된다.
    """
    detail = "; ".join(
        f"{n.name}(남은 CPU {n.c_free}m, 남은 메모리 {format_k8s_memory_from_bytes(n.m_free)})" for n in nodes
    )
    need = f"CPU {cpu_req_millicores}m·메모리 {format_k8s_memory_from_bytes(mem_req_bytes)}"
    if gpu_path_cpu_millicores is not None and gpu_path_mem_bytes is not None:
        need = (
            f"GPU 배치 시 CPU {gpu_path_cpu_millicores}m·"
            f"메모리 {format_k8s_memory_from_bytes(gpu_path_mem_bytes)}, "
            f"CPU 전용 배치 시 {need}"
        )
    head = f"서빙에 필요한 자원({need})을 받아줄 노드가 없어 배포를 시작하지 않았습니다."
    if cpu_headroom_millicores > 0:
        head += f" (각 노드에서 시스템 몫 {cpu_headroom_millicores}m를 빼고 판단)"
    tail = f" 현재 노드 상태: {detail or '(대상 노드 없음)'}."
    if log_context:
        tail += f" [{log_context}]"
    return head + tail + " 실행 중인 다른 워크플로우를 정리한 뒤 다시 시도해 주세요."


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
    cpu_lim: str,
    node: NodeInventory,
) -> tuple[str, str, str, str]:
    """노드 allocatable 을 넘지 않도록 request/limit 캡. CPU limit 은 request 와 별개로 캡한다."""
    req_mem = k8s_memory_quantity_to_bytes(mem_req)
    req_mc = _cpu_string_to_millicores(cpu_req)
    lim_mc = max(req_mc, _cpu_string_to_millicores(cpu_lim))
    # allocatable 상한 캡 (노드 총량 기준)
    cap_mem = max(1, min(req_mem, node.alloc_memory_bytes or req_mem))
    cap_req_mc = max(1, min(req_mc, node.alloc_cpu_millicores or req_mc))
    cap_lim_mc = max(cap_req_mc, min(lim_mc, node.alloc_cpu_millicores or lim_mc))
    ms = format_k8s_memory_from_bytes(cap_mem)
    return ms, millicores_to_k8s_cpu(cap_req_mc), ms, millicores_to_k8s_cpu(cap_lim_mc)


def _choose_gpu_node(
    nodes: list[NodeInventory],
    v_need: int,
    whitelist: list[str],
    *,
    cpu_req_millicores: int,
    mem_req_bytes: int,
    cpu_headroom_millicores: int = 0,
) -> Optional[tuple[str, int, int]]:
    """(node_name, k, slack_bytes). 없으면 None.

    GPU 슬롯이 남아도 그 노드의 CPU·메모리 잔여가 Pod 요청을 못 받치면 후보에서 뺀다.
    이 검사를 빼면 스케줄 후 kubelet 이 OutOfcpu 로 Pod 를 떨어뜨린다.
    """
    best: Optional[tuple[str, int, int]] = None
    best_key: Optional[tuple] = None

    for n in nodes:
        if n.alloc_gpu <= 0 or not n.v_card_bytes or n.v_card_bytes <= 0:
            continue
        if n.c_free - cpu_headroom_millicores < cpu_req_millicores or n.m_free < mem_req_bytes:
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
    *,
    cpu_headroom_millicores: int = 0,
) -> Optional[str]:
    """노드 이름 하나 또는 None."""
    tier1: list[tuple[tuple, str]] = []
    tier2: list[tuple[tuple, str]] = []

    for n in nodes:
        if n.m_free < mem_req_bytes or n.c_free - cpu_headroom_millicores < cpu_req_millicores:
            continue
        margin_m = n.m_free - mem_req_bytes
        margin_c = n.c_free - cpu_headroom_millicores - cpu_req_millicores
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
    """인벤토리 리스트가 비어 있지 않을 때 GPU/CPU 배치 계획을 세우고 ServingResourcePlan 을 반환."""
    v_need = int(normalized_meta["serving_vram_need_bytes"])
    mem_gpu = str(normalized_meta["serving_memory_request_gpu"])
    mem_cpu = str(normalized_meta["serving_memory_request_cpu"])
    gpu_pod_mc = int(normalized_meta["serving_gpu_pod_cpu_request_millicores"])
    cpu_mc = int(normalized_meta["serving_cpu_request_millicores"])
    gpu_pod_lim_mc = max(gpu_pod_mc, int(normalized_meta.get(SERVING_CPU_LIMIT_KEYS[0]) or gpu_pod_mc))
    cpu_lim_mc = max(cpu_mc, int(normalized_meta.get(SERVING_CPU_LIMIT_KEYS[1]) or cpu_mc))

    try_gpu, fb_try = decide_try_gpu_path(
        task=task,
        execute_parameters=execute_parameters,
        default_try_gpu=settings.DEFAULT_TRY_GPU,
    )

    wl = [x.strip() for x in (settings.SERVING_NODE_NAMES or "").split(",") if x.strip()]
    headroom = max(0, int(getattr(settings, "SERVING_NODE_CPU_HEADROOM_MILLICORES", 0) or 0))

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
        _log_gpu_placement_summary(
            log_context,
            node_list,
            v_need,
            wl,
            cpu_req_millicores=gpu_pod_mc,
            mem_req_bytes=k8s_memory_quantity_to_bytes(mem_gpu),
            cpu_headroom_millicores=headroom,
        )
        picked = _choose_gpu_node(
            node_list,
            v_need,
            wl,
            cpu_req_millicores=gpu_pod_mc,
            mem_req_bytes=k8s_memory_quantity_to_bytes(mem_gpu),
            cpu_headroom_millicores=headroom,
        )
        if picked:
            node_name, k, slack = picked
            node = next(n for n in node_list if n.name == node_name)
            mr, cr, ml, cl = _cap_memory_cpu_to_node(
                mem_gpu, millicores_to_k8s_cpu(gpu_pod_mc), millicores_to_k8s_cpu(gpu_pod_lim_mc), node
            )
            v_card = int(node.v_card_bytes or 0)
            slot_remain = node.g_free - k
            slack_s = format_k8s_memory_from_bytes(slack)
            pin_msg = f"스케줄러에 노드 '{node_name}' 고정" if pin else "노드 고정 없이 요청만 전달"
            logger.info(
                "%s%s GPU 배정 확정 — 노드 '%s', 이 Pod에 쓸 GPU %d장(모델 VRAM %s 기준·장당 VRAM %s), "
                "남는 VRAM 여유 약 %s, 같은 노드에 남은 GPU 슬롯 %d장, Pod 메모리·CPU 요청 %s / %s (CPU 상한 %s), %s",
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
                cl,
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
                ephemeral_storage_limit=str(normalized_meta.get("serving_ephemeral_storage_limit", "1Gi")),
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
    _log_cpu_placement_summary(log_context, node_list, mem_req_b, cpu_mc, headroom)
    cpu_node = _choose_cpu_node(node_list, mem_req_b, cpu_mc, wl, cpu_headroom_millicores=headroom)
    if cpu_node:
        node = next(n for n in node_list if n.name == cpu_node)
        mr, cr, ml, cl = _cap_memory_cpu_to_node(
            mem_cpu, millicores_to_k8s_cpu(cpu_mc), millicores_to_k8s_cpu(cpu_lim_mc), node
        )
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
            "요청 메모리·CPU %s / %s (CPU 상한 %s), 노드 잔여에서 남는 여유 약 메모리 %s·CPU %s, %s",
            _LOG_TAG,
            suf,
            transition,
            cpu_node,
            pool_ko,
            mr,
            cr,
            cl,
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
            ephemeral_storage_limit=str(normalized_meta.get("serving_ephemeral_storage_limit", "1Gi")),
        )

    # 여기까지 왔다는 것은 GPU·CPU 어느 쪽으로도 받아줄 노드가 없다는 뜻이다. 예전에는 인벤토리를 무시한
    # 기본 공식으로 플랜을 만들어 그대로 배포했는데, 그러면 Pod 가 만들어진 뒤 kubelet 이 OutOfcpu 로
    # 떨어뜨려 원인 파악도 어렵고 실패한 Pod 도 쌓인다. 배포를 시작하기 전에 거부한다.
    msg = _capacity_rejection_message(
        node_list,
        mem_req_bytes=mem_req_b,
        cpu_req_millicores=cpu_mc,
        cpu_headroom_millicores=headroom,
        log_context=log_context,
        gpu_path_cpu_millicores=gpu_pod_mc if try_gpu else None,
        gpu_path_mem_bytes=k8s_memory_quantity_to_bytes(mem_gpu) if try_gpu else None,
    )
    logger.warning("%s%s %s", _LOG_TAG, suf, msg)
    raise ServingCapacityError(msg)


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
