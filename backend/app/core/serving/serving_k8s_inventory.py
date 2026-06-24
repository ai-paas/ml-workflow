"""
§7.3 Kubernetes 노드 인벤토리 (백엔드 전용).

§7.3.1: allocatable에 더해, 전 네임스페이스 Pod의 resources.requests 합산으로
G_used/M_used/C_used를 구하고 G_free/M_free/C_free를 근사한다.
Pod 조회 실패 시 allocatable-only 폴백(planner_note 등으로 상위에서 기록 가능).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from core.serving.gpu_placement import parse_gpu_pool_profiles, resolve_placement

logger = logging.getLogger(__name__)

_GIB = 1024**3
_GPU_RESOURCE_KEY = "nvidia.com/gpu"


@dataclass(frozen=True)
class NodeInventory:
    """단일 노드의 스케줄링에 쓰는 스냅샷."""

    name: str
    schedulable: bool
    alloc_gpu: int
    alloc_memory_bytes: int
    alloc_cpu_millicores: int
    v_card_bytes: Optional[int]
    g_free: int
    m_free: int
    c_free: int
    pod_requests_applied: bool
    # GPU 노드풀 프로파일 기반 배치(§ docs/k8s-gpu-allocation). frozen 친화 위해 JSON 문자열로 보관.
    gpu_resource_key: str = "nvidia.com/gpu"
    node_selector_json: str = "{}"
    tolerations_json: str = "[]"


def _try_load_core_v1():
    try:
        from kubernetes import client
        from kubernetes import config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        return client.CoreV1Api()
    except Exception as e:
        logger.warning("Kubernetes 클라이언트 초기화 실패(인벤토리 생략): %s", e)
        return None


def _node_ready(node: Any) -> bool:
    for c in node.status.conditions or []:
        if getattr(c, "type", None) == "Ready" and getattr(c, "status", None) == "True":
            return True
    return False


# Kubernetes 표준 control-plane 식별(라벨·taints). 워커만 서빙 인벤토리에 넣기 위해 제외한다.
_CONTROL_PLANE_LABEL_KEYS = (
    "node-role.kubernetes.io/control-plane",
    "node-role.kubernetes.io/master",
)
_CONTROL_PLANE_TAINT_KEYS = (
    "node-role.kubernetes.io/control-plane",
    "node-role.kubernetes.io/master",
)


def _is_kubernetes_control_plane_node(node: Any) -> bool:
    meta = getattr(node, "metadata", None)
    labels = (meta.labels if meta else None) or {}
    for k in _CONTROL_PLANE_LABEL_KEYS:
        if k in labels:
            return True
    spec = getattr(node, "spec", None)
    for t in getattr(spec, "taints", None) or []:
        if (getattr(t, "key", None) or "") in _CONTROL_PLANE_TAINT_KEYS:
            return True
    return False


def _parse_quantity_cpu_to_millicores(q: str) -> int:
    s = (q or "").strip()
    if not s:
        return 0
    if s.endswith("m"):
        return max(0, int(float(s[:-1])))
    return max(0, int(math.ceil(float(s) * 1000)))


def _parse_quantity_memory_to_bytes(q: str) -> int:
    """Kubernetes quantity 문자열을 바이트로 (kubernetes.utils 우선)."""
    try:
        from kubernetes.utils import quantity

        return int(quantity.parse_quantity(q))
    except Exception:
        from core.serving.serving_resource_meta import k8s_memory_quantity_to_bytes

        try:
            return k8s_memory_quantity_to_bytes(q)
        except Exception:
            return 0


def _parse_alloc_gpu(alloc: dict[str, str], resource_key: str = _GPU_RESOURCE_KEY) -> int:
    raw = (alloc or {}).get(resource_key, "")
    if not raw:
        return 0
    s = str(raw).strip()
    if not s:
        return 0
    try:
        f = float(s)
        return max(0, int(math.floor(f)))
    except ValueError:
        return 0


def _parse_gpu_request_quantity(raw: Any) -> int:
    """Pod container request의 GPU 칸 수(내림 정수)."""
    if raw is None:
        return 0
    s = str(raw).strip()
    if not s:
        return 0
    try:
        return max(0, int(math.floor(float(s))))
    except ValueError:
        return 0


def _container_triplet(container: Any, gpu_key: str) -> tuple[int, int, int]:
    """(gpu_slots, memory_bytes, cpu_millicores) 단일 컨테이너."""
    req = (container.resources and container.resources.requests) or {}
    if not isinstance(req, dict):
        req = dict(req) if req else {}
    gpu = _parse_gpu_request_quantity(req.get(gpu_key))
    mem = _parse_quantity_memory_to_bytes(str(req.get("memory") or ""))
    cpu = _parse_quantity_cpu_to_millicores(str(req.get("cpu") or ""))
    return gpu, mem, cpu


def _sum_container_list(containers: Any, gpu_key: str) -> tuple[int, int, int]:
    tg, tm, tc = 0, 0, 0
    for ctr in containers or []:
        g, m, c = _container_triplet(ctr, gpu_key)
        tg += g
        tm += m
        tc += c
    return tg, tm, tc


def _effective_pod_requests(pod: Any, gpu_key: str) -> tuple[int, int, int]:
    """
    §7.3.1: 스케줄러와 동일하게 max(sum(init), sum(containers)) per resource.
    """
    ig, im, ic = _sum_container_list(getattr(pod.spec, "init_containers", None) or [], gpu_key)
    ag, am, ac = _sum_container_list(getattr(pod.spec, "containers", None) or [], gpu_key)
    return max(ig, ag), max(im, am), max(ic, ac)


def _pod_counts_for_usage(pod: Any) -> bool:
    """바인딩된 Running/Pending 만 점유로 친다."""
    if pod.metadata and getattr(pod.metadata, "deletion_timestamp", None):
        return False
    phase = (pod.status and pod.status.phase) or ""
    if phase not in ("Pending", "Running"):
        return False
    spec = pod.spec
    if not spec or not getattr(spec, "node_name", None):
        return False
    return True


def aggregate_pod_requests_by_node(
    core: Any,
    *,
    gpu_resource_key: str = _GPU_RESOURCE_KEY,
    page_limit: int = 500,
) -> tuple[dict[str, tuple[int, int, int]], bool]:
    """
    노드명 -> (g_used, m_used_bytes, c_used_millicores).
    반환 (usage_map, applied_ok). applied_ok False 이면 빈 dict 또는 부분일 수 있음 — 호출부에서 alloc-only 폴백.
    """
    usage: dict[str, tuple[int, int, int]] = {}
    cont: Optional[str] = None
    try:
        while True:
            kwargs: dict[str, Any] = {"limit": page_limit}
            if cont:
                kwargs["_continue"] = cont
            resp = core.list_pod_for_all_namespaces(**kwargs)
            for pod in resp.items or []:
                if not _pod_counts_for_usage(pod):
                    continue
                node = pod.spec.node_name
                g, m, c = _effective_pod_requests(pod, gpu_resource_key)
                if g == 0 and m == 0 and c == 0:
                    continue
                prev = usage.get(node, (0, 0, 0))
                usage[node] = (prev[0] + g, prev[1] + m, prev[2] + c)
            meta = resp.metadata
            cont = getattr(meta, "_continue", None) if meta else None
            if not cont:
                break
        return usage, True
    except Exception as e:
        logger.warning("list_pod_for_all_namespaces(점유 합산) 실패, allocatable-only 폴백: %s", e)
        return {}, False


def _v_card_bytes_for_node(
    node_name: str,
    labels: dict[str, str],
    *,
    default_vram_bytes: int,
    vram_overrides_gib: dict[str, int],
) -> Optional[int]:
    """GPU 1장당 VRAM 바이트. 라벨 nvidia.com/gpu.memory(MiB) → 오버라이드 → 기본."""
    if node_name in vram_overrides_gib:
        return int(vram_overrides_gib[node_name]) * _GIB
    mem_mi = labels.get("nvidia.com/gpu.memory")
    if mem_mi is not None and str(mem_mi).strip() != "":
        try:
            mi = int(float(str(mem_mi).strip()))
            return max(1, mi) * (1024**2)
        except ValueError:
            pass
    return default_vram_bytes if default_vram_bytes > 0 else None


def parse_vram_overrides_json(raw: str) -> dict[str, int]:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("SERVING_NODE_VRAM_OVERRIDES_JSON 파싱 실패, 무시: %s", raw[:200])
        return {}
    out: dict[str, int] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    return out


def collect_node_inventory(
    *,
    default_vram_bytes: int,
    vram_overrides_json: str,
    serving_node_names_csv: str,
    gpu_resource_key: str = _GPU_RESOURCE_KEY,
    gpu_pool_profiles_json: str = "[]",
    exclude_control_plane_nodes: bool = True,
) -> list[NodeInventory]:
    """
    Ready·스케줄 가능 노드 목록. SERVING_NODE_NAMES 비면 클러스터 전체(필터: Ready).
    §7.3.1: Pod requests 합산으로 g_free / m_free / c_free 채움.
    exclude_control_plane_nodes: control-plane/master 역할 노드는 서빙 후보에서 제외(기본 True).
    gpu_pool_profiles_json: GPU 노드풀 프로파일(MIG taint 등). 매칭 노드에 nodeSelector/tolerations/자원키 부여.
    """
    core = _try_load_core_v1()
    if core is None:
        return []

    whitelist = [n.strip() for n in (serving_node_names_csv or "").split(",") if n.strip()]
    vram_gib = parse_vram_overrides_json(vram_overrides_json)
    gpu_profiles = parse_gpu_pool_profiles(gpu_pool_profiles_json)

    try:
        resp = core.list_node()
        items = resp.items or []
    except Exception as e:
        logger.warning("list_node 실패: %s", e)
        return []

    pod_by_node, pod_ok = aggregate_pod_requests_by_node(core, gpu_resource_key=gpu_resource_key)

    out: list[NodeInventory] = []
    for node in items:
        name = node.metadata.name
        if not name:
            continue
        if whitelist and name not in whitelist:
            continue
        if exclude_control_plane_nodes and _is_kubernetes_control_plane_node(node):
            continue
        if node.spec and getattr(node.spec, "unschedulable", False):
            continue
        if not _node_ready(node):
            continue

        alloc = node.status.allocatable or {}
        labels = node.metadata.labels or {}

        # GPU 노드풀 프로파일 매칭 → 자원키/nodeSelector/tolerations 결정 (MIG taint 노드 등)
        taints = [
            {"key": t.key, "value": getattr(t, "value", None), "effect": getattr(t, "effect", None)}
            for t in ((node.spec.taints if node.spec else None) or [])
        ]
        placement = resolve_placement(labels, profiles=gpu_profiles, taints=taints, default_gpu_key=gpu_resource_key)
        node_gpu_key = placement["gpu_resource_key"]

        alloc_gpu = _parse_alloc_gpu(alloc, node_gpu_key)
        mem_s = alloc.get("memory", "0")
        cpu_s = alloc.get("cpu", "0")
        alloc_mem = _parse_quantity_memory_to_bytes(mem_s)
        alloc_cpu = _parse_quantity_cpu_to_millicores(cpu_s)

        g_used, m_used, c_used = pod_by_node.get(name, (0, 0, 0))
        if pod_ok:
            g_free = max(0, alloc_gpu - g_used)
            m_free = max(0, alloc_mem - m_used)
            c_free = max(0, alloc_cpu - c_used)
        else:
            g_free, m_free, c_free = alloc_gpu, alloc_mem, alloc_cpu

        v_card: Optional[int] = None
        if alloc_gpu > 0:
            # 프로파일에 slice_vram_gib 가 있으면 강제(예: gpu.memory 라벨 없는 MIG-미적용 노드 보정)
            slice_gib = placement.get("slice_vram_gib")
            if slice_gib:
                v_card = int(slice_gib) * _GIB
            else:
                v_card = _v_card_bytes_for_node(
                    name, labels, default_vram_bytes=default_vram_bytes, vram_overrides_gib=vram_gib
                )
            if v_card is None or v_card <= 0:
                v_card = default_vram_bytes if default_vram_bytes > 0 else None

        out.append(
            NodeInventory(
                name=name,
                schedulable=True,
                alloc_gpu=alloc_gpu,
                alloc_memory_bytes=alloc_mem,
                alloc_cpu_millicores=alloc_cpu,
                v_card_bytes=v_card,
                g_free=g_free,
                m_free=m_free,
                c_free=c_free,
                pod_requests_applied=pod_ok,
                gpu_resource_key=node_gpu_key,
                node_selector_json=json.dumps(placement["node_selector"]),
                tolerations_json=json.dumps(placement["tolerations"]),
            )
        )

    return out


def node_inventories_with_free_overrides(
    base: list[NodeInventory],
    frees: dict[str, tuple[int, int, int]],
) -> list[NodeInventory]:
    """
    §7.3.2 연쇄 잔여: 동일 스냅샷 alloc/v_card는 유지하고 g_free/m_free/c_free만 덮어쓴다.
    frees 키는 노드명; 없는 노드는 base 값 유지.
    """
    out: list[NodeInventory] = []
    for n in base:
        g, m, c = frees.get(n.name, (n.g_free, n.m_free, n.c_free))
        out.append(
            NodeInventory(
                name=n.name,
                schedulable=n.schedulable,
                alloc_gpu=n.alloc_gpu,
                alloc_memory_bytes=n.alloc_memory_bytes,
                alloc_cpu_millicores=n.alloc_cpu_millicores,
                v_card_bytes=n.v_card_bytes,
                g_free=g,
                m_free=m,
                c_free=c,
                pod_requests_applied=n.pod_requests_applied,
                gpu_resource_key=n.gpu_resource_key,
                node_selector_json=n.node_selector_json,
                tolerations_json=n.tolerations_json,
            )
        )
    return out


def whitelist_rank(name: str, order: list[str]) -> int:
    if not order:
        return 0
    try:
        return order.index(name)
    except ValueError:
        return len(order) + hash(name) % 1000
