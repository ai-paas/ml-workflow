"""워크플로 서빙 배포 정책: §2.5·§2.6 deployment_type·URL, §7.3.2 MODEL 확정 순서."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any, Optional

from config.db.enums import ModelProviderEnum
from config.settings import Settings
from db.models.model import Model
from db.models.model_workflow_deployment import ServingDeviceType, WorkflowServingDeploymentType
from db.models.service import ComponentType, Workflow, WorkflowComponent


def parse_remote_serving_model_map(settings: Settings) -> dict[str, str]:
    """§6: `REMOTE_SERVING_MODEL_MAP` JSON → repo_id → 원격 서버 모델명. 잘못된 JSON은 빈 dict."""
    raw = (settings.REMOTE_SERVING_MODEL_MAP or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        ks, vs = str(k).strip(), str(v).strip()
        if ks and vs:
            out[ks] = vs
    return out


def resolve_workflow_serving_deployment_type(
    model: Optional[Model],
    _settings: Settings,
) -> WorkflowServingDeploymentType:
    """MODEL 컴포넌트의 deployment_type (§2.5).

    우선순위: §6 REMOTE(`REMOTE_SERVING_MODEL_MAP`에 repo_id 존재) → Ollama → KServe.
    """
    rid = (model.repo_id or "").strip() if model is not None else ""
    if rid:
        if rid in parse_remote_serving_model_map(_settings):
            return WorkflowServingDeploymentType.REMOTE
    if model is not None and model.provider_info is not None:
        if model.provider_info.name.lower() == ModelProviderEnum.OLLAMA.value.lower():
            return WorkflowServingDeploymentType.OLLAMA
    return WorkflowServingDeploymentType.KSERVE


def parse_serving_device_type(plan: Optional[dict[str, Any]]) -> Optional[ServingDeviceType]:
    if not plan:
        return None
    raw = plan.get("device_type")
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if s == "GPU":
        return ServingDeviceType.GPU
    if s == "CPU":
        return ServingDeviceType.CPU
    return None


def kserve_public_infer_url(gateway_base: str, model_name: str) -> Optional[str]:
    g = (gateway_base or "").strip().rstrip("/")
    if not g:
        return None
    mn = (model_name or "").strip()
    if not mn:
        return None
    return f"{g}/v2/models/{mn}/infer"


def backend_api_url_from_internal(internal_url: Optional[str]) -> Optional[str]:
    u = (internal_url or "").strip()
    return u if u else None


def topological_order_model_components(workflow: Workflow) -> list[WorkflowComponent]:
    """
    §7.3.2: 동일 워크플로 내 MODEL 컴포넌트 확정 순서.

    START/END를 제외한 컴포넌트 그래프로 위상 정렬 후, MODEL만 순서를 추출한다.
    순환 시 설계서대로 MODEL은 component.id 사전순으로 처리한다.

    MODEL + model_id 있는 컴포넌트만, 워크플로 의존성(비-MODEL 경유 간선 포함) 순서.
    동일 깊이(동시 ready)는 component.id 사전순.
    """
    models = [c for c in workflow.components if c.type == ComponentType.MODEL and c.model_id]
    if not models:
        return []
    if len(models) == 1:
        return models

    skip = {ComponentType.START, ComponentType.END}
    comps = [c for c in workflow.components if c.type not in skip]
    comp_ids = {c.id for c in comps}

    out_edges: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {c.id: 0 for c in comps}

    for conn in workflow.component_connections:
        s, t = conn.source_component_id, conn.target_component_id
        if s in comp_ids and t in comp_ids:
            out_edges[s].append(t)
            in_degree[t] += 1

    ready = deque(sorted([c for c in comps if in_degree[c.id] == 0], key=lambda x: x.id))
    order_all: list[WorkflowComponent] = []

    while ready:
        u = ready.popleft()
        order_all.append(u)
        newly: list[WorkflowComponent] = []
        for v in sorted(out_edges[u.id]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                vc = next(c for c in comps if c.id == v)
                newly.append(vc)
        newly.sort(key=lambda x: x.id)
        ready.extend(newly)

    if len(order_all) != len(comps):
        return sorted(models, key=lambda x: x.id)

    pos = {c.id: i for i, c in enumerate(order_all)}
    return sorted(models, key=lambda m: (pos[m.id], m.id))
