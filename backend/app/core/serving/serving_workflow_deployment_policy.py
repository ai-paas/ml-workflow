"""워크플로 서빙 배포 정책: deployment_type·URL, MODEL 확정 순서."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional

from config.db.enums import ModelProviderEnum
from config.settings import Settings
from core.serving.serving_mode import ServingMode
from db.models.model import Model
from db.models.model_workflow_deployment import ServingDeviceType, WorkflowServingDeploymentType
from db.models.service import ComponentType, Workflow, WorkflowComponent


def parse_remote_serving_model_map(settings: Settings) -> dict[str, str]:
    """`REMOTE_SERVING_MODEL_MAP` JSON → repo_id → 원격 서버 모델명. 잘못된 JSON은 빈 dict."""
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


@dataclass(frozen=True)
class WorkflowServingDecision:
    """MODEL 컴포넌트 하나에 대한 서빙 경로 판정 결과."""

    deployment_type: WorkflowServingDeploymentType  # reject_reason 이 있으면 의미 없음
    serving_mode: ServingMode
    remote_ready: bool
    remote_model_name: Optional[str]  # 원격 서버가 아는 모델명. 치환하지 않은 원문.
    remote_base_url: Optional[str]
    reject_reason: Optional[str]  # None 이면 배포 가능
    note: str  # 로깅·응답용 판정 근거


def _predefined_config_for(repo_id: str) -> dict[str, Any]:
    """`PREDEFINED_MODEL_CONFIGS` 조회. 순환 import 를 피하려 호출 시점에 가져온다."""
    if not repo_id:
        return {}
    from services.model import PREDEFINED_MODEL_CONFIGS

    return PREDEFINED_MODEL_CONFIGS.get(repo_id) or {}


def resolve_serving_mode_by_repo_id(repo_id: str, settings: Optional[Settings] = None) -> ServingMode:
    """repo_id 하나에 대한 서빙 모드. 잘못된 태그 문자열은 ValueError."""
    rid = (repo_id or "").strip()
    raw = str(_predefined_config_for(rid).get("serving_mode", "")).strip().lower()
    if raw:
        return ServingMode(raw)
    # PREDEFINED 미등록 커스텀 모델: 맵에 있으면 기존 동작(무조건 REMOTE)을 보존한다.
    if rid and settings is not None and rid in parse_remote_serving_model_map(settings):
        return ServingMode.REMOTE_PREFERRED
    return ServingMode.LOCAL


def resolve_serving_mode(model: Optional[Model], settings: Settings) -> ServingMode:
    rid = (model.repo_id or "").strip() if model is not None else ""
    return resolve_serving_mode_by_repo_id(rid, settings)


def _resolve_local_type(model: Optional[Model]) -> WorkflowServingDeploymentType:
    """실물 가중치 경로: provider=ollama → OLLAMA, 그 외 → KSERVE."""
    if model is not None and model.provider_info is not None:
        if model.provider_info.name.lower() == ModelProviderEnum.OLLAMA.value.lower():
            return WorkflowServingDeploymentType.OLLAMA
    return WorkflowServingDeploymentType.KSERVE


def _local_serving_ready(
    model: Optional[Model],
    local_type: WorkflowServingDeploymentType,
) -> tuple[bool, str]:
    """로컬 서빙 산출물이 실제로 있는지 DB 값으로 판정. (가능여부, 불가 사유)."""
    reg = getattr(model, "registry", None) if model is not None else None
    if reg is None:
        return False, "모델 레지스트리 정보가 없습니다"
    if local_type is WorkflowServingDeploymentType.OLLAMA:
        return bool((reg.pvc or "").strip()), "Ollama 가중치 PVC 가 없습니다"
    return bool((reg.artifact_path or "").strip()), "MLflow 아티팩트 경로가 없습니다"


def decide_workflow_serving(model: Optional[Model], settings: Settings) -> WorkflowServingDecision:
    """MODEL 컴포넌트의 서빙 경로를 판정한다. `reject_reason` 이 있으면 배포를 막아야 한다."""
    mode = resolve_serving_mode(model, settings)
    rid = (model.repo_id or "").strip() if model is not None else ""
    mmap = parse_remote_serving_model_map(settings)
    base = (settings.REMOTE_SERVING_API_URL or "").strip().rstrip("/")
    remote_ready = bool(base) and bool(rid) and rid in mmap

    if mode is ServingMode.REMOTE_ONLY and not remote_ready:
        return WorkflowServingDecision(
            deployment_type=WorkflowServingDeploymentType.KSERVE,
            serving_mode=mode,
            remote_ready=False,
            remote_model_name=None,
            remote_base_url=base or None,
            reject_reason=(
                f"'{rid}' 는 원격 서빙 전용 모델입니다. 서버 설정(REMOTE_SERVING_API_URL / "
                f"REMOTE_SERVING_MODEL_MAP)에 등록되지 않아 배포할 수 없습니다."
            ),
            note="remote_only 이나 원격 엔드포인트 미설정",
        )

    if mode in (ServingMode.REMOTE_ONLY, ServingMode.REMOTE_PREFERRED) and remote_ready:
        return WorkflowServingDecision(
            deployment_type=WorkflowServingDeploymentType.REMOTE,
            serving_mode=mode,
            remote_ready=True,
            remote_model_name=mmap[rid],
            remote_base_url=base,
            reject_reason=None,
            note=f"{mode.value} + 원격 설정 있음 → REMOTE({mmap[rid]})",
        )

    # LOCAL, 또는 REMOTE_PREFERRED 인데 원격 미설정 → 실물 가중치 경로
    local_type = _resolve_local_type(model)
    ready, why = _local_serving_ready(model, local_type)
    if not ready:
        if mode is ServingMode.REMOTE_PREFERRED:
            reason = (
                f"'{rid}' 는 원격 우선 모델이나 원격 엔드포인트가 설정되지 않았고, "
                f"로컬 폴백도 불가합니다({why}). 서버 설정을 등록하거나 가중치를 재등록하세요."
            )
        else:
            reason = f"'{rid}' 로컬 서빙 불가: {why}"
        return WorkflowServingDecision(
            deployment_type=local_type,
            serving_mode=mode,
            remote_ready=remote_ready,
            remote_model_name=None,
            remote_base_url=base or None,
            reject_reason=reason,
            note=f"{mode.value} + 로컬 산출물 없음({why})",
        )

    return WorkflowServingDecision(
        deployment_type=local_type,
        serving_mode=mode,
        remote_ready=remote_ready,
        remote_model_name=None,
        remote_base_url=base or None,
        reject_reason=None,
        note=f"{mode.value} → {local_type.value}",
    )


def resolve_workflow_serving_deployment_type(
    model: Optional[Model],
    _settings: Settings,
) -> WorkflowServingDeploymentType:
    """MODEL 컴포넌트의 deployment_type. `decide_workflow_serving` 의 얇은 래퍼."""
    return decide_workflow_serving(model, _settings).deployment_type


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
    동일 워크플로 내 MODEL 컴포넌트 확정 순서.

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
