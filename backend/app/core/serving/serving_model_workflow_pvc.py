"""
§3.3~3.4: Ollama 워크플로 PVC 선택·복제, WorkflowServingVolumeLock 직렬화, 복제본 삭제.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from config.settings import get_settings
from core.serving.workflow_serving_volume_lock import (
    WorkflowServingVolumeLock,
    get_workflow_serving_volume_lock,
    serving_volume_lock_key,
)

if TYPE_CHECKING:
    from db.models.service import Workflow
    from kubernetes.client import CoreV1Api
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PvcReplicationConflict(Exception):
    """§3.3.1: 동일 (model_id, serving_node)에서 복제·삭제 직렬화 충돌."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _k8s_sanitize_segment(s: str, max_len: int = 40) -> str:
    x = re.sub(r"[^a-z0-9-]+", "-", (s or "").lower())
    x = re.sub(r"-+", "-", x).strip("-")
    return (x or "node")[:max_len]


def _clone_pvc_name(model_id: int, node: str) -> str:
    suf = uuid.uuid4().hex[:8]
    node_part = _k8s_sanitize_segment(node, 24)
    base = f"ollama-mw-m{model_id}-{node_part}-{suf}"
    if len(base) > 253:
        base = base[:253].rstrip("-")
    return base


def _load_core_v1():
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.CoreV1Api()


def _load_storage_v1():
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.StorageV1Api()


def _cinder_zone_sc_map(settings: Any) -> Dict[str, str]:
    raw = (settings.CINDER_ZONE_TO_STORAGE_CLASS_JSON or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("CINDER_ZONE_TO_STORAGE_CLASS_JSON 파싱 실패, 빈 맵 사용")
        return {}


def _get_node_topology_zone(core_v1: "CoreV1Api", node_name: str, label_key: str) -> Optional[str]:
    try:
        node = core_v1.read_node(name=node_name)
    except Exception as e:
        logger.warning("노드 조회 실패 (zone 생략): %s %s", node_name, e)
        return None
    labels = (node.metadata.labels or {}) if node and node.metadata else {}
    z = labels.get(label_key)
    return str(z).strip() if z is not None and str(z).strip() else None


def _get_storage_class_availability(storage_v1: Any, sc_name: str) -> Optional[str]:
    try:
        sc = storage_v1.read_storage_class(name=sc_name)
    except Exception as e:
        logger.warning("StorageClass 조회 실패: %s %s", sc_name, e)
        return None
    params = sc.parameters or {}
    if not isinstance(params, dict):
        return None
    av = params.get("availability")
    return str(av).strip() if av is not None and str(av).strip() else None


def _list_cinder_sc_availability_pairs(provisioner: str) -> List[Tuple[str, str]]:
    """(storageClassName, parameters.availability) — provisioner 일치하는 SC만."""
    storage_v1 = _load_storage_v1()
    out: List[Tuple[str, str]] = []
    try:
        lst = storage_v1.list_storage_class()
    except Exception as e:
        logger.warning("StorageClass 목록 조회 실패: %s", e)
        return []
    for sc in lst.items or []:
        if (sc.provisioner or "") != provisioner:
            continue
        name = sc.metadata.name if sc.metadata else None
        if not name:
            continue
        params = sc.parameters or {}
        if not isinstance(params, dict):
            continue
        av = params.get("availability")
        if av is None or not str(av).strip():
            continue
        out.append((name, str(av).strip()))
    return out


def _select_storage_class_name_for_cinder_zone(node_zone: str) -> str:
    """§12.3.4: node_zone 과 parameters.availability 가 일치하는 Cinder SC 이름. 없거나 모호하면 ValueError."""
    from config.settings import get_settings

    s = get_settings()
    zone_map = _cinder_zone_sc_map(s)
    if node_zone in zone_map:
        want = zone_map[node_zone]
        storage_v1 = _load_storage_v1()
        try:
            sc = storage_v1.read_storage_class(name=want)
        except Exception as e:
            raise ValueError(
                f"Cinder zone 정합성: CINDER_ZONE_TO_STORAGE_CLASS_JSON 의 SC '{want}' 를 조회할 수 없습니다: {e}"
            ) from e
        if (sc.provisioner or "") != s.CINDER_CSI_PROVISIONER:
            raise ValueError(
                f"Cinder zone 정합성: SC '{want}' 의 provisioner 가 기대와 다릅니다 ({s.CINDER_CSI_PROVISIONER})."
            )
        params = sc.parameters or {}
        av = params.get("availability") if isinstance(params, dict) else None
        if not av or str(av).strip() != node_zone:
            raise ValueError(
                f"Cinder zone 정합성: SC '{want}' 의 availability({av!r}) 가 node_zone({node_zone!r}) 과 일치하지 않습니다."
            )
        return want

    pairs = _list_cinder_sc_availability_pairs(s.CINDER_CSI_PROVISIONER)
    cands = [name for name, av in pairs if av == node_zone]
    if len(cands) == 0:
        raise ValueError(
            f"Cinder zone 정합성: node_zone='{node_zone}' 에 해당하는 StorageClass가 없습니다 "
            f"(provisioner={s.CINDER_CSI_PROVISIONER}, parameters.availability 일치). "
            f"클러스터에 해당 availability 의 SC를 추가하거나 CINDER_ZONE_TO_STORAGE_CLASS_JSON 을 설정하세요."
        )
    if len(cands) == 1:
        return cands[0]
    raise ValueError(
        f"Cinder zone 정합성: availability='{node_zone}' 인 StorageClass가 복수입니다: {cands}. "
        f"CINDER_ZONE_TO_STORAGE_CLASS_JSON 에 해당 zone 키로 하나만 지정하세요."
    )


def _registry_pvc_storage_class_and_zone(
    core_v1: "CoreV1Api", namespace: str, registry_pvc: str
) -> Tuple[Optional[str], Optional[str]]:
    """(storageClassName, SC parameters.availability) — PVC spec SC 가 없으면 (None, None)."""
    try:
        pvc = core_v1.read_namespaced_persistent_volume_claim(name=registry_pvc, namespace=namespace)
    except Exception as e:
        logger.warning("레지스트리 PVC 조회 실패: %s %s", registry_pvc, e)
        return None, None
    scn = None
    if pvc.spec and pvc.spec.storage_class_name:
        scn = pvc.spec.storage_class_name.strip() or None
    if not scn:
        return None, None
    storage_v1 = _load_storage_v1()
    av = _get_storage_class_availability(storage_v1, scn)
    return scn, av


def _wait_pvc_bound(
    core_v1: "CoreV1Api",
    namespace: str,
    pvc_name: str,
    *,
    timeout_sec: int = 600,
    interval_sec: int = 5,
) -> None:
    """Longhorn 등 dataSource 클론이 Pending→Bound 될 때까지 대기. 파이프라인보다 앞서 볼륨이 준비되도록 함."""
    from kubernetes import client as k8s_client

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            pvc = core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            phase = (pvc.status.phase or "") if pvc.status else ""
            if phase == "Bound":
                logger.info("PVC %s is Bound", pvc_name)
                return
            if phase == "Lost":
                raise RuntimeError(f"PVC {pvc_name} phase=Lost, 볼륨 바인딩 실패")
            logger.info("PVC %s waiting for Bound (phase=%s)", pvc_name, phase or "?")
        except k8s_client.exceptions.ApiException as e:
            if e.status != 404:
                raise
            logger.info("PVC %s not found yet, waiting…", pvc_name)
        time.sleep(interval_sec)

    raise RuntimeError(
        f"PVC {pvc_name} 가 {timeout_sec}s 내에 Bound 되지 않았습니다. "
        "Longhorn 클론 지연·스토리지 용량·소스 PVC 상태를 확인하세요."
    )


def _create_pvc_clone_from_source(
    core_v1: "CoreV1Api",
    namespace: str,
    source_pvc_name: str,
    new_pvc_name: str,
    model_id: int,
    serving_node_name: str,
    storage_class_name_override: Optional[str] = None,
) -> None:
    from kubernetes import client

    src = core_v1.read_namespaced_persistent_volume_claim(name=source_pvc_name, namespace=namespace)
    spec = src.spec
    if spec is None:
        raise RuntimeError(f"PVC {source_pvc_name} has empty spec")

    sc_name = storage_class_name_override if storage_class_name_override else spec.storage_class_name

    ds = client.V1TypedLocalObjectReference(
        api_group="",
        kind="PersistentVolumeClaim",
        name=source_pvc_name,
    )
    new_spec = client.V1PersistentVolumeClaimSpec(
        access_modes=list(spec.access_modes or ["ReadWriteOnce"]),
        resources=spec.resources,
        storage_class_name=sc_name,
        volume_mode=spec.volume_mode,
        data_source=ds,
    )
    meta = client.V1ObjectMeta(
        name=new_pvc_name,
        namespace=namespace,
        labels={
            "app": "ollama-workflow",
            "ml-workflow/ollama-workflow-clone": "true",
            "ml-workflow/source-model-id": str(model_id),
            "ml-workflow/serving-node": _k8s_sanitize_segment(serving_node_name, 60),
            "ml-workflow/clone-of": source_pvc_name[:60],
        },
    )
    body = client.V1PersistentVolumeClaim(metadata=meta, spec=new_spec)
    try:
        core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=body)
        logger.info("Created workflow Ollama PVC clone %s from %s", new_pvc_name, source_pvc_name)
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("PVC clone %s already exists", new_pvc_name)
        else:
            raise
    _wait_pvc_bound(core_v1, namespace, new_pvc_name)


def _delete_pvc_if_exists(core_v1: "CoreV1Api", namespace: str, pvc_name: str) -> None:
    try:
        core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        logger.info("Deleted workflow replica PVC %s", pvc_name)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "404" in err:
            return
        logger.warning("PVC delete %s: %s", pvc_name, e)


def _any_live_uses_original_on_other_node(
    live_rows: List[Any],
    registry_pvc: str,
    target_node: str,
) -> bool:
    for d in live_rows:
        if not d.serving_node_name or not d.pvc:
            continue
        if d.pvc == registry_pvc and d.serving_node_name != target_node:
            return True
    return False


def _find_reuse_pvc_same_model_node(live_rows: List[Any], target_node: str) -> Optional[str]:
    """live_rows는 이미 해당 model_id로 필터된 배포 행."""
    for d in live_rows:
        if (d.serving_node_name or "") != target_node:
            continue
        if d.pvc:
            return d.pvc
    return None


def _live_deployments_for_model(db: Any, model_id: int) -> Any:
    """workflow_components 조인으로 model_id 기준 실사용 배포 행 조회."""
    from db.models.model_workflow_deployment import DeploymentStatus, ModelWorkflowDeployment
    from db.models.service import WorkflowComponent

    return (
        db.query(ModelWorkflowDeployment)
        .join(
            WorkflowComponent,
            (ModelWorkflowDeployment.workflow_id == WorkflowComponent.workflow_id)
            & (ModelWorkflowDeployment.component_id == WorkflowComponent.id),
        )
        .filter(
            WorkflowComponent.model_id == model_id,
            ModelWorkflowDeployment.status.in_([DeploymentStatus.DEPLOYING, DeploymentStatus.DEPLOYED]),
        )
    )


def _clone_and_track(
    model_id: int,
    serving_node_name: str,
    registry_pvc: str,
    rollback_clone_pvcs: List[str],
    storage_class_name_override: Optional[str] = None,
) -> str:
    new_name = _clone_pvc_name(model_id, serving_node_name)
    ns = get_settings().KUBEFLOW_NAMESPACE
    core = _load_core_v1()
    _create_pvc_clone_from_source(
        core,
        ns,
        registry_pvc,
        new_name,
        model_id,
        serving_node_name,
        storage_class_name_override=storage_class_name_override,
    )
    rollback_clone_pvcs.append(new_name)
    return new_name


def _resolve_ollama_pvc_execute_legacy(
    db: "Session",
    *,
    model_id: int,
    registry_pvc: str,
    serving_node_name: str,
    rollback_clone_pvcs: List[str],
    vol_lock: Optional[WorkflowServingVolumeLock] = None,
) -> str:
    """§3.3: Cinder zone 매칭 비활성 시 동작."""
    lock = vol_lock or get_workflow_serving_volume_lock()

    live_rows = _live_deployments_for_model(db, model_id).all()

    reuse = _find_reuse_pvc_same_model_node(live_rows, serving_node_name)
    if reuse:
        return reuse

    if not live_rows:
        return registry_pvc

    if not _any_live_uses_original_on_other_node(live_rows, registry_pvc, serving_node_name):
        return registry_pvc

    lk = serving_volume_lock_key(model_id, serving_node_name)
    if not lock.try_acquire_nonblocking(lk):
        raise PvcReplicationConflict(
            "동일 model·serving_node에 대해 복제 PVC 생성 또는 삭제가 진행 중입니다. 잠시 후 다시 시도해 주세요."
        )

    try:
        db.expire_all()
        live_rows2 = _live_deployments_for_model(db, model_id).all()
        reuse2 = _find_reuse_pvc_same_model_node(live_rows2, serving_node_name)
        if reuse2:
            return reuse2
        if not live_rows2:
            return registry_pvc
        if not _any_live_uses_original_on_other_node(live_rows2, registry_pvc, serving_node_name):
            return registry_pvc

        return _clone_and_track(
            model_id, serving_node_name, registry_pvc, rollback_clone_pvcs, storage_class_name_override=None
        )
    finally:
        lock.release(lk)


def resolve_ollama_pvc_for_execute(
    db: "Session",
    *,
    model_id: int,
    registry_pvc: Optional[str],
    serving_node_name: str,
    rollback_clone_pvcs: List[str],
    vol_lock: Optional[WorkflowServingVolumeLock] = None,
) -> str:
    """§3.3 / §12: 이번 Ollama 배포에 쓸 PVC 이름. 복제 시 vol_lock으로 직렬화."""
    lock = vol_lock or get_workflow_serving_volume_lock()
    s = get_settings()

    if not registry_pvc:
        return ""

    if not serving_node_name:
        logger.warning(
            "serving_node_name 비어 있음 — §3.3 노드 분기 생략, 레지스트리 원본 PVC 사용 (model_id=%s)", model_id
        )
        return registry_pvc

    if not s.CINDER_ZONE_MATCH_ENABLED:
        return _resolve_ollama_pvc_execute_legacy(
            db,
            model_id=model_id,
            registry_pvc=registry_pvc,
            serving_node_name=serving_node_name,
            rollback_clone_pvcs=rollback_clone_pvcs,
            vol_lock=lock,
        )

    core = _load_core_v1()
    ns = s.KUBEFLOW_NAMESPACE
    node_zone = _get_node_topology_zone(core, serving_node_name, s.CINDER_NODE_TOPOLOGY_KEY)
    _, reg_av = _registry_pvc_storage_class_and_zone(core, ns, registry_pvc)
    cinder_data_ok = bool(node_zone and reg_av is not None)

    live_rows = _live_deployments_for_model(db, model_id).all()

    if cinder_data_ok and node_zone is not None and reg_av is not None:
        if node_zone == reg_av:
            if not _any_live_uses_original_on_other_node(live_rows, registry_pvc, serving_node_name):
                return registry_pvc
            lk = serving_volume_lock_key(model_id, serving_node_name)
            if not lock.try_acquire_nonblocking(lk):
                raise PvcReplicationConflict(
                    "동일 model·serving_node에 대해 복제 PVC 생성 또는 삭제가 진행 중입니다. 잠시 후 다시 시도해 주세요."
                )
            try:
                db.expire_all()
                live_rows2 = _live_deployments_for_model(db, model_id).all()
                if not _any_live_uses_original_on_other_node(live_rows2, registry_pvc, serving_node_name):
                    return registry_pvc
                reuse2 = _find_reuse_pvc_same_model_node(live_rows2, serving_node_name)
                if reuse2:
                    return reuse2
                return _clone_and_track(
                    model_id, serving_node_name, registry_pvc, rollback_clone_pvcs, storage_class_name_override=None
                )
            finally:
                lock.release(lk)

        reuse = _find_reuse_pvc_same_model_node(live_rows, serving_node_name)
        if reuse:
            return reuse

        lk2 = serving_volume_lock_key(model_id, serving_node_name)
        if not lock.try_acquire_nonblocking(lk2):
            raise PvcReplicationConflict(
                "동일 model·serving_node에 대해 복제 PVC 생성 또는 삭제가 진행 중입니다. 잠시 후 다시 시도해 주세요."
            )
        try:
            db.expire_all()
            live_rows2 = _live_deployments_for_model(db, model_id).all()
            reuse2 = _find_reuse_pvc_same_model_node(live_rows2, serving_node_name)
            if reuse2:
                return reuse2
            target_sc = _select_storage_class_name_for_cinder_zone(node_zone)
            return _clone_and_track(
                model_id,
                serving_node_name,
                registry_pvc,
                rollback_clone_pvcs,
                storage_class_name_override=target_sc,
            )
        finally:
            lock.release(lk2)

    if not cinder_data_ok:
        logger.info(
            "Cinder zone: 노드 label 또는 레지스트리 SC availability 미확인 — §3.3 legacy 경로 (model_id=%s)",
            model_id,
        )
    return _resolve_ollama_pvc_execute_legacy(
        db,
        model_id=model_id,
        registry_pvc=registry_pvc,
        serving_node_name=serving_node_name,
        rollback_clone_pvcs=rollback_clone_pvcs,
        vol_lock=lock,
    )


def _cinder_sc_override_for_clone(
    serving_node_name: str,
    registry_pvc: str,
) -> Optional[str]:
    """§12: 존 불일치 시에만 target StorageClass, 일치·비활성·미조회 시 None. 매칭 SC 없으면 ValueError."""
    s = get_settings()
    if not s.CINDER_ZONE_MATCH_ENABLED:
        return None
    try:
        core = _load_core_v1()
        ns = s.KUBEFLOW_NAMESPACE
        nz = _get_node_topology_zone(core, serving_node_name, s.CINDER_NODE_TOPOLOGY_KEY)
        _, rav = _registry_pvc_storage_class_and_zone(core, ns, registry_pvc)
    except Exception as e:
        logger.warning("Cinder zone 조회 실패(클론 SC는 소스 기준 상속): %s", e)
        return None
    if not nz or rav is None or nz == rav:
        return None
    return _select_storage_class_name_for_cinder_zone(nz)


def _clone_registry_pvc_for_node(
    *,
    model_id: int,
    serving_node_name: str,
    registry_pvc: str,
    rollback_clone_pvcs: List[str],
    vol_lock: WorkflowServingVolumeLock,
) -> str:
    """레지스트리 원본 PVC를 소스로 해당 노드 전용 클론 PVC 생성(§3.3.1 락)."""
    sc_override = _cinder_sc_override_for_clone(serving_node_name, registry_pvc)

    lk = serving_volume_lock_key(model_id, serving_node_name)
    if not vol_lock.try_acquire_nonblocking(lk):
        raise PvcReplicationConflict(
            "동일 model·serving_node에 대해 복제 PVC 생성 또는 삭제가 진행 중입니다. 잠시 후 다시 시도해 주세요."
        )
    try:
        new_name = _clone_pvc_name(model_id, serving_node_name)
        ns = get_settings().KUBEFLOW_NAMESPACE
        core = _load_core_v1()
        _create_pvc_clone_from_source(
            core,
            ns,
            registry_pvc,
            new_name,
            model_id,
            serving_node_name,
            storage_class_name_override=sc_override,
        )
        rollback_clone_pvcs.append(new_name)
        return new_name
    finally:
        vol_lock.release(lk)


def _distinct_ollama_serving_nodes_by_model_id(
    db: "Session", workflow: "Workflow", parameters: dict[str, Any]
) -> dict[int, set[str]]:
    """이번 워크플로 Ollama(GGUF) MODEL 컴포넌트별 확정 노드 → model_id별 서로 다른 노드 집합."""
    from config.db.enums import ModelFormatEnum, ModelProviderEnum
    from db.models.model import Model
    from db.models.service import ComponentType
    from sqlalchemy.orm import joinedload

    plans = parameters.get("serving_plans") or {}
    out: dict[int, set[str]] = {}
    for c in workflow.components:
        if c.type != ComponentType.MODEL or not c.model_id:
            continue
        model = (
            db.query(Model)
            .options(joinedload(Model.provider_info), joinedload(Model.format_info), joinedload(Model.registry))
            .filter(Model.id == c.model_id)
            .first()
        )
        if not model or not model.registry or not model.registry.pvc:
            continue
        is_ollama = (
            getattr(model, "provider_info", None)
            and model.provider_info.name.lower() == ModelProviderEnum.OLLAMA.value.lower()
            and getattr(model, "format_info", None)
            and model.format_info.name.lower() == ModelFormatEnum.GGUF.value.lower()
        )
        if not is_ollama:
            continue
        plan = plans.get(c.id) or {}
        node = (plan.get("serving_node_name") or parameters.get("node_name") or "").strip()
        if not node:
            continue
        out.setdefault(model.id, set()).add(node)
    return out


def prepare_ollama_pvc_for_workflow_models(db: "Session", workflow: "Workflow", parameters: dict[str, Any]) -> None:
    """parameters['serving_plans'][component_id]['resolved_ollama_pvc'] 설정."""
    from config.db.enums import ModelFormatEnum, ModelProviderEnum
    from db.models.model import Model
    from db.models.service import ComponentType
    from sqlalchemy.orm import joinedload

    rollback_list: List[str] = []
    parameters["_ollama_clone_pvcs_rollback"] = rollback_list
    resolve_cache: dict[tuple[int, str], str] = {}
    parameters["_ollama_pvc_resolve_cache"] = resolve_cache

    plans: dict[str, dict[str, Any]] = parameters.setdefault("serving_plans", {})
    vol_lock = get_workflow_serving_volume_lock()

    nodes_by_model = _distinct_ollama_serving_nodes_by_model_id(db, workflow, parameters)

    for component in workflow.components:
        if component.type != ComponentType.MODEL or not component.model_id:
            continue
        model = (
            db.query(Model)
            .options(joinedload(Model.provider_info), joinedload(Model.format_info), joinedload(Model.registry))
            .filter(Model.id == component.model_id)
            .first()
        )
        if not model or not model.registry or not model.registry.pvc:
            continue
        is_ollama = (
            getattr(model, "provider_info", None)
            and model.provider_info.name.lower() == ModelProviderEnum.OLLAMA.value.lower()
            and getattr(model, "format_info", None)
            and model.format_info.name.lower() == ModelFormatEnum.GGUF.value.lower()
        )
        if not is_ollama:
            continue

        plan = plans.get(component.id) or {}
        node = (plan.get("serving_node_name") or parameters.get("node_name") or "").strip()
        reg_pvc = model.registry.pvc
        cache_key = (model.id, node)
        if cache_key in resolve_cache:
            resolved = resolve_cache[cache_key]
        else:
            resolved = resolve_ollama_pvc_for_execute(
                db,
                model_id=model.id,
                registry_pvc=reg_pvc,
                serving_node_name=node,
                rollback_clone_pvcs=rollback_list,
                vol_lock=vol_lock,
            )
            resolve_cache[cache_key] = resolved

        # 동일 워크플로·동일 모델이 서로 다른 노드: RWO 원본은 한 노드만. DB에 아직 행이 없을 때 둘 다 원본이 되는 것을 막음.
        multi_nodes = nodes_by_model.get(model.id) or set()
        if len(multi_nodes) >= 2 and reg_pvc:
            primary_node = min(multi_nodes)
            if node != primary_node and resolved == reg_pvc:
                resolved = _clone_registry_pvc_for_node(
                    model_id=model.id,
                    serving_node_name=node,
                    registry_pvc=reg_pvc,
                    rollback_clone_pvcs=rollback_list,
                    vol_lock=vol_lock,
                )
                resolve_cache[cache_key] = resolved

        plan = dict(plan)
        plan["resolved_ollama_pvc"] = resolved
        plans[component.id] = plan


def rollback_new_clone_pvcs(pvc_names: List[str]) -> None:
    if not pvc_names:
        return
    ns = get_settings().KUBEFLOW_NAMESPACE
    try:
        core = _load_core_v1()
    except Exception as e:
        logger.warning("rollback clone pvcs: k8s client unavailable: %s", e)
        return
    for name in pvc_names:
        try:
            _delete_pvc_if_exists(core, ns, name)
        except Exception as e:
            logger.warning("rollback delete %s: %s", name, e)


def delete_replica_pvc_if_last_consumer(db: "Session", deployment_id: str) -> None:
    """§3.4·§3.4.1: Ollama 복제 PVC, 마지막 소비자일 때만 K8s 삭제.

    DEPLOYING/DEPLOYED뿐 아니라 FAILED도 처리한다(실패 전·후에 생성된 복제 PVC 정리).
    """
    from datetime import datetime

    from db.models.model import Model
    from db.models.model_workflow_deployment import (
        DeploymentStatus,
        ModelWorkflowDeployment,
        WorkflowServingDeploymentType,
    )
    from db.models.service import WorkflowComponent

    lock = get_workflow_serving_volume_lock()
    dep = db.get(ModelWorkflowDeployment, deployment_id)
    # FAILED도 prepare 단계에서 만든 Ollama 복제 PVC가 남을 수 있어 워크플로 삭제 시 정리 대상에 포함한다.
    if not dep or dep.status not in (
        DeploymentStatus.DEPLOYING,
        DeploymentStatus.DEPLOYED,
        DeploymentStatus.FAILED,
    ):
        return
    if dep.deployment_type != WorkflowServingDeploymentType.OLLAMA:
        return
    comp = (
        db.query(WorkflowComponent)
        .filter(
            WorkflowComponent.workflow_id == dep.workflow_id,
            WorkflowComponent.id == dep.component_id,
        )
        .first()
    )
    mid = comp.model_id if comp else None
    node = (dep.serving_node_name or "").strip()
    pvc_used = (dep.pvc or "").strip()
    if not mid or not node or not pvc_used:
        return

    registry_original_pvc: Optional[str] = None
    m = db.get(Model, mid)
    if m and m.registry:
        registry_original_pvc = m.registry.pvc
    if registry_original_pvc and pvc_used == registry_original_pvc:
        dep.status = DeploymentStatus.DELETED
        dep.deleted_at = datetime.utcnow()
        db.commit()
        return

    lk = serving_volume_lock_key(mid, node)
    if not lock.acquire_blocking(lk, timeout_sec=30.0):
        logger.warning("delete PVC: could not acquire lock model_id=%s node=%s", mid, node)
        dep.status = DeploymentStatus.DELETED
        dep.deleted_at = datetime.utcnow()
        db.commit()
        return

    try:
        dep.status = DeploymentStatus.DELETED
        dep.deleted_at = datetime.utcnow()
        db.commit()

        cnt = _live_deployments_for_model(db, mid).filter(ModelWorkflowDeployment.serving_node_name == node).count()

        is_replica = bool(pvc_used) and (not registry_original_pvc or pvc_used != registry_original_pvc)
        if cnt == 0 and is_replica:
            try:
                core = _load_core_v1()
                _delete_pvc_if_exists(core, get_settings().KUBEFLOW_NAMESPACE, pvc_used)
            except Exception as e:
                logger.warning("replica PVC delete skipped: %s", e)
    finally:
        lock.release(lk)


def process_deployments_before_hard_delete(db: "Session", workflow_id: str) -> None:
    from db.models.model_workflow_deployment import ModelWorkflowDeployment

    rows = db.query(ModelWorkflowDeployment).filter(ModelWorkflowDeployment.workflow_id == workflow_id).all()
    for dep in sorted(rows, key=lambda r: (r.workflow_id, r.component_id, r.id)):
        delete_replica_pvc_if_last_consumer(db, dep.id)
