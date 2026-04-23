"""
§3.3~3.4: Ollama 워크플로 PVC 선택·복제, WorkflowServingVolumeLock 직렬화, 복제본 삭제.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional

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
) -> None:
    from kubernetes import client

    src = core_v1.read_namespaced_persistent_volume_claim(name=source_pvc_name, namespace=namespace)
    spec = src.spec
    if spec is None:
        raise RuntimeError(f"PVC {source_pvc_name} has empty spec")

    ds = client.V1TypedLocalObjectReference(
        api_group="",
        kind="PersistentVolumeClaim",
        name=source_pvc_name,
    )
    new_spec = client.V1PersistentVolumeClaimSpec(
        access_modes=list(spec.access_modes or ["ReadWriteOnce"]),
        resources=spec.resources,
        storage_class_name=spec.storage_class_name,
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


def resolve_ollama_pvc_for_execute(
    db: "Session",
    *,
    model_id: int,
    registry_pvc: Optional[str],
    serving_node_name: str,
    rollback_clone_pvcs: List[str],
    vol_lock: Optional[WorkflowServingVolumeLock] = None,
) -> str:
    """§3.3: 이번 Ollama 배포에 쓸 PVC 이름. 복제 시 vol_lock으로 직렬화."""
    lock = vol_lock or get_workflow_serving_volume_lock()

    if not registry_pvc:
        return ""

    if not serving_node_name:
        logger.warning(
            "serving_node_name 비어 있음 — §3.3 노드 분기 생략, 레지스트리 원본 PVC 사용 (model_id=%s)", model_id
        )
        return registry_pvc

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

        new_name = _clone_pvc_name(model_id, serving_node_name)
        ns = get_settings().KUBEFLOW_NAMESPACE
        core = _load_core_v1()
        _create_pvc_clone_from_source(core, ns, registry_pvc, new_name, model_id, serving_node_name)
        rollback_clone_pvcs.append(new_name)
        return new_name
    finally:
        lock.release(lk)


def _clone_registry_pvc_for_node(
    *,
    model_id: int,
    serving_node_name: str,
    registry_pvc: str,
    rollback_clone_pvcs: List[str],
    vol_lock: WorkflowServingVolumeLock,
) -> str:
    """레지스트리 원본 PVC를 소스로 해당 노드 전용 클론 PVC 생성(§3.3.1 락)."""
    lk = serving_volume_lock_key(model_id, serving_node_name)
    if not vol_lock.try_acquire_nonblocking(lk):
        raise PvcReplicationConflict(
            "동일 model·serving_node에 대해 복제 PVC 생성 또는 삭제가 진행 중입니다. 잠시 후 다시 시도해 주세요."
        )
    try:
        new_name = _clone_pvc_name(model_id, serving_node_name)
        ns = get_settings().KUBEFLOW_NAMESPACE
        core = _load_core_v1()
        _create_pvc_clone_from_source(core, ns, registry_pvc, new_name, model_id, serving_node_name)
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
