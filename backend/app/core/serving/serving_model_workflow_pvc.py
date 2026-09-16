"""
Ollama 워크플로 PVC 선택·복제, WorkflowServingVolumeLock 직렬화, 복제본 삭제.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from config.settings import get_settings
from core.serving.workflow_serving_volume_lock import (
    WorkflowServingVolumeLock,
    get_workflow_serving_volume_lock,
    serving_volume_lock_key,
)

if TYPE_CHECKING:
    from db.models.service import Workflow
    from kubernetes.client import CoreV1Api
    from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# 복사 Job 이 데이터를 모두 옮긴 뒤에만 남기는 파일. 존재하지 않으면 복제가 중단된 것으로 본다.
PVC_COPY_COMPLETE_MARKER = ".mlw-copy-complete"

# 복제본이 어떤 방식으로 채워졌는지 구분한다. Job 복사본만 완료 마커를 갖는다.
CLONE_LABEL_COPY_MODE = "ml-workflow/copy-mode"
COPY_MODE_JOB = "job"
COPY_MODE_CSI = "csi"

# 복사가 끝났음을 PVC 자체에 남기는 표시. 복사 Job 은 TTL 로 사라지므로 Job 상태만으로는
# "끝났는지" 와 "실패한 뒤 정리되었는지" 를 구분할 수 없다. 이 annotation 이 유일한 완료 근거다.
CLONE_ANNOTATION_COPY_COMPLETED = "ml-workflow/copy-completed"


class PvcReplicationConflict(Exception):
    """동일 (model_id, serving_node)에서 복제·삭제 직렬화 충돌."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ResolvedOllamaPvc(NamedTuple):
    """이번 배포에 쓸 PVC 와, 서빙 전에 파이프라인이 확인해야 할 복사 상태."""

    pvc_name: str
    # 복사 Job 이 아직 남아 있으면 그 이름. 완료 후 정리되었거나 CSI 클론이면 빈 문자열.
    copy_job_name: str = ""
    # Job 으로 채우는 복제본이면 True. 서빙 Pod 은 완료 마커를 확인한 뒤 떠야 한다.
    copy_marker_required: bool = False


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


def _load_batch_v1():
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.BatchV1Api()


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
    """node_zone 과 parameters.availability 가 일치하는 Cinder SC 이름. 없거나 모호하면 ValueError."""
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


def _is_nfs_fallback_sc(sc_name: Optional[str]) -> bool:
    """주어진 StorageClass 가 NFS fallback 으로 설정된 SC 인지 판별."""
    if not sc_name:
        return False
    s = get_settings()
    if not s.CINDER_ZONE_MISMATCH_USE_NFS_FALLBACK_ENABLED:
        return False
    fb = (s.CINDER_ZONE_MISMATCH_FALLBACK_STORAGE_CLASS or "").strip()
    return bool(fb and fb == sc_name.strip())


def _resolve_clone_storage_class_when_cinder_zone_mismatch(node_zone: str) -> str:
    """node_zone≠원본 availability 일 때 클론용 SC. 임시 NFS fallback 또는 Cinder zone 매칭."""
    from config.settings import get_settings

    s = get_settings()
    if s.CINDER_ZONE_MISMATCH_USE_NFS_FALLBACK_ENABLED:
        scn = (s.CINDER_ZONE_MISMATCH_FALLBACK_STORAGE_CLASS or "").strip()
        if not scn:
            raise ValueError(
                "Cinder zone 불일치 NFS fallback 이 활성화되어 있으나 "
                "CINDER_ZONE_MISMATCH_FALLBACK_STORAGE_CLASS 가 비어 있습니다."
            )
        storage_v1 = _load_storage_v1()
        try:
            storage_v1.read_storage_class(name=scn)
        except Exception as e:
            raise ValueError(f"Cinder zone 불일치: fallback StorageClass '{scn}' 을(를) 조회할 수 없습니다: {e}") from e
        logger.info(
            "Cinder zone 불일치 클론: Cinder zone 매칭 SC 대신 임시 fallback SC '%s' 사용 (node_zone=%s)",
            scn,
            node_zone,
        )
        return scn
    return _select_storage_class_name_for_cinder_zone(node_zone)


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


def _find_node_attached_to_pvc(core_v1: "CoreV1Api", namespace: str, pvc_name: str) -> Optional[str]:
    """PVC 를 사용 중인 Pod 의 nodeName. 없으면 None (어느 노드든 스케줄 가능)."""
    try:
        pods = core_v1.list_namespaced_pod(namespace=namespace).items or []
    except Exception as e:
        logger.warning("Pod 목록 조회 실패 (PVC attach 노드 탐지 생략): %s %s", pvc_name, e)
        return None
    for pod in pods:
        if not pod.spec or not pod.spec.volumes:
            continue
        for v in pod.spec.volumes:
            pvc_ref = getattr(v, "persistent_volume_claim", None)
            if pvc_ref and getattr(pvc_ref, "claim_name", None) == pvc_name:
                node = pod.spec.node_name
                if node:
                    return node
    return None


def _pvc_mounted_by_any_pod(core_v1: "CoreV1Api", namespace: str, pvc_name: str) -> Optional[bool]:
    """PVC 를 마운트한 Pod 가 있는지. 조회에 실패하면 None(알 수 없음)을 돌려준다.

    삭제 판단에 쓰므로 "조회 실패"와 "사용 안 함"을 반드시 구분해야 한다. 둘을 뭉뚱그리면
    API 가 잠깐 흔들린 순간에 사용 중인 볼륨을 지우게 된다.
    """
    try:
        pods = core_v1.list_namespaced_pod(namespace=namespace).items or []
    except Exception as e:
        logger.warning("Pod 목록 조회 실패 (PVC %s 사용 여부 확인 불가): %s", pvc_name, e)
        return None
    for pod in pods:
        if not pod.spec or not pod.spec.volumes:
            continue
        for v in pod.spec.volumes:
            pvc_ref = getattr(v, "persistent_volume_claim", None)
            if pvc_ref and getattr(pvc_ref, "claim_name", None) == pvc_name:
                return True
    return False


def _copy_pvc_job_name(dst_pvc: str) -> str:
    suf = uuid.uuid4().hex[:8]
    base = f"pvc-copy-{dst_pvc}-{suf}"
    if len(base) > 63:
        base = base[:63].rstrip("-")
    return base


def _copy_pvc_data_via_job(
    core_v1: "CoreV1Api",
    namespace: str,
    *,
    src_pvc: str,
    dst_pvc: str,
    target_node: Optional[str],
) -> str:
    """src_pvc → dst_pvc 데이터를 복사하는 일회성 Job 을 생성하고 Job 이름을 반환한다.

    - src 는 RO 마운트 (Longhorn 같은 노드 다중 Pod RO 공유 가능)
    - dst 는 RW 마운트 (NFS PVC, 어디서든 RW 마운트 가능)
    - target_node 가 있으면 nodeSelector 로 강제 (원본이 attach 된 노드)
    - 완료 대기는 하지 않는다. 서빙 파이프라인이 Ollama Pod 을 띄우기 전에 이 Job 의 종료를 기다린다.
    - 복사가 끝나면 dst 루트에 완료 마커를 남겨, 중단된 복제본을 나중에 재사용하지 않도록 한다.
    """
    from kubernetes import client as k8s_client

    s = get_settings()
    image = (s.PVC_DATA_COPY_IMAGE or "busybox:1.36").strip() or "busybox:1.36"
    timeout_sec = int(s.PVC_DATA_COPY_TIMEOUT_SEC or 1800)

    job_name = _copy_pvc_job_name(dst_pvc)

    pod_spec_kwargs: Dict[str, Any] = {
        "restart_policy": "Never",
        "containers": [
            k8s_client.V1Container(
                name="copier",
                image=image,
                command=["/bin/sh", "-c"],
                args=[
                    # cp -r: 재귀 복사. busybox cp -p 는 ownership 도 시도하므로 NFS root_squash 거부 회피 위해 -p 제외.
                    # 앞뒤로 마커를 지웠다 만든다. src 가 이미 마커를 갖고 있으면 cp 가 그것을 먼저 옮겨
                    # 복사 도중인데도 완료로 보이므로, 복사가 끝난 시점에 새로 쓴 것만 남긴다.
                    "set -e && rm -f /dst/" + PVC_COPY_COMPLETE_MARKER + " && sync"
                    " && cp -r /src/. /dst/"
                    " && rm -f /dst/" + PVC_COPY_COMPLETE_MARKER + " && sync"
                    " && touch /dst/" + PVC_COPY_COMPLETE_MARKER + " && sync"
                    " && echo 'pvc-copy: done'",
                ],
                resources=k8s_client.V1ResourceRequirements(
                    requests={"memory": "256Mi", "cpu": "200m"},
                    limits={"memory": "1Gi", "cpu": "1000m"},
                ),
                volume_mounts=[
                    k8s_client.V1VolumeMount(name="src", mount_path="/src", read_only=True),
                    k8s_client.V1VolumeMount(name="dst", mount_path="/dst"),
                ],
            )
        ],
        "volumes": [
            k8s_client.V1Volume(
                name="src",
                persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=src_pvc, read_only=True
                ),
            ),
            k8s_client.V1Volume(
                name="dst",
                persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=dst_pvc),
            ),
        ],
    }
    if target_node:
        pod_spec_kwargs["node_selector"] = {"kubernetes.io/hostname": target_node}

    job = k8s_client.V1Job(
        metadata=k8s_client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={
                "app": "ollama-workflow",
                "ml-workflow/pvc-copy": "true",
                "ml-workflow/dst-pvc": dst_pvc[:60],
            },
        ),
        spec=k8s_client.V1JobSpec(
            backoff_limit=2,
            ttl_seconds_after_finished=600,
            active_deadline_seconds=timeout_sec,
            template=k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(
                    labels={
                        "app": "ollama-workflow",
                        "ml-workflow/pvc-copy": "true",
                    },
                ),
                spec=k8s_client.V1PodSpec(**pod_spec_kwargs),
            ),
        ),
    )

    batch_v1 = _load_batch_v1()
    try:
        batch_v1.create_namespaced_job(namespace=namespace, body=job)
        logger.info(
            "Created PVC copy Job %s (src=%s → dst=%s, node=%s)",
            job_name,
            src_pvc,
            dst_pvc,
            target_node or "<any>",
        )
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("PVC copy Job %s already exists", job_name)
        else:
            raise

    return job_name


def _job_outcome(job: Any) -> str:
    """복사 Job 의 종료 상태 → "succeeded" | "failed" | "running".

    conditions 를 먼저 본다. activeDeadlineSeconds 초과(DeadlineExceeded)는 파드 실패 수를
    backoff_limit 이상으로 올리지 않으므로, 실패 수만 세면 끝난 Job 을 진행 중으로 오인한다.
    """
    status = getattr(job, "status", None)
    for c in (getattr(status, "conditions", None) or []) if status else []:
        ctype = getattr(c, "type", "")
        cstatus = str(getattr(c, "status", "")).lower()
        if cstatus != "true":
            continue
        if ctype == "Failed":
            return "failed"
        if ctype == "Complete":
            return "succeeded"
    if status and (status.succeeded or 0) >= 1:
        return "succeeded"
    spec = getattr(job, "spec", None)
    backoff_limit = spec.backoff_limit if spec and spec.backoff_limit is not None else 0
    if status and (status.failed or 0) > backoff_limit:
        return "failed"
    return "running"


def _find_active_copy_job_for_pvc(namespace: str, dst_pvc: str) -> Tuple[Optional[str], str]:
    """dst_pvc 를 대상으로 하는 복사 Job 조회 → (job_name, outcome).

    job_name 이 None 이면 Job 이 없다. 완료 후 TTL 로 정리된 것일 수도, 실패한 뒤 정리된 것일
    수도 있어 그 자체로는 완료 근거가 되지 않는다. 완료 판단은 PVC annotation 으로 한다.
    """
    batch_v1 = _load_batch_v1()
    selector = "ml-workflow/pvc-copy=true," + "ml-workflow/dst-pvc=" + dst_pvc[:60]
    try:
        lst = batch_v1.list_namespaced_job(namespace=namespace, label_selector=selector)
    except Exception as e:
        logger.warning("복사 Job 조회 실패 (dst=%s): %s", dst_pvc, e)
        return None, "unknown"

    for j in lst.items or []:
        name = j.metadata.name if j.metadata else None
        if not name:
            continue
        return name, _job_outcome(j)
    return None, "absent"


def reclaim_reason_for_incomplete_clone(pvc: Any, namespace: str, job_outcome: str) -> Optional[str]:
    """복사가 끝나지 않은 복제본을 회수해도 되는지 판단한다. 회수 가능하면 사유, 아니면 None.

    호출 전에 완료 표시가 없다는 것이 확인된 복제본만 들어온다. 여기서는 "정말 아무도 쓰지 않고
    앞으로도 채워질 일이 없는가" 만 본다.
    """
    from datetime import timezone

    meta = pvc.metadata
    name = meta.name

    if getattr(meta, "deletion_timestamp", None):
        return None  # 이미 삭제가 진행 중

    if job_outcome == "running":
        return None  # 지금 채워지는 중
    if job_outcome == "unknown":
        return None  # Job 조회 자체가 실패 — 판단 근거가 없으니 손대지 않는다

    created = getattr(meta, "creation_timestamp", None)
    if created is None:
        return None
    grace = int(get_settings().OLLAMA_CLONE_RECLAIM_GRACE_SEC or 3600)
    age_sec = (datetime.now(timezone.utc) - created).total_seconds()
    if age_sec < grace:
        # 만들어진 직후에는 복사 Job 이 아직 목록에 안 잡힐 수 있다. 그 창에서 지우면
        # 진행 중인 배포의 볼륨을 빼앗는다.
        return None

    # 마운트한 Pod 이 있으면 누군가 쓰는 중이다.
    try:
        core = _load_core_v1()
        attached = _find_node_attached_to_pvc(core, namespace, name)
    except Exception as e:
        logger.warning("복제본 %s 사용 중인 Pod 확인 실패 — 회수 보류: %s", name, e)
        return None
    if attached:
        return None

    return f"복사 미완료(job={job_outcome}), 생성 후 {int(age_sec)}s 경과, 사용 중인 Pod 없음"


def _reclaim_incomplete_clone(pvc: Any, namespace: str, job_outcome: str) -> None:
    """회수 조건을 만족하면 삭제하거나(설정 시) 회수 후보로 남긴다."""
    reason = reclaim_reason_for_incomplete_clone(pvc, namespace, job_outcome)
    if not reason:
        return
    name = pvc.metadata.name
    if not get_settings().OLLAMA_CLONE_RECLAIM_ENABLED:
        logger.warning(
            "복제본 %s 는 회수 대상입니다(%s). OLLAMA_CLONE_RECLAIM_ENABLED=false 라 삭제하지 않습니다.",
            name,
            reason,
        )
        return
    logger.info("불완전 복제본 %s 회수: %s", name, reason)
    try:
        core = _load_core_v1()
        _delete_pvc_if_exists(core, namespace, name)
    except Exception as e:
        logger.warning("복제본 %s 회수 실패: %s", name, e)


def mark_clone_copy_completed(namespace: str, pvc_name: str) -> None:
    """복사가 끝난 복제본에 완료 표시를 남긴다. 서빙 파이프라인이 Job 종료를 확인한 뒤 호출한다."""
    core = _load_core_v1()
    body = {"metadata": {"annotations": {CLONE_ANNOTATION_COPY_COMPLETED: "true"}}}
    core.patch_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace, body=body)
    logger.info("복제본 %s 복사 완료 표시", pvc_name)


def _create_pvc_clone_from_source(
    core_v1: "CoreV1Api",
    namespace: str,
    source_pvc_name: str,
    new_pvc_name: str,
    model_id: int,
    serving_node_name: str,
    storage_class_name_override: Optional[str] = None,
) -> Optional[str]:
    """복제본 PVC 를 만들고, 데이터 복사 Job 이 필요하면 생성해 그 Job 이름을 반환한다.

    PVC Bound 와 복사 완료를 기다리지 않는다. 두 대기는 서빙 파이프라인이 Ollama Pod 을 띄우기 직전에 수행한다.
    """
    from kubernetes import client

    src = core_v1.read_namespaced_persistent_volume_claim(name=source_pvc_name, namespace=namespace)
    spec = src.spec
    if spec is None:
        raise RuntimeError(f"PVC {source_pvc_name} has empty spec")

    sc_name = storage_class_name_override if storage_class_name_override else spec.storage_class_name

    # NFS fallback SC 는 CSI volume cloning 미지원 → data_source 를 비우고 빈 PVC 생성 후 별도 Job 으로 복사.
    is_nfs_fallback = _is_nfs_fallback_sc(sc_name)
    ds = (
        None
        if is_nfs_fallback
        else client.V1TypedLocalObjectReference(
            api_group="",
            kind="PersistentVolumeClaim",
            name=source_pvc_name,
        )
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
            CLONE_LABEL_COPY_MODE: COPY_MODE_JOB if is_nfs_fallback else COPY_MODE_CSI,
        },
    )
    body = client.V1PersistentVolumeClaim(metadata=meta, spec=new_spec)
    try:
        core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=body)
        logger.info(
            "Created workflow Ollama PVC clone %s from %s (nfs_fallback=%s)",
            new_pvc_name,
            source_pvc_name,
            is_nfs_fallback,
        )
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("PVC clone %s already exists", new_pvc_name)
        else:
            raise

    if not is_nfs_fallback:
        return None

    try:
        target_node = _find_node_attached_to_pvc(core_v1, namespace, source_pvc_name)
        return _copy_pvc_data_via_job(
            core_v1,
            namespace,
            src_pvc=source_pvc_name,
            dst_pvc=new_pvc_name,
            target_node=target_node,
        )
    except Exception:
        # Job 생성 자체가 실패하면 채워질 일이 없는 빈 PVC 이므로 남기지 않는다.
        logger.warning("PVC copy Job 생성 실패 → 빈 클론 PVC %s 정리 후 예외 전파", new_pvc_name)
        _delete_pvc_if_exists(core_v1, namespace, new_pvc_name)
        raise


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


def _find_existing_clone_pvc(model_id: int, serving_node_name: str) -> Optional["ResolvedOllamaPvc"]:
    """같은 (model, node) 로 이미 만들어 둔 복제본을 클러스터에서 찾는다.

    배포 행이 아직 기록되기 전에도 복제본을 알아보기 위한 경로다. 배포 행 기준으로만 판단하면
    복제 직후·배포 행 생성 직전 사이에 들어온 요청이 같은 복제본을 또 만들어 모델을 두 번 내려받는다.
    복사가 실패한 채 남은 복제본은 되돌려주지 않는다.
    """
    s = get_settings()
    ns = s.KUBEFLOW_NAMESPACE
    selector = (
        "ml-workflow/ollama-workflow-clone=true,"
        + "ml-workflow/source-model-id="
        + str(int(model_id))
        + ","
        + "ml-workflow/serving-node="
        + _k8s_sanitize_segment(serving_node_name, 60)
    )
    try:
        core = _load_core_v1()
        lst = core.list_namespaced_persistent_volume_claim(namespace=ns, label_selector=selector)
    except Exception as e:
        logger.warning("기존 복제본 조회 실패 (model_id=%s node=%s): %s", model_id, serving_node_name, e)
        return None

    items = [p for p in (lst.items or []) if p.metadata and p.metadata.name]
    # 생성이 이른 것부터 본다. 같은 조합에 복제본이 여러 개면 가장 먼저 만들어진 것으로 수렴시킨다.
    items.sort(key=lambda p: (p.metadata.creation_timestamp is None, p.metadata.creation_timestamp))

    for pvc in items:
        name = pvc.metadata.name
        phase = (pvc.status.phase or "") if pvc.status else ""
        if phase not in ("Bound", "Pending"):
            continue
        labels = pvc.metadata.labels or {}
        annotations = pvc.metadata.annotations or {}
        filled_by_job = labels.get(CLONE_LABEL_COPY_MODE) == COPY_MODE_JOB

        # CSI 클론은 볼륨이 Bound 되는 순간 원본 내용을 그대로 갖는다. 따로 채울 것이 없다.
        if not filled_by_job:
            logger.info("기존 복제본 재사용: %s (CSI 클론, phase=%s)", name, phase or "?")
            return ResolvedOllamaPvc(pvc_name=name)

        if annotations.get(CLONE_ANNOTATION_COPY_COMPLETED) == "true":
            logger.info("기존 복제본 재사용: %s (복사 완료됨)", name)
            return ResolvedOllamaPvc(pvc_name=name, copy_marker_required=True)

        job_name, outcome = _find_active_copy_job_for_pvc(ns, name)
        if outcome == "running":
            logger.info("기존 복제본 재사용: %s (복사 진행 중, job=%s)", name, job_name)
            return ResolvedOllamaPvc(pvc_name=name, copy_job_name=job_name or "", copy_marker_required=True)

        # 완료 표시도 없고 진행 중인 Job 도 없다. 복사가 끝났다고 볼 근거가 없으므로 넘긴다.
        # (실패했거나, 중간에 프로세스가 끊겼거나, 끝난 Job 이 TTL 로 사라졌는데 표시가 안 된 경우)
        logger.warning(
            "복제본 %s 는 복사 완료 표시가 없고 진행 중인 Job 도 없음(job=%s, outcome=%s) — 재사용하지 않음",
            name,
            job_name or "<none>",
            outcome,
        )
        # 배포 행에 한 번도 기록되지 않은 복제본은 기존 정리 경로가 찾지 못한다. 이 자리가 유일한 회수 지점이다.
        _reclaim_incomplete_clone(pvc, ns, outcome)
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
) -> ResolvedOllamaPvc:
    """복제본이 이미 있으면 그것을 쓰고, 없을 때만 새로 만든다. 반드시 락 안에서 호출한다."""
    existing = _find_existing_clone_pvc(model_id, serving_node_name)
    if existing:
        return existing

    new_name = _clone_pvc_name(model_id, serving_node_name)
    ns = get_settings().KUBEFLOW_NAMESPACE
    core = _load_core_v1()
    copy_job = _create_pvc_clone_from_source(
        core,
        ns,
        registry_pvc,
        new_name,
        model_id,
        serving_node_name,
        storage_class_name_override=storage_class_name_override,
    )
    rollback_clone_pvcs.append(new_name)
    return ResolvedOllamaPvc(
        pvc_name=new_name,
        copy_job_name=copy_job or "",
        copy_marker_required=bool(copy_job),
    )


def _describe_pvc_for_serving(
    pvc_name: str,
    registry_pvc: str,
    model_id: int,
    serving_node_name: str,
) -> ResolvedOllamaPvc:
    """이미 정해진 PVC 이름에 복사 대기 정보를 채운다. 원본을 그대로 쓰면 기다릴 것이 없다."""
    if not pvc_name or pvc_name == registry_pvc:
        return ResolvedOllamaPvc(pvc_name=pvc_name)
    found = _find_existing_clone_pvc(model_id, serving_node_name)
    if found and found.pvc_name == pvc_name:
        return found
    return ResolvedOllamaPvc(pvc_name=pvc_name)


def _acquire_clone_lock_or_conflict(lock: WorkflowServingVolumeLock, model_id: int, serving_node_name: str) -> str:
    lk = serving_volume_lock_key(model_id, serving_node_name)
    if not lock.try_acquire_nonblocking(lk):
        raise PvcReplicationConflict(
            "동일 model·serving_node에 대해 복제 PVC 생성 또는 삭제가 진행 중입니다. 잠시 후 다시 시도해 주세요."
        )
    return lk


def _resolve_ollama_pvc_execute_legacy(
    db: "Session",
    *,
    model_id: int,
    registry_pvc: str,
    serving_node_name: str,
    rollback_clone_pvcs: List[str],
    vol_lock: Optional[WorkflowServingVolumeLock] = None,
) -> ResolvedOllamaPvc:
    """Cinder zone 매칭 비활성 시 동작."""
    lock = vol_lock or get_workflow_serving_volume_lock()

    live_rows = _live_deployments_for_model(db, model_id).all()

    reuse = _find_reuse_pvc_same_model_node(live_rows, serving_node_name)
    if reuse:
        return _describe_pvc_for_serving(reuse, registry_pvc, model_id, serving_node_name)

    if not live_rows:
        return ResolvedOllamaPvc(pvc_name=registry_pvc)

    if not _any_live_uses_original_on_other_node(live_rows, registry_pvc, serving_node_name):
        return ResolvedOllamaPvc(pvc_name=registry_pvc)

    lk = _acquire_clone_lock_or_conflict(lock, model_id, serving_node_name)
    try:
        db.expire_all()
        live_rows2 = _live_deployments_for_model(db, model_id).all()
        reuse2 = _find_reuse_pvc_same_model_node(live_rows2, serving_node_name)
        if reuse2:
            return _describe_pvc_for_serving(reuse2, registry_pvc, model_id, serving_node_name)
        if not live_rows2:
            return ResolvedOllamaPvc(pvc_name=registry_pvc)
        if not _any_live_uses_original_on_other_node(live_rows2, registry_pvc, serving_node_name):
            return ResolvedOllamaPvc(pvc_name=registry_pvc)

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
) -> ResolvedOllamaPvc:
    """이번 Ollama 배포에 쓸 PVC 와 복사 대기 정보. 복제 시 vol_lock으로 직렬화."""
    lock = vol_lock or get_workflow_serving_volume_lock()
    s = get_settings()

    if not registry_pvc:
        return ResolvedOllamaPvc(pvc_name="")

    if not serving_node_name:
        logger.warning("serving_node_name 비어 있음 — 노드 분기 생략, 레지스트리 원본 PVC 사용 (model_id=%s)", model_id)
        return ResolvedOllamaPvc(pvc_name=registry_pvc)

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
                return ResolvedOllamaPvc(pvc_name=registry_pvc)
            lk = _acquire_clone_lock_or_conflict(lock, model_id, serving_node_name)
            try:
                db.expire_all()
                live_rows2 = _live_deployments_for_model(db, model_id).all()
                if not _any_live_uses_original_on_other_node(live_rows2, registry_pvc, serving_node_name):
                    return ResolvedOllamaPvc(pvc_name=registry_pvc)
                reuse2 = _find_reuse_pvc_same_model_node(live_rows2, serving_node_name)
                if reuse2:
                    return _describe_pvc_for_serving(reuse2, registry_pvc, model_id, serving_node_name)
                return _clone_and_track(
                    model_id, serving_node_name, registry_pvc, rollback_clone_pvcs, storage_class_name_override=None
                )
            finally:
                lock.release(lk)

        reuse = _find_reuse_pvc_same_model_node(live_rows, serving_node_name)
        if reuse:
            return _describe_pvc_for_serving(reuse, registry_pvc, model_id, serving_node_name)

        lk2 = _acquire_clone_lock_or_conflict(lock, model_id, serving_node_name)
        try:
            db.expire_all()
            live_rows2 = _live_deployments_for_model(db, model_id).all()
            reuse2 = _find_reuse_pvc_same_model_node(live_rows2, serving_node_name)
            if reuse2:
                return _describe_pvc_for_serving(reuse2, registry_pvc, model_id, serving_node_name)
            target_sc = _resolve_clone_storage_class_when_cinder_zone_mismatch(node_zone)
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
            "Cinder zone: 노드 label 또는 레지스트리 SC availability 미확인 — legacy 경로 (model_id=%s)",
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
    """존 불일치 시에만 target StorageClass, 일치·비활성·미조회 시 None. 매칭 SC 없으면 ValueError."""
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
    return _resolve_clone_storage_class_when_cinder_zone_mismatch(nz)


def _clone_registry_pvc_for_node(
    *,
    model_id: int,
    serving_node_name: str,
    registry_pvc: str,
    rollback_clone_pvcs: List[str],
    vol_lock: WorkflowServingVolumeLock,
) -> ResolvedOllamaPvc:
    """레지스트리 원본 PVC를 소스로 해당 노드 전용 클론 PVC 생성(락)."""
    sc_override = _cinder_sc_override_for_clone(serving_node_name, registry_pvc)

    lk = _acquire_clone_lock_or_conflict(vol_lock, model_id, serving_node_name)
    try:
        return _clone_and_track(
            model_id,
            serving_node_name,
            registry_pvc,
            rollback_clone_pvcs,
            storage_class_name_override=sc_override,
        )
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
    """serving_plans[component_id] 에 이번 배포용 PVC 와 복사 대기 정보를 채운다.

    복제본 생성과 복사 Job 기동까지만 하고 완료를 기다리지 않는다. 복사가 끝났는지는 서빙 파이프라인이
    Ollama Pod 을 띄우기 직전에 확인한다.
    """
    from config.db.enums import ModelFormatEnum, ModelProviderEnum
    from db.models.model import Model
    from db.models.service import ComponentType
    from sqlalchemy.orm import joinedload

    rollback_list: List[str] = []
    parameters["_ollama_clone_pvcs_rollback"] = rollback_list
    resolve_cache: dict[tuple[int, str], ResolvedOllamaPvc] = {}
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
            if node != primary_node and resolved.pvc_name == reg_pvc:
                resolved = _clone_registry_pvc_for_node(
                    model_id=model.id,
                    serving_node_name=node,
                    registry_pvc=reg_pvc,
                    rollback_clone_pvcs=rollback_list,
                    vol_lock=vol_lock,
                )
                resolve_cache[cache_key] = resolved

        plan = dict(plan)
        plan["resolved_ollama_pvc"] = resolved.pvc_name
        plan["ollama_pvc_copy_job"] = resolved.copy_job_name
        plan["ollama_pvc_copy_marker_required"] = resolved.copy_marker_required
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
    """Ollama 복제 PVC, 마지막 소비자일 때만 K8s 삭제.

    DEPLOYING/DEPLOYED뿐 아니라 FAILED도 처리한다(실패 전·후에 생성된 복제 PVC 정리).
    """
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
        dep.deleted_at = func.now()
        db.commit()
        return

    lk = serving_volume_lock_key(mid, node)
    if not lock.acquire_blocking(lk, timeout_sec=30.0):
        logger.warning("delete PVC: could not acquire lock model_id=%s node=%s", mid, node)
        dep.status = DeploymentStatus.DELETED
        dep.deleted_at = func.now()
        db.commit()
        return

    try:
        dep.status = DeploymentStatus.DELETED
        dep.deleted_at = func.now()
        db.commit()

        cnt = _live_deployments_for_model(db, mid).filter(ModelWorkflowDeployment.serving_node_name == node).count()

        is_replica = bool(pvc_used) and (not registry_original_pvc or pvc_used != registry_original_pvc)
        if cnt == 0 and is_replica:
            try:
                core = _load_core_v1()
                ns = get_settings().KUBEFLOW_NAMESPACE
                # 배포 행 기준으로는 마지막 소비자라도, 실제로 그 볼륨을 물고 있는 Pod 가 있으면 지우지 않는다.
                # 배포 행은 API 프로세스마다 따로 보이고 실행 구간 락도 프로세스 안에서만 유효하므로,
                # 클러스터의 실제 사용 여부가 마지막 판단 근거다.
                mounted = _pvc_mounted_by_any_pod(core, ns, pvc_used)
                if mounted is None:
                    logger.warning(
                        "복제 PVC %s 사용 여부를 확인하지 못해 삭제를 보류합니다 "
                        "(workflow_id=%s, deployment_id=%s). 다음 정리 때 다시 시도됩니다.",
                        pvc_used,
                        dep.workflow_id,
                        deployment_id,
                    )
                elif mounted:
                    logger.warning(
                        "복제 PVC %s 를 사용 중인 Pod 가 있어 삭제하지 않습니다 " "(workflow_id=%s, deployment_id=%s).",
                        pvc_used,
                        dep.workflow_id,
                        deployment_id,
                    )
                else:
                    _delete_pvc_if_exists(core, ns, pvc_used)
            except Exception as e:
                logger.warning("replica PVC delete skipped: %s", e)
    finally:
        lock.release(lk)


def process_deployments_before_hard_delete(
    db: "Session", workflow_id: str, allowed_deployment_ids: Optional[List[str]] = None
) -> None:
    """워크플로의 배포 행을 훑어 복제 PVC 를 정리한다.

    allowed_deployment_ids 를 주면 그 행만 대상으로 한다. 정리가 진행되는 동안 같은 워크플로가
    다시 실행되면 새 배포 행과 새 복제 PVC 가 생기는데, 그것까지 지우면 실행 중인 파이프라인이
    참조하는 볼륨이 사라져 PVC not found 로 실패한다. 호출부에서 정리 시작 시점의 행 목록을
    넘겨 그 시점 이후에 생긴 행을 제외한다.
    """
    from db.models.model_workflow_deployment import ModelWorkflowDeployment

    rows = db.query(ModelWorkflowDeployment).filter(ModelWorkflowDeployment.workflow_id == workflow_id).all()
    if allowed_deployment_ids is not None:
        allowed = set(allowed_deployment_ids)
        skipped = [r.id for r in rows if r.id not in allowed]
        if skipped:
            logger.info(
                "정리 대상에서 제외: 정리 시작 이후 생긴 배포 행 %s (workflow_id=%s) — 새 실행의 볼륨을 보존한다.",
                skipped,
                workflow_id,
            )
        rows = [r for r in rows if r.id in allowed]
    for dep in sorted(rows, key=lambda r: (r.workflow_id, r.component_id, r.id)):
        delete_replica_pvc_if_last_consumer(db, dep.id)
