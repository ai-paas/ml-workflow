"""KServe InferenceService 조회 헬퍼 함수"""

import logging
from typing import Dict, List, Optional

from kubernetes import client, config

logger = logging.getLogger(__name__)


def get_inference_services_by_labels(
    namespace: str,
    workflow_id: Optional[str] = None,
    component_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> List[Dict]:
    """
    Label selector를 사용하여 InferenceService를 효율적으로 조회

    Args:
        namespace: Kubernetes namespace
        workflow_id: workflow-id label로 필터링 (선택)
        component_id: component-id label로 필터링 (선택)
        model_id: model-id label로 필터링 (선택)

    Returns:
        매칭되는 InferenceService 목록
    """
    try:
        # Kubernetes config 로드
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        # Custom Object API 사용 (KServe InferenceService는 CRD)
        api = client.CustomObjectsApi()

        # Label selector 구성
        label_selectors = []
        if workflow_id:
            label_selectors.append(f"workflow-id={workflow_id}")
        if component_id:
            label_selectors.append(f"component-id={component_id}")
        if model_id:
            label_selectors.append(f"model-id={model_id}")

        label_selector = ",".join(label_selectors) if label_selectors else None

        logger.info(f"Querying InferenceServices with label_selector: {label_selector}")

        # KServe InferenceService 조회 (label selector 적용)
        result = api.list_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            label_selector=label_selector,
        )

        services = result.get("items", [])
        logger.info(f"Found {len(services)} InferenceServices matching labels")

        return services

    except Exception as e:
        logger.error(f"Failed to query InferenceServices: {e}")
        return []


def get_inference_service_status(namespace: str, workflow_id: str, component_id: str) -> Optional[Dict]:
    """
    특정 workflow_id와 component_id로 InferenceService 상태 조회

    Args:
        namespace: Kubernetes namespace
        workflow_id: Workflow ID
        component_id: Component ID

    Returns:
        InferenceService 정보 (없으면 None)
    """
    services = get_inference_services_by_labels(
        namespace=namespace,
        workflow_id=workflow_id,
        component_id=component_id,
    )

    if not services:
        logger.warning(f"No InferenceService found for workflow={workflow_id}, component={component_id}")
        return None

    if len(services) > 1:
        logger.warning(
            f"Multiple InferenceServices found for workflow={workflow_id}, component={component_id}. "
            f"Using the first one."
        )

    service = services[0]
    metadata = service.get("metadata", {})
    status = service.get("status", {})

    # Ready 상태 확인
    conditions = status.get("conditions", [])
    is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

    return {
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "labels": metadata.get("labels", {}),
        "service_name": metadata.get("name"),
        "internal_url": status.get("address", {}).get("url", ""),
        "is_ready": is_ready,
        "status": status,
        "raw_service": service,
    }


def get_all_workflow_services(namespace: str, workflow_id: str) -> List[Dict]:
    """
    워크플로우의 모든 InferenceService 조회

    Args:
        namespace: Kubernetes namespace
        workflow_id: Workflow ID

    Returns:
        워크플로우의 모든 InferenceService 목록
    """
    return get_inference_services_by_labels(namespace=namespace, workflow_id=workflow_id)
