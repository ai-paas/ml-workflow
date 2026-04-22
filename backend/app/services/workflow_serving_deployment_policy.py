"""§2.5 deployment_type 결정(OLLAMA/KSERVE), §2.6 URL 생성에 쓰는 순수 정책."""

from __future__ import annotations

from typing import Any, Optional

from config.db.enums import ModelProviderEnum
from config.settings import Settings
from db.models.model import Model
from db.models.model_workflow_deployment import ServingDeviceType, WorkflowServingDeploymentType


def resolve_workflow_serving_deployment_type(
    model: Optional[Model],
    _settings: Settings,
) -> WorkflowServingDeploymentType:
    """MODEL 컴포넌트의 deployment_type (§2.5).

    REMOTE 1순위는 §6: `REMOTE_SERVING_MODEL_MAP`에 모델 repo_id가 있으면 REMOTE (구현 예정). 예시만 주석으로 둔다:
      # if repo_id_in_remote_serving_model_map(model, _settings):
      #     return WorkflowServingDeploymentType.REMOTE
    """
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
