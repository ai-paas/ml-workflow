"""최적화/경량화 소스 모델(repo_id) 설정.

- 키: `opt_enable_yn`을 켤 수 있는 HuggingFace 등 `repo_id`
- 값: 허용하는 `task_type`(최적화 서버 `optimizer_name`) 집합
"""

from __future__ import annotations

# repo_id -> 허용 기법. 맵에 없으면 최적화 소스로 등록되지 않음(opt_enable false).
OPTIMIZATION_SOURCES: dict[str, frozenset[str]] = {
    "facebook/detr-resnet-50": frozenset({"tensorrt", "openvino", "pruning"}),
    "facebook/detr-resnet-101": frozenset({"tensorrt", "openvino", "pruning"}),
}


def _allowlist_for_id_string(raw: str | None) -> frozenset[str] | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    v = OPTIMIZATION_SOURCES.get(s)
    if v is not None:
        return v
    sl = s.lower()
    for k, vv in OPTIMIZATION_SOURCES.items():
        if k.lower() == sl:
            return vv
    return None


def is_optimization_eligible(repo_id: str | None, model_name: str | None = None) -> bool:
    """repo_id(또는 동일 규칙으로 model_name)이 최적화/경량화 소스로 허용되는지."""
    return optimization_task_allowlist_for_repo(repo_id, model_name=model_name) is not None


def optimization_task_allowlist_for_repo(
    repo_id: str | None, *, model_name: str | None = None
) -> frozenset[str] | None:
    """소스로 등록된 repo면 허용 기법 집합, 아니면 None(목록 API에서 전체 기준 필터 없음).

    DB에 공백이 붙은 repo_id, 대소문자만 다른 값, 또는 repo_id 비어 있고 name만 HF id인 경우를
    `model_name` 폴백·정규화로 맞춘다.
    """
    v = _allowlist_for_id_string(repo_id)
    if v is not None:
        return v
    return _allowlist_for_id_string(model_name)
