"""
E2E: 최적화/경량화 (model-improvements)

- 빠른 검증 (`-m model_improvement`): task-types, POST 422 등
- 종단 시나리오 (`-m model_improvement_scenario`): 작업 생성 → 상태 폴링 → SUCCEEDED·result_model_id
  소스는 E2E_OPTIMIZATION_SOURCE_MODEL_NAME (/models 의 name, 워크플로 E2E_WORKFLOW_TARGET_LLM_MODEL 과 동일 패턴). 예:
  make e2e-model-improvement-scenario ENV=dev \\
    E2E_OPTIMIZATION_SOURCE_MODEL_NAME=facebook-detr-resnet-50 \\
    E2E_OPTIMIZATION_TASK_TYPE=pruning
"""

from __future__ import annotations

import time

import pytest
import requests
from config import (
    OPTIMIZATION_POLL_INTERVAL_SEC,
    OPTIMIZATION_POLL_TIMEOUT_SEC,
    OPTIMIZATION_SOURCE_MODEL_NAME,
    OPTIMIZATION_TASK_TYPE,
    find_optimization_source_model,
)

DETR_REPO_IDS = frozenset({"facebook/detr-resnet-50", "facebook/detr-resnet-101"})
DETR_ALLOWED = frozenset({"pruning", "tensorrt", "openvino"})


@pytest.mark.model_improvement
class TestModelImprovementApi:
    """GET/POST model-improvements (최적화 서버 연동 가정)."""

    @staticmethod
    def _get_models(api_url: str, headers: dict) -> list[dict]:
        r = requests.get(f"{api_url}/models", headers=headers)
        assert r.status_code == 200, r.text
        return r.json()

    @staticmethod
    def _find_detr_model(models: list[dict]) -> dict | None:
        """백엔드와 동일하게 repo_id 또는 name 이 DETR HF id 인 모델을 고른다."""
        for m in models:
            if m.get("opt_enable_yn") is not True:
                continue
            rid = (m.get("repo_id") or "").strip()
            nm = (m.get("name") or "").strip()
            if rid in DETR_REPO_IDS or nm in DETR_REPO_IDS:
                return m
        return None

    @staticmethod
    def _find_non_opt_model(models: list[dict]) -> dict | None:
        for m in models:
            if m.get("opt_enable_yn") is not True:
                return m
        return None

    def test_task_types_list(self, api_url: str, auth_headers: dict):
        """기법 목록 조회 (200)."""
        r = requests.get(f"{api_url}/model-improvements/task-types", headers=auth_headers)
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list)
        for item in data:
            assert "name" in item
            assert "category" in item

    def test_task_types_category_filter(self, api_url: str, auth_headers: dict):
        """category=optimization|lightweight 필터."""
        for cat in ("optimization", "lightweight"):
            r = requests.get(
                f"{api_url}/model-improvements/task-types",
                params={"category": cat},
                headers=auth_headers,
            )
            assert r.status_code == 200, r.text
            for item in r.json():
                assert item.get("category") == cat

    def test_task_types_invalid_category(self, api_url: str, auth_headers: dict):
        r = requests.get(
            f"{api_url}/model-improvements/task-types",
            params={"category": "invalid"},
            headers=auth_headers,
        )
        assert r.status_code == 422

    def test_task_types_filtered_for_detr_source(self, api_url: str, auth_headers: dict):
        """source_model_id=DETR 이면 허용 기법만 노출."""
        models = self._get_models(api_url, auth_headers)
        detr = self._find_detr_model(models)
        if not detr:
            pytest.skip("opt_enable_yn=true 인 DETR 모델이 환경에 없습니다.")

        r = requests.get(
            f"{api_url}/model-improvements/task-types",
            params={"source_model_id": detr["id"]},
            headers=auth_headers,
        )
        assert r.status_code == 200, r.text
        names = {item["name"].lower() for item in r.json()}
        assert names <= {x.lower() for x in DETR_ALLOWED}, f"허용 초과: {names - DETR_ALLOWED}"

    def test_create_rejects_disallowed_task_type_for_detr(self, api_url: str, auth_headers: dict):
        """DETR 소스에 허용 목록 밖 task_type → 422."""
        models = self._get_models(api_url, auth_headers)
        detr = self._find_detr_model(models)
        if not detr:
            pytest.skip("opt_enable_yn=true 인 DETR 모델이 환경에 없습니다.")

        r = requests.post(
            f"{api_url}/model-improvements",
            json={"source_model_id": detr["id"], "task_type": "ptq"},
            headers=auth_headers,
        )
        assert r.status_code == 422, r.text
        payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        assert "OPTIMIZATION_TASK_TYPE_NOT_ALLOWED" in str(payload.get("detail", r.text))

    def test_create_rejects_ineligible_source(self, api_url: str, auth_headers: dict):
        """opt_enable_yn=false 인 모델 → 422."""
        models = self._get_models(api_url, auth_headers)
        bad = self._find_non_opt_model(models)
        if not bad:
            pytest.skip("opt_enable_yn=false 인 모델이 목록에 없습니다.")

        r = requests.post(
            f"{api_url}/model-improvements",
            json={"source_model_id": bad["id"], "task_type": "tensorrt"},
            headers=auth_headers,
        )
        assert r.status_code == 422, r.text


def _task_type_allowed_for_model(api_url: str, headers: dict, source_model_id: int, task_type: str) -> bool:
    r = requests.get(
        f"{api_url}/model-improvements/task-types",
        params={"source_model_id": source_model_id},
        headers=headers,
    )
    if r.status_code != 200:
        return False
    names = {item.get("name", "").lower() for item in r.json()}
    return task_type.lower() in names


def _poll_until_terminal(api_url: str, headers: dict, task_id: str) -> dict:
    deadline = time.monotonic() + OPTIMIZATION_POLL_TIMEOUT_SEC
    last: dict = {}
    while time.monotonic() < deadline:
        r = requests.get(
            f"{api_url}/model-improvements/status",
            params={"task_id": task_id},
            headers=headers,
        )
        assert r.status_code == 200, f"상태 조회 실패: {r.status_code} {r.text}"
        last = r.json()
        st = (last.get("status") or "").upper()
        if st == "SUCCEEDED":
            return last
        if st == "FAILED":
            err = last.get("error") or last.get("message") or last
            pytest.fail(f"최적화 작업 실패(status=FAILED): {err}")
        time.sleep(OPTIMIZATION_POLL_INTERVAL_SEC)
    pytest.fail(f"{OPTIMIZATION_POLL_TIMEOUT_SEC}s 내 종료 상태 미도달. 마지막 응답: {last!r}")


@pytest.mark.model_improvement_scenario
class TestModelImprovementScenario:
    """작업 생성(202) → 폴링 → SUCCEEDED·result_model_id."""

    def test_create_and_poll_until_succeeded(self, api_url: str, auth_headers: dict):
        if not OPTIMIZATION_SOURCE_MODEL_NAME:
            pytest.skip(
                "소스 모델을 지정하세요: E2E_OPTIMIZATION_SOURCE_MODEL_NAME "
                "(워크플로 E2E_WORKFLOW_TARGET_LLM_MODEL 또는 E2E_TARGET_MODEL_NAME 과 같이 /models 의 name)"
            )

        r = requests.get(f"{api_url}/models", headers=auth_headers)
        assert r.status_code == 200, r.text
        models = r.json()

        source = find_optimization_source_model(models)
        assert (
            source is not None
        ), f"name={OPTIMIZATION_SOURCE_MODEL_NAME!r} 인 모델을 /models 목록에서 찾지 못했습니다."

        assert source.get("opt_enable_yn") is True, "소스 모델은 opt_enable_yn=true 여야 합니다."
        reg = source.get("registry") or {}
        assert reg.get("run_id"), "레지스트리에 MLflow run_id가 있어야 최적화 서버로 요청할 수 있습니다."
        assert (reg.get("artifact_path") or "").strip() or (
            reg.get("uri") or ""
        ).strip(), "레지스트리에 artifact_path 또는 uri가 있어야 합니다."

        assert _task_type_allowed_for_model(api_url, auth_headers, source["id"], OPTIMIZATION_TASK_TYPE), (
            f"task_type={OPTIMIZATION_TASK_TYPE!r} 은 이 소스에 허용되지 않습니다. "
            "task-types?source_model_id= 로 확인하세요."
        )

        r = requests.post(
            f"{api_url}/model-improvements",
            json={"source_model_id": source["id"], "task_type": OPTIMIZATION_TASK_TYPE},
            headers=auth_headers,
        )
        if r.status_code == 409:
            pytest.skip("동일 모델에 진행 중인 작업이 있어 409입니다. 완료 후 다시 실행하세요.")
        assert r.status_code == 202, f"작업 생성 실패: {r.status_code} {r.text}"
        body = r.json()
        task_id = body.get("task_id")
        assert task_id, f"응답에 task_id 없음: {body!r}"
        assert body.get("status") == "PENDING"

        print(
            f"\n▶ 최적화 작업 생성: task_id={task_id}, source_model_id={source['id']}, "
            f"task_type={OPTIMIZATION_TASK_TYPE}"
        )

        final = _poll_until_terminal(api_url, auth_headers, task_id)
        assert final.get("status") == "SUCCEEDED"
        rid = final.get("result_model_id")
        assert rid is not None and isinstance(rid, int), f"result_model_id 기대: {final!r}"
        print(f"▶ 완료: result_model_id={rid}")
