"""
E2E: 서비스 모니터링 metric 기록 검증 (단순 LLM 자기완결형)

목적: 워크플로우(시작→LLM→종료)를 **서비스에 연결**하여 배포하고, 추론을 수행한 뒤
서비스 상세 조회의 monitoring_data(1h/1d/1w)에 metric이 기록되는지 확인한다.

핵심: 추론 시 모니터링은 workflow.service_id 가 있을 때만 기록되므로,
**서비스를 먼저 만들고 그 service_id 로 워크플로우를 생성**하여 연결한다.

흐름:
  1.  GET  /models                          — LLM 타깃 모델
  2.  POST /prompts                          — 시나리오 프롬프트 (시나리오1은 없음 → no-op)
  3.  POST /services                         — 서비스 생성
  4.  POST /workflows (service_id 연결)      — 시작→LLM→종료, 서비스에 연결
  5.  POST /workflows/{id}/execute           — 배포
  6.  GET  /workflows/{id}/status            — 배포 완료 폴링
  7.  GET  /workflows/{id}/status            — ACTIVE 확인
  8.  POST /workflows/{id}/test/rag          — 추론 (서비스 연결되어 metric 기록)
  9.  GET  /services/{id}                    — monitoring_data.total_metrics["1h"] 검증
  10. (확인) 삭제 진행 여부 입력             — 거절 시 이후 정리 스킵, 리소스 유지(수동 점검용)
  11~13. DELETE /workflows/{id} → finalize → 404
  14~15. DELETE /services/{id} → 404
  16. DELETE /prompts/{id}                   — 프롬프트 정리 (있는 경우)

전제: 대상 서버가 1h/1d/1w 기간별 모니터링(정규화 마이그레이션 포함)으로 배포되어 있어야 한다.
시나리오: E2E_SCENARIO(기본 1). KB가 필요한 시나리오는 본 테스트 범위 밖이라 스킵한다.
"""

import time
import uuid

import pytest
import requests
from cleanup_prompt import confirm_cleanup
from config import (
    DELETE_TIMEOUT_SEC,
    DEPLOY_TIMEOUT_SEC,
    KB_TOP_K,
    POLL_INTERVAL_SEC,
    SCENARIO_NUM,
    workflow_primary_target_model_name,
)
from workflow.definitions import build_workflow_definition, get_scenario
from workflow.deploy_wait import expected_model_component_count, workflow_deploy_poll_should_fail

SCENARIO = get_scenario(SCENARIO_NUM)
INFERENCE_TIMEOUT_SEC = 120


class TestServiceMetricLifecycle:
    """서비스 연결 워크플로우 추론 → 서비스 metric 기록 검증 → 정리"""

    model: dict | None = None
    service_id: str | None = None
    workflow_id: str | None = None
    prompt_ids: dict = {}
    proceed_cleanup: bool | None = None

    def _skip_if_no_cleanup(self):
        """삭제 확인 단계에서 사용자가 거절하면 이후 정리 단계를 모두 스킵한다."""
        if not self.__class__.proceed_cleanup:
            pytest.skip("사용자가 삭제를 건너뜀 — 리소스 유지(수동 확인용)")

    @staticmethod
    def _find_model_by_name(api_url: str, headers: dict, model_name: str) -> dict:
        resp = requests.get(f"{api_url}/models", headers=headers)
        assert resp.status_code == 200, f"모델 목록 조회 실패: {resp.status_code} {resp.text}"
        matched = [m for m in resp.json() if m.get("name") == model_name]
        assert matched, f"name='{model_name}' 인 모델을 찾을 수 없습니다."
        return matched[0]

    # ── Phase 1: 사전 준비 ──────────────────────────────────

    def test_01_preconditions(self, api_url: str, auth_headers: dict):
        """단순 LLM(비-KB) 시나리오인지 확인하고 타깃 모델을 찾는다."""
        print(f"\n{'=' * 60}")
        print(f"  서비스 metric E2E — 시나리오 #{SCENARIO_NUM}: {SCENARIO['name']}")
        print(f"  구성: {SCENARIO['graph']}")
        print(f"{'=' * 60}")

        if SCENARIO["kb_count"] > 0:
            pytest.skip(
                f"시나리오 #{SCENARIO_NUM} 은 KB가 필요해 서비스 metric 테스트 범위 밖입니다. (LLM 전용, 예: SCENARIO=1)"
            )
        if SCENARIO.get("inference_kind") == "ml":
            pytest.skip(f"시나리오 #{SCENARIO_NUM} 은 ODM(ml) 이라 본 LLM metric 테스트 범위 밖입니다.")

        model = self._find_model_by_name(api_url, auth_headers, workflow_primary_target_model_name())
        assert model["id"]
        self.__class__.model = model
        print(f"\n✔ LLM 모델 발견: id={model['id']}, name={model['name']}")

    def test_02_create_prompts(self, api_url: str, auth_headers: dict):
        """시나리오 프롬프트를 생성한다 (시나리오1은 프롬프트가 없어 no-op)."""
        created: dict[str, int] = {}
        for comp_name, prompt_def in SCENARIO["prompts"].items():
            payload: dict = {
                "prompt": {
                    "name": f"E2E-SVC-S{SCENARIO_NUM}-{prompt_def['name']}-{uuid.uuid4().hex[:8]}",
                    "description": f"서비스 metric E2E 시나리오 #{SCENARIO_NUM} {comp_name}",
                    "content": prompt_def["content"],
                },
            }
            if prompt_def.get("has_context"):
                payload["prompt_variable"] = ["context"]
            resp = requests.post(f"{api_url}/prompts", json=payload, headers=auth_headers)
            assert resp.status_code == 201, f"프롬프트 생성 실패 ({comp_name}): {resp.status_code} {resp.text}"
            created[comp_name] = resp.json()["id"]
            print(f"  ✔ 프롬프트 생성: {comp_name} → id={created[comp_name]}")
        self.__class__.prompt_ids = created
        print(f"\n✔ 프롬프트 {len(created)}개 생성 (없으면 0)")

    def test_03_create_service(self, api_url: str, auth_headers: dict):
        """모니터링 대상 서비스를 생성한다."""
        payload = {
            "name": f"E2E-SVC-{uuid.uuid4().hex[:8]}",
            "description": "서비스 모니터링 metric E2E",
            "tags": ["e2e", "service-metric"],
        }
        resp = requests.post(f"{api_url}/services", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"서비스 생성 실패: {resp.status_code} {resp.text}"
        data = resp.json()
        self.__class__.service_id = data["id"]
        print(f"\n✔ 서비스 생성: id={data['id']}, name={data['name']}")

    # ── Phase 2: 서비스 연결 워크플로우 생성 & 배포 ─────────

    def test_04_create_workflow_linked_to_service(self, api_url: str, auth_headers: dict):
        """service_id 를 지정해 워크플로우를 서비스에 연결한 채로 생성한다."""
        model = self.__class__.model
        service_id = self.__class__.service_id
        assert model and service_id

        definition = build_workflow_definition(SCENARIO_NUM, model["id"], [], KB_TOP_K, self.__class__.prompt_ids)
        payload = {
            "name": f"E2E-SVC-WF-S{SCENARIO_NUM}-{uuid.uuid4().hex[:8]}",
            "description": f"서비스 metric E2E 워크플로우 (시나리오 #{SCENARIO_NUM})",
            "service_id": service_id,
            "workflow_definition": definition,
        }
        resp = requests.post(f"{api_url}/workflows", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"
        data = resp.json()
        self.__class__.workflow_id = data["id"]
        assert data.get("service_id") == service_id, f"서비스 연결 누락: service_id={data.get('service_id')}"
        print(f"\n✔ 워크플로우 생성(서비스 연결): id={data['id']}, service_id={data['service_id']}")

    def test_05_execute_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우를 실행(배포)한다."""
        wf_id = self.__class__.workflow_id
        resp = requests.post(f"{api_url}/workflows/{wf_id}/execute", headers=auth_headers)
        assert resp.status_code == 200, f"실행 실패: {resp.status_code} {resp.text}"
        assert resp.json()["status"] == "running"
        print(f"\n✔ 워크플로우 실행 시작: kubeflow_run_id={resp.json().get('kubeflow_run_id')}")

    def test_06_wait_for_deployment(self, api_url: str, auth_headers: dict):
        """배포 완료까지 폴링한다."""
        wf_id = self.__class__.workflow_id
        wf_read = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert wf_read.status_code == 200
        expected = expected_model_component_count(wf_read.json())
        assert expected > 0

        deadline = time.time() + DEPLOY_TIMEOUT_SEC
        last_statuses: list[str] = []
        while time.time() < deadline:
            resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            fail_reason = workflow_deploy_poll_should_fail(data, expected_model_deployments=expected)
            if fail_reason:
                pytest.fail(fail_reason)
            deployed = data.get("deployed_models", [])
            last_statuses = [m.get("status", "UNKNOWN") for m in deployed]
            if last_statuses and all(s == "DEPLOYED" for s in last_statuses):
                print(f"\n✔ 배포 완료! deployed_models={len(deployed)}")
                return
            if any(s == "FAILED" for s in last_statuses):
                pytest.fail(f"배포 실패: {last_statuses}")
            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 {int(deadline - time.time())}s)")
            time.sleep(POLL_INTERVAL_SEC)
        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s). 마지막 statuses={last_statuses}")

    def test_07_verify_active(self, api_url: str, auth_headers: dict):
        """배포 후 ACTIVE 상태 확인."""
        wf_id = self.__class__.workflow_id
        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json().get("status") == "ACTIVE", f"status={resp.json().get('status')}"
        print("\n✔ 상태 확인: ACTIVE")

    # ── Phase 3: 추론 → metric 검증 ─────────────────────────

    def test_08_inference(self, api_url: str, auth_headers: dict):
        """추론(test/rag) — 서비스 연결 워크플로우이므로 모니터링이 기록된다."""
        wf_id = self.__class__.workflow_id
        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/test/rag",
            data={"text": SCENARIO["inference_text"]},
            headers=auth_headers,
            timeout=INFERENCE_TIMEOUT_SEC,
        )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"
        data = resp.json()
        assert data["workflow_id"] == wf_id
        assert isinstance(data["results"], list) and len(data["results"]) > 0
        assert len(data.get("final_result") or "") > 0
        print(f"\n✔ 추론 성공: {(data.get('final_result') or '')[:80]}…")

    def test_09_verify_service_metrics(self, api_url: str, auth_headers: dict):
        """서비스 상세 조회로 1h/1d/1w 기간별 metric 기록을 검증한다."""
        service_id = self.__class__.service_id
        wf_id = self.__class__.workflow_id

        resp = requests.get(f"{api_url}/services/{service_id}", headers=auth_headers)
        assert resp.status_code == 200, f"서비스 조회 실패: {resp.status_code} {resp.text}"
        monitoring = resp.json().get("monitoring_data")
        assert monitoring, "monitoring_data 가 비어 있습니다."

        # 새 구조(1h/1d/1w + aggregated_at) 확인 — 옛 평면 구조면 서버가 구버전임
        assert "aggregated_at" in monitoring, "aggregated_at 없음 — 서버가 기간별 모니터링 코드가 아닌 듯합니다."
        total = monitoring["total_metrics"]
        assert {"1h", "1d", "1w"} <= set(total.keys()), f"기간 키 누락: {list(total.keys())}"

        h1 = total["1h"]
        print(f"\n  total_metrics.1h = {h1}")
        assert h1["message_count"] >= 1, f"최근 1시간 메시지 수가 0입니다: {h1}"
        assert h1["active_users"] >= 1, f"활성 사용자 수가 0입니다: {h1}"
        assert h1["error_count"] == 0, f"오류가 기록됨: {h1}"
        assert h1["success_rate"] == 100.0, f"성공률이 100이 아님: {h1}"
        # 누적 윈도우이므로 1d/1w 도 1h 이상 포함
        assert total["1d"]["message_count"] >= h1["message_count"]
        assert total["1w"]["message_count"] >= h1["message_count"]

        # 워크플로우별 metric 에도 해당 워크플로우가 잡혀야 한다
        wf_metrics = {w["workflow_id"]: w for w in monitoring["workflow_metrics"]}
        assert wf_id in wf_metrics, f"workflow_metrics 에 {wf_id} 없음: {list(wf_metrics.keys())}"
        assert wf_metrics[wf_id]["metrics"]["1h"]["message_count"] >= 1
        print(f"✔ 서비스 metric 기록 확인: 1h.message_count={h1['message_count']}, active_users={h1['active_users']}")

    # ── Phase 4: 정리 (삭제 전 확인) ─────────────────────────

    def test_10_confirm_cleanup(self, api_url: str, auth_headers: dict):
        """삭제 전에 사용자에게 확인. 거절하면 이후 정리 단계를 스킵하고 리소스를 남긴다.

        남겨두면 DB 상태 확인·추가 추론으로 상태를 직접 더 점검할 수 있다.
        비대화형(파이프/CI)이면 자동으로 정리를 진행한다.
        """
        service_id = self.__class__.service_id
        wf_id = self.__class__.workflow_id
        self.__class__.proceed_cleanup = confirm_cleanup(
            [
                f"service_id = {service_id}",
                f"workflow_id = {wf_id}",
                f"상태 확인 예: GET {api_url}/services/{service_id}",
                f"추가 추론 예: POST {api_url}/workflows/{wf_id}/test/rag",
            ]
        )
        if not self.__class__.proceed_cleanup:
            print(f"   워크플로우 삭제: DELETE {api_url}/workflows/{wf_id} → finalize-deletion")
            print(f"   서비스 삭제: DELETE {api_url}/services/{service_id}")

    def test_11_delete_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우 삭제 시작 (모니터링 행은 FK CASCADE 로 함께 삭제됨)."""
        self._skip_if_no_cleanup()
        wf_id = self.__class__.workflow_id
        resp = requests.delete(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"
        print(f"\n✔ 워크플로우 삭제 시작: cleanup_run_id={resp.json().get('cleanup_run_id')}")

    def test_12_finalize_deletion(self, api_url: str, auth_headers: dict):
        """K8s 리소스 정리 + DB 삭제 완료를 폴링한다."""
        self._skip_if_no_cleanup()
        wf_id = self.__class__.workflow_id
        deadline = time.time() + DELETE_TIMEOUT_SEC
        while time.time() < deadline:
            resp = requests.post(f"{api_url}/workflows/{wf_id}/finalize-deletion", headers=auth_headers)
            assert resp.status_code in (200, 404)
            if resp.status_code == 404:
                print("\n✔ 워크플로우 이미 삭제됨")
                return
            status_val = resp.json().get("status")
            if status_val == "completed":
                print("\n✔ 워크플로우 삭제 완료")
                return
            if status_val == "failed":
                pytest.fail(f"워크플로우 삭제 실패: {resp.json().get('message', 'N/A')}")
            print(f"  ⏳ 정리 대기 중… status={status_val} (남은 {int(deadline - time.time())}s)")
            time.sleep(POLL_INTERVAL_SEC)
        pytest.fail(f"삭제 타임아웃 ({DELETE_TIMEOUT_SEC}s)")

    def test_13_verify_workflow_deleted(self, api_url: str, auth_headers: dict):
        """워크플로우 404 확인."""
        self._skip_if_no_cleanup()
        wf_id = self.__class__.workflow_id
        resp = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 404, f"워크플로우가 아직 존재: {resp.status_code}"
        print("\n✔ 워크플로우 404 확인")

    def test_14_delete_service(self, api_url: str, auth_headers: dict):
        """서비스 삭제 (모니터링 행은 FK CASCADE 로 함께 삭제, 다른 워크플로우는 SET NULL 로 보존)."""
        self._skip_if_no_cleanup()
        service_id = self.__class__.service_id
        resp = requests.delete(f"{api_url}/services/{service_id}", headers=auth_headers)
        assert resp.status_code == 204, f"서비스 삭제 실패: {resp.status_code} {resp.text}"
        print(f"\n✔ 서비스 삭제 완료: {service_id}")

    def test_15_verify_service_deleted(self, api_url: str, auth_headers: dict):
        """서비스 404 확인."""
        self._skip_if_no_cleanup()
        service_id = self.__class__.service_id
        resp = requests.get(f"{api_url}/services/{service_id}", headers=auth_headers)
        assert resp.status_code == 404, f"서비스가 아직 존재: {resp.status_code}"
        print("\n✔ 서비스 404 확인")

    def test_16_delete_prompts(self, api_url: str, auth_headers: dict):
        """프롬프트 정리 (있는 경우)."""
        self._skip_if_no_cleanup()
        prompt_ids = self.__class__.prompt_ids
        if not prompt_ids:
            pytest.skip("프롬프트가 없어 스킵합니다.")
        for comp_name, pid in prompt_ids.items():
            resp = requests.delete(f"{api_url}/prompts/{pid}", headers=auth_headers)
            assert resp.status_code == 204, f"프롬프트 삭제 실패 ({comp_name}, id={pid}): {resp.status_code}"
            print(f"  ✔ 프롬프트 삭제: {comp_name} (id={pid})")
