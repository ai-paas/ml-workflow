"""
E2E 시나리오: LLM 워크플로우 전체 생명주기

LLM 모델만을 컴포넌트로 사용하는 워크플로우의 전체 생명주기를 검증한다.

흐름:
  1. GET  /models                       — 대상 LLM 모델 검색
  2. POST /workflows                    — 워크플로우 생성 (시작→LLM→끝)
  3. POST /workflows/{id}/execute       — 워크플로우 실행 (KServe 배포)
  4. GET  /workflows/{id}/status        — 배포 완료 폴링
  5. GET  /workflows/{id}               — ACTIVE 상태 확인
  6. POST /workflows/{id}/test/rag      — LLM 추론 테스트 + 응답 형식 검증
  7. DELETE /workflows/{id}             — 삭제 시작 (리소스 정리 파이프라인)
  8. POST /workflows/{id}/finalize-deletion — 삭제 완료 폴링
  9. GET  /workflows/{id}               — 404 확인 (완전 삭제)
"""

import time
import uuid

import pytest
import requests
from config import DELETE_TIMEOUT_SEC, DEPLOY_TIMEOUT_SEC, POLL_INTERVAL_SEC, TARGET_MODEL_NAME

INFERENCE_TIMEOUT_SEC = 120


@pytest.mark.llm_workflow_lifecycle
class TestLlmWorkflowLifecycle:
    """LLM 워크플로우 생성 → 배포 → 추론 테스트 → 삭제 전체 흐름"""

    workflow_id: str | None = None
    model: dict | None = None

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _find_model_by_name(api_url: str, headers: dict, model_name: str) -> dict:
        resp = requests.get(f"{api_url}/models", headers=headers)
        assert resp.status_code == 200, f"모델 목록 조회 실패: {resp.status_code} {resp.text}"

        models = resp.json()
        matched = [m for m in models if m.get("name") == model_name]
        if not matched:
            available = [f"  id={m.get('id')}, name={m.get('name')}" for m in models]
            listing = "\n".join(available) if available else "  (없음)"
            raise AssertionError(
                f"name='{model_name}' 인 모델을 찾을 수 없습니다.\n"
                f"등록된 모델 목록 ({len(models)}개): \n{listing}\n\n"
                f".env 의 E2E_TARGET_MODEL_NAME 을 위 목록에서 선택하세요."
            )
        return matched[0]

    @staticmethod
    def _build_workflow_definition(model_id: int) -> dict:
        start_ref = f"start-{uuid.uuid4().hex[:8]}"
        model_ref = f"model-{uuid.uuid4().hex[:8]}"
        end_ref = f"end-{uuid.uuid4().hex[:8]}"

        return {
            "components": [
                {"ref_id": start_ref, "name": "시작", "type": "START"},
                {
                    "ref_id": model_ref,
                    "name": "GPT-OSS-20B 모델",
                    "type": "MODEL",
                    "model_id": model_id,
                    "config": {"temperature": 0.7, "max_tokens": 512},
                },
                {"ref_id": end_ref, "name": "끝", "type": "END"},
            ],
            "connections": [
                {"source_ref_id": start_ref, "target_ref_id": model_ref},
                {"source_ref_id": model_ref, "target_ref_id": end_ref},
            ],
        }

    # ── Phase 1: 생성 & 배포 ────────────────────────────────

    def test_01_find_model(self, api_url: str, auth_headers: dict):
        """대상 모델이 존재해야 한다."""
        model = self._find_model_by_name(api_url, auth_headers, TARGET_MODEL_NAME)
        assert model["id"], "모델 ID가 비어 있습니다."
        self.__class__.model = model
        print(f"\n✔ 모델 발견: id={model['id']}, name={model['name']}")

    def test_02_create_workflow(self, api_url: str, auth_headers: dict):
        """시작 → LLM → 끝 워크플로우를 생성한다."""
        model = self.__class__.model
        assert model, "test_01 에서 모델을 찾지 못했습니다."

        definition = self._build_workflow_definition(model["id"])
        payload = {
            "name": f"E2E-lifecycle-{uuid.uuid4().hex[:8]}",
            "description": "E2E 테스트: 전체 생명주기 시나리오",
            "workflow_definition": definition,
        }

        resp = requests.post(f"{api_url}/workflows", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.workflow_id = data["id"]
        assert data["status"] == "DRAFT"
        print(f"\n✔ 워크플로우 생성 완료: id={data['id']}, status={data['status']}")

    def test_03_execute_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우를 실행(배포)한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id, "test_02 에서 워크플로우가 생성되지 않았습니다."

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/execute",
            json={"parameters": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"워크플로우 실행 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data["status"] == "running", f"예상 status='running', 실제='{data['status']}'"
        print(f"\n✔ 워크플로우 실행 시작: kubeflow_run_id={data.get('kubeflow_run_id')}")

    def test_04_wait_for_deployment(self, api_url: str, auth_headers: dict):
        """배포가 완료될 때까지 폴링한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id, "test_03 에서 워크플로우가 실행되지 않았습니다."

        deadline = time.time() + DEPLOY_TIMEOUT_SEC
        last_statuses: list[str] = []

        while time.time() < deadline:
            resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
            assert resp.status_code == 200, f"상태 조회 실패: {resp.status_code} {resp.text}"

            data = resp.json()
            deployed_models = data.get("deployed_models", [])
            last_statuses = [m.get("status", "UNKNOWN") for m in deployed_models]

            all_deployed = all(s == "DEPLOYED" for s in last_statuses) and len(last_statuses) > 0
            any_failed = any(s == "FAILED" for s in last_statuses)

            if all_deployed:
                print(f"\n✔ 배포 완료! deployed_models={len(deployed_models)}")
                for m in deployed_models:
                    print(f" - {m.get('model_name')}: status={m.get('status')}")
                return

            if any_failed:
                failed = [m for m in deployed_models if m.get("status") == "FAILED"]
                msgs = [f"{m.get('model_name')}: {m.get('error_message', 'N/A')}" for m in failed]
                pytest.fail(f"배포 실패 감지: {'; '.join(msgs)}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s 초과). 마지막 statuses={last_statuses}")

    def test_05_verify_active(self, api_url: str, auth_headers: dict):
        """배포 완료 후 워크플로우 상태가 ACTIVE 인지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200

        data = resp.json()
        assert data.get("status") == "ACTIVE", f"예상 status='ACTIVE', 실제='{data.get('status')}'"
        print(f"\n✔ 상태 확인: workflow_id={wf_id}, status=ACTIVE")

    # ── Phase 2: 추론 테스트 ────────────────────────────────

    def test_06_rag_inference(self, api_url: str, auth_headers: dict):
        """LLM 추론 테스트 — /test/rag API를 호출하고 응답 형식을 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/test/rag",
            data={"text": "안녕하세요?"},
            headers=auth_headers,
            timeout=INFERENCE_TIMEOUT_SEC,
        )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"

        data = resp.json()

        # 최상위 필드 검증
        assert data["workflow_id"] == wf_id
        assert isinstance(data["execution_order"], list) and len(data["execution_order"]) > 0
        assert isinstance(data["results"], list) and len(data["results"]) > 0
        assert data["final_result"] is not None and len(data["final_result"]) > 0

        print(f"\n✔ 추론 성공: workflow_id={wf_id}")
        print(f"  execution_order: {data['execution_order']}")
        print(f"  final_result: {data['final_result'][:100]}{'…' if len(data['final_result']) > 100 else ''}")

    def test_07_rag_response_structure(self, api_url: str, auth_headers: dict):
        """추론 응답의 results[] 내부 구조를 상세 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/test/rag",
            data={"text": "대한민국의 수도는 어디인가요?"},
            headers=auth_headers,
            timeout=INFERENCE_TIMEOUT_SEC,
        )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        results = data["results"]
        assert len(results) >= 1, "results 가 비어 있습니다."

        for idx, result in enumerate(results):
            # 공통 필드
            assert "component_id" in result, f"results[{idx}]: component_id 누락"
            assert "component_name" in result, f"results[{idx}]: component_name 누락"
            assert "component_type" in result, f"results[{idx}]: component_type 누락"
            assert "model_type" in result, f"results[{idx}]: model_type 누락"

            if result["component_type"] == "MODEL":
                assert (
                    result["model_type"] == "LLM"
                ), f"results[{idx}]: LLM 워크플로우인데 model_type='{result['model_type']}'"
                assert "result" in result, f"results[{idx}]: result 필드 누락"

                llm_result = result["result"]
                assert "response" in llm_result, f"results[{idx}].result: response 누락"
                assert isinstance(llm_result["response"], str), f"results[{idx}].result.response 가 문자열이 아님"
                assert len(llm_result["response"]) > 0, f"results[{idx}].result.response 가 빈 문자열"

                assert "full_response" in llm_result, f"results[{idx}].result: full_response 누락"
                full_resp = llm_result["full_response"]
                assert isinstance(full_resp, dict), f"results[{idx}].result.full_response 가 dict 이 아님"
                assert full_resp.get("done") is True, f"results[{idx}]: Ollama 응답이 완료되지 않음 (done≠true)"
                assert "model" in full_resp, f"results[{idx}].result.full_response: model 누락"

                print(f"\n✔ results[{idx}] 검증 통과: ")
                print(f"  component: {result['component_name']} ({result['component_type']}/{result['model_type']})")
                print(f"  response: {llm_result['response'][:80]}{'…' if len(llm_result['response']) > 80 else ''}")
                print(f"  model: {full_resp.get('model')}")
                print(f"  eval_count: {full_resp.get('eval_count')}, done_reason: {full_resp.get('done_reason')}")

        # final_result 는 마지막 LLM의 response 와 일치해야 함
        last_llm = [r for r in results if r.get("component_type") == "MODEL"]
        if last_llm:
            expected_final = last_llm[-1]["result"]["response"]
            assert (
                data["final_result"] == expected_final
            ), f"final_result 불일치: '{data['final_result'][:50]}' ≠ '{expected_final[:50]}'"
            print("\n✔ final_result 검증 통과 (마지막 LLM response 와 일치)")

    # ── Phase 3: 삭제 ───────────────────────────────────────

    def test_08_delete_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우 삭제를 시작한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.delete(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        print(f"\n✔ 삭제 시작: status={data.get('status')}, cleanup_run_id={data.get('cleanup_run_id')}")

    def test_09_finalize_deletion(self, api_url: str, auth_headers: dict):
        """K8s 리소스가 정리될 때까지 폴링한 뒤 DB 삭제를 완료한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        deadline = time.time() + DELETE_TIMEOUT_SEC

        while time.time() < deadline:
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/finalize-deletion",
                headers=auth_headers,
            )
            assert resp.status_code in (200, 404), f"finalize-deletion 실패: {resp.status_code} {resp.text}"

            if resp.status_code == 404:
                print(f"\n✔ 워크플로우가 이미 삭제됨: workflow_id={wf_id}")
                return

            data = resp.json()
            status_val = data.get("status")

            if status_val == "completed":
                print(f"\n✔ 삭제 완료: workflow_id={wf_id}, deleted_from_db={data.get('deleted_from_db')}")
                return

            if status_val == "failed":
                pytest.fail(f"삭제 실패: {data.get('message', 'N/A')}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 리소스 정리 대기 중… status={status_val} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"삭제 타임아웃 ({DELETE_TIMEOUT_SEC}s 초과)")

    def test_10_verify_deleted(self, api_url: str, auth_headers: dict):
        """워크플로우가 완전히 삭제되었는지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 404, f"워크플로우가 아직 존재합니다: {resp.status_code} {resp.text}"
        print(f"\n✔ 404 확인: workflow_id={wf_id} 이(가) 완전히 삭제되었습니다.")
