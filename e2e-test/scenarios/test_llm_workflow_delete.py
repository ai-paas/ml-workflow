"""
E2E 시나리오: LLM 워크플로우 삭제 → 삭제 완료 확인

흐름:
  1. .state.json 에서 llm_workflow_id 를 읽음
  2. DELETE /workflows/{id}              — 삭제 시작 (KServe 리소스 정리 파이프라인)
  3. POST   /workflows/{id}/finalize-deletion — 폴링하여 삭제 완료 확인
"""

import time

import pytest
import requests
from config import DELETE_TIMEOUT_SEC, POLL_INTERVAL_SEC, clear_state, load_state

STATE_KEY = "llm_workflow_id"


@pytest.mark.llm_workflow_delete
class TestLlmWorkflowDelete:
    """배포된 LLM 워크플로우를 삭제하고 완전히 정리되었는지 확인하는 시나리오"""

    workflow_id: str | None = None

    def test_01_load_workflow_id(self):
        """상태 파일에서 삭제 대상 workflow_id 를 읽는다."""
        wf_id = load_state(STATE_KEY)
        assert wf_id, f"삭제할 {STATE_KEY} 가 없습니다. " "먼저 make e2e-llm-workflow-deploy 를 실행하세요."
        self.__class__.workflow_id = wf_id
        print(f"\n✔ 삭제 대상: workflow_id={wf_id}")

    def test_02_delete_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우 삭제를 시작한다 (KServe 리소스 정리 파이프라인 실행)."""
        wf_id = self.__class__.workflow_id
        assert wf_id, "test_01 에서 workflow_id 를 불러오지 못했습니다."

        resp = requests.delete(
            f"{api_url}/workflows/{wf_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        print(f"\n✔ 삭제 시작: status={data.get('status')}, cleanup_run_id={data.get('cleanup_run_id')}")

    def test_03_finalize_deletion(self, api_url: str, auth_headers: dict):
        """K8s 리소스가 정리될 때까지 폴링한 뒤 DB 삭제를 완료한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id, "test_02 에서 삭제가 시작되지 않았습니다."

        deadline = time.time() + DELETE_TIMEOUT_SEC

        while time.time() < deadline:
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/finalize-deletion",
                headers=auth_headers,
            )
            assert resp.status_code in (200, 404), f"finalize-deletion 실패: {resp.status_code} {resp.text}"

            if resp.status_code == 404:
                print(f"\n✔ 워크플로우가 이미 삭제되었습니다: workflow_id={wf_id}")
                clear_state(STATE_KEY)
                return

            data = resp.json()
            status = data.get("status")

            if status == "completed":
                print(f"\n✔ 삭제 완료: workflow_id={wf_id}, deleted_from_db={data.get('deleted_from_db')}")
                clear_state(STATE_KEY)
                return

            if status == "failed":
                pytest.fail(f"삭제 실패: {data.get('message', 'N/A')}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 리소스 정리 대기 중… status={status} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"삭제 타임아웃 ({DELETE_TIMEOUT_SEC}s 초과)")

    def test_04_verify_deleted(self, api_url: str, auth_headers: dict):
        """워크플로우가 실제로 조회되지 않는지 확인한다."""
        wf_id = self.__class__.workflow_id
        if not wf_id:
            pytest.skip("이미 삭제 확인 완료")

        resp = requests.get(
            f"{api_url}/workflows/{wf_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 404, f"워크플로우가 아직 존재합니다: {resp.status_code} {resp.text}"
        print(f"\n✔ 404 확인: workflow_id={wf_id} 이(가) 완전히 삭제되었습니다.")
        clear_state(STATE_KEY)
