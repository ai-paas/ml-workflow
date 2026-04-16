"""
E2E 시나리오: RAG 워크플로우 삭제 → KB 삭제

.state.json 에서 rag_workflow_id, rag_kb_id 를 읽어 삭제한다.
사전 조건: make e2e-rag-workflow-deploy 로 배포된 상태

흐름:
  1. .state.json 에서 rag_workflow_id / rag_kb_id 를 읽음
  2. DELETE /workflows/{id}              — 워크플로우 삭제 시작
  3. POST   /workflows/{id}/finalize-deletion — 삭제 완료 폴링
  4. GET    /workflows/{id}              — 워크플로우 404 확인
  5. DELETE /knowledge-bases/{id}         — KB 삭제
  6. GET    /knowledge-bases/{id}         — KB 404 확인
"""

import time

import pytest
import requests
from config import DELETE_TIMEOUT_SEC, POLL_INTERVAL_SEC, clear_state, load_state

STATE_KEY_WF = "rag_workflow_id"
STATE_KEY_KB = "rag_kb_id"


@pytest.mark.rag_workflow_delete
class TestRagWorkflowDelete:
    """배포된 RAG 워크플로우 + KB를 삭제하는 시나리오"""

    workflow_id: str | None = None
    kb_id: int | None = None

    # ── 상태 로드 ────────────────────────────────────────────

    def test_01_load_state(self):
        """상태 파일에서 삭제 대상 ID를 읽는다."""
        wf_id = load_state(STATE_KEY_WF)
        kb_id = load_state(STATE_KEY_KB)
        assert wf_id or kb_id, (
            f"삭제할 {STATE_KEY_WF} / {STATE_KEY_KB} 가 없습니다. " "먼저 make e2e-rag-workflow-deploy 를 실행하세요."
        )
        self.__class__.workflow_id = wf_id
        self.__class__.kb_id = int(kb_id) if kb_id else None
        print(f"\n✔ 삭제 대상: workflow_id={wf_id}, kb_id={kb_id}")

    # ── 워크플로우 삭제 ──────────────────────────────────────

    def test_02_delete_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우 삭제를 시작한다."""
        wf_id = self.__class__.workflow_id
        if not wf_id:
            pytest.skip("워크플로우 ID가 없어 스킵합니다.")

        resp = requests.delete(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        print(f"\n✔ 워크플로우 삭제 시작: cleanup_run_id={data.get('cleanup_run_id')}")

    def test_03_finalize_deletion(self, api_url: str, auth_headers: dict):
        """K8s 리소스 정리를 폴링한다."""
        wf_id = self.__class__.workflow_id
        if not wf_id:
            pytest.skip("워크플로우 ID가 없어 스킵합니다.")

        deadline = time.time() + DELETE_TIMEOUT_SEC

        while time.time() < deadline:
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/finalize-deletion",
                headers=auth_headers,
            )
            assert resp.status_code in (200, 404)

            if resp.status_code == 404:
                print(f"\n✔ 워크플로우가 이미 삭제됨: {wf_id}")
                clear_state(STATE_KEY_WF)
                return

            data = resp.json()
            status_val = data.get("status")

            if status_val == "completed":
                print(f"\n✔ 워크플로우 삭제 완료: {wf_id}")
                clear_state(STATE_KEY_WF)
                return
            if status_val == "failed":
                pytest.fail(f"워크플로우 삭제 실패: {data.get('message', 'N/A')}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 리소스 정리 대기 중… status={status_val} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"삭제 타임아웃 ({DELETE_TIMEOUT_SEC}s 초과)")

    def test_04_verify_workflow_deleted(self, api_url: str, auth_headers: dict):
        """워크플로우가 완전히 삭제되었는지 확인한다."""
        wf_id = self.__class__.workflow_id
        if not wf_id:
            pytest.skip("워크플로우 ID가 없어 스킵합니다.")

        resp = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 404, f"워크플로우가 아직 존재합니다: {resp.status_code}"
        print(f"\n✔ 워크플로우 404 확인: {wf_id}")
        clear_state(STATE_KEY_WF)

    # ── KB 삭제 ──────────────────────────────────────────────

    def test_05_delete_knowledge_base(self, api_url: str, auth_headers: dict):
        """Knowledge Base를 삭제한다."""
        kb_id = self.__class__.kb_id
        if not kb_id:
            pytest.skip("KB ID가 없어 스킵합니다.")

        resp = requests.delete(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.status_code == 200, f"KB 삭제 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data.get("success") is True
        print(f"\n✔ KB 삭제 완료: kb_id={kb_id}")

    def test_06_verify_kb_deleted(self, api_url: str, auth_headers: dict):
        """KB가 완전히 삭제되었는지 확인한다."""
        kb_id = self.__class__.kb_id
        if not kb_id:
            pytest.skip("KB ID가 없어 스킵합니다.")

        resp = requests.get(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.status_code == 404, f"KB가 아직 존재합니다: {resp.status_code}"
        print(f"\n✔ KB 404 확인: kb_id={kb_id}")
        clear_state(STATE_KEY_KB)
