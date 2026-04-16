"""
E2E 시나리오: 워크플로우 시나리오 삭제 테스트

.state.json 에서 해당 시나리오의 workflow_id, kb_ids, prompt_ids 를 읽어 삭제한다.
사전 조건: make e2e-wf-scenario-deploy SCENARIO=N 으로 배포된 상태

시나리오 선택: E2E_SCENARIO 환경변수 (필수, Makefile에서 SCENARIO 인자로 주입)

흐름:
  1. .state.json 에서 scenario_N_workflow_id / kb_ids / prompt_ids 를 읽음
  2. DELETE /workflows/{id}              — 워크플로우 삭제 시작
  3. POST   /workflows/{id}/finalize-deletion — 삭제 완료 폴링
  4. GET    /workflows/{id}              — 워크플로우 404 확인
  5. DELETE /knowledge-bases/{id}         — KB 삭제 (1~2개, 있는 경우)
  6. GET    /knowledge-bases/{id}         — KB 404 확인 (있는 경우)
  7. DELETE /prompts/{id}                 — 프롬프트 삭제
"""

import json
import time

import pytest
import requests
from config import DELETE_TIMEOUT_SEC, POLL_INTERVAL_SEC, SCENARIO_NUM, clear_state, load_state
from workflow_scenarios import get_scenario, state_key_kb_ids, state_key_prompts, state_key_wf

SCENARIO = get_scenario(SCENARIO_NUM)


@pytest.mark.workflow_scenario_delete
class TestWorkflowScenarioDelete:
    """배포된 워크플로우 시나리오를 삭제하는 테스트"""

    workflow_id: str | None = None
    kb_ids: list[int] = []
    prompt_ids: dict = {}

    # ── 상태 로드 ────────────────────────────────────────────

    def test_01_load_state(self):
        """상태 파일에서 삭제 대상 ID를 읽는다."""
        print(f"\n{'=' * 60}")
        print(f"  시나리오 #{SCENARIO_NUM}: {SCENARIO['name']}")
        print(f"  구성: {SCENARIO['graph']}")
        print(f"{'=' * 60}")

        wf_id = load_state(state_key_wf(SCENARIO_NUM))
        kb_ids_str = load_state(state_key_kb_ids(SCENARIO_NUM))
        prompt_ids_str = load_state(state_key_prompts(SCENARIO_NUM))

        assert wf_id, (
            f"시나리오 #{SCENARIO_NUM} 의 워크플로우 ID가 없습니다.\n"
            f"먼저 make e2e-wf-scenario-deploy SCENARIO={SCENARIO_NUM} 을 실행하세요."
        )

        self.__class__.workflow_id = wf_id
        self.__class__.kb_ids = json.loads(kb_ids_str) if kb_ids_str else []
        self.__class__.prompt_ids = json.loads(prompt_ids_str) if prompt_ids_str else {}
        print(
            f"\n✔ 삭제 대상: workflow_id={wf_id}, "
            f"kb_ids={self.__class__.kb_ids}, "
            f"prompts={len(self.__class__.prompt_ids)}개"
        )

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
                clear_state(state_key_wf(SCENARIO_NUM))
                return

            data = resp.json()
            status_val = data.get("status")

            if status_val == "completed":
                print(f"\n✔ 워크플로우 삭제 완료: {wf_id}")
                clear_state(state_key_wf(SCENARIO_NUM))
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
        clear_state(state_key_wf(SCENARIO_NUM))

    # ── KB 삭제 ──────────────────────────────────────────────

    def test_05_delete_knowledge_bases(self, api_url: str, auth_headers: dict):
        """Knowledge Base를 삭제한다 (1~2개, 있는 경우)."""
        kb_ids = self.__class__.kb_ids
        if not kb_ids:
            pytest.skip("KB ID가 없어 스킵합니다.")

        for idx, kb_id in enumerate(kb_ids):
            resp = requests.delete(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
            assert resp.status_code == 200, f"KB 삭제 실패 (KB{idx + 1}, id={kb_id}): {resp.status_code} {resp.text}"

            data = resp.json()
            assert data.get("success") is True
            print(f"  ✔ KB{idx + 1} 삭제 완료: kb_id={kb_id}")

        print(f"\n✔ KB {len(kb_ids)}개 삭제 완료")

    def test_06_verify_kbs_deleted(self, api_url: str, auth_headers: dict):
        """KB가 완전히 삭제되었는지 확인한다."""
        kb_ids = self.__class__.kb_ids
        if not kb_ids:
            pytest.skip("KB ID가 없어 스킵합니다.")

        for idx, kb_id in enumerate(kb_ids):
            resp = requests.get(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
            assert resp.status_code == 404, f"KB{idx + 1}이 아직 존재합니다 (id={kb_id}): {resp.status_code}"
            print(f"  ✔ KB{idx + 1} 404 확인: kb_id={kb_id}")

        clear_state(state_key_kb_ids(SCENARIO_NUM))

    # ── 프롬프트 삭제 ────────────────────────────────────────

    def test_07_delete_prompts(self, api_url: str, auth_headers: dict):
        """시나리오에서 생성한 프롬프트를 삭제한다."""
        prompt_ids = self.__class__.prompt_ids
        if not prompt_ids:
            pytest.skip("프롬프트 ID가 없어 스킵합니다.")

        for comp_name, pid in prompt_ids.items():
            resp = requests.delete(f"{api_url}/prompts/{pid}", headers=auth_headers)
            assert resp.status_code == 204, f"프롬프트 삭제 실패 ({comp_name}, id={pid}): {resp.status_code}"
            print(f"  ✔ 프롬프트 삭제: {comp_name} (id={pid})")

        clear_state(state_key_prompts(SCENARIO_NUM))
        print(f"\n✔ 프롬프트 {len(prompt_ids)}개 삭제 완료")
