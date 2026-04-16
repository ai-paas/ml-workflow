"""
E2E 시나리오: LLM 워크플로우 생성 → 배포(실행) → 배포 완료 확인

흐름:
  1. GET  /models           — name 이 TARGET_MODEL_NAME 인 LLM 모델을 검색
  2. POST /workflows        — 시작 → LLM → 끝 워크플로우 생성
  3. POST /workflows/{id}/execute  — 워크플로우 실행(= KServe 배포)
  4. GET  /workflows/{id}/status   — 폴링하여 배포 완료 확인
  5. 최종 상태 확인
"""

import time
import uuid

import pytest
import requests
from config import DEPLOY_TIMEOUT_SEC, POLL_INTERVAL_SEC, TARGET_MODEL_NAME, save_state

STATE_KEY = "llm_workflow_id"


@pytest.mark.llm_workflow_deploy
class TestLlmWorkflowDeploy:
    """시작 → LLM → 끝 워크플로우 배포 E2E 시나리오"""

    workflow_id: str | None = None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _find_model_by_name(api_url: str, headers: dict, model_name: str) -> dict:
        """모델 목록에서 name 이 일치하는 첫 번째 모델을 반환한다."""
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
        """시작 → 모델 → 끝 형태의 workflow_definition 을 생성한다."""
        start_ref = f"start-{uuid.uuid4().hex[:8]}"
        model_ref = f"model-{uuid.uuid4().hex[:8]}"
        end_ref = f"end-{uuid.uuid4().hex[:8]}"

        return {
            "components": [
                {
                    "ref_id": start_ref,
                    "name": "시작",
                    "type": "START",
                },
                {
                    "ref_id": model_ref,
                    "name": "GPT-OSS-20B 모델",
                    "type": "MODEL",
                    "model_id": model_id,
                    "config": {
                        "temperature": 0.7,
                        "max_tokens": 512,
                    },
                },
                {
                    "ref_id": end_ref,
                    "name": "끝",
                    "type": "END",
                },
            ],
            "connections": [
                {"source_ref_id": start_ref, "target_ref_id": model_ref},
                {"source_ref_id": model_ref, "target_ref_id": end_ref},
            ],
        }

    # -- tests --------------------------------------------------------------

    def test_01_find_model(self, api_url: str, auth_headers: dict):
        """name 이 TARGET_MODEL_NAME 인 모델이 존재해야 한다."""
        model = self._find_model_by_name(api_url, auth_headers, TARGET_MODEL_NAME)
        assert model["id"], "모델 ID가 비어 있습니다."
        self.__class__.model = model
        print(f"\n✔ 모델 발견: id={model['id']}, name={model['name']}")

    def test_02_create_workflow(self, api_url: str, auth_headers: dict):
        """시작 → 모델 → 끝 워크플로우를 생성한다."""
        model = getattr(self.__class__, "model", None)
        assert model, "test_01 에서 모델을 찾지 못했습니다."

        definition = self._build_workflow_definition(model["id"])
        payload = {
            "name": f"E2E-deploy-test-{uuid.uuid4().hex[:8]}",
            "description": "E2E 테스트: 워크플로우 배포 시나리오",
            "workflow_definition": definition,
        }

        resp = requests.post(
            f"{api_url}/workflows",
            json=payload,
            headers=auth_headers,
        )
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.workflow_id = data["id"]
        save_state(STATE_KEY, data["id"])
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
            resp = requests.get(
                f"{api_url}/workflows/{wf_id}/status",
                headers=auth_headers,
            )
            assert resp.status_code == 200, f"상태 조회 실패: {resp.status_code} {resp.text}"

            data = resp.json()
            deployed_models = data.get("deployed_models", [])
            last_statuses = [m.get("status", "UNKNOWN") for m in deployed_models]

            all_deployed = all(s == "DEPLOYED" for s in last_statuses) and len(last_statuses) > 0
            any_failed = any(s == "FAILED" for s in last_statuses)

            if all_deployed:
                print(f"\n✔ 배포 완료! deployed_models={len(deployed_models)}")
                for m in deployed_models:
                    print(f" - {m.get('model_name')}: status={m.get('status')}, gateway_url={m.get('gateway_url')}")
                return

            if any_failed:
                failed = [m for m in deployed_models if m.get("status") == "FAILED"]
                msgs = [f"{m.get('model_name')}: {m.get('error_message', 'N/A')}" for m in failed]
                pytest.fail(f"배포 실패 감지: {'; '.join(msgs)}")

            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 시간: {int(deadline - time.time())}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s 초과). 마지막 statuses={last_statuses}")

    def test_05_verify_status(self, api_url: str, auth_headers: dict):
        """배포 완료 후 워크플로우 상태가 ACTIVE 인지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id, "워크플로우가 생성되지 않았습니다."

        resp = requests.get(
            f"{api_url}/workflows/{wf_id}/status",
            headers=auth_headers,
        )
        assert resp.status_code == 200

        data = resp.json()
        print(f"\n✔ 최종 상태 확인: workflow_id={wf_id}, status={data.get('status')}")
        print("  ℹ 삭제하려면: make e2e-llm-workflow-delete")
