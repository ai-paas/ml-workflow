"""
E2E 시나리오: 워크플로우 템플릿 생성 → 복제 → 실행 → 추론 → 정리

가장 단순한 정의(시나리오 #1: START → LLM → END, KB/프롬프트 불필요)로 다음을 검증한다.
  1.  GET    /models                                 — 타깃 LLM 모델 검색
  2.  POST   /workflows/templates                    — 템플릿 생성 (workflow_definition 포함)
  3.  GET    /workflows/templates/{id}               — 템플릿 상세 조회 (x, y 좌표 검증)
  4.  POST   /workflows/templates/{id}/clone         — 템플릿으로부터 워크플로우 복제
  5.  POST   /workflows/{id}/execute                 — 복제 워크플로우 실행 (KServe 배포)
  6.  GET    /workflows/{id}/status                  — 배포 완료 폴링
  7.  GET    /workflows/{id}/status                  — ACTIVE 상태 확인
  8.  POST   /workflows/{id}/test/rag                — 추론 테스트
  9.  DELETE /workflows/{id}                         — 워크플로우 삭제 시작
  10. POST   /workflows/{id}/finalize-deletion       — 삭제 완료 폴링
  11. GET    /workflows/{id}                         — 워크플로우 404 확인
  12. DELETE /workflows/templates/{id}               — 템플릿 삭제

핵심 검증: 템플릿 생성 시 보낸 컴포넌트 x, y 좌표가 (1) 템플릿 조회, (2) 복제된 워크플로우 조회에서
그대로 유지되는지.
"""

import time
import uuid

import pytest
import requests
from config import DELETE_TIMEOUT_SEC, DEPLOY_TIMEOUT_SEC, POLL_INTERVAL_SEC, WORKFLOW_TARGET_LLM_MODEL
from workflow.definitions import build_workflow_definition, get_scenario
from workflow.deploy_wait import expected_model_component_count, workflow_deploy_poll_should_fail

# 가장 단순한 시나리오(단순 LLM) 사용 — KB·프롬프트 불필요
TEMPLATE_SCENARIO_NUM = 1
TEMPLATE_SCENARIO = get_scenario(TEMPLATE_SCENARIO_NUM)
INFERENCE_TIMEOUT_SEC = 120


@pytest.mark.workflow_template_clone
class TestWorkflowTemplateClone:
    """템플릿 생성 → 복제 → 실행 → 추론 → 정리 전체 흐름을 검증한다."""

    model: dict | None = None
    definition: dict | None = None
    template_id: str | None = None
    workflow_id: str | None = None

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
                f"name='{model_name}' 인 모델을 찾을 수 없습니다.\n등록된 모델 목록 ({len(models)}개): \n{listing}"
            )
        return matched[0]

    @staticmethod
    def _coords_by_name(components: list[dict]) -> dict[str, tuple[int | None, int | None]]:
        return {c["name"]: (c.get("x"), c.get("y")) for c in components}

    # ── Phase 1: 사전 준비 ───────────────────────────────────

    def test_01_find_llm_model(self, api_url: str, auth_headers: dict):
        """타깃 LLM 모델이 존재해야 한다."""
        print(f"\n{'=' * 60}")
        print(f"  템플릿 시나리오: 단순 LLM (시나리오 #{TEMPLATE_SCENARIO_NUM})")
        print(f"  구성: {TEMPLATE_SCENARIO['graph']}")
        print(f"{'=' * 60}")

        model = self._find_model_by_name(api_url, auth_headers, WORKFLOW_TARGET_LLM_MODEL)
        assert model["id"]
        self.__class__.model = model
        print(f"\n✔ LLM 모델 발견: id={model['id']}, name={model['name']}")

    # ── Phase 2: 템플릿 생성 & 조회 ──────────────────────────

    def test_02_create_template(self, api_url: str, auth_headers: dict):
        """workflow_definition 을 포함하여 템플릿을 생성한다."""
        model = self.__class__.model
        assert model

        definition = build_workflow_definition(num=TEMPLATE_SCENARIO_NUM, model_id=model["id"])
        self.__class__.definition = definition

        payload = {
            "name": f"E2E-template-{uuid.uuid4().hex[:8]}",
            "description": "E2E 템플릿 생성 → 복제 → 실행 흐름 검증용",
            "category": "e2e-test",
            "workflow_definition": definition,
        }
        resp = requests.post(f"{api_url}/workflows/templates", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"템플릿 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data["is_template"] is True, f"is_template 가 True 가 아닙니다: {data['is_template']}"
        assert data["template_id"] is None, f"신규 템플릿의 template_id 는 null 이어야 합니다: {data['template_id']}"

        self.__class__.template_id = data["id"]
        print(f"\n✔ 템플릿 생성 완료: id={data['id']}, name={data['name']}")

    def test_03_read_template(self, api_url: str, auth_headers: dict):
        """템플릿 상세 조회 — components/connections 개수 및 x, y 좌표를 검증한다."""
        template_id = self.__class__.template_id
        assert template_id
        definition = self.__class__.definition
        assert definition

        resp = requests.get(f"{api_url}/workflows/templates/{template_id}", headers=auth_headers)
        assert resp.status_code == 200, f"템플릿 조회 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data["is_template"] is True
        assert len(data["components"]) == len(definition["components"]), "템플릿 컴포넌트 개수 불일치"
        assert len(data["component_connections"]) == len(definition["connections"]), "템플릿 연결 개수 불일치"

        sent = self._coords_by_name(definition["components"])
        for comp in data["components"]:
            assert comp["name"] in sent, f"예상치 못한 템플릿 컴포넌트: {comp['name']}"
            assert (comp["x"], comp["y"]) == sent[comp["name"]], (
                f"템플릿 컴포넌트 '{comp['name']}' 좌표 불일치: "
                f"보낸 값={sent[comp['name']]}, 응답={(comp['x'], comp['y'])}"
            )
        print(f"\n✔ 템플릿 조회: components {len(data['components'])}개 x, y 좌표 일치 확인")

    # ── Phase 3: 템플릿 복제 → 워크플로우 ────────────────────

    def test_04_clone_template_to_workflow(self, api_url: str, auth_headers: dict):
        """템플릿으로부터 워크플로우를 복제한다 (template_id 보존, x/y 좌표 동일성 검증)."""
        template_id = self.__class__.template_id
        assert template_id
        definition = self.__class__.definition
        assert definition

        workflow_name = f"E2E-cloned-{uuid.uuid4().hex[:8]}"
        resp = requests.post(
            f"{api_url}/workflows/templates/{template_id}/clone",
            params={"workflow_name": workflow_name},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"템플릿 복제 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        # 이후 검증이 실패해도 cleanup(test_09~11)이 동작하도록 workflow_id 를 먼저 저장한다.
        self.__class__.workflow_id = data.get("id")
        assert self.__class__.workflow_id, f"복제 응답에 id 가 없습니다: {data}"

        assert (
            data["is_template"] is False
        ), f"복제 워크플로우의 is_template 는 False 여야 합니다: {data['is_template']}"
        assert (
            data["template_id"] == template_id
        ), f"복제 워크플로우의 template_id 가 원본과 다릅니다: {data['template_id']} != {template_id}"
        assert data["name"] == workflow_name
        assert data["status"] == "DRAFT", f"복제 직후 상태는 DRAFT 여야 합니다: {data['status']}"
        assert len(data["components"]) == len(definition["components"]), "복제 후 컴포넌트 개수 불일치"

        sent = self._coords_by_name(definition["components"])
        for comp in data["components"]:
            assert comp["name"] in sent, f"예상치 못한 복제 컴포넌트: {comp['name']}"
            assert (comp["x"], comp["y"]) == sent[comp["name"]], (
                f"복제 워크플로우 컴포넌트 '{comp['name']}' 좌표 불일치: "
                f"보낸 값={sent[comp['name']]}, 응답={(comp['x'], comp['y'])}"
            )

        print(f"\n✔ 템플릿 복제 완료: workflow_id={data['id']}, template_id={data['template_id']}")
        print(f"  ✔ 복제된 components {len(data['components'])}개 x, y 좌표 일치 확인")

    # ── Phase 4: 복제 워크플로우 실행 & 배포 ─────────────────

    def test_05_execute_workflow(self, api_url: str, auth_headers: dict):
        """복제된 워크플로우를 실행(배포)한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(f"{api_url}/workflows/{wf_id}/execute", headers=auth_headers)
        assert resp.status_code == 200, f"실행 실패: {resp.status_code} {resp.text}"
        assert resp.json()["status"] == "running"
        print(f"\n✔ 워크플로우 실행 시작: kubeflow_run_id={resp.json().get('kubeflow_run_id')}")

    def test_06_wait_for_deployment(self, api_url: str, auth_headers: dict):
        """배포가 완료될 때까지 폴링한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        wf_read = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert wf_read.status_code == 200, f"워크플로 조회 실패: {wf_read.status_code} {wf_read.text}"
        expected_deployments = expected_model_component_count(wf_read.json())
        assert expected_deployments > 0, "MODEL 컴포넌트가 없으면 배포 대기를 진행할 수 없습니다."

        deadline = time.time() + DEPLOY_TIMEOUT_SEC
        last_statuses: list[str] = []

        while time.time() < deadline:
            resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
            assert resp.status_code == 200

            data = resp.json()
            fail_reason = workflow_deploy_poll_should_fail(data, expected_model_deployments=expected_deployments)
            if fail_reason:
                pytest.fail(fail_reason)

            deployed_models = data.get("deployed_models", [])
            last_statuses = [m.get("status", "UNKNOWN") for m in deployed_models]

            if all(s == "DEPLOYED" for s in last_statuses) and last_statuses:
                print(f"\n✔ 배포 완료! deployed_models={len(deployed_models)}")
                for m in deployed_models:
                    print(f" - {m.get('model_name')}: status={m.get('status')}")
                return

            if any(s == "FAILED" for s in last_statuses):
                failed = [m for m in deployed_models if m.get("status") == "FAILED"]
                msgs = [f"{m.get('model_name')}: {m.get('error_message', 'N/A')}" for m in failed]
                pytest.fail(f"배포 실패: {'; '.join(msgs)}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s 초과). 마지막 statuses={last_statuses}")

    def test_07_verify_active(self, api_url: str, auth_headers: dict):
        """배포 완료 후 워크플로우 상태가 ACTIVE 인지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json().get("status") == "ACTIVE", f"예상 status='ACTIVE', 실제='{resp.json().get('status')}'"
        print(f"\n✔ 상태 확인: workflow_id={wf_id}, status=ACTIVE")

    # ── Phase 5: 추론 테스트 ─────────────────────────────────

    def test_08_inference(self, api_url: str, auth_headers: dict):
        """추론 테스트 — 복제된 워크플로우의 파이프라인을 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/test/rag",
            data={"text": TEMPLATE_SCENARIO["inference_text"]},
            headers=auth_headers,
            timeout=INFERENCE_TIMEOUT_SEC,
        )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data["workflow_id"] == wf_id
        assert isinstance(data["results"], list) and len(data["results"]) > 0
        fr = data.get("final_result") or ""
        assert len(fr) > 0

        print(f"\n✔ 추론 성공: workflow_id={wf_id}")
        print(f"  execution_order: {data.get('execution_order', [])}")
        print(f"  final_result: {fr[:120]}{'…' if len(fr) > 120 else ''}")

    # ── Phase 6: 정리 ────────────────────────────────────────

    def test_09_delete_workflow(self, api_url: str, auth_headers: dict):
        """복제 워크플로우 삭제를 시작한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.delete(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        print(f"\n✔ 워크플로우 삭제 시작: cleanup_run_id={data.get('cleanup_run_id')}")

    def test_10_finalize_deletion(self, api_url: str, auth_headers: dict):
        """K8s 리소스 정리를 폴링한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        deadline = time.time() + DELETE_TIMEOUT_SEC

        while time.time() < deadline:
            resp = requests.post(f"{api_url}/workflows/{wf_id}/finalize-deletion", headers=auth_headers)
            assert resp.status_code in (200, 404)

            if resp.status_code == 404:
                print(f"\n✔ 워크플로우가 이미 삭제됨: {wf_id}")
                return

            data = resp.json()
            status_val = data.get("status")

            if status_val == "completed":
                print(f"\n✔ 워크플로우 삭제 완료: {wf_id}")
                return
            if status_val == "failed":
                pytest.fail(f"워크플로우 삭제 실패: {data.get('message', 'N/A')}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 리소스 정리 대기 중… status={status_val} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"삭제 타임아웃 ({DELETE_TIMEOUT_SEC}s 초과)")

    def test_11_verify_workflow_deleted(self, api_url: str, auth_headers: dict):
        """복제 워크플로우가 완전히 삭제되었는지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 404, f"워크플로우가 아직 존재합니다: {resp.status_code}"
        print(f"\n✔ 워크플로우 404 확인: {wf_id}")

    def test_12_delete_template(self, api_url: str, auth_headers: dict):
        """템플릿을 삭제한다 (파생 워크플로우가 정리된 후이므로 가능)."""
        template_id = self.__class__.template_id
        assert template_id

        resp = requests.delete(f"{api_url}/workflows/templates/{template_id}", headers=auth_headers)
        assert resp.status_code in (200, 204), f"템플릿 삭제 실패: {resp.status_code} {resp.text}"
        print(f"\n✔ 템플릿 삭제 완료: id={template_id}")
