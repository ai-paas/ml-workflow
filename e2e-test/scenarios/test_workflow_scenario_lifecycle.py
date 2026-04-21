"""
E2E 시나리오: 워크플로우 시나리오 전체 생명주기 테스트

§5.2 시나리오 1~7 중 하나를 선택하여 생성 → 배포 → 추론 → 삭제 전체를 검증한다.
시나리오 선택: E2E_SCENARIO 환경변수 (필수, Makefile에서 SCENARIO 인자로 주입)

흐름:
  1.  GET  /models                          — LLM 모델 검색
  2.  GET  /models                          — 임베딩 모델 검색 (KB 필요 시)
  3~4. KB 메타데이터 조회 → KB 생성 (1~2개) — (KB 필요 시)
  5.  POST /prompts                          — 시나리오별 LLM 프롬프트 생성
  6.  POST /workflows                        — 워크플로우 생성
  7.  POST /workflows/{id}/execute           — 워크플로우 실행 (KServe 배포)
  8.  GET  /workflows/{id}/status            — 배포 완료 폴링
  9.  GET  /workflows/{id}/status            — ACTIVE 상태 확인
  10. POST /workflows/{id}/test/rag          — 추론 테스트
  11. DELETE /workflows/{id}                 — 워크플로우 삭제 시작
  12. POST /workflows/{id}/finalize-deletion — 삭제 완료 폴링
  13. GET  /workflows/{id}                   — 워크플로우 404 확인
  14. DELETE /knowledge-bases/{id}            — KB 삭제 (1~2개, 있는 경우)
  15. GET  /knowledge-bases/{id}              — KB 404 확인 (있는 경우)
  16. DELETE /prompts/{id}                    — 프롬프트 삭제
"""

import json
import time
import uuid

import pytest
import requests
from config import (
    DELETE_TIMEOUT_SEC,
    DEPLOY_TIMEOUT_SEC,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_FILES,
    KB_THRESHOLD,
    KB_TOP_K,
    POLL_INTERVAL_SEC,
    SCENARIO_NUM,
    TARGET_EMBEDDING_MODEL_NAME,
    TARGET_MODEL_NAME,
)
from workflow_scenarios import build_workflow_definition, get_scenario

SCENARIO = get_scenario(SCENARIO_NUM)
INFERENCE_TIMEOUT_SEC = 120


@pytest.mark.workflow_scenario_lifecycle
class TestWorkflowScenarioLifecycle:
    """워크플로우 시나리오 전체 생명주기: 생성 → 배포 → 추론 → 삭제"""

    model: dict | None = None
    embedding_model: dict | None = None
    kb_ids: list[int] = []
    workflow_id: str | None = None
    prompt_ids: dict = {}
    chunk_type_id: int | None = None
    language_id: int | None = None
    search_method_id: int | None = None

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
                f"name='{model_name}' 인 모델을 찾을 수 없습니다.\n" f"등록된 모델 목록 ({len(models)}개): \n{listing}"
            )
        return matched[0]

    # ── Phase 1: 사전 준비 (모델 & KB & 프롬프트) ────────────

    def test_01_find_llm_model(self, api_url: str, auth_headers: dict):
        """LLM 모델이 존재해야 한다."""
        kb_count = SCENARIO["kb_count"]
        kb_info = f"{kb_count}개 ({', '.join(SCENARIO['kb_labels'])})" if kb_count > 0 else "아니오"
        print(f"\n{'=' * 60}")
        print(f"  시나리오 #{SCENARIO_NUM}: {SCENARIO['name']}")
        print(f"  구성: {SCENARIO['graph']}")
        print(f"  KB: {kb_info}")
        print(f"  프롬프트: {len(SCENARIO['prompts'])}개")
        print(f"{'=' * 60}")

        model = self._find_model_by_name(api_url, auth_headers, TARGET_MODEL_NAME)
        assert model["id"]
        self.__class__.model = model
        print(f"\n✔ LLM 모델 발견: id={model['id']}, name={model['name']}")

    def test_02_find_embedding_model(self, api_url: str, auth_headers: dict):
        """임베딩 모델이 존재해야 한다 (KB 필요 시)."""
        if SCENARIO["kb_count"] == 0:
            pytest.skip(f"시나리오 #{SCENARIO_NUM} 은 KB가 필요 없습니다.")

        model = self._find_model_by_name(api_url, auth_headers, TARGET_EMBEDDING_MODEL_NAME)
        assert model["id"]
        self.__class__.embedding_model = model
        print(f"\n✔ 임베딩 모델 발견: id={model['id']}, name={model['name']}")

    def test_03_lookup_kb_metadata(self, api_url: str, auth_headers: dict):
        """KB 생성에 필요한 메타데이터를 조회한다 (KB 필요 시)."""
        if SCENARIO["kb_count"] == 0:
            pytest.skip(f"시나리오 #{SCENARIO_NUM} 은 KB가 필요 없습니다.")

        base = f"{api_url}/knowledge-bases"

        resp = requests.get(f"{base}/chunk-types", headers=auth_headers)
        assert resp.status_code == 200 and resp.json()
        self.__class__.chunk_type_id = resp.json()[0]["id"]

        resp = requests.get(f"{base}/languages", headers=auth_headers)
        assert resp.status_code == 200
        langs = resp.json()
        ko = [lang for lang in langs if lang.get("name") == "KO"]
        self.__class__.language_id = (ko[0] if ko else langs[0])["id"]

        resp = requests.get(f"{base}/search-methods", headers=auth_headers)
        assert resp.status_code == 200 and resp.json()
        self.__class__.search_method_id = resp.json()[0]["id"]

        print("\n✔ KB 메타데이터 조회 완료")

    def test_04_create_knowledge_bases(self, api_url: str, auth_headers: dict):
        """Knowledge Base를 생성한다 (KB 필요 시, 시나리오에 따라 1~2개)."""
        kb_count = SCENARIO["kb_count"]
        if kb_count == 0:
            pytest.skip(f"시나리오 #{SCENARIO_NUM} 은 KB가 필요 없습니다.")

        emb = self.__class__.embedding_model
        assert emb, "test_02 에서 임베딩 모델을 찾지 못했습니다."

        created_ids: list[int] = []
        for i in range(kb_count):
            kb_file = KB_FILES[i]
            assert kb_file.exists(), f"KB 파일이 없습니다: {kb_file}"

            label = SCENARIO["kb_labels"][i]
            form_data = {
                "name": f"E2E-S{SCENARIO_NUM}-lifecycle-KB{i + 1}-{uuid.uuid4().hex[:8]}",
                "description": f"E2E 시나리오 #{SCENARIO_NUM} ({SCENARIO['name']}) lifecycle KB{i + 1}: {label}",
                "language_id": str(self.__class__.language_id),
                "embedding_model_id": str(emb["id"]),
                "chunk_size": str(KB_CHUNK_SIZE),
                "chunk_overlap": str(KB_CHUNK_OVERLAP),
                "chunk_type_id": str(self.__class__.chunk_type_id),
                "search_method_id": str(self.__class__.search_method_id),
                "top_k": str(KB_TOP_K),
                "threshold": str(KB_THRESHOLD),
            }

            with open(kb_file, "rb") as f:
                resp = requests.post(
                    f"{api_url}/knowledge-bases",
                    data=form_data,
                    files={"file": (kb_file.name, f, "application/pdf")},
                    headers=auth_headers,
                )
            assert resp.status_code == 200, f"KB 생성 실패 (KB{i + 1}): {resp.status_code} {resp.text}"

            data = resp.json()
            created_ids.append(data["id"])
            print(f"  ✔ KB{i + 1} 생성: id={data['id']}, name={data['name']} ({label})")

        self.__class__.kb_ids = created_ids
        print(f"\n✔ KB {len(created_ids)}개 생성 완료")

    def test_05_create_prompts(self, api_url: str, auth_headers: dict):
        """시나리오별 LLM 프롬프트를 생성한다."""
        created: dict[str, int] = {}

        for comp_name, prompt_def in SCENARIO["prompts"].items():
            payload: dict = {
                "prompt": {
                    "name": f"E2E-S{SCENARIO_NUM}-{prompt_def['name']}-{uuid.uuid4().hex[:8]}",
                    "description": f"시나리오 #{SCENARIO_NUM} {comp_name} 프롬프트",
                    "content": prompt_def["content"],
                },
            }
            if prompt_def.get("has_context"):
                payload["prompt_variable"] = ["context"]

            resp = requests.post(f"{api_url}/prompts", json=payload, headers=auth_headers)
            assert resp.status_code == 201, f"프롬프트 생성 실패 ({comp_name}): {resp.status_code} {resp.text}"

            data = resp.json()
            created[comp_name] = data["id"]
            print(f"  ✔ 프롬프트 생성: {comp_name} → id={data['id']}")

        self.__class__.prompt_ids = created
        print(f"\n✔ 프롬프트 {len(created)}개 생성 완료")

    # ── Phase 2: 워크플로우 생성 & 배포 ─────────────────────

    def test_06_create_workflow(self, api_url: str, auth_headers: dict):
        """시나리오에 맞는 워크플로우를 생성한다."""
        model = self.__class__.model
        assert model, "LLM 모델이 없습니다."

        definition = build_workflow_definition(
            SCENARIO_NUM, model["id"], self.__class__.kb_ids, KB_TOP_K, self.__class__.prompt_ids
        )
        payload = {
            "name": f"E2E-S{SCENARIO_NUM}-lifecycle-{uuid.uuid4().hex[:8]}",
            "description": f"E2E 시나리오 #{SCENARIO_NUM}: {SCENARIO['name']} (lifecycle)",
            "workflow_definition": definition,
        }

        resp = requests.post(f"{api_url}/workflows", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.workflow_id = data["id"]
        assert data["status"] == "DRAFT"
        print(f"\n✔ 워크플로우 생성 완료: id={data['id']}, status={data['status']}")

    def test_07_execute_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우를 실행(배포)한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/execute",
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"실행 실패: {resp.status_code} {resp.text}"
        assert resp.json()["status"] == "running"
        print(f"\n✔ 워크플로우 실행 시작: kubeflow_run_id={resp.json().get('kubeflow_run_id')}")

    def test_08_wait_for_deployment(self, api_url: str, auth_headers: dict):
        """배포가 완료될 때까지 폴링한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        deadline = time.time() + DEPLOY_TIMEOUT_SEC
        last_statuses: list[str] = []

        while time.time() < deadline:
            resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
            assert resp.status_code == 200

            data = resp.json()
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

    def test_09_verify_active(self, api_url: str, auth_headers: dict):
        """배포 완료 후 워크플로우 상태가 ACTIVE 인지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json().get("status") == "ACTIVE", f"예상 status='ACTIVE', 실제='{resp.json().get('status')}'"
        print(f"\n✔ 상태 확인: workflow_id={wf_id}, status=ACTIVE")

    # ── Phase 3: 추론 테스트 ────────────────────────────────

    def test_10_inference(self, api_url: str, auth_headers: dict):
        """추론 테스트 — 워크플로우 파이프라인을 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

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
        assert data["final_result"] is not None and len(data["final_result"]) > 0

        print(f"\n✔ 추론 성공: workflow_id={wf_id}")
        print(f"  execution_order: {data.get('execution_order', [])}")
        print(f"  final_result: {data['final_result'][:120]}{'…' if len(data['final_result']) > 120 else ''}")

    # ── Phase 4: 삭제 ───────────────────────────────────────

    def test_11_delete_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우 삭제를 시작한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.delete(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        print(f"\n✔ 워크플로우 삭제 시작: cleanup_run_id={data.get('cleanup_run_id')}")

    def test_12_finalize_deletion(self, api_url: str, auth_headers: dict):
        """K8s 리소스 정리를 폴링한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        deadline = time.time() + DELETE_TIMEOUT_SEC

        while time.time() < deadline:
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/finalize-deletion",
                headers=auth_headers,
            )
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

    def test_13_verify_workflow_deleted(self, api_url: str, auth_headers: dict):
        """워크플로우가 완전히 삭제되었는지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
        assert resp.status_code == 404, f"워크플로우가 아직 존재합니다: {resp.status_code}"
        print(f"\n✔ 워크플로우 404 확인: {wf_id}")

    def test_14_delete_knowledge_bases(self, api_url: str, auth_headers: dict):
        """Knowledge Base를 삭제한다 (1~2개, 있는 경우)."""
        kb_ids = self.__class__.kb_ids
        if not kb_ids:
            pytest.skip("KB가 없는 시나리오이므로 스킵합니다.")

        for idx, kb_id in enumerate(kb_ids):
            resp = requests.delete(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
            assert resp.status_code == 200, f"KB 삭제 실패 (KB{idx + 1}, id={kb_id}): {resp.status_code} {resp.text}"

            data = resp.json()
            assert data.get("success") is True
            print(f"  ✔ KB{idx + 1} 삭제 완료: kb_id={kb_id}")

        print(f"\n✔ KB {len(kb_ids)}개 삭제 완료")

    def test_15_verify_kbs_deleted(self, api_url: str, auth_headers: dict):
        """KB가 완전히 삭제되었는지 확인한다 (있는 경우)."""
        kb_ids = self.__class__.kb_ids
        if not kb_ids:
            pytest.skip("KB가 없는 시나리오이므로 스킵합니다.")

        for idx, kb_id in enumerate(kb_ids):
            resp = requests.get(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
            assert resp.status_code == 404, f"KB{idx + 1}이 아직 존재합니다 (id={kb_id}): {resp.status_code}"
            print(f"  ✔ KB{idx + 1} 404 확인: kb_id={kb_id}")

    def test_16_delete_prompts(self, api_url: str, auth_headers: dict):
        """시나리오에서 생성한 프롬프트를 삭제한다."""
        prompt_ids = self.__class__.prompt_ids
        if not prompt_ids:
            pytest.skip("프롬프트가 없어 스킵합니다.")

        for comp_name, pid in prompt_ids.items():
            resp = requests.delete(f"{api_url}/prompts/{pid}", headers=auth_headers)
            assert resp.status_code == 204, f"프롬프트 삭제 실패 ({comp_name}, id={pid}): {resp.status_code}"
            print(f"  ✔ 프롬프트 삭제: {comp_name} (id={pid})")

        print(f"\n✔ 프롬프트 {len(prompt_ids)}개 삭제 완료")
