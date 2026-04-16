"""
E2E 시나리오: RAG 워크플로우 전체 생명주기

Knowledge Base + LLM 모델을 조합한 RAG 워크플로우의 전체 생명주기를 검증한다.

흐름:
  1.  GET  /models                          — LLM 모델 검색
  2.  GET  /models                          — 임베딩 모델 검색
  3.  GET  /knowledge-bases/chunk-types      — chunk_type_id 조회
  4.  GET  /knowledge-bases/languages        — language_id 조회
  5.  GET  /knowledge-bases/search-methods   — search_method_id 조회
  6.  POST /knowledge-bases                  — Knowledge Base 생성 (파일 업로드 포함)
  7.  POST /workflows                        — RAG 워크플로우 생성 (시작→KB→LLM→끝)
  8.  POST /workflows/{id}/execute           — 워크플로우 실행 (KServe 배포)
  9.  GET  /workflows/{id}/status            — 배포 완료 폴링
  10. GET  /workflows/{id}                   — ACTIVE 상태 확인
  11. POST /workflows/{id}/test/rag          — RAG 추론 테스트 + 응답 형식 검증
  12. DELETE /workflows/{id}                 — 워크플로우 삭제 시작
  13. POST /workflows/{id}/finalize-deletion — 삭제 완료 폴링
  14. GET  /workflows/{id}                   — 워크플로우 404 확인
  15. DELETE /knowledge-bases/{id}            — Knowledge Base 삭제
"""

import time
import uuid
from pathlib import Path

import pytest
import requests
from config import (
    DELETE_TIMEOUT_SEC,
    DEPLOY_TIMEOUT_SEC,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_FILE_DIR,
    KB_THRESHOLD,
    KB_TOP_K,
    POLL_INTERVAL_SEC,
    TARGET_EMBEDDING_MODEL_NAME,
    TARGET_MODEL_NAME,
)

INFERENCE_TIMEOUT_SEC = 120


@pytest.mark.rag_workflow_lifecycle
class TestRagWorkflowLifecycle:
    """KB 생성 → RAG 워크플로우 배포 → 추론 테스트 → 삭제 전체 흐름"""

    model: dict | None = None
    embedding_model: dict | None = None
    kb_id: int | None = None
    workflow_id: str | None = None
    # KB 생성에 필요한 메타 ID
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

    @staticmethod
    def _pick_first(items: list, label: str) -> dict:
        assert len(items) > 0, f"{label} 목록이 비어 있습니다."
        return items[0]

    KB_FILE = KB_FILE_DIR / "고압가스 안전관리법(법률)(제21065호)(20251001).pdf"

    @staticmethod
    def _get_kb_file() -> Path:
        assert TestRagWorkflowLifecycle.KB_FILE.exists(), f"KB 파일이 없습니다: {TestRagWorkflowLifecycle.KB_FILE}"
        return TestRagWorkflowLifecycle.KB_FILE

    @staticmethod
    def _build_rag_workflow_definition(model_id: int, kb_id: int) -> dict:
        start_ref = f"start-{uuid.uuid4().hex[:8]}"
        kb_ref = f"kb-{uuid.uuid4().hex[:8]}"
        model_ref = f"model-{uuid.uuid4().hex[:8]}"
        end_ref = f"end-{uuid.uuid4().hex[:8]}"

        return {
            "components": [
                {"ref_id": start_ref, "name": "시작", "type": "START"},
                {
                    "ref_id": kb_ref,
                    "name": "법률 Knowledge Base",
                    "type": "KNOWLEDGE_BASE",
                    "knowledge_base_id": kb_id,
                    "config": {"top_k": KB_TOP_K},
                },
                {
                    "ref_id": model_ref,
                    "name": "LLM 모델",
                    "type": "MODEL",
                    "model_id": model_id,
                    "config": {"temperature": 0.7, "max_tokens": 1024},
                },
                {"ref_id": end_ref, "name": "끝", "type": "END"},
            ],
            "connections": [
                {"source_ref_id": start_ref, "target_ref_id": kb_ref},
                {"source_ref_id": kb_ref, "target_ref_id": model_ref},
                {"source_ref_id": model_ref, "target_ref_id": end_ref},
            ],
        }

    # ── Phase 1: 사전 준비 (모델 & KB) ──────────────────────

    def test_01_find_llm_model(self, api_url: str, auth_headers: dict):
        """LLM 모델이 존재해야 한다."""
        model = self._find_model_by_name(api_url, auth_headers, TARGET_MODEL_NAME)
        assert model["id"]
        self.__class__.model = model
        print(f"\n✔ LLM 모델 발견: id={model['id']}, name={model['name']}")

    def test_02_find_embedding_model(self, api_url: str, auth_headers: dict):
        """임베딩 모델이 존재해야 한다."""
        model = self._find_model_by_name(api_url, auth_headers, TARGET_EMBEDDING_MODEL_NAME)
        assert model["id"]
        self.__class__.embedding_model = model
        print(f"\n✔ 임베딩 모델 발견: id={model['id']}, name={model['name']}")

    def test_03_lookup_kb_metadata(self, api_url: str, auth_headers: dict):
        """KB 생성에 필요한 chunk_type, language, search_method ID를 조회한다."""
        base = f"{api_url}/knowledge-bases"

        resp = requests.get(f"{base}/chunk-types", headers=auth_headers)
        assert resp.status_code == 200
        ct = self._pick_first(resp.json(), "chunk-types")
        self.__class__.chunk_type_id = ct["id"]

        resp = requests.get(f"{base}/languages", headers=auth_headers)
        assert resp.status_code == 200
        langs = resp.json()
        ko = [lang for lang in langs if lang.get("name") == "KO"]
        lang = ko[0] if ko else langs[0]
        self.__class__.language_id = lang["id"]

        resp = requests.get(f"{base}/search-methods", headers=auth_headers)
        assert resp.status_code == 200
        sm = self._pick_first(resp.json(), "search-methods")
        self.__class__.search_method_id = sm["id"]

        print(
            f"\n✔ KB 메타데이터 조회 완료: "
            f"chunk_type_id={self.__class__.chunk_type_id}, "
            f"language_id={self.__class__.language_id}, "
            f"search_method_id={self.__class__.search_method_id}"
        )

    def test_04_create_knowledge_base(self, api_url: str, auth_headers: dict):
        """Knowledge Base를 생성하고 파일을 업로드한다."""
        emb = self.__class__.embedding_model
        assert emb, "test_02 에서 임베딩 모델을 찾지 못했습니다."

        kb_file = self._get_kb_file()
        form_data = {
            "name": f"E2E-RAG-KB-{uuid.uuid4().hex[:8]}",
            "description": "E2E RAG 테스트용 Knowledge Base",
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
        assert resp.status_code == 200, f"KB 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.kb_id = data["id"]
        print(f"\n✔ KB 생성 완료: id={data['id']}, name={data['name']}, files={len(data.get('files', []))}")

    # ── Phase 2: 워크플로우 생성 & 배포 ─────────────────────

    def test_05_create_workflow(self, api_url: str, auth_headers: dict):
        """시작 → KB → LLM → 끝 RAG 워크플로우를 생성한다."""
        model = self.__class__.model
        kb_id = self.__class__.kb_id
        assert model, "LLM 모델이 없습니다."
        assert kb_id, "KB 가 생성되지 않았습니다."

        definition = self._build_rag_workflow_definition(model["id"], kb_id)
        payload = {
            "name": f"E2E-RAG-lifecycle-{uuid.uuid4().hex[:8]}",
            "description": "E2E 테스트: RAG 워크플로우 전체 생명주기",
            "workflow_definition": definition,
        }

        resp = requests.post(f"{api_url}/workflows", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.workflow_id = data["id"]
        assert data["status"] == "DRAFT"
        print(f"\n✔ RAG 워크플로우 생성 완료: id={data['id']}, status={data['status']}")

    def test_06_execute_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우를 실행(배포)한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/execute",
            json={"parameters": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"워크플로우 실행 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data["status"] == "running"
        print(f"\n✔ 워크플로우 실행 시작: kubeflow_run_id={data.get('kubeflow_run_id')}")

    def test_07_wait_for_deployment(self, api_url: str, auth_headers: dict):
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
                pytest.fail(f"배포 실패 감지: {'; '.join(msgs)}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s 초과). 마지막 statuses={last_statuses}")

    def test_08_verify_active(self, api_url: str, auth_headers: dict):
        """배포 완료 후 워크플로우 상태가 ACTIVE 인지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200

        data = resp.json()
        assert data.get("status") == "ACTIVE", f"예상 status='ACTIVE', 실제='{data.get('status')}'"
        print(f"\n✔ 상태 확인: workflow_id={wf_id}, status=ACTIVE")

    # ── Phase 3: RAG 추론 테스트 ────────────────────────────

    def test_09_rag_inference(self, api_url: str, auth_headers: dict):
        """RAG 추론 테스트 — KB 검색 + LLM 추론 파이프라인을 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/test/rag",
            data={"text": "고압가스 안전관리법의 목적은 무엇인가요?"},
            headers=auth_headers,
            timeout=INFERENCE_TIMEOUT_SEC,
        )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"

        data = resp.json()

        assert data["workflow_id"] == wf_id
        assert isinstance(data["execution_order"], list) and len(data["execution_order"]) > 0
        assert isinstance(data["results"], list) and len(data["results"]) > 0
        assert data["final_result"] is not None and len(data["final_result"]) > 0

        print(f"\n✔ RAG 추론 성공: workflow_id={wf_id}")
        print(f"  execution_order ({len(data['execution_order'])} 개 컴포넌트): {data['execution_order']}")
        print(f"  final_result: {data['final_result'][:120]}{'…' if len(data['final_result']) > 120 else ''}")

    def test_10_rag_response_structure(self, api_url: str, auth_headers: dict):
        """RAG 추론 응답의 results[] 내부 구조를 상세 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/test/rag",
            data={"text": "고압가스의 저장 기준에 대해 설명해주세요."},
            headers=auth_headers,
            timeout=INFERENCE_TIMEOUT_SEC,
        )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        results = data["results"]

        # RAG 워크플로우 = KB + MODEL 결과가 모두 있어야 함
        comp_types = [r["component_type"] for r in results]
        assert "KNOWLEDGE_BASE" in comp_types, "KB 컴포넌트 결과가 없습니다."
        assert "MODEL" in comp_types, "MODEL 컴포넌트 결과가 없습니다."

        for idx, result in enumerate(results):
            assert "component_id" in result, f"results[{idx}]: component_id 누락"
            assert "component_name" in result, f"results[{idx}]: component_name 누락"
            assert "component_type" in result, f"results[{idx}]: component_type 누락"

            if result["component_type"] == "KNOWLEDGE_BASE":
                assert "result" in result, f"results[{idx}]: result 필드 누락"
                kb_result = result["result"]
                assert "search_result" in kb_result, f"results[{idx}].result: search_result 누락"
                assert isinstance(
                    kb_result["search_result"], str
                ), f"results[{idx}].result.search_result 가 문자열이 아님"
                assert len(kb_result["search_result"]) > 0, f"results[{idx}]: KB 검색 결과가 비어 있습니다."
                assert "total" in kb_result, f"results[{idx}].result: total 누락"
                assert kb_result["total"] > 0, f"results[{idx}]: total 이 0 입니다."
                print(
                    f"\n✔ results[{idx}] KB 검증 통과:"
                    f" total={kb_result['total']}건,"
                    f" search_method={kb_result.get('search_method')}"
                )
                print(f"  search_result (앞 100자): {kb_result['search_result'][:100]}…")

            elif result["component_type"] == "MODEL":
                assert "result" in result, f"results[{idx}]: result 필드 누락"
                llm_result = result["result"]
                assert "response" in llm_result, f"results[{idx}].result: response 누락"
                assert isinstance(llm_result["response"], str) and len(llm_result["response"]) > 0

                assert "full_response" in llm_result, f"results[{idx}].result: full_response 누락"
                full_resp = llm_result["full_response"]
                assert full_resp.get("done") is True, f"results[{idx}]: Ollama 응답 미완료 (done≠true)"

                print(f"\n✔ results[{idx}] LLM 검증 통과: ")
                print(f"  response: {llm_result['response'][:80]}{'…' if len(llm_result['response']) > 80 else ''}")
                print(f"  model: {full_resp.get('model')}, done_reason: {full_resp.get('done_reason')}")

        # final_result 는 마지막 LLM의 response 와 일치
        last_llm = [r for r in results if r.get("component_type") == "MODEL"]
        if last_llm:
            expected_final = last_llm[-1]["result"]["response"]
            assert (
                data["final_result"] == expected_final
            ), f"final_result 불일치: '{data['final_result'][:50]}' ≠ '{expected_final[:50]}'"
            print("\n✔ final_result 검증 통과 (마지막 LLM response 와 일치)")

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
                print(f"\n✔ 워크플로우가 이미 삭제됨: workflow_id={wf_id}")
                return

            data = resp.json()
            status_val = data.get("status")

            if status_val == "completed":
                print(f"\n✔ 워크플로우 삭제 완료: workflow_id={wf_id}")
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

    def test_14_delete_knowledge_base(self, api_url: str, auth_headers: dict):
        """Knowledge Base를 삭제한다."""
        kb_id = self.__class__.kb_id
        assert kb_id

        resp = requests.delete(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.status_code == 200, f"KB 삭제 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data.get("success") is True
        print(f"\n✔ KB 삭제 완료: kb_id={kb_id}")

    def test_15_verify_kb_deleted(self, api_url: str, auth_headers: dict):
        """KB가 완전히 삭제되었는지 확인한다."""
        kb_id = self.__class__.kb_id
        assert kb_id

        resp = requests.get(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.status_code == 404, f"KB가 아직 존재합니다: {resp.status_code}"
        print(f"\n✔ KB 404 확인: kb_id={kb_id}")
