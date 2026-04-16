"""
E2E 시나리오: RAG 워크플로우 생성 → 배포 → 추론 테스트

삭제는 포함하지 않는다. 별도 삭제: make e2e-rag-workflow-delete

흐름:
  1. LLM / 임베딩 모델 검색
  2. KB 메타데이터(chunk_type, language, search_method) 조회
  3. Knowledge Base 생성 (파일 업로드)
  4. RAG 워크플로우 생성 (시작→KB→LLM→끝)
  5. 워크플로우 실행 (KServe 배포)
  6. 배포 완료 폴링
  7. ACTIVE 상태 확인
  8. RAG 추론 테스트
"""

import time
import uuid
from pathlib import Path

import pytest
import requests
from config import (
    DEPLOY_TIMEOUT_SEC,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_FILE_DIR,
    KB_THRESHOLD,
    KB_TOP_K,
    POLL_INTERVAL_SEC,
    TARGET_EMBEDDING_MODEL_NAME,
    TARGET_MODEL_NAME,
    save_state,
)

STATE_KEY_WF = "rag_workflow_id"
STATE_KEY_KB = "rag_kb_id"

INFERENCE_TIMEOUT_SEC = 120


@pytest.mark.rag_workflow_deploy
class TestRagWorkflowDeploy:
    """KB 생성 → RAG 워크플로우 배포 → 추론 테스트 (삭제 제외)"""

    model: dict | None = None
    embedding_model: dict | None = None
    kb_id: int | None = None
    workflow_id: str | None = None
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

    KB_FILE = KB_FILE_DIR / "고압가스 안전관리법(법률)(제21065호)(20251001).pdf"

    @staticmethod
    def _get_kb_file() -> Path:
        assert TestRagWorkflowDeploy.KB_FILE.exists(), f"KB 파일이 없습니다: {TestRagWorkflowDeploy.KB_FILE}"
        return TestRagWorkflowDeploy.KB_FILE

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

    # ── tests ────────────────────────────────────────────────

    def test_01_find_llm_model(self, api_url: str, auth_headers: dict):
        model = self._find_model_by_name(api_url, auth_headers, TARGET_MODEL_NAME)
        assert model["id"]
        self.__class__.model = model
        print(f"\n✔ LLM 모델 발견: id={model['id']}, name={model['name']}")

    def test_02_find_embedding_model(self, api_url: str, auth_headers: dict):
        model = self._find_model_by_name(api_url, auth_headers, TARGET_EMBEDDING_MODEL_NAME)
        assert model["id"]
        self.__class__.embedding_model = model
        print(f"\n✔ 임베딩 모델 발견: id={model['id']}, name={model['name']}")

    def test_03_lookup_kb_metadata(self, api_url: str, auth_headers: dict):
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

    def test_04_create_knowledge_base(self, api_url: str, auth_headers: dict):
        emb = self.__class__.embedding_model
        assert emb

        kb_file = self._get_kb_file()
        form_data = {
            "name": f"E2E-RAG-KB-{uuid.uuid4().hex[:8]}",
            "description": "E2E RAG deploy 테스트용 KB",
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
        save_state(STATE_KEY_KB, str(data["id"]))
        print(f"\n✔ KB 생성 완료: id={data['id']}, name={data['name']}")

    def test_05_create_workflow(self, api_url: str, auth_headers: dict):
        model = self.__class__.model
        kb_id = self.__class__.kb_id
        assert model and kb_id

        definition = self._build_rag_workflow_definition(model["id"], kb_id)
        payload = {
            "name": f"E2E-RAG-deploy-{uuid.uuid4().hex[:8]}",
            "description": "E2E 테스트: RAG 워크플로우 배포",
            "workflow_definition": definition,
        }

        resp = requests.post(f"{api_url}/workflows", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.workflow_id = data["id"]
        save_state(STATE_KEY_WF, data["id"])
        print(f"\n✔ RAG 워크플로우 생성 완료: id={data['id']}")

    def test_06_execute_workflow(self, api_url: str, auth_headers: dict):
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/execute",
            json={"parameters": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"실행 실패: {resp.status_code} {resp.text}"
        assert resp.json()["status"] == "running"
        print("\n✔ 워크플로우 실행 시작")

    def test_07_wait_for_deployment(self, api_url: str, auth_headers: dict):
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
                return

            if any(s == "FAILED" for s in last_statuses):
                failed = [m for m in deployed_models if m.get("status") == "FAILED"]
                msgs = [f"{m.get('model_name')}: {m.get('error_message', 'N/A')}" for m in failed]
                pytest.fail(f"배포 실패: {'; '.join(msgs)}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s 초과)")

    def test_08_verify_active(self, api_url: str, auth_headers: dict):
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json().get("status") == "ACTIVE"
        print("\n✔ 상태 확인: ACTIVE")

    def test_09_rag_inference(self, api_url: str, auth_headers: dict):
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
        assert len(data["results"]) > 0
        assert data["final_result"] and len(data["final_result"]) > 0

        print("\n✔ RAG 추론 성공")
        print(f"  final_result: {data['final_result'][:100]}{'…' if len(data['final_result']) > 100 else ''}")
        print("\n  ℹ 삭제하려면: make e2e-rag-workflow-delete")
