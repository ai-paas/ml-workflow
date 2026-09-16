"""
E2E 시나리오: 워크플로우 시나리오 배포 테스트

시나리오 1~10 중 하나를 선택하여 프롬프트 생성 → KB 생성 → 배포 → 추론까지 검증한다.
삭제는 포함하지 않는다. 별도 삭제: make e2e-wf-scenario-delete SCENARIO=N (저장된 배포 건별 확인)
같은 SCENARIO 로 재실행 시 이전 배포를 덮어쓰지 않고 상태 파일의 scenario_N_deployments 목록에 추가된다.
(ENV= 지정 시 e2e-test/.state.{ENV}.json, 미지정 시 .state.json)

시나리오 선택: E2E_SCENARIO 환경변수 (필수, Makefile에서 SCENARIO 인자로 주입)

흐름:
  1. 타깃 모델 검색 (LLM 또는 ODM)
  2. (KB 필요 시) 임베딩 모델 검색 → KB 메타데이터 조회 → KB 생성 (1~2개)
  3. 시나리오별 LLM 프롬프트 생성
  4. 워크플로우 생성 (prompt_id 포함)
  5. 워크플로우 실행 (KServe 배포)
  6. 배포 완료 폴링
  7. ACTIVE 상태 확인
  8. 추론 테스트
"""

import json
import os
import time
import uuid

import pytest
import requests
from config import (
    DEPLOY_TIMEOUT_SEC,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_FILES,
    KB_THRESHOLD,
    KB_TOP_K,
    POLL_INTERVAL_SEC,
    SCENARIO_NUM,
    STATE_FILE,
    TARGET_EMBEDDING_MODEL_NAME,
    WORKFLOW_FILLMASK_TEST_SEQUENCE,
    WORKFLOW_FILLMASK_TOP_K,
    WORKFLOW_ODM_TEST_IMAGE,
    WORKFLOW_PLM_TEST_SAMPLE,
    WORKFLOW_STRUCTURE_NUM_LOOPS,
    WORKFLOW_STRUCTURE_NUM_SAMPLING_STEPS,
    WORKFLOW_STRUCTURE_TEST_SEQUENCE,
    workflow_primary_target_model_name,
)
from workflow.definitions import (
    append_deployment_entry,
    build_workflow_definition,
    get_scenario,
    load_deployment_entries,
)
from workflow.deploy_wait import expected_model_component_count, workflow_deploy_poll_should_fail

SCENARIO = get_scenario(SCENARIO_NUM)
INFERENCE_TIMEOUT_SEC = 120


@pytest.mark.workflow_scenario_deploy
class TestWorkflowScenarioDeploy:
    """워크플로우 시나리오 배포 테스트 (삭제 제외)"""

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

    # ── tests ────────────────────────────────────────────────

    def test_01_find_llm_model(self, api_url: str, auth_headers: dict):
        """시나리오 타깃 모델(LLM 또는 ODM)이 존재해야 한다."""
        kb_count = SCENARIO["kb_count"]
        kb_info = f"{kb_count}개 ({', '.join(SCENARIO['kb_labels'])})" if kb_count > 0 else "아니오"
        print(f"\n{'=' * 60}")
        print(f"  시나리오 #{SCENARIO_NUM}: {SCENARIO['name']}")
        print(f"  구성: {SCENARIO['graph']}")
        print(f"  KB: {kb_info}")
        print(f"  프롬프트: {len(SCENARIO['prompts'])}개")
        print(f"  상태 파일: {STATE_FILE.name}")
        print(f"{'=' * 60}")

        target = workflow_primary_target_model_name()
        model = self._find_model_by_name(api_url, auth_headers, target)
        assert model["id"]
        self.__class__.model = model
        kind = {10: "ODM", 11: "pLM", 12: "BFM(fill-mask)", 13: "BFM(structure-prediction)"}.get(SCENARIO_NUM, "LLM")
        print(f"\n✔ {kind} 모델 발견: id={model['id']}, name={model['name']}")

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
        assert emb

        created_ids: list[int] = []
        for i in range(kb_count):
            kb_file = KB_FILES[i]
            assert kb_file.exists(), f"KB 파일이 없습니다: {kb_file}"

            label = SCENARIO["kb_labels"][i]
            form_data = {
                "name": f"E2E-S{SCENARIO_NUM}-KB{i + 1}-{uuid.uuid4().hex[:8]}",
                "description": f"E2E 시나리오 #{SCENARIO_NUM} ({SCENARIO['name']}) KB{i + 1}: {label}",
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
            print(f"  ✔ 프롬프트 생성: {comp_name} → id={data['id']}, name={data['name']}")

        self.__class__.prompt_ids = created
        print(f"\n✔ 프롬프트 {len(created)}개 생성 완료")

    def test_06_create_workflow(self, api_url: str, auth_headers: dict):
        """시나리오에 맞는 워크플로우를 생성한다."""
        model = self.__class__.model
        assert model

        definition = build_workflow_definition(
            SCENARIO_NUM, model["id"], self.__class__.kb_ids, KB_TOP_K, self.__class__.prompt_ids
        )
        payload = {
            "name": f"E2E-S{SCENARIO_NUM}-deploy-{uuid.uuid4().hex[:8]}",
            "description": f"E2E 시나리오 #{SCENARIO_NUM}: {SCENARIO['name']}",
            "workflow_definition": definition,
        }

        resp = requests.post(f"{api_url}/workflows", json=payload, headers=auth_headers)
        assert resp.status_code == 201, f"워크플로우 생성 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        self.__class__.workflow_id = data["id"]
        append_deployment_entry(
            SCENARIO_NUM,
            {
                "workflow_id": data["id"],
                "workflow_name": payload["name"],
                "kb_ids": list(self.__class__.kb_ids),
                "prompt_ids": dict(self.__class__.prompt_ids),
            },
        )
        n_saved = len(load_deployment_entries(SCENARIO_NUM))
        print(f"\n✔ 워크플로우 생성 완료: id={data['id']} (시나리오 #{SCENARIO_NUM} 저장 건수: {n_saved})")

        # 생성한 컴포넌트의 x, y 좌표가 조회 응답에 그대로 반환되는지 검증한다 (음수 좌표 포함).
        wf_read = requests.get(f"{api_url}/workflows/{data['id']}", headers=auth_headers)
        assert wf_read.status_code == 200, f"워크플로우 조회 실패: {wf_read.status_code} {wf_read.text}"
        sent_by_name = {c["name"]: (c["x"], c["y"]) for c in definition["components"]}
        for comp in wf_read.json()["components"]:
            assert comp["name"] in sent_by_name, f"예상치 못한 컴포넌트: {comp['name']}"
            assert (comp["x"], comp["y"]) == sent_by_name[comp["name"]], (
                f"컴포넌트 '{comp['name']}' 의 좌표 불일치: "
                f"보낸 값={sent_by_name[comp['name']]}, 응답={(comp['x'], comp['y'])}"
            )
        print(f"  ✔ 컴포넌트 {len(definition['components'])}개 x, y 좌표 일치 확인")

    def test_07_execute_workflow(self, api_url: str, auth_headers: dict):
        """워크플로우를 실행(배포)한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/execute",
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"실행 실패: {resp.status_code} {resp.text}"
        # MODEL 이 전부 원격 서빙이면 백엔드가 Kubeflow 파이프라인을 만들지 않고 즉시 배포를 끝내고
        # "succeeded" 를 돌려준다(클러스터에 만들 리소스가 없다). 실물 가중치를 올리는 경로만 "running".
        exec_status = resp.json()["status"]
        assert exec_status in ("running", "succeeded"), f"예상 밖 실행 상태: {exec_status} ({resp.text})"
        if exec_status == "succeeded":
            print("\n✔ 워크플로우 실행 완료 (원격 서빙 전용 — 파이프라인 없이 즉시 배포)")
        else:
            print("\n✔ 워크플로우 실행 시작")

    def test_08_wait_for_deployment(self, api_url: str, auth_headers: dict):
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
                return

            if any(s == "FAILED" for s in last_statuses):
                failed = [m for m in deployed_models if m.get("status") == "FAILED"]
                msgs = [f"{m.get('model_name')}: {m.get('error_message', 'N/A')}" for m in failed]
                pytest.fail(f"배포 실패: {'; '.join(msgs)}")

            remaining = int(deadline - time.time())
            print(f"  ⏳ 배포 대기 중… statuses={last_statuses} (남은 시간: {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)

        pytest.fail(f"배포 타임아웃 ({DEPLOY_TIMEOUT_SEC}s 초과)")

    def test_09_verify_active(self, api_url: str, auth_headers: dict):
        """배포 완료 후 워크플로우 상태가 ACTIVE 인지 확인한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        resp = requests.get(f"{api_url}/workflows/{wf_id}/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json().get("status") == "ACTIVE"
        print("\n✔ 상태 확인: ACTIVE")

    def test_10_inference(self, api_url: str, auth_headers: dict):
        """추론 테스트 — 워크플로우 파이프라인을 검증한다."""
        wf_id = self.__class__.workflow_id
        assert wf_id

        inference_kind = SCENARIO.get("inference_kind")
        if inference_kind == "ml":
            img = WORKFLOW_ODM_TEST_IMAGE
            assert img.is_file(), f"ODM 테스트 이미지가 없습니다: {img}"
            with open(img, "rb") as f:
                resp = requests.post(
                    f"{api_url}/workflows/{wf_id}/test/ml",
                    files={"image": (img.name, f, "image/png")},
                    headers=auth_headers,
                    timeout=INFERENCE_TIMEOUT_SEC,
                )
        elif inference_kind == "plm":
            sample_path = WORKFLOW_PLM_TEST_SAMPLE
            assert sample_path.is_file(), f"pLM 테스트 샘플이 없습니다: {sample_path}"
            sample = json.loads(sample_path.read_text())
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/test/protein-classification",
                json={"epitope": sample["epitope"], "cdr3b": sample["cdr3b"]},
                headers=auth_headers,
                timeout=INFERENCE_TIMEOUT_SEC,
            )
        elif inference_kind == "fill_mask":
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/test/fill-mask",
                json={"sequence": WORKFLOW_FILLMASK_TEST_SEQUENCE, "top_k": WORKFLOW_FILLMASK_TOP_K},
                headers=auth_headers,
                timeout=INFERENCE_TIMEOUT_SEC,
            )
        elif inference_kind == "structure_prediction":
            # 구조예측은 콜드스타트+확산 샘플링이라 fill-mask 보다 오래 걸린다 → 타임아웃 상향(5분).
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/test/protein-structure-prediction",
                json={
                    "sequence": WORKFLOW_STRUCTURE_TEST_SEQUENCE,
                    "num_loops": WORKFLOW_STRUCTURE_NUM_LOOPS,
                    "num_sampling_steps": WORKFLOW_STRUCTURE_NUM_SAMPLING_STEPS,
                },
                headers=auth_headers,
                timeout=300,
            )
        else:
            resp = requests.post(
                f"{api_url}/workflows/{wf_id}/test/rag",
                data={"text": SCENARIO["inference_text"]},
                headers=auth_headers,
                timeout=INFERENCE_TIMEOUT_SEC,
            )
        assert resp.status_code == 200, f"추론 실패: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data["workflow_id"] == wf_id
        assert len(data["results"]) > 0

        print("\n✔ 추론 성공")
        if inference_kind == "plm":
            # protein-classification 응답은 final_result 없이 results[].result.predictions 를 검증
            result = data["results"][0]
            assert (
                result.get("task") == "protein-classification"
            ), f"task 기대=protein-classification, 실제={result.get('task')}"
            preds = (result.get("result") or {}).get("predictions") or []
            assert preds, f"predictions 가 비어 있습니다: {result}"
            top = preds[0]
            assert "label" in top and "score" in top, f"predictions 형식 오류: {top}"
            print(f"  predictions[0]: label={top.get('label')}, score={top.get('score')}")
        elif inference_kind == "fill_mask":
            # fill-mask 응답도 final_result 없이 results[].result.predictions(마스크 위치별 top-k) 를 검증
            result = data["results"][0]
            assert result.get("task") == "fill-mask", f"task 기대=fill-mask, 실제={result.get('task')}"
            preds = (result.get("result") or {}).get("predictions") or []
            assert preds, f"fill-mask predictions 가 비어 있습니다: {result}"
            top = preds[0]
            assert "position" in top and top.get("predictions"), f"fill-mask 형식 오류(position/predictions): {top}"
            tok0 = top["predictions"][0]
            assert "token" in tok0 and "score" in tok0, f"토큰 예측 형식 오류: {tok0}"
            print(f"  mask@pos{top.get('position')} top1: token={tok0.get('token')!r}, score={tok0.get('score')}")
        elif inference_kind == "structure_prediction":
            # 구조예측 응답: results[].result.predictions[0] 에 pdb 문자열 + plddt/ptm/iptm 신뢰도
            result = data["results"][0]
            assert (
                result.get("task") == "protein-structure-prediction"
            ), f"task 기대=protein-structure-prediction, 실제={result.get('task')}"
            preds = (result.get("result") or {}).get("predictions") or []
            assert preds, f"structure predictions 가 비어 있습니다: {result}"
            top = preds[0]
            pdb = top.get("pdb") or ""
            assert pdb and "ATOM" in pdb, f"PDB 문자열이 비었거나 ATOM 레코드가 없습니다: keys={list(top.keys())}"
            print(
                f"  구조예측 성공: pdb_len={len(pdb)}, plddt_mean={top.get('plddt_mean')}, "
                f"ptm={top.get('ptm')}, iptm={top.get('iptm')}"
            )
        else:
            fr = data.get("final_result") or ""
            assert len(fr) > 0
            if inference_kind == "ml":
                print(f"  final_result: (base64 이미지 문자열, 길이 {len(fr)})")
            else:
                print(f"  final_result: {fr[:100]}{'…' if len(fr) > 100 else ''}")
        _env = (os.environ.get("ENV") or "").strip()
        if _env:
            print(f"\n  ℹ 삭제하려면: make e2e-wf-scenario-delete SCENARIO={SCENARIO_NUM} ENV={_env}")
        else:
            print(f"\n  ℹ 삭제하려면: make e2e-wf-scenario-delete SCENARIO={SCENARIO_NUM}")
