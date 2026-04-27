"""
E2E 시나리오: 워크플로우 정의 검증 오류 케이스

잘못된 워크플로우 정의가 validate API에서 정상적으로 오류를 반환하는지 확인한다.
validate API와 create/update API는 동일한 _validate_workflow_definition_checks 를
공유하므로, validate 통과 = create 통과가 보장된다.

검증 규칙:
  1. ref_id_validity        — 연결에 존재하지 않는 ref_id 사용
  2. no_self_connection     — 자기 자신에게 연결
  3. no_cycle               — 순환 참조
  4. model_count_limit      — MODEL 4개 초과
  5. connection_type_rules  — END→*, *→START, KB→KB 연결
  6. required_fields        — MODEL에 model_id 누락
  7. config_validation      — temperature 범위 초과, 허용되지 않는 키 등
"""

import uuid

import pytest
import requests
from config import TARGET_MODEL_NAME


@pytest.mark.workflow_validation
class TestWorkflowValidation:
    """잘못된 워크플로우 정의가 올바르게 거부되는지 검증"""

    model: dict | None = None

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _find_model_by_name(api_url: str, headers: dict, model_name: str) -> dict:
        resp = requests.get(f"{api_url}/models", headers=headers)
        assert resp.status_code == 200
        matched = [m for m in resp.json() if m.get("name") == model_name]
        assert matched, f"모델 '{model_name}' 을 찾을 수 없습니다."
        return matched[0]

    @staticmethod
    def _validate(api_url: str, headers: dict, definition: dict) -> dict:
        resp = requests.post(
            f"{api_url}/workflows/validate",
            json={"workflow_definition": definition},
            headers=headers,
        )
        assert resp.status_code == 200, f"validate API 호출 실패: {resp.status_code} {resp.text}"
        return resp.json()

    @staticmethod
    def _assert_valid(data: dict):
        assert data["valid"] is True, (
            f"valid=False — 실패한 규칙: "
            f"{[c['rule'] + ': ' + (c.get('message') or '') for c in data['checks'] if not c['passed']]}"
        )

    @staticmethod
    def _assert_rule_failed(data: dict, rule_name: str):
        assert data["valid"] is False, f"valid=True 이지만 {rule_name} 실패를 기대함"
        failed = {c["rule"]: c.get("message") for c in data["checks"] if not c["passed"]}
        assert rule_name in failed, f"규칙 '{rule_name}' 이 실패 목록에 없습니다. 실패: {list(failed.keys())}"
        print(f"    → {rule_name}: {failed[rule_name]}")

    @staticmethod
    def _base_definition(model_id: int) -> tuple[dict, str, str, str]:
        s = f"start-{uuid.uuid4().hex[:8]}"
        m = f"model-{uuid.uuid4().hex[:8]}"
        e = f"end-{uuid.uuid4().hex[:8]}"
        defn = {
            "components": [
                {"ref_id": s, "name": "시작", "type": "START"},
                {"ref_id": m, "name": "모델", "type": "MODEL", "model_id": model_id},
                {"ref_id": e, "name": "끝", "type": "END"},
            ],
            "connections": [
                {"source_ref_id": s, "target_ref_id": m},
                {"source_ref_id": m, "target_ref_id": e},
            ],
        }
        return defn, s, m, e

    # ── 사전 준비 ────────────────────────────────────────────

    def test_00_find_model(self, api_url: str, auth_headers: dict):
        """테스트에 사용할 모델을 조회한다."""
        model = self._find_model_by_name(api_url, auth_headers, TARGET_MODEL_NAME)
        self.__class__.model = model
        print(f"\n✔ 모델 발견: id={model['id']}, name={model['name']}")

    # ── 정상 케이스 (baseline) ───────────────────────────────

    def test_01_valid_definition(self, api_url: str, auth_headers: dict):
        """정상적인 정의는 valid=true 이어야 한다."""
        model = self.__class__.model
        assert model
        defn, *_ = self._base_definition(model["id"])

        data = self._validate(api_url, auth_headers, defn)
        self._assert_valid(data)
        print("\n✔ 정상 정의 — 모든 규칙 통과")

    # ── 오류 케이스들 ────────────────────────────────────────

    def test_02_invalid_ref_id(self, api_url: str, auth_headers: dict):
        """연결에 존재하지 않는 ref_id 를 사용하면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        defn["connections"].append({"source_ref_id": m, "target_ref_id": "nonexistent-ref"})

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "ref_id_validity")
        print("\n✔ ref_id_validity 검증 통과")

    def test_03_self_connection(self, api_url: str, auth_headers: dict):
        """컴포넌트가 자기 자신에게 연결되면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        defn["connections"].append({"source_ref_id": m, "target_ref_id": m})

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "no_self_connection")
        print("\n✔ no_self_connection 검증 통과")

    def test_04_cycle_detection(self, api_url: str, auth_headers: dict):
        """순환 참조가 있으면 오류"""
        model = self.__class__.model
        assert model
        s = f"start-{uuid.uuid4().hex[:8]}"
        m1 = f"model1-{uuid.uuid4().hex[:8]}"
        m2 = f"model2-{uuid.uuid4().hex[:8]}"
        e = f"end-{uuid.uuid4().hex[:8]}"
        defn = {
            "components": [
                {"ref_id": s, "name": "시작", "type": "START"},
                {"ref_id": m1, "name": "모델1", "type": "MODEL", "model_id": model["id"]},
                {"ref_id": m2, "name": "모델2", "type": "MODEL", "model_id": model["id"]},
                {"ref_id": e, "name": "끝", "type": "END"},
            ],
            "connections": [
                {"source_ref_id": s, "target_ref_id": m1},
                {"source_ref_id": m1, "target_ref_id": m2},
                {"source_ref_id": m2, "target_ref_id": m1},
                {"source_ref_id": m2, "target_ref_id": e},
            ],
        }

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "no_cycle")
        print("\n✔ no_cycle 검증 통과")

    def test_05_model_count_limit(self, api_url: str, auth_headers: dict):
        """MODEL 컴포넌트가 4개 이상이면 오류"""
        model = self.__class__.model
        assert model
        s = f"start-{uuid.uuid4().hex[:8]}"
        e = f"end-{uuid.uuid4().hex[:8]}"
        models = [f"model{i}-{uuid.uuid4().hex[:8]}" for i in range(4)]

        components = [{"ref_id": s, "name": "시작", "type": "START"}]
        for ref in models:
            components.append({"ref_id": ref, "name": ref, "type": "MODEL", "model_id": model["id"]})
        components.append({"ref_id": e, "name": "끝", "type": "END"})

        connections = [{"source_ref_id": s, "target_ref_id": models[0]}]
        for i in range(len(models) - 1):
            connections.append({"source_ref_id": models[i], "target_ref_id": models[i + 1]})
        connections.append({"source_ref_id": models[-1], "target_ref_id": e})

        defn = {"components": components, "connections": connections}

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "model_count_limit")
        print("\n✔ model_count_limit 검증 통과")

    def test_06_end_outgoing_connection(self, api_url: str, auth_headers: dict):
        """END 컴포넌트에서 나가는 연결이 있으면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        defn["connections"].append({"source_ref_id": e, "target_ref_id": m})

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "connection_type_rules")
        print("\n✔ connection_type_rules (END→*) 검증 통과")

    def test_07_start_incoming_connection(self, api_url: str, auth_headers: dict):
        """START 컴포넌트로 들어오는 연결이 있으면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        defn["connections"].append({"source_ref_id": m, "target_ref_id": s})

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "connection_type_rules")
        print("\n✔ connection_type_rules (*→START) 검증 통과")

    def test_08_model_without_model_id(self, api_url: str, auth_headers: dict):
        """MODEL 컴포넌트에 model_id 가 없으면 오류"""
        s = f"start-{uuid.uuid4().hex[:8]}"
        m = f"model-{uuid.uuid4().hex[:8]}"
        e = f"end-{uuid.uuid4().hex[:8]}"
        defn = {
            "components": [
                {"ref_id": s, "name": "시작", "type": "START"},
                {"ref_id": m, "name": "모델", "type": "MODEL"},
                {"ref_id": e, "name": "끝", "type": "END"},
            ],
            "connections": [
                {"source_ref_id": s, "target_ref_id": m},
                {"source_ref_id": m, "target_ref_id": e},
            ],
        }

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "required_fields")
        print("\n✔ required_fields (model_id 누락) 검증 통과")

    def test_09_temperature_out_of_range(self, api_url: str, auth_headers: dict):
        """temperature 가 1.0 초과이면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        for comp in defn["components"]:
            if comp["type"] == "MODEL":
                comp["config"] = {"temperature": 2.0}

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "config_validation")
        print("\n✔ config_validation (temperature 범위 초과) 검증 통과")

    def test_10_top_p_out_of_range(self, api_url: str, auth_headers: dict):
        """top_p 가 1.0 초과이면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        for comp in defn["components"]:
            if comp["type"] == "MODEL":
                comp["config"] = {"top_p": 1.5}

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "config_validation")
        print("\n✔ config_validation (top_p 범위 초과) 검증 통과")

    def test_11_max_tokens_out_of_range(self, api_url: str, auth_headers: dict):
        """max_tokens 가 4096 초과이면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        for comp in defn["components"]:
            if comp["type"] == "MODEL":
                comp["config"] = {"max_tokens": 9999}

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "config_validation")
        print("\n✔ config_validation (max_tokens 범위 초과) 검증 통과")

    def test_12_unknown_config_key(self, api_url: str, auth_headers: dict):
        """허용되지 않는 config 키를 사용하면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        for comp in defn["components"]:
            if comp["type"] == "MODEL":
                comp["config"] = {"unknown_key": 123}

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "config_validation")
        print("\n✔ config_validation (허용되지 않는 키) 검증 통과")

    def test_13_start_end_with_config(self, api_url: str, auth_headers: dict):
        """START/END 에 config 를 지정하면 오류"""
        model = self.__class__.model
        assert model
        defn, s, m, e = self._base_definition(model["id"])
        for comp in defn["components"]:
            if comp["type"] == "START":
                comp["config"] = {"temperature": 0.5}

        data = self._validate(api_url, auth_headers, defn)
        self._assert_rule_failed(data, "config_validation")
        print("\n✔ config_validation (START에 config 지정) 검증 통과")
