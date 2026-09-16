"""원격 LLM 서빙 판정(`decide_workflow_serving`)과 호출 어댑터 단위 테스트.

pytest 미설치 환경을 고려해 stdlib unittest 로 작성한다. DB·네트워크를 쓰지 않도록
Model·Settings 는 필요한 속성만 가진 합성 객체로 대체한다.

실행:
    cd backend/app && ENV=local PYTHONPATH=. python -m unittest tests.test_remote_serving -v
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from core.serving.remote_workflow_serving import (
    REMOTE_PATH_URL_BUDGET_BYTES,
    SLASH_SUBSTITUTE,
    build_path_chat_url,
    flatten_messages_to_prompt,
    parse_path_chat_response,
    path_chat_error_message,
    prepare_path_chat_request,
    sanitize_prompt_for_path,
)
from core.serving.serving_mode import ServingMode
from core.serving.serving_workflow_deployment_policy import decide_workflow_serving
from db.models.model_workflow_deployment import WorkflowServingDeploymentType as DType
from services.model import PREDEFINED_MODEL_CONFIGS

BASE = "http://remote.example.com:8001"


def make_settings(api_url: str = "", model_map: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        REMOTE_SERVING_API_URL=api_url,
        REMOTE_SERVING_MODEL_MAP=json.dumps(model_map or {}),
    )


def make_model(
    repo_id: str,
    *,
    provider: str = "ollama",
    pvc: str | None = "pvc-x",
    artifact_path: str = "s3://bucket/model",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        repo_id=repo_id,
        provider_info=SimpleNamespace(name=provider),
        registry=SimpleNamespace(pvc=pvc, artifact_path=artifact_path),
    )


class DecideWorkflowServingTest(unittest.TestCase):
    """§8 진리표. local 결과는 registry 값이 채워진 정상 모델 기준."""

    LOCAL_OLLAMA = "qwq:32b"  # PREDEFINED: serving_mode=local, provider=ollama
    REMOTE_ONLY = "qwen2.5:0.5b"  # PREDEFINED: serving_mode=remote_only

    def test_local_tag_ignores_remote_map(self):
        s = make_settings(BASE, {self.LOCAL_OLLAMA: "gw-qwq"})
        d = decide_workflow_serving(make_model(self.LOCAL_OLLAMA), s)
        self.assertEqual(d.deployment_type, DType.OLLAMA)
        self.assertEqual(d.serving_mode, ServingMode.LOCAL)
        self.assertIsNone(d.reject_reason)

    def test_local_tag_without_remote_config(self):
        d = decide_workflow_serving(make_model(self.LOCAL_OLLAMA), make_settings())
        self.assertEqual(d.deployment_type, DType.OLLAMA)
        self.assertIsNone(d.reject_reason)

    def test_remote_only_with_config(self):
        s = make_settings(BASE, {self.REMOTE_ONLY: self.REMOTE_ONLY})
        d = decide_workflow_serving(make_model(self.REMOTE_ONLY, provider="custom"), s)
        self.assertEqual(d.deployment_type, DType.REMOTE)
        self.assertEqual(d.remote_model_name, self.REMOTE_ONLY)
        self.assertEqual(d.remote_base_url, BASE)
        self.assertIsNone(d.reject_reason)

    def test_remote_only_without_map_entry_rejects(self):
        d = decide_workflow_serving(make_model(self.REMOTE_ONLY, provider="custom"), make_settings(BASE, {}))
        self.assertIsNotNone(d.reject_reason)
        self.assertIn("원격 서빙 전용", d.reject_reason)

    def test_remote_only_without_api_url_rejects(self):
        s = make_settings("", {self.REMOTE_ONLY: self.REMOTE_ONLY})
        d = decide_workflow_serving(make_model(self.REMOTE_ONLY, provider="custom"), s)
        self.assertIsNotNone(d.reject_reason)

    def test_remote_only_placeholder_registry_is_not_rejected(self):
        """등록 시 만든 placeholder(artifact_path='', pvc=None)가 로컬 검사에 걸리면 안 된다."""
        s = make_settings(BASE, {self.REMOTE_ONLY: self.REMOTE_ONLY})
        model = make_model(self.REMOTE_ONLY, provider="custom", pvc=None, artifact_path="")
        d = decide_workflow_serving(model, s)
        self.assertEqual(d.deployment_type, DType.REMOTE)
        self.assertIsNone(d.reject_reason)

    def test_unregistered_repo_id_in_map_is_remote(self):
        """PREDEFINED 미등록 모델은 하위호환으로 맵에 있으면 REMOTE."""
        s = make_settings(BASE, {"vendor/custom-llm": "vendor/custom-llm"})
        d = decide_workflow_serving(make_model("vendor/custom-llm", provider="custom"), s)
        self.assertEqual(d.deployment_type, DType.REMOTE)
        self.assertEqual(d.serving_mode, ServingMode.REMOTE_PREFERRED)

    def test_remote_preferred_falls_back_to_local(self):
        """맵에서 빠지면 실물 가중치 경로로 폴백한다."""
        d = decide_workflow_serving(make_model("vendor/custom-llm"), make_settings())
        self.assertEqual(d.deployment_type, DType.OLLAMA)
        self.assertIsNone(d.reject_reason)

    def test_remote_preferred_without_local_artifacts_rejects(self):
        s = make_settings("", {"vendor/custom-llm": "x"})
        model = make_model("vendor/custom-llm", pvc=None)
        d = decide_workflow_serving(model, s)
        self.assertIsNotNone(d.reject_reason)
        self.assertIn("로컬 폴백도 불가", d.reject_reason)

    def test_local_without_artifacts_rejects(self):
        model = make_model(self.LOCAL_OLLAMA, pvc="")
        d = decide_workflow_serving(model, make_settings())
        self.assertIsNotNone(d.reject_reason)
        self.assertIn("Ollama 가중치 PVC", d.reject_reason)

    def test_kserve_model_uses_artifact_path(self):
        model = make_model("facebook/detr-resnet-50", provider="huggingface", pvc=None)
        d = decide_workflow_serving(model, make_settings())
        self.assertEqual(d.deployment_type, DType.KSERVE)
        self.assertIsNone(d.reject_reason)


class PredefinedTagTest(unittest.TestCase):
    def test_all_serving_mode_values_are_valid(self):
        for key, cfg in PREDEFINED_MODEL_CONFIGS.items():
            raw = cfg.get("serving_mode")
            if raw is None:
                continue
            ServingMode(str(raw))  # 잘못된 문자열이면 ValueError

    def test_remote_only_entries_are_consistent(self):
        for key, cfg in PREDEFINED_MODEL_CONFIGS.items():
            if cfg.get("serving_mode") != ServingMode.REMOTE_ONLY.value:
                continue
            self.assertEqual(cfg["repo_id"], key, f"{key}: repo_id 가 키와 다르다")
            self.assertEqual(cfg["type_name"], "LLM", f"{key}: 원격 전용은 LLM 타입만 허용")
            self.assertNotIn("ollama_pvc_storage", cfg, f"{key}: 원격 전용에 PVC 용량 키가 있다")
            self.assertNotIn("serving_vram_need_bytes", cfg, f"{key}: 원격 전용에 서빙 자원 키가 있다")


class PathChatAdapterTest(unittest.TestCase):
    MODEL = "deepseek-r1:1.5b"

    def test_flatten_messages(self):
        messages = [
            {"role": "system", "content": "너는 도우미다"},
            {"role": "system", "content": "[참고자료]\n자료 본문"},
            {"role": "user", "content": "질문입니다"},
        ]
        prompt = flatten_messages_to_prompt(messages)
        self.assertIn("[지시]", prompt)
        self.assertIn("너는 도우미다", prompt)
        self.assertIn("자료 본문", prompt)
        self.assertTrue(prompt.endswith("질문입니다"))

    def test_slash_is_substituted(self):
        out, replaced = sanitize_prompt_for_path("1/2 는 얼마?")
        self.assertTrue(replaced)
        self.assertNotIn("/", out)
        self.assertIn(SLASH_SUBSTITUTE, out)

    def test_prepare_request_encodes_and_substitutes(self):
        messages = [{"role": "user", "content": "경로 a/b 설명"}]
        url, err = prepare_path_chat_request(BASE, self.MODEL, messages)
        self.assertIsNone(err)
        self.assertTrue(url.startswith(f"{BASE}/model/chat/"))
        self.assertNotIn("%2F", url.split("/model/chat/")[1])

    def test_prepare_request_requires_base_and_model(self):
        messages = [{"role": "user", "content": "질문"}]
        self.assertIsNotNone(prepare_path_chat_request("", self.MODEL, messages)[1])
        self.assertIsNotNone(prepare_path_chat_request(BASE, "", messages)[1])

    def test_oversized_prompt_without_reference_is_rejected(self):
        messages = [{"role": "user", "content": "가" * 5000}]
        url, err = prepare_path_chat_request(BASE, self.MODEL, messages)
        self.assertIsNone(url)
        self.assertIn("2,600자", err)

    def test_oversized_reference_block_is_truncated(self):
        messages = [
            {"role": "system", "content": "[참고자료]\n" + "나" * 5000},
            {"role": "user", "content": "요약해줘"},
        ]
        url, err = prepare_path_chat_request(BASE, self.MODEL, messages)
        self.assertIsNone(err)
        self.assertLessEqual(len(url.encode("utf-8")), REMOTE_PATH_URL_BUDGET_BYTES)

    def test_parse_response_unwraps_json_string(self):
        self.assertEqual(parse_path_chat_response(json.dumps("안녕하세요")), "안녕하세요")

    def test_parse_response_strips_thinking(self):
        body = json.dumps("<think>내부 추론</think>최종 답변")
        self.assertEqual(parse_path_chat_response(body), "최종 답변")
        body2 = json.dumps("<unused94>thought\n생각 과정\n\n실제 답변")
        self.assertEqual(parse_path_chat_response(body2), "실제 답변")

    def test_error_messages(self):
        self.assertIn("없습니다", path_chat_error_message(500, '{"detail":"\'x:1b\'"}', "x:1b"))
        self.assertIn(
            "서빙할 수 없습니다",
            path_chat_error_message(500, '{"detail":"no_available_chat_model: ..."}', "glm-ocr:q8_0"),
        )
        self.assertIn("2,600자", path_chat_error_message(400, "", self.MODEL))
        self.assertIn("처리할 수 없는 문자", path_chat_error_message(404, '{"detail":"Not Found"}', self.MODEL))

    def test_model_name_is_not_slash_substituted(self):
        url = build_path_chat_url(BASE, "org/model", "hi")
        self.assertIn("org%2Fmodel", url)


if __name__ == "__main__":
    unittest.main()
