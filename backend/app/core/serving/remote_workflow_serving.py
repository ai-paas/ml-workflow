"""원격 LLM 서버 호출 어댑터.

두 가지 요청 규약을 지원한다.

- `path_chat`: `GET {base}/model/chat/{model}/{prompt}` → 본문이 JSON 문자열 하나.
  프롬프트를 URL 경로에 싣기 때문에 '/' 를 전송할 수 없고 요청 길이 상한도 URL 바이트로 걸린다.
- `openai`  : `POST {base}/v1/chat/completions` → `choices[0].message.content`.

호출부는 규약을 몰라도 되도록 `remote_chat_completion_async` / `_sync` 시그니처를 유지하고,
내부에서 `settings.REMOTE_SERVING_PROTOCOL` 로 분기한다.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum as PyEnum
from typing import Any, Optional, Tuple
from urllib.parse import quote

import httpx
import requests
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RemoteServingProtocol(str, PyEnum):
    """원격 LLM 서버의 요청 규약."""

    PATH_CHAT = "path_chat"
    OPENAI = "openai"

    def __str__(self) -> str:
        return self.value


# '/' 는 경로 구분자로 해석돼 퍼센트 인코딩(%2F)해도 404 가 된다. 시각적으로 같은 별도 글리프로 바꿔 보낸다.
SLASH_SUBSTITUTE = "∕"  # DIVISION SLASH

# 원격 서버가 받아들이는 요청 URL 실측 상한은 약 23.7KB 다. 프록시·헤더 여유를 두고 잘라 쓴다.
REMOTE_PATH_URL_BUDGET_BYTES = 20_000

# 사고 과정(thinking)을 그대로 실어 보내는 모델이 있어 응답에서 걷어낸다.
_THINK_PATTERNS = (
    re.compile(r"<think>.*?</think>\s*", re.S),
    re.compile(r"<unused\d+>\s*thought\b.*?(?=\n\n|\Z)", re.S),
)

_LENGTH_ERROR_MESSAGE = (
    "원격 서버는 요청 1건당 약 2,600자(한글 기준)까지만 받습니다. 프롬프트 또는 참고자료를 줄여 주세요."
)


def _timeout_message(timeout: float) -> str:
    """flake8 이 f-string 의 포맷 스펙 콜론을 오탐하므로 메시지 생성을 함수로 뺀다."""
    return f"원격 모델 응답이 지연됩니다(타임아웃 {int(timeout)}초)."


def resolve_remote_protocol(protocol: Optional[RemoteServingProtocol] = None) -> RemoteServingProtocol:
    if protocol is not None:
        return protocol
    raw = (get_settings().REMOTE_SERVING_PROTOCOL or "").strip().lower()
    return RemoteServingProtocol(raw) if raw else RemoteServingProtocol.PATH_CHAT


def resolve_remote_timeout(timeout_sec: Optional[float] = None) -> float:
    if timeout_sec is not None:
        return float(timeout_sec)
    return float(get_settings().REMOTE_SERVING_TIMEOUT_SEC)


def flatten_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """system/user 메시지를 원격 서버가 받는 단일 프롬프트로 합친다."""
    system_parts = [str(m.get("content")) for m in messages if m.get("role") == "system" and m.get("content")]
    user_parts = [str(m.get("content")) for m in messages if m.get("role") == "user" and m.get("content")]
    blocks: list[str] = []
    if system_parts:
        blocks.append("[지시]\n" + "\n\n".join(system_parts))
    if user_parts:
        blocks.append("[질문]\n" + "\n\n".join(user_parts))
    return "\n\n".join(blocks)


def sanitize_prompt_for_path(prompt: str) -> Tuple[str, bool]:
    """URL 경로에 실을 수 있도록 '/' 를 대체 글리프로 바꾼다. (변환된 프롬프트, 치환 여부)."""
    if "/" not in prompt:
        return prompt, False
    return prompt.replace("/", SLASH_SUBSTITUTE), True


def build_path_chat_url(base_url: str, remote_model_name: str, prompt: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    return f"{b}/model/chat/{quote(remote_model_name, safe='')}/{quote(prompt, safe='')}"


def _url_bytes(base_url: str, remote_model_name: str, prompt: str) -> int:
    return len(build_path_chat_url(base_url, remote_model_name, prompt).encode("utf-8"))


def fits_in_path_budget(base_url: str, remote_model_name: str, prompt: str) -> bool:
    return _url_bytes(base_url, remote_model_name, prompt) <= REMOTE_PATH_URL_BUDGET_BYTES


def _truncate_reference_block(prompt: str, base_url: str, remote_model_name: str) -> Optional[str]:
    """예산을 넘으면 참고자료(지식베이스 검색 결과) 블록을 뒤에서부터 잘라 맞춘다.

    참고자료가 없거나 잘라도 모자라면 None(호출 불가)을 돌려준다.
    """
    marker = "[참고자료]"
    idx = prompt.find(marker)
    if idx < 0:
        return None

    head = prompt[: idx + len(marker)]
    tail = prompt[idx + len(marker) :]
    lo, hi = 0, len(tail)
    best: Optional[str] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = head + tail[:mid]
        if fits_in_path_budget(base_url, remote_model_name, candidate):
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None or not best[len(head) :].strip():
        return None
    return best


def parse_path_chat_response(body: str) -> str:
    """응답 본문(JSON 문자열 리터럴)에서 답변 텍스트를 뽑고 thinking 구간을 제거한다."""
    text = json.loads(body)
    if not isinstance(text, str):
        raise ValueError("원격 응답이 문자열이 아닙니다.")
    for pat in _THINK_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def path_chat_error_message(status_code: int, body: str, remote_model_name: str) -> str:
    """원격 서버는 4xx 를 거의 쓰지 않으므로 상태코드와 본문을 함께 보고 사유를 고른다."""
    snippet = (body or "").strip()[:500]
    if status_code == 404:
        return "프롬프트에 원격 서버가 처리할 수 없는 문자가 있습니다."
    if status_code == 400:
        return _LENGTH_ERROR_MESSAGE
    if status_code >= 500:
        if "no_available_chat_model" in snippet:
            return f"원격 서버가 '{remote_model_name}' 을 현재 서빙할 수 없습니다."
        if remote_model_name and remote_model_name in snippet:
            return (
                f"원격 서버에 '{remote_model_name}' 모델이 없습니다. "
                f"서버 설정(REMOTE_SERVING_MODEL_MAP)을 확인하세요."
            )
    return f"REMOTE 추론 HTTP {status_code}: {snippet}"


def prepare_path_chat_request(
    base_url: str,
    remote_model_name: str,
    messages: list[dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """(요청 URL, 에러 메시지). URL 이 None 이면 호출하지 않고 에러를 돌려준다."""
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return None, "원격 추론 베이스 URL이 비어 있습니다."
    if not (remote_model_name or "").strip():
        return None, "원격 추론 모델명이 비어 있습니다."

    prompt, replaced = sanitize_prompt_for_path(flatten_messages_to_prompt(messages))
    if replaced:
        logger.info("REMOTE path_chat: 프롬프트의 '/' 를 대체 글리프로 치환했습니다(원격 경로 제약).")
    if not prompt.strip():
        return None, "원격 추론 프롬프트가 비어 있습니다."

    if not fits_in_path_budget(b, remote_model_name, prompt):
        shortened = _truncate_reference_block(prompt, b, remote_model_name)
        if shortened is None:
            return None, _LENGTH_ERROR_MESSAGE
        logger.warning(
            "REMOTE path_chat: 요청 URL 예산(%d bytes) 초과로 참고자료를 잘랐습니다.",
            REMOTE_PATH_URL_BUDGET_BYTES,
        )
        prompt = shortened

    return build_path_chat_url(b, remote_model_name, prompt), None


def _parse_openai_style_content(result: dict[str, Any]) -> str:
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    return c
    return ""


def _openai_request(base_url: str, remote_model_name: str, messages: list[dict[str, Any]]) -> Tuple[str, dict]:
    b = (base_url or "").strip().rstrip("/")
    url = f"{b}/v1/chat/completions"
    body: dict[str, Any] = {"model": remote_model_name, "messages": messages, "stream": False}
    return url, body


def remote_chat_completion_sync(
    base_url: str,
    remote_model_name: str,
    messages: list[dict[str, Any]],
    *,
    timeout_sec: Optional[float] = None,
    protocol: Optional[RemoteServingProtocol] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """동기 원격 추론. (응답 텍스트, 에러 메시지)."""
    proto = resolve_remote_protocol(protocol)
    timeout = resolve_remote_timeout(timeout_sec)

    if proto is RemoteServingProtocol.PATH_CHAT:
        url, err = prepare_path_chat_request(base_url, remote_model_name, messages)
        if err:
            return None, err
        try:
            r = requests.get(url, headers={"accept": "application/json"}, timeout=timeout)
            if r.status_code >= 400:
                return None, path_chat_error_message(r.status_code, r.text, remote_model_name)
            return parse_path_chat_response(r.text), None
        except requests.Timeout:
            return None, _timeout_message(timeout)
        except Exception as e:
            return None, f"REMOTE 추론 요청 실패: {e}"

    if not (base_url or "").strip():
        return None, "원격 추론 베이스 URL이 비어 있습니다."
    url, body = _openai_request(base_url, remote_model_name, messages)
    try:
        r = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=timeout)
        if r.status_code >= 400:
            return None, f"REMOTE 추론 HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        if not isinstance(data, dict):
            return None, "REMOTE 추론 응답이 JSON 객체가 아닙니다."
        return _parse_openai_style_content(data), None
    except requests.Timeout:
        return None, _timeout_message(timeout)
    except Exception as e:
        return None, f"REMOTE 추론 요청 실패: {e}"


async def remote_chat_completion_async(
    base_url: str,
    remote_model_name: str,
    messages: list[dict[str, Any]],
    *,
    timeout_sec: Optional[float] = None,
    protocol: Optional[RemoteServingProtocol] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """비동기 원격 추론. (응답 텍스트, 에러 메시지)."""
    proto = resolve_remote_protocol(protocol)
    timeout = resolve_remote_timeout(timeout_sec)

    if proto is RemoteServingProtocol.PATH_CHAT:
        url, err = prepare_path_chat_request(base_url, remote_model_name, messages)
        if err:
            return None, err
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, headers={"accept": "application/json"})
            if r.status_code >= 400:
                return None, path_chat_error_message(r.status_code, r.text, remote_model_name)
            return parse_path_chat_response(r.text), None
        except httpx.TimeoutException:
            return None, _timeout_message(timeout)
        except Exception as e:
            return None, f"REMOTE 추론 요청 실패: {e}"

    if not (base_url or "").strip():
        return None, "원격 추론 베이스 URL이 비어 있습니다."
    url, body = _openai_request(base_url, remote_model_name, messages)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body, headers={"Content-Type": "application/json"})
        if r.status_code >= 400:
            return None, f"REMOTE 추론 HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        if not isinstance(data, dict):
            return None, "REMOTE 추론 응답이 JSON 객체가 아닙니다."
        return _parse_openai_style_content(data), None
    except httpx.TimeoutException:
        return None, _timeout_message(timeout)
    except Exception as e:
        return None, f"REMOTE 추론 요청 실패: {e}"
