"""Remote LLM 호출 최소 구현. 인증·요청 스키마·스트리밍 등은 스텁(고정 OpenAI류 /v1/chat/completions)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import httpx
import requests


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


def remote_chat_completion_sync(
    base_url: str,
    remote_model_name: str,
    messages: list[dict[str, Any]],
    *,
    timeout_sec: float = 300.0,
) -> Tuple[Optional[str], Optional[str]]:
    """동기 POST /v1/chat/completions. (응답 텍스트, 에러 메시지)"""
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return None, "원격 추론 베이스 URL이 비어 있습니다."
    url = f"{b}/v1/chat/completions"
    body: dict[str, Any] = {"model": remote_model_name, "messages": messages, "stream": False}
    try:
        r = requests.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_sec,
        )
        if r.status_code >= 400:
            return None, f"REMOTE 추론 HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        if not isinstance(data, dict):
            return None, "REMOTE 추론 응답이 JSON 객체가 아닙니다."
        return _parse_openai_style_content(data), None
    except Exception as e:
        return None, f"REMOTE 추론 요청 실패: {e}"


async def remote_chat_completion_async(
    base_url: str,
    remote_model_name: str,
    messages: list[dict[str, Any]],
    *,
    timeout_sec: float = 300.0,
) -> Tuple[Optional[str], Optional[str]]:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return None, "원격 추론 베이스 URL이 비어 있습니다."
    url = f"{b}/v1/chat/completions"
    body: dict[str, Any] = {"model": remote_model_name, "messages": messages, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            r = await client.post(url, json=body, headers={"Content-Type": "application/json"})
        if r.status_code >= 400:
            return None, f"REMOTE 추론 HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        if not isinstance(data, dict):
            return None, "REMOTE 추론 응답이 JSON 객체가 아닙니다."
        return _parse_openai_style_content(data), None
    except Exception as e:
        return None, f"REMOTE 추론 요청 실패: {e}"
