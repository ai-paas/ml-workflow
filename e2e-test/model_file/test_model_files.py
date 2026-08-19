"""
E2E: 모델 파일 목록·다운로드 URL (GET /models/{id}/files, /files/download-url)

대상 모델은 자동으로 고른다. `/models` 를 훑어 저장 유형별(MLFLOW/OLLAMA/NONE) 첫 모델을 잡으며,
특정 모델로 고정하고 싶으면 환경변수로 지정한다.

  E2E_MODEL_FILES_MLFLOW_MODEL   MLflow 아티팩트를 쓰는 모델 name (파일 목록·다운로드 검증용)
  E2E_MODEL_FILES_OLLAMA_MODEL   Ollama 모델 name (빈 목록 + 사유 검증용)
  E2E_MODEL_FILES_REMOTE_MODEL   원격 서빙 전용 모델 name (빈 목록 + 사유 검증용)
  E2E_MODEL_FILES_SCAN_LIMIT     자동 탐색 시 훑어볼 모델 수 (기본 40)

실행:
  make e2e-model-files ENV=prod
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
import requests

SCAN_LIMIT = int(os.environ.get("E2E_MODEL_FILES_SCAN_LIMIT", "40"))
FIXED = {
    "MLFLOW": (os.environ.get("E2E_MODEL_FILES_MLFLOW_MODEL") or "").strip(),
    "OLLAMA": (os.environ.get("E2E_MODEL_FILES_OLLAMA_MODEL") or "").strip(),
    "NONE": (os.environ.get("E2E_MODEL_FILES_REMOTE_MODEL") or "").strip(),
}

# 저장 유형별로 한 번만 찾아 재사용한다(모델이 많으면 탐색이 느리다).
_resolved: dict[str, Optional[dict]] = {}


def _get(api_url: str, headers: dict, path: str, **kwargs) -> requests.Response:
    return requests.get(f"{api_url}{path}", headers=headers, timeout=60, **kwargs)


def _files(api_url: str, headers: dict, model_id: int, **params) -> requests.Response:
    return _get(api_url, headers, f"/models/{model_id}/files", params=params or None)


def _find_by_storage(api_url: str, headers: dict, storage: str) -> Optional[dict]:
    """저장 유형이 `storage` 인 모델 1건. {"id", "name", "body"}(files 응답) 또는 None."""
    if storage in _resolved:
        return _resolved[storage]

    r = _get(api_url, headers, "/models")
    assert r.status_code == 200, f"모델 목록 조회 실패: {r.status_code} {r.text}"
    models = r.json()

    fixed_name = FIXED[storage]
    if fixed_name:
        models = [m for m in models if (m.get("name") or "").strip() == fixed_name]
        assert models, f"지정한 모델을 찾을 수 없습니다: {fixed_name}"

    found = None
    for m in models[:SCAN_LIMIT]:
        fr = _files(api_url, headers, m["id"])
        if fr.status_code != 200:
            continue
        body = fr.json()
        if body.get("storage_type") == storage:
            found = {"id": m["id"], "name": m.get("name"), "body": body}
            break

    _resolved[storage] = found
    return found


def _require(api_url: str, headers: dict, storage: str) -> dict:
    found = _find_by_storage(api_url, headers, storage)
    if not found:
        pytest.skip(f"{storage} 저장 유형 모델이 없어 건너뜁니다(앞 {SCAN_LIMIT}건 기준).")
    return found


@pytest.mark.model_files
class TestModelFiles:
    """모델 파일 목록 API."""

    def test_00_endpoint_deployed(self, api_url: str, auth_headers: dict):
        """엔드포인트가 배포되어 있는지 먼저 확인한다.

        미배포 서버는 모든 경로에 404 를 주므로, 이 확인 없이는 '없는 모델 404' 같은 검사가
        엔드포인트가 없어도 통과해 버린다. 실제 모델 하나로 200 이 오는지 본다.
        """
        r = _get(api_url, auth_headers, "/models")
        assert r.status_code == 200, f"모델 목록 조회 실패: {r.status_code} {r.text}"
        models = r.json()
        assert models, "등록된 모델이 없어 검증할 수 없습니다."

        first = models[0]
        fr = _files(api_url, auth_headers, first["id"])
        assert fr.status_code == 200, (
            f"모델 파일 API 가 응답하지 않습니다(model_id={first['id']}, {fr.status_code}). "
            "백엔드가 이 기능을 포함해 배포되었는지 확인하세요."
        )
        body = fr.json()
        assert body.get("storage_type") in ("MLFLOW", "OLLAMA", "NONE"), f"예상 밖 응답: {body}"
        print(f"\n✔ 엔드포인트 확인 (model_id={first['id']}, storage_type={body['storage_type']})")

    def test_01_not_found(self, api_url: str, auth_headers: dict):
        """없는 모델은 404."""
        r = _files(api_url, auth_headers, 99999999)
        assert r.status_code == 404, f"기대 404, 실제 {r.status_code} {r.text}"
        print("\n✔ 없는 모델 404")

    def test_02_unauthorized(self, api_url: str):
        """인증 없이 호출하면 401."""
        r = requests.get(f"{api_url}/models/1/files", timeout=30)
        assert r.status_code == 401, f"기대 401, 실제 {r.status_code}"
        print("✔ 미인증 401")

    def test_03_mlflow_list(self, api_url: str, auth_headers: dict):
        """MLflow 모델: 파일 목록·필드·정렬·`.cache` 제외."""
        target = _require(api_url, auth_headers, "MLFLOW")
        body = target["body"]
        print(f"\n  대상: id={target['id']} name={target['name']}")
        print(f"  location={body.get('location')}")

        assert body["storage_type"] == "MLFLOW"
        assert isinstance(body["files"], list), "files 는 항상 배열이어야 한다"

        if not body["files"]:
            assert body.get("message"), "빈 목록이면 사유(message)가 있어야 한다"
            pytest.skip(f"아티팩트가 비어 있어 건너뜁니다: {body.get('message')}")

        names = [f["name"] for f in body["files"]]
        assert names == sorted(names), "name 오름차순이어야 한다"
        head = names[:3]
        assert not any(".cache" in n.split("/") for n in names), f".cache 가 목록에 섞였다: {head}"
        assert body.get("message") is None, "목록이 있으면 message 는 null 이어야 한다"

        first = body["files"][0]
        for key in ("name", "size_bytes", "last_modified", "download_url"):
            assert key in first, f"필드 누락: {key}"
        assert first["size_bytes"] >= 0
        assert first["last_modified"].endswith("Z"), f"UTC Z 표기여야 한다: {first['last_modified']}"
        assert first["download_url"].startswith(
            f"/api/v1/models/{target['id']}/files/download-url"
        ), f"download_url 은 발급 API 경로여야 한다: {first['download_url']}"
        assert "Signature" not in first["download_url"], "목록에 서명 URL 이 들어가면 안 된다"

        print(f"✔ 파일 {len(body['files'])}건, 정렬·필드·.cache 제외 확인")
        for f in body["files"][:3]:
            size = format(f["size_bytes"], ",").rjust(14)
            print(f"    {size}  {f['last_modified']}  {f['name']}")

    def test_04_pagination_contract(self, api_url: str, auth_headers: dict):
        """next_cursor 규약: 빈 목록과 토큰이 함께 나오지 않고, 이어받으면 중복이 없다."""
        target = _require(api_url, auth_headers, "MLFLOW")
        seen: list[str] = []
        cursor = None
        pages = 0

        while True:
            r = _files(api_url, auth_headers, target["id"], **({"cursor": cursor} if cursor else {}))
            assert r.status_code == 200, r.text
            body = r.json()
            pages += 1

            if not body["files"]:
                assert body.get("next_cursor") is None, "files 가 비었는데 next_cursor 가 있으면 안 된다"
            seen += [f["name"] for f in body["files"]]

            cursor = body.get("next_cursor")
            if not cursor:
                break
            assert pages < 50, "쪽이 지나치게 많다(무한 반복 의심)"

        assert len(seen) == len(set(seen)), "쪽 경계에서 중복이 발생했다"
        assert seen == sorted(seen), "쪽을 넘어가며 정렬이 깨졌다"
        print(f"\n✔ {pages}쪽, 총 {len(seen)}건, 중복·정렬 이상 없음")

    def test_05_download_url_and_fetch(self, api_url: str, auth_headers: dict):
        """다운로드 URL 발급 → 실제 내려받기(앞부분만)."""
        target = _require(api_url, auth_headers, "MLFLOW")
        files = target["body"]["files"]
        if not files:
            pytest.skip("아티팩트가 비어 있어 건너뜁니다.")

        # 너무 큰 파일은 피하고 가장 작은 것으로 확인한다.
        entry = min(files, key=lambda f: f["size_bytes"])
        r = _get(api_url, auth_headers, f"/models/{target['id']}/files/download-url", params={"name": entry["name"]})
        assert r.status_code == 200, f"발급 실패: {r.status_code} {r.text}"
        body = r.json()

        assert body["name"] == entry["name"]
        assert body["size_bytes"] == entry["size_bytes"]
        assert body["expires_at"].endswith("Z")
        url = body["download_url"]
        url_head = url[:80]
        # SigV4 는 X-Amz-Signature, SigV2 는 Signature 파라미터를 쓴다. 서명이 붙었는지만 본다.
        assert ("X-Amz-Signature" in url) or ("Signature=" in url), f"서명 URL 이 아니다: {url_head}"
        size = format(entry["size_bytes"], ",")
        print(f"\n  발급: {entry['name']} ({size} bytes), 만료 {body['expires_at']}")

        got = requests.get(url, headers={"Range": "bytes=0-63"}, timeout=60)
        assert got.status_code in (200, 206), f"다운로드 실패: {got.status_code} {got.text[:200]}"
        assert got.content, "빈 응답"
        print(f"✔ 실제 다운로드 HTTP {got.status_code}, {len(got.content)}바이트 수신")

    def test_06_download_url_rejects_traversal(self, api_url: str, auth_headers: dict):
        """경로를 벗어나거나 제외된 파일은 404."""
        target = _require(api_url, auth_headers, "MLFLOW")
        bad_names = [
            "../../../../etc/passwd",
            "../" * 8 + "mlflow/other-model/config.json",
            "does-not-exist-" + "x" * 12 + ".bin",
        ]
        for name in bad_names:
            r = _get(api_url, auth_headers, f"/models/{target['id']}/files/download-url", params={"name": name})
            assert r.status_code == 404, f"차단되지 않음({r.status_code}): {name}"
            assert "Signature" not in r.text, "차단해야 할 요청에 서명 URL 이 나갔다"
        print(f"\n✔ 경로 이탈·미존재 {len(bad_names)}건 모두 404")

    @pytest.mark.parametrize("storage", ["OLLAMA", "NONE"])
    def test_07_no_file_storage(self, api_url: str, auth_headers: dict, storage: str):
        """Ollama·원격 전용: 200 + 빈 목록 + 사유 message."""
        target = _require(api_url, auth_headers, storage)
        body = target["body"]
        print(f"\n  {storage} 대상: id={target['id']} name={target['name']}")

        assert body["files"] == [], "파일 목록을 제공하지 않아야 한다"
        assert body.get("next_cursor") is None
        assert body.get("message"), "빈 목록이면 사유가 있어야 한다"
        print(f"✔ 빈 목록 + message: {body['message']}")

    @pytest.mark.parametrize("storage", ["OLLAMA", "NONE"])
    def test_08_download_url_conflict(self, api_url: str, auth_headers: dict, storage: str):
        """다운로드를 제공하지 않는 저장소에 발급 요청하면 409."""
        target = _require(api_url, auth_headers, storage)
        r = _get(api_url, auth_headers, f"/models/{target['id']}/files/download-url", params={"name": "any-file"})
        assert r.status_code == 409, f"기대 409, 실제 {r.status_code} {r.text}"
        print(f"\n✔ {storage} 발급 요청 409")
