"""E2E 테스트 설정 — .env 파일에서 로딩, 환경 변수로 오버라이드 가능"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE_URL: str = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
API_PREFIX: str = "/api/v1"

USERNAME: str = os.environ.get("E2E_USERNAME", "admin")
PASSWORD: str = os.environ.get("E2E_PASSWORD", "admin")

TARGET_MODEL_NAME: str = os.environ.get("E2E_TARGET_MODEL_NAME", "gpt-oss-20b")
TARGET_EMBEDDING_MODEL_NAME: str = os.environ.get("E2E_TARGET_EMBEDDING_MODEL_NAME", "bge-m3")

_scenario_raw = os.environ.get("E2E_SCENARIO")
if not _scenario_raw:
    raise RuntimeError(
        "E2E_SCENARIO 환경변수가 설정되지 않았습니다. " "make e2e-wf-scenario-deploy SCENARIO=N 형태로 실행하세요."
    )
SCENARIO_NUM: int = int(_scenario_raw)

DEPLOY_TIMEOUT_SEC: int = int(os.environ.get("E2E_DEPLOY_TIMEOUT_SEC", "600"))
POLL_INTERVAL_SEC: int = int(os.environ.get("E2E_POLL_INTERVAL_SEC", "10"))
DELETE_TIMEOUT_SEC: int = int(os.environ.get("E2E_DELETE_TIMEOUT_SEC", "300"))

# ── Knowledge Base 기본값 ────────────────────────────────────────────────
KB_CHUNK_SIZE: int = int(os.environ.get("E2E_KB_CHUNK_SIZE", "500"))
KB_CHUNK_OVERLAP: int = int(os.environ.get("E2E_KB_CHUNK_OVERLAP", "50"))
KB_TOP_K: int = int(os.environ.get("E2E_KB_TOP_K", "3"))
KB_THRESHOLD: float = float(os.environ.get("E2E_KB_THRESHOLD", "0.5"))

# ── Knowledge Base 파일 경로 ─────────────────────────────────────────────
KB_FILE_DIR: Path = Path(__file__).parent / "knowledge-base-file"
KB_FILES: list[Path] = [
    KB_FILE_DIR / "고압가스 안전관리법(법률)(제21065호)(20251001).pdf",
    KB_FILE_DIR / "건축서비스산업 진흥법(법률)(제19990호)(20240710).pdf",
]

# ── 시나리오 간 상태 공유 ──────────────────────────────────────────────────
STATE_FILE: Path = Path(__file__).parent / ".state.json"


def save_state(key: str, value: str) -> None:
    """상태 파일에 key=value 를 저장한다."""
    data = {}
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
    data[key] = value
    STATE_FILE.write_text(json.dumps(data, indent=2))


def load_state(key: str) -> str | None:
    """상태 파일에서 key 에 해당하는 값을 읽는다."""
    if not STATE_FILE.exists():
        return None
    data = json.loads(STATE_FILE.read_text())
    return data.get(key)


def clear_state(key: str) -> None:
    """상태 파일에서 key 를 제거한다."""
    if not STATE_FILE.exists():
        return
    data = json.loads(STATE_FILE.read_text())
    data.pop(key, None)
    STATE_FILE.write_text(json.dumps(data, indent=2))
