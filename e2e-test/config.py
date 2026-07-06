"""E2E 테스트 설정 — .env 파일에서 로딩, 환경 변수로 오버라이드 가능

도트엔브 로딩 순서 (백엔드 make … ENV=dev 과 동일한 환경 변수 ENV 사용):

1. e2e-test/.env 가 있으면 먼저 로드 (공통 기본값, 기존 키는 덮어쓰지 않음).
2. ENV 가 비어 있지 않으면 e2e-test/.env.{ENV} 를 로드 (같은 키는 위 설정을 덮어씀).
   해당 파일이 없으면 경고만 하고, 셸에 이미 설정된 환경 변수만 사용.

예: make e2e-wf-scenario-deploy SCENARIO=1 ENV=dev → .env 후 .env.dev 적용.

배포/삭제 시나리오 상태 파일(.state):
  - ENV 미지정: e2e-test/.state.json (기존과 동일)
  - ENV 지정: e2e-test/.state.{ENV}.json (환경별 분리)
  delete / lifecycle 중 배포 기록을 쓰는 단계는 deploy 때와 동일한 ENV 로 실행해야 한다.

E2E_SCENARIO: 시나리오 배포/삭제/생명주기(make e2e-wf-scenario-*) 에서 필수.
  그 외(워크플로 validate, model_improvement 패키지 등)는 미설정 시 기본 1로 두어 import 만 통과한다.

최적화 E2E 시나리오(make e2e-model-improvement-scenario): 워크플로 LLM 타깃과 같이
  E2E_OPTIMIZATION_SOURCE_MODEL_NAME 으로 소스 모델을 지정(/models 의 name 정확 일치).
  E2E_OPTIMIZATION_TASK_TYPE (기본 pruning), E2E_OPTIMIZATION_POLL_TIMEOUT_SEC, E2E_OPTIMIZATION_POLL_INTERVAL_SEC
"""

import json
import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv

_E2E_ROOT = Path(__file__).parent
_SHARED_ENV = _E2E_ROOT / ".env"
_ENV_SUFFIX = (os.environ.get("ENV") or "").strip()
# 파일명에 쓸 ENV 슬러그 (경로 이탈·특수문자 방지)
_STATE_ENV_SLUG = ""
if _ENV_SUFFIX:
    if "/" in _ENV_SUFFIX or "\\" in _ENV_SUFFIX:
        warnings.warn(
            "E2E: ENV에 경로 구분자가 있어 상태 파일 분리를 건너뜁니다. (.state.json 사용)",
            stacklevel=1,
        )
    else:
        cand = re.sub(r"[^a-zA-Z0-9._-]+", "_", _ENV_SUFFIX).strip("._")
        if cand and ".." not in cand:
            _STATE_ENV_SLUG = cand
        else:
            warnings.warn(
                f"E2E: ENV={_ENV_SUFFIX!r} 은 상태 파일명에 부적합하여 .state.json 만 사용합니다.",
                stacklevel=1,
            )

if _SHARED_ENV.is_file():
    load_dotenv(_SHARED_ENV, override=False)

if _ENV_SUFFIX:
    _specific = _E2E_ROOT / f".env.{_ENV_SUFFIX}"
    if _specific.is_file():
        load_dotenv(_specific, override=True)
    else:
        warnings.warn(
            f"E2E: ENV={_ENV_SUFFIX!r} 인데 {_specific.name} 이(가) 없습니다. "
            f"공통 {_SHARED_ENV.name} 및 셸 환경 변수만 사용합니다.",
            stacklevel=1,
        )

BASE_URL: str = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
API_PREFIX: str = "/api/v1"

USERNAME: str = os.environ.get("E2E_USERNAME", "admin")
PASSWORD: str = os.environ.get("E2E_PASSWORD", "admin")

# 워크플로 시나리오: LLM(1~9)용 / ODM(10)용 타깃 모델 display name (/models 의 name)
WORKFLOW_TARGET_LLM_MODEL: str = (
    (os.environ.get("E2E_WORKFLOW_TARGET_LLM_MODEL") or os.environ.get("E2E_TARGET_MODEL_NAME") or "gpt-oss-20b")
).strip()
WORKFLOW_TARGET_ODM_MODEL: str = (os.environ.get("E2E_WORKFLOW_TARGET_ODM_MODEL") or "facebook/detr-resnet-50").strip()
# 시나리오 #11(pLM) 타깃: 파인튜닝된 ESM2 자식 모델 name (학습→등록 e2e 산출물)
WORKFLOW_TARGET_PLM_MODEL: str = (os.environ.get("E2E_WORKFLOW_TARGET_PLM_MODEL") or "").strip()
# 시나리오 #12(BFM fill-mask) 타깃: base BFM 모델 name (parent 없는 원본, 어댑터 없이 직접 서빙)
WORKFLOW_TARGET_FILLMASK_MODEL: str = (
    os.environ.get("E2E_WORKFLOW_TARGET_FILLMASK_MODEL") or "facebook/esm2_t6_8M_UR50D"
).strip()

TARGET_EMBEDDING_MODEL_NAME: str = os.environ.get("E2E_TARGET_EMBEDDING_MODEL_NAME", "bge-m3")

# 시나리오 #10(ODM) ML 추론 시 업로드할 이미지 (PNG/JPEG 등)
WORKFLOW_ODM_TEST_IMAGE: Path = Path(
    os.environ.get("E2E_WORKFLOW_ODM_TEST_IMAGE") or str(_E2E_ROOT / "workflow" / "fixtures" / "odm_sample.jpg")
).expanduser()

# 시나리오 #11(pLM) 추론 입력 단백질 서열 샘플 (JSON: {"epitope": ..., "cdr3b": ...})
WORKFLOW_PLM_TEST_SAMPLE: Path = Path(
    os.environ.get("E2E_WORKFLOW_PLM_TEST_SAMPLE") or str(_E2E_ROOT / "workflow" / "fixtures" / "plm_sample.json")
).expanduser()

# 시나리오 #12(BFM fill-mask) 추론 입력: 마스크 토큰(<mask>) 포함 단백질 서열 + 마스크 위치별 top-k
WORKFLOW_FILLMASK_TEST_SEQUENCE: str = (
    os.environ.get("E2E_WORKFLOW_FILLMASK_TEST_SEQUENCE") or "MKTAYIAKQR<mask>ISFVKSHFSRQLEE"
).strip()
WORKFLOW_FILLMASK_TOP_K: int = int(os.environ.get("E2E_WORKFLOW_FILLMASK_TOP_K", "5"))

# ── 최적화/경량화 E2E (model-improvements 시나리오) ─────────────────────────
# 소스 모델: E2E_WORKFLOW_TARGET_LLM_MODEL 과 동일하게 등록된 name 으로만 지정
OPTIMIZATION_SOURCE_MODEL_NAME: str = (os.environ.get("E2E_OPTIMIZATION_SOURCE_MODEL_NAME") or "").strip()

OPTIMIZATION_TASK_TYPE: str = (os.environ.get("E2E_OPTIMIZATION_TASK_TYPE") or "pruning").strip()

OPTIMIZATION_POLL_TIMEOUT_SEC: int = int(os.environ.get("E2E_OPTIMIZATION_POLL_TIMEOUT_SEC", "3600"))
OPTIMIZATION_POLL_INTERVAL_SEC: int = int(os.environ.get("E2E_OPTIMIZATION_POLL_INTERVAL_SEC", "15"))

# 워크플로 시나리오 배포/삭제/생명주기: make … SCENARIO=N 로 필수 지정.
# 검증·최적화 등 다른 E2E는 미설정 시 1로 두어 import 만 통과시킨다(해당 테스트는 SCENARIO 미사용).
_scenario_raw = (os.environ.get("E2E_SCENARIO") or "").strip()
if _scenario_raw:
    SCENARIO_NUM: int = int(_scenario_raw)
else:
    SCENARIO_NUM = 1


def workflow_primary_target_model_name() -> str:
    """워크플로 시나리오 배포/생명주기: #10 은 ODM, #11 은 pLM, #12 는 BFM fill-mask, 그 외는 LLM."""
    if SCENARIO_NUM == 10:
        return WORKFLOW_TARGET_ODM_MODEL
    if SCENARIO_NUM == 11:
        return WORKFLOW_TARGET_PLM_MODEL
    if SCENARIO_NUM == 12:
        return WORKFLOW_TARGET_FILLMASK_MODEL
    return WORKFLOW_TARGET_LLM_MODEL


# PVC 복제(Longhorn 등)·배포 지연을 감안해 기본 20분 (환경변수 E2E_DEPLOY_TIMEOUT_SEC로 조정)
DEPLOY_TIMEOUT_SEC: int = int(os.environ.get("E2E_DEPLOY_TIMEOUT_SEC", "1200"))
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
STATE_FILE: Path = _E2E_ROOT / f".state.{_STATE_ENV_SLUG}.json" if _STATE_ENV_SLUG else _E2E_ROOT / ".state.json"


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


def find_optimization_source_model(models: list[dict]) -> dict | None:
    """E2E_OPTIMIZATION_SOURCE_MODEL_NAME 과 name 이 일치하는 모델 1건."""
    if not OPTIMIZATION_SOURCE_MODEL_NAME:
        return None
    for m in models:
        if m.get("name") == OPTIMIZATION_SOURCE_MODEL_NAME:
            return m
    return None
