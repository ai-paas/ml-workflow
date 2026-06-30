"""
E2E: 통합 학습 → 등록 파이프라인 (YOLOX / ESM2 공통).

학습→등록은 모델군에 관계없이 같은 API(`POST /pipeline/training`, `POST /pipeline/model/registration`)를
쓰므로 ESM2 전용이 아니라 **모델 무관 제네릭 테스트**다. 모델군은 .env.{ENV}(또는 환경변수)로 선택한다.

설정:
  E2E_TRAINING_REFERENCE_MODEL_NAME   학습할 reference 모델 name (/models)         (필수)
  E2E_TRAINING_DATASET_ID             재사용할 데이터셋 id (있으면 파일 업로드 대신 사용)
  E2E_TRAINING_DATASET_FILE           업로드할 데이터셋 파일 (e2e-test/dataset-file/ 하위)
  E2E_TRAINING_DATASET_KIND           업로드 데이터의 분류 (dataset_file 동반 시 필수, 데이터 파일과 짝)
  E2E_TRAINING_EPOCHS                 빠른 검증용 epoch (기본 2)
  E2E_TRAINING_CHILD_MODEL_NAME       등록할 자식 모델 name 의 베이스 (뒤에 uuid 접미로 유니크화; 기본 e2e-ft)

dataset_kind 는 '업로드 데이터의 분류'라 dataset_file 동반 시 필수다(api-spec §2.1) — 데이터 파일과 짝지어
env 로 명시한다(모델에서 도출하면 호환성 검사가 동어반복이 됨). 자식은 reference 의
type/format/task/parameter/sample_code 를 상속하며, 검증은 자식이 reference 의 type/format 을 상속했는지 확인한다.

예 (ESM2):
  make e2e-training ENV=dev \\
    E2E_TRAINING_REFERENCE_MODEL_NAME=facebook/esm2_t6_8M_UR50D \\
    E2E_TRAINING_DATASET_FILE=protein_sample.zip

산출: 각 학습 런(experiment/child model/auto dataset)을 .state.{ENV}.json 의 training_runs 리스트에
누적 기록한다 → make e2e-training-clean 이 일괄 삭제. (시나리오 #11 배포는 E2E_WORKFLOW_TARGET_PLM_MODEL
env 로 모델을 이름 조회하며 상태 파일과 무관하다.)
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
import requests
from config import _E2E_ROOT
from training.state import append_training_run, update_training_run

REFERENCE_MODEL_NAME = (os.environ.get("E2E_TRAINING_REFERENCE_MODEL_NAME") or "").strip()
DATASET_ID = (os.environ.get("E2E_TRAINING_DATASET_ID") or "").strip()
DATASET_FILE = (os.environ.get("E2E_TRAINING_DATASET_FILE") or "").strip()
# dataset_file 업로드 시 필수(데이터의 분류; api-spec §2.1). 데이터 파일과 짝지어 env 로 명시(모델 도출 아님).
DATASET_KIND = (os.environ.get("E2E_TRAINING_DATASET_KIND") or "").strip()
EPOCHS = (os.environ.get("E2E_TRAINING_EPOCHS") or "2").strip()
# 재등록은 모델명/실험명이 유니크해야 통과한다(백엔드가 중복 시 409 거부).
# 베이스 이름(env 또는 기본 e2e-ft) 뒤에 uuid 를 잘라 붙여 런마다 유니크하게 만든다.
_BASE_CHILD_MODEL_NAME = (os.environ.get("E2E_TRAINING_CHILD_MODEL_NAME") or "e2e-ft").strip()
CHILD_MODEL_NAME = f"{_BASE_CHILD_MODEL_NAME}-{uuid.uuid4().hex[:8]}"

TRAIN_TIMEOUT_SEC = int(os.environ.get("E2E_TRAINING_TIMEOUT_SEC", "2400"))
REGISTER_TIMEOUT_SEC = int(os.environ.get("E2E_TRAINING_REGISTER_TIMEOUT_SEC", "1200"))
POLL_INTERVAL_SEC = int(os.environ.get("E2E_TRAINING_POLL_INTERVAL_SEC", "10"))

DATASET_DIR = _E2E_ROOT / "dataset-file"


@pytest.mark.training
class TestTrainingRegister:
    """reference 모델 → 학습(experiment) → 자식 모델 등록 종단 검증 (YOLOX/ESM2 공통)."""

    reference_model: dict | None = None
    experiment_id: int | None = None
    child_model_id: int | None = None

    @staticmethod
    def _find_model_by_name(api_url: str, headers: dict, name: str) -> dict:
        resp = requests.get(f"{api_url}/models", headers=headers)
        assert resp.status_code == 200, f"모델 목록 조회 실패: {resp.status_code} {resp.text}"
        for m in resp.json():
            if m.get("name") == name or (m.get("repo_id") or "") == name:
                return m
        listing = "\n".join(f"  id={m.get('id')} name={m.get('name')}" for m in resp.json())
        raise AssertionError(f"name='{name}' 모델을 찾지 못했습니다.\n{listing}")

    def _poll_experiment(
        self, api_url: str, headers: dict, key: str, terminal_ok: set, terminal_fail: set, timeout: int
    ):
        deadline = time.time() + timeout
        last: dict = {}
        while time.time() < deadline:
            r = requests.get(f"{api_url}/experiments/{self.__class__.experiment_id}", headers=headers)
            assert r.status_code == 200, f"실험 조회 실패: {r.status_code} {r.text}"
            last = r.json()
            val = (last.get(key) or "").upper()
            if val in terminal_ok:
                return last
            if val in terminal_fail:
                msg = last.get("train_msg") or last.get("model_register_msg") or last
                pytest.fail(f"실험 {key}={val} (실패): {msg}")
            remaining = int(deadline - time.time())
            print(f"  ⏳ {key}={val or '대기'} … (남은 {remaining}s)")
            time.sleep(POLL_INTERVAL_SEC)
        pytest.fail(f"{timeout}s 내 {key} 종료 상태 미도달. 마지막: {last!r}")

    def test_01_find_reference_model(self, api_url: str, auth_headers: dict):
        if not REFERENCE_MODEL_NAME:
            pytest.skip("E2E_TRAINING_REFERENCE_MODEL_NAME 를 지정하세요.")
        model = self._find_model_by_name(api_url, auth_headers, REFERENCE_MODEL_NAME)
        assert model.get("learning_enable_yn") is True, f"학습 불가 모델입니다(learning_enable_yn != True): {model}"
        self.__class__.reference_model = model
        print(
            f"\n✔ reference 모델: id={model['id']} name={model['name']} "
            f"type={(model.get('type_info') or {}).get('name')} format={(model.get('format_info') or {}).get('name')}"
        )

    def test_02_start_training(self, api_url: str, auth_headers: dict):
        model = self.__class__.reference_model
        assert model, "reference 모델 미발견"

        data = {
            "model_id": str(model["id"]),
            "train_name": CHILD_MODEL_NAME,
            "description": "e2e training",
            "epochs": EPOCHS,
        }
        files = None
        fh = None
        if DATASET_ID:
            data["dataset_id"] = DATASET_ID
            print(f"\n▶ 학습 시작 (dataset_id={DATASET_ID}, epochs={EPOCHS})")
            resp = requests.post(f"{api_url}/pipeline/training", data=data, headers=auth_headers)
        else:
            assert DATASET_FILE, "E2E_TRAINING_DATASET_ID 또는 E2E_TRAINING_DATASET_FILE 중 하나는 필요합니다."
            assert DATASET_KIND, "dataset_file 업로드 시 E2E_TRAINING_DATASET_KIND(데이터의 분류)가 필요합니다."
            ds_path = DATASET_DIR / DATASET_FILE
            assert ds_path.is_file(), f"데이터셋 파일이 없습니다: {ds_path}"
            # dataset_kind 는 '업로드 데이터의 분류' 라 필수(api-spec §2.1). 데이터 파일과 짝지은 env 값을 명시한다.
            data["dataset_kind"] = DATASET_KIND
            fh = open(ds_path, "rb")
            files = {"dataset_file": (ds_path.name, fh, "application/zip")}
            print(f"\n▶ 학습 시작 (file={DATASET_FILE}, kind={DATASET_KIND}, epochs={EPOCHS})")
            resp = requests.post(f"{api_url}/pipeline/training", data=data, files=files, headers=auth_headers)
        if fh:
            fh.close()

        assert resp.status_code == 200, f"학습 요청 실패: {resp.status_code} {resp.text}"
        exp_id = resp.json().get("experiment_id")
        assert exp_id, f"experiment_id 없음: {resp.text}"
        self.__class__.experiment_id = exp_id

        # 자동 생성된 데이터셋 id 조회(파일 업로드 시에만 정리 대상; 재사용 dataset_id 는 제외)
        auto_dataset_id = None
        if not DATASET_ID:
            exp_detail = requests.get(f"{api_url}/experiments/{exp_id}", headers=auth_headers).json()
            auto_dataset_id = (exp_detail.get("dataset") or {}).get("id") or exp_detail.get("dataset_id")
        # 학습 런을 상태 파일에 누적 기록(여러 런 + cleanup 용). 등록 성공 시 child_model_id 를 채운다.
        append_training_run(
            {
                "experiment_id": exp_id,
                "child_model_id": None,
                "child_model_name": CHILD_MODEL_NAME,
                "dataset_id": auto_dataset_id,
            }
        )
        print(f"  experiment_id={exp_id} (auto_dataset_id={auto_dataset_id})")

    def test_03_wait_training_completed(self, api_url: str, auth_headers: dict):
        assert self.__class__.experiment_id, "experiment_id 미설정"
        final = self._poll_experiment(
            api_url, auth_headers, "status", {"COMPLETED"}, {"FAILED", "ERROR"}, TRAIN_TIMEOUT_SEC
        )
        print(f"\n✔ 학습 COMPLETED (loss={final.get('loss')}, accuracy={final.get('accuracy')})")

    def test_04_register_child_model(self, api_url: str, auth_headers: dict):
        assert self.__class__.experiment_id, "experiment_id 미설정"
        payload = {
            "model_name": CHILD_MODEL_NAME,
            "description": "e2e fine-tuned child",
            "experiment_id": self.__class__.experiment_id,
        }
        resp = requests.post(f"{api_url}/pipeline/model/registration", json=payload, headers=auth_headers)
        assert resp.status_code in (200, 201, 202), f"등록 요청 실패: {resp.status_code} {resp.text}"

        # registration_status=SUCCESS 가 registered_model_id 보다 먼저 떨어질 수 있으므로 id 자체를 폴링한다.
        deadline = time.time() + REGISTER_TIMEOUT_SEC
        child_id = None
        last: dict = {}
        while time.time() < deadline:
            r = requests.get(f"{api_url}/experiments/{self.__class__.experiment_id}", headers=auth_headers)
            assert r.status_code == 200, f"실험 조회 실패: {r.status_code} {r.text}"
            last = r.json()
            if last.get("registered_model_id"):
                child_id = last["registered_model_id"]
                break
            if (last.get("registration_status") or "").upper() in {"FAILED", "ERROR"}:
                pytest.fail(f"모델 등록 실패: {last.get('model_register_msg') or last}")
            print(f"  ⏳ registration_status={last.get('registration_status')} registered_model_id=대기 …")
            time.sleep(POLL_INTERVAL_SEC)
        assert child_id, f"{REGISTER_TIMEOUT_SEC}s 내 registered_model_id 미생성. 마지막: {last!r}"
        self.__class__.child_model_id = child_id
        update_training_run(self.__class__.experiment_id, child_model_id=child_id)
        print(f"\n✔ 자식 모델 등록 SUCCESS: id={child_id} name={CHILD_MODEL_NAME}")

    def test_05_verify_child_inherits_meta(self, api_url: str, auth_headers: dict):
        assert self.__class__.child_model_id, "child_model_id 미설정"
        resp = requests.get(f"{api_url}/models/{self.__class__.child_model_id}", headers=auth_headers)
        assert resp.status_code == 200, f"자식 모델 조회 실패: {resp.status_code} {resp.text}"
        child = resp.json()
        ctype = (child.get("type_info") or {}).get("name")
        cformat = (child.get("format_info") or {}).get("name")
        print(
            f"\n✔ 자식 모델 메타: type={ctype} format={cformat} "
            f"parent={child.get('parent_model_id')} task={child.get('task')}"
        )
        # 자식은 reference 모델의 type/format 을 그대로 상속해야 한다(기대값 = reference).
        ref = self.__class__.reference_model or {}
        expect_type = (ref.get("type_info") or {}).get("name")
        expect_format = (ref.get("format_info") or {}).get("name")
        assert ctype == expect_type, f"type 상속 기대={expect_type} 실제={ctype}"
        assert cformat == expect_format, f"format 상속 기대={expect_format} 실제={cformat}"
