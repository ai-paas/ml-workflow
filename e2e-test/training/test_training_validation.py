"""E2E: 학습 요청 검증 negative 케이스 (POST /pipeline/training → 400).

학습 검증 리팩터로 정합된 거부 케이스. 모두 dataset/experiment 생성 전에 400 으로 막히므로 부작용 없음.
  · dataset_file 동반 + dataset_kind 생략 → 400 (api-spec §2.1: dataset_file 시 dataset_kind 필수)
  · dataset_kind ↔ 모델 요구 분류 불일치 → 400

모델: ESM2 reference(facebook/esm2_t6_8M_UR50D, 학습가능 pLM, 요구 분류=protein-classification).
실행: ENV=dev uv run --group e2e pytest e2e-test/training/test_training_validation.py -v -s -m training_validation
"""

from __future__ import annotations

import os
import uuid

import pytest
import requests
from config import _E2E_ROOT

DATASET_DIR = _E2E_ROOT / "dataset-file"
PROTEIN_ZIP = "protein_sample.zip"
PLM_MODEL_NAME = (os.environ.get("E2E_TRAINING_REFERENCE_MODEL_NAME") or "facebook/esm2_t6_8M_UR50D").strip()


def _find_model(api_url: str, headers: dict, name: str) -> dict | None:
    resp = requests.get(f"{api_url}/models", headers=headers)
    assert resp.status_code == 200, f"모델 목록 조회 실패: {resp.status_code} {resp.text}"
    for m in resp.json():
        if m.get("name") == name or (m.get("repo_id") or "") == name:
            return m
    return None


@pytest.mark.training_validation
class TestTrainingValidation:
    plm: dict | None = None

    def test_00_resolve_model(self, api_url: str, auth_headers: dict):
        TestTrainingValidation.plm = _find_model(api_url, auth_headers, PLM_MODEL_NAME)
        assert self.plm, f"pLM 모델 미발견: {PLM_MODEL_NAME}"
        assert self.plm.get("learning_enable_yn") is True, f"학습 불가 모델: {self.plm}"
        print(f"\n✔ pLM 학습 모델: id={self.plm['id']} ({PLM_MODEL_NAME})")

    def _post_training(self, api_url: str, auth_headers: dict, extra_data: dict):
        ds = DATASET_DIR / PROTEIN_ZIP
        assert ds.is_file(), f"데이터셋 파일 없음: {ds}"
        data = {
            "model_id": str(self.plm["id"]),
            "train_name": f"guard-{uuid.uuid4().hex[:8]}",
            "epochs": "2",
            **extra_data,
        }
        with open(ds, "rb") as fh:
            files = {"dataset_file": (ds.name, fh, "application/zip")}
            return requests.post(f"{api_url}/pipeline/training", data=data, files=files, headers=auth_headers)

    def test_01_missing_dataset_kind(self, api_url: str, auth_headers: dict):
        """dataset_file 동반 + dataset_kind 생략 → 400"""
        resp = self._post_training(api_url, auth_headers, {})
        assert resp.status_code == 400, f"dataset_kind 생략이 거부되지 않음: {resp.status_code} {resp.text}"
        assert "dataset_kind" in resp.text, f"메시지 불일치: {resp.text}"
        print(f"\n✔ dataset_kind 생략 거부(400): {resp.text}")

    def test_02_dataset_kind_mismatch(self, api_url: str, auth_headers: dict):
        """dataset_kind 가 모델 요구 분류와 불일치(pLM 모델에 object-detection) → 400"""
        resp = self._post_training(api_url, auth_headers, {"dataset_kind": "object-detection"})
        assert resp.status_code == 400, f"kind 불일치가 거부되지 않음: {resp.status_code} {resp.text}"
        assert "일치하지 않" in resp.text, f"메시지 불일치: {resp.text}"
        print(f"\n✔ kind 불일치 거부(400): {resp.text}")
