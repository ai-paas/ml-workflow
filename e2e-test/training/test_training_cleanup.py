"""
E2E: 통합 학습→등록 e2e 산출물 일괄 정리.

make e2e-training-clean ENV=dev — .state.{ENV}.json 의 training_runs 에 누적된 모든 학습 런
(experiment / child model / auto dataset)을 삭제한다.

삭제 순서는 FK 참조 역순: experiment → child model → dataset.
(experiment 가 registered_model_id·dataset_id 를 참조하므로 먼저 지워야 child model/dataset 삭제 가능)
이미 없는 리소스(404)는 성공으로 본다. 일부 실패한 런만 training_runs 에 남겨 재시도할 수 있게 한다.
"""

from __future__ import annotations

import pytest
import requests
from training.state import load_training_runs, save_training_runs

_OK = {200, 201, 204, 404}  # 404 = 이미 삭제됨


def _delete(api_url: str, headers: dict, path: str) -> tuple[bool, str]:
    r = requests.delete(f"{api_url}{path}", headers=headers, timeout=60)
    ok = r.status_code in _OK
    return ok, f"{path} -> {r.status_code} {'' if ok else r.text[:160]}"


@pytest.mark.training_clean
class TestTrainingCleanup:
    """training_runs 에 기록된 학습 산출물을 일괄 삭제한다."""

    def test_cleanup_all_training_runs(self, api_url: str, auth_headers: dict):
        runs = load_training_runs()
        if not runs:
            pytest.skip("정리할 학습 런 기록(training_runs)이 없습니다.")

        print(f"\n▶ 학습 런 {len(runs)}건 정리 시작")
        remaining: list[dict] = []
        for run in runs:
            exp_id = run.get("experiment_id")
            child_id = run.get("child_model_id")
            ds_id = run.get("dataset_id")
            logs: list[str] = []
            failed = False

            # 순서: experiment → child model → dataset
            for path in (
                f"/experiments/{exp_id}" if exp_id else None,
                f"/models/{child_id}" if child_id else None,
                f"/datasets/{ds_id}" if ds_id else None,
            ):
                if not path:
                    continue
                ok, msg = _delete(api_url, auth_headers, path)
                logs.append(msg)
                failed = failed or not ok

            status = "❌ 일부 실패" if failed else "✔ 삭제"
            print(f"  {status} exp={exp_id} child={child_id} dataset={ds_id}")
            for m in logs:
                print(f"      {m}")
            if failed:
                remaining.append(run)

        # 실패한 런만 남겨 재시도 가능하게 한다.
        save_training_runs(remaining)
        assert not remaining, f"{len(remaining)}건 정리 실패(남김): {remaining}"
        print(f"\n✔ 학습 런 전체 정리 완료 ({len(runs)}건)")
