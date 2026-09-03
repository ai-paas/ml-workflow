"""학습→등록 e2e 산출물(experiment / child model / auto dataset)을 상태 파일에 누적 기록.

시나리오 배포 기록(scenario_N_deployments)과 같은 패턴으로, 여러 학습 런을 리스트로 모아
cleanup 테스트(test_training_cleanup)가 일괄 삭제할 수 있게 한다.

엔트리 형식:
  {
    "experiment_id": int,
    "child_model_id": int | None,   # 등록 성공 시 채워짐
    "child_model_name": str,
    "dataset_id": int | None,       # 파일 업로드로 자동 생성된 경우만(재사용 dataset_id 는 정리 대상 아님)
  }
"""

import json

from config import clear_state, load_state, save_state

_RUNS_KEY = "training_runs"


def load_training_runs() -> list[dict]:
    """누적된 학습 런 기록을 읽는다."""
    raw = load_state(_RUNS_KEY)
    return json.loads(raw) if raw else []


def save_training_runs(runs: list[dict]) -> None:
    """학습 런 기록 전체를 저장(빈 리스트면 키 제거)."""
    if runs:
        save_state(_RUNS_KEY, json.dumps(runs))
    else:
        clear_state(_RUNS_KEY)


def append_training_run(entry: dict) -> None:
    """학습 런 1건을 목록 끝에 추가(이전 런을 덮어쓰지 않음)."""
    runs = load_training_runs()
    runs.append(entry)
    save_training_runs(runs)


def update_training_run(experiment_id, **fields) -> None:
    """experiment_id 가 일치하는 런 엔트리의 필드를 갱신(등록 성공 후 child_model_id 채우기 등)."""
    runs = load_training_runs()
    for r in runs:
        if str(r.get("experiment_id")) == str(experiment_id):
            r.update(fields)
    save_training_runs(runs)
