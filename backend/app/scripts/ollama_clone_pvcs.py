"""Ollama 워크플로 복제 PVC 현황 점검·정리.

복제본은 배포 행에 기록되기 전에 만들어지므로, 프로세스가 중간에 끊기면 어느 정리 경로에도
걸리지 않는 복제본이 남는다. 이 스크립트는 그런 복제본을 찾아 보여주고, 요청하면 지운다.

사용법 (리포지토리 루트에서):
    cd backend/app && PYTHONPATH=. uv run --project .. python scripts/ollama_clone_pvcs.py [ENV=dev]
    ... --delete        회수 가능한 것만 실제로 삭제
    ... --all           완료된 복제본까지 모두 표로 출력(기본은 전부 출력이므로 표시용)

상태 의미
    complete    복사 완료. 다음 배포가 재사용한다(지우면 모델을 다시 내려받는다).
    copying     복사 Job 이 진행 중. 건드리지 않는다.
    incomplete  복사가 끝났다고 볼 근거가 없다. 회수 후보.
    in-use      복사는 미완료지만 마운트한 Pod 이 있다. 건드리지 않는다.
    fresh       미완료지만 생성된 지 얼마 안 됐다. grace period 안이라 보류.
"""

from __future__ import annotations

import argparse
import sys

from config.settings import get_settings
from core.serving.serving_model_workflow_pvc import (
    CLONE_ANNOTATION_COPY_COMPLETED,
    CLONE_LABEL_COPY_MODE,
    COPY_MODE_JOB,
    _delete_pvc_if_exists,
    _find_active_copy_job_for_pvc,
    _load_core_v1,
    reclaim_reason_for_incomplete_clone,
)

CLONE_SELECTOR = "ml-workflow/ollama-workflow-clone=true"


def classify(pvc, namespace: str) -> tuple[str, str, str]:
    """(상태, 복사 Job, 비고) 판정."""
    meta = pvc.metadata
    labels = meta.labels or {}
    annotations = meta.annotations or {}

    if getattr(meta, "deletion_timestamp", None):
        return "deleting", "-", "삭제 진행 중"

    if labels.get(CLONE_LABEL_COPY_MODE) != COPY_MODE_JOB:
        return "complete", "-", "CSI 클론(복사 불필요)"

    if annotations.get(CLONE_ANNOTATION_COPY_COMPLETED) == "true":
        return "complete", "-", "완료 표시 있음"

    job_name, outcome = _find_active_copy_job_for_pvc(namespace, meta.name)
    if outcome == "running":
        return "copying", job_name or "-", "복사 진행 중"

    reason = reclaim_reason_for_incomplete_clone(pvc, namespace, outcome)
    if reason:
        return "incomplete", job_name or "-", reason
    return "hold", job_name or "-", f"회수 보류(job={outcome})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama 복제 PVC 점검·정리")
    parser.add_argument("--delete", action="store_true", help="회수 가능한(incomplete) 복제본을 삭제한다")
    args = parser.parse_args()

    ns = get_settings().KUBEFLOW_NAMESPACE
    core = _load_core_v1()
    lst = core.list_namespaced_persistent_volume_claim(namespace=ns, label_selector=CLONE_SELECTOR)
    items = [p for p in (lst.items or []) if p.metadata and p.metadata.name]
    if not items:
        print(f"복제 PVC 없음 (namespace={ns})")
        return 0

    items.sort(key=lambda p: (p.metadata.creation_timestamp is None, p.metadata.creation_timestamp))

    rows = []
    for pvc in items:
        state, job, note = classify(pvc, ns)
        labels = pvc.metadata.labels or {}
        rows.append(
            {
                "name": pvc.metadata.name,
                "phase": (pvc.status.phase or "?") if pvc.status else "?",
                "model": labels.get("ml-workflow/source-model-id", "-"),
                "node": labels.get("ml-workflow/serving-node", "-"),
                "state": state,
                "note": note,
            }
        )

    width = max(len(r["name"]) for r in rows)
    print(f"namespace={ns}  복제 PVC {len(rows)}건\n")
    header = "NAME".ljust(width) + "  " + "PHASE".ljust(8) + " " + "MODEL".ljust(6) + " " + "STATE".ljust(11) + " NOTE"
    print(header)
    for r in rows:
        line = (
            r["name"].ljust(width)
            + "  "
            + r["phase"].ljust(8)
            + " "
            + r["model"].ljust(6)
            + " "
            + r["state"].ljust(11)
            + " "
            + r["note"]
        )
        print(line)

    targets = [r for r in rows if r["state"] == "incomplete"]
    print(f"\n회수 가능: {len(targets)}건")
    if not targets:
        return 0

    if not args.delete:
        print("삭제하려면 --delete 를 붙여 다시 실행하세요.")
        return 0

    for r in targets:
        print(f"삭제: {r['name']}")
        _delete_pvc_if_exists(core, ns, r["name"])
    print(f"완료: {len(targets)}건 삭제 요청")
    return 0


if __name__ == "__main__":
    sys.exit(main())
