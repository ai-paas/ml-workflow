"""워크플로 배포 완료 폴링 시 조기 실패 판단 (시나리오 테스트 공통)."""

from __future__ import annotations

from typing import Any


def expected_model_component_count(workflow_read: dict[str, Any]) -> int:
    """GET /workflows/{id} 응답에서 배포 대상 MODEL 컴포넌트 수 (model_id 있는 것만)."""
    return sum(
        1 for c in (workflow_read.get("components") or []) if c.get("type") == "MODEL" and c.get("model_id") is not None
    )


def workflow_deploy_poll_should_fail(
    status_payload: dict[str, Any],
    *,
    expected_model_deployments: int,
) -> str | None:
    """
    폴링을 더 이상 이어가면 안 되는 경우 비실패(None) 대신 실패 사유 문자열을 반환한다.

    - execute 성공 후에는 kubeflow_run_id와 함께 model_workflow_deployments 행이 있어야 한다.
    - 워크플로 ERROR이면 배포 완료를 기다릴 수 없다.
    - ACTIVE인데 배포 행이 전혀 없으면 불일치다.
    """
    err = status_payload.get("error")
    if err:
        return f"상태 조회 오류: {err}"

    wf_status = (status_payload.get("status") or "").strip()
    deployed = status_payload.get("deployed_models") or []
    n = len(deployed)
    kf_run = status_payload.get("kubeflow_run_id")

    if wf_status == "ERROR":
        return (
            f"워크플로 상태가 ERROR입니다. deployed_models={n}건, kubeflow_run_id={kf_run!r}. "
            "배포 완료를 기다리지 않습니다."
        )

    if expected_model_deployments <= 0:
        return None

    if n == 0 and kf_run:
        return (
            "Kubeflow 실행 ID는 있으나 model_workflow_deployments 행이 조회되지 않습니다. "
            "배포 레코드 누락·DB 불일치 등으로 더 이상 상태만 폴링하지 않습니다."
        )

    if n == 0 and wf_status == "ACTIVE":
        return f"워크플로가 ACTIVE인데 배포 행이 없습니다. " f"기대 MODEL 배포 수={expected_model_deployments}."

    return None
