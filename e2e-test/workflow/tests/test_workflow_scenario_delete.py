"""
E2E 시나리오: 워크플로우 시나리오 삭제 테스트

상태 파일(.state.json 또는 ENV별 .state.{ENV}.json)의 scenario_N_deployments (또는 레거시 단일 키)에서 배포 기록을 읽는다.
deploy 때와 동일한 ENV 로 실행해야 해당 환경의 기록을 읽는다.
같은 시나리오로 여러 번 배포한 경우 건별로 삭제 여부를 묻고, y 일 때만 API 삭제를 수행한다.

시나리오 선택: E2E_SCENARIO 환경변수 (필수, Makefile에서 SCENARIO 인자로 주입)
표준 입력: 각 배포마다 삭제 확인 (y 만 삭제, 그 외는 건너뜀)

흐름 (사용자가 y 로 확인한 배포마다):
  1. DELETE /workflows/{id}              — 워크플로우 삭제 시작
  2. POST   /workflows/{id}/finalize-deletion — 삭제 완료 폴링
  3. GET    /workflows/{id}              — 워크플로우 404 확인
  4. DELETE /knowledge-bases/{id}       — KB 삭제 (있는 경우)
  5. GET    /knowledge-bases/{id}       — KB 404 확인
  6. DELETE /prompts/{id}               — 프롬프트 삭제
  7. 상태 파일에서 해당 workflow_id 기록 제거
"""

from __future__ import annotations

import time

import pytest
import requests
from cleanup_prompt import is_yes
from config import DELETE_TIMEOUT_SEC, POLL_INTERVAL_SEC, SCENARIO_NUM, STATE_FILE
from workflow.definitions import get_scenario, load_deployment_entries, remove_deployment_entry_by_workflow_id

SCENARIO = get_scenario(SCENARIO_NUM)


def _prompt_delete_one(index: int, total: int, dep: dict) -> bool:
    wf_id = dep.get("workflow_id", "")
    name = dep.get("workflow_name") or "(이름 없음)"
    kb_ids = dep.get("kb_ids") or []
    prompts = dep.get("prompt_ids") or {}
    print(f"\n--- [{index}/{total}] ---")
    print(f"  workflow_id: {wf_id}")
    print(f"  workflow_name: {name}")
    print(f"  kb_ids: {kb_ids}")
    print(f"  prompt_ids: {len(prompts)}개")
    try:
        answer = input("이 배포를 삭제할까요? 삭제하려면 y 입력 (그 외: 건너뜀): ")
    except EOFError:
        print("\n(EOF — 건너뜀)")
        return False
    return is_yes(answer)


def _delete_workflow_and_wait(api_url: str, auth_headers: dict, wf_id: str) -> None:
    resp = requests.delete(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
    assert resp.status_code == 202, f"삭제 시작 실패: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  ✔ 워크플로우 삭제 시작: cleanup_run_id={data.get('cleanup_run_id')}")

    deadline = time.time() + DELETE_TIMEOUT_SEC
    while time.time() < deadline:
        resp = requests.post(
            f"{api_url}/workflows/{wf_id}/finalize-deletion",
            headers=auth_headers,
        )
        assert resp.status_code in (200, 404)

        if resp.status_code == 404:
            print(f"  ✔ 워크플로우가 이미 삭제됨: {wf_id}")
            return

        fin = resp.json()
        status_val = fin.get("status")
        if status_val == "completed":
            print(f"  ✔ 워크플로우 삭제 완료(K8s 정리): {wf_id}")
            return
        if status_val == "failed":
            pytest.fail(f"워크플로우 삭제 실패: {fin.get('message', 'N/A')}")

        remaining = int(deadline - time.time())
        print(f"  ⏳ 리소스 정리 대기… status={status_val} (남은 시간: {remaining}s)")
        time.sleep(POLL_INTERVAL_SEC)

    pytest.fail(f"삭제 타임아웃 ({DELETE_TIMEOUT_SEC}s 초과)")


def _verify_workflow_gone(api_url: str, auth_headers: dict, wf_id: str) -> None:
    resp = requests.get(f"{api_url}/workflows/{wf_id}", headers=auth_headers)
    assert resp.status_code == 404, f"워크플로우가 아직 존재합니다: {resp.status_code}"
    print(f"  ✔ 워크플로우 404 확인: {wf_id}")


def _delete_knowledge_bases(api_url: str, auth_headers: dict, kb_ids: list) -> None:
    for idx, kb_id in enumerate(kb_ids):
        resp = requests.delete(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.status_code == 200, f"KB 삭제 실패 (KB{idx + 1}, id={kb_id}): {resp.status_code} {resp.text}"
        body = resp.json()
        assert body.get("success") is True
        print(f"  ✔ KB{idx + 1} 삭제 완료: kb_id={kb_id}")


def _verify_kbs_gone(api_url: str, auth_headers: dict, kb_ids: list) -> None:
    for idx, kb_id in enumerate(kb_ids):
        resp = requests.get(f"{api_url}/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.status_code == 404, f"KB{idx + 1}이 아직 존재합니다 (id={kb_id}): {resp.status_code}"
        print(f"  ✔ KB{idx + 1} 404 확인: kb_id={kb_id}")


def _delete_prompts(api_url: str, auth_headers: dict, prompt_ids: dict) -> None:
    for comp_name, pid in prompt_ids.items():
        resp = requests.delete(f"{api_url}/prompts/{pid}", headers=auth_headers)
        assert resp.status_code == 204, f"프롬프트 삭제 실패 ({comp_name}, id={pid}): {resp.status_code}"
        print(f"  ✔ 프롬프트 삭제: {comp_name} (id={pid})")


def _delete_one_deployment(api_url: str, auth_headers: dict, dep: dict) -> None:
    wf_id = dep["workflow_id"]
    kb_ids = list(dep.get("kb_ids") or [])
    prompt_ids = dict(dep.get("prompt_ids") or {})

    _delete_workflow_and_wait(api_url, auth_headers, wf_id)
    _verify_workflow_gone(api_url, auth_headers, wf_id)

    if kb_ids:
        _delete_knowledge_bases(api_url, auth_headers, kb_ids)
        _verify_kbs_gone(api_url, auth_headers, kb_ids)

    if prompt_ids:
        _delete_prompts(api_url, auth_headers, prompt_ids)

    remove_deployment_entry_by_workflow_id(SCENARIO_NUM, wf_id)
    print(f"  ✔ {STATE_FILE.name} 에서 기록 제거: workflow_id={wf_id}")


@pytest.mark.workflow_scenario_delete
def test_interactive_delete_scenario_deployments(api_url: str, auth_headers: dict):
    """저장된 배포를 순서대로 안내하고, y 로 확인한 항목만 삭제한다."""
    print(f"\n{'=' * 60}")
    print(f"  시나리오 #{SCENARIO_NUM}: {SCENARIO['name']}")
    print(f"  구성: {SCENARIO['graph']}")
    print(f"  상태 파일: {STATE_FILE.name}")
    print(f"{'=' * 60}")

    entries = load_deployment_entries(SCENARIO_NUM)
    if not entries:
        pytest.fail(
            f"시나리오 #{SCENARIO_NUM} 에 저장된 배포 기록이 없습니다. ({STATE_FILE.name})\n"
            f"먼저 deploy 를 실행하세요. 예: make e2e-wf-scenario-deploy SCENARIO={SCENARIO_NUM}"
            + (" ENV=… (deploy 때와 동일)" if STATE_FILE.name != ".state.json" else "")
        )

    total = len(entries)
    print(f"\n저장된 배포 {total}건 (순서대로 확인합니다).")
    deleted_count = 0
    skipped_count = 0

    for i, dep in enumerate(entries, start=1):
        if not dep.get("workflow_id"):
            print(f"\n--- [{i}/{total}] --- (workflow_id 없음 — 건너뜀)")
            skipped_count += 1
            continue

        if not _prompt_delete_one(i, total, dep):
            print("  → 건너뜀")
            skipped_count += 1
            continue

        print(f"  … 삭제 진행 중 (workflow_id={dep['workflow_id']})")
        _delete_one_deployment(api_url, auth_headers, dep)
        deleted_count += 1

    print(f"\n{'=' * 60}")
    print(f"  완료: 삭제 {deleted_count}건, 건너뜀 {skipped_count}건 (조회 시점 기준 총 {total}건)")
    remaining = len(load_deployment_entries(SCENARIO_NUM))
    print(f"  시나리오 #{SCENARIO_NUM} 남은 저장 건수: {remaining}")
    print(f"{'=' * 60}")
