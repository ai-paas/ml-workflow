"""E2E 정리(삭제) 전 확인 프롬프트 공용 헬퍼.

여러 e2e 테스트(서비스 metric, 워크플로우 시나리오 lifecycle 등)에서
"삭제 전에 한 번 물어보고, 거절하면 리소스를 남겨 수동 점검" 동작을 공유한다.
허용 응답을 바꾸려면 여기 한 곳만 고치면 된다.
"""

from __future__ import annotations

import sys

# y / Y / yes / YES(입력은 lower 처리), 한글 IME 'y'→'ㅛ' 오타, 한국어 '예' 허용
_YES_ANSWERS = ("y", "yes", "ㅛ", "예")


def is_yes(answer: str) -> bool:
    """응답 문자열이 'yes' 계열인지 판정 (y/Y/yes/YES, 한글 IME 'ㅛ', 한국어 '예')."""
    return answer.strip().lower() in _YES_ANSWERS


def confirm_cleanup(resource_lines: list[str] | None = None) -> bool:
    """삭제(정리) 진행 여부를 묻는다.

    - 대화형(tty): [y/N] 입력을 받아 y 계열이면 True.
    - 비대화형(CI/파이프): 자동 True (정리 진행, 리소스 잔존 방지).

    resource_lines: 프롬프트 전에 출력할 현재 리소스 정보(수동 점검용).
    반환: True 면 정리 진행, False 면 건너뜀(리소스 유지).
    """
    print("\n" + "=" * 60)
    print("  현재 리소스 (수동 확인용)")
    for line in resource_lines or []:
        print(f"   {line}")
    print("=" * 60)

    if not sys.stdin.isatty():
        print("\n(비대화형 환경 — 자동으로 정리를 진행합니다)")
        return True

    try:
        answer = input("\n리소스를 삭제(정리)하시겠습니까? [y/N]: ")
    except EOFError:
        answer = "y"

    proceed = is_yes(answer)
    if proceed:
        print("\n▶ 삭제를 진행합니다.")
    else:
        print("\n⏸  삭제를 건너뜁니다 — 리소스가 남아 있습니다. 직접 확인 후 정리하세요.")
    return proceed
