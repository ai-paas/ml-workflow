#!/usr/bin/env python3
"""Alembic versions 디렉터리에서 revision: str = \"NNNN\" 최댓값 기준 다음 4자리 ID를 stdout에 출력한다."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_VERSIONS = Path(__file__).resolve().parent.parent / "alembic" / "versions"
_REVISION_RE = re.compile(r'^revision:\s*str\s*=\s*["\'](\d{4})["\']', re.MULTILINE)


def main() -> None:
    if not _VERSIONS.is_dir():
        print("alembic/versions 디렉터리를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    max_n = 0
    for path in sorted(_VERSIONS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        m = _REVISION_RE.search(text)
        if m:
            max_n = max(max_n, int(m.group(1)))
    sys.stdout.write(f"{max_n + 1:04d}")


if __name__ == "__main__":
    main()
