# CLAUDE.md

ml-workflow 백엔드 작업 시 지켜야 할 규칙.

## Alembic 마이그레이션

- **마이그레이션 파일은 직접 손으로 작성하지 않는다.** 모델(`backend/app/db/models/*.py`)을 먼저 변경한 뒤 **`make alembic-autogen-file` 로 자동 생성**한다.
  - 명령: `make alembic-autogen-file msg="<변경 요약>" [ENV=local|dev]`
  - `msg` 는 **변경되는 테이블·컬럼 내역을 참고해 간결한 영문**으로 지정한다.
    예) `msg="normalize service_monitoring add user_id success drop metric columns"`
  - revision id 는 `backend/app/scripts/next_alembic_rev.py` 가 4자리(`NNNN`)로 자동 부여한다.
  - autogenerate 는 **DB 연결·reflection 이 필요**하므로, 대상 DB 가 head 리비전 상태여야 한다(아니면 "Target database is not up to date").
- **생성된 파일은 초안이므로 반드시 검토·보정한다.** autogenerate 가 놓치는 것들:
  - 데이터 백필(예: `NOT NULL` 컬럼을 기존 데이터가 있는 테이블에 추가할 때는 `nullable=True` 로 추가 → `UPDATE` 백필 → `NOT NULL` 로 alter 순서로 분리).
  - drop/backfill **순서**(백필에 쓰는 컬럼을 먼저 drop 하면 안 됨).
  - FK 대상 테이블명 확인. 테이블명은 `BaseModel.__tablename__`(PascalCase→snake, `Model` 접미사 제거)으로 자동 생성되거나 클래스에서 명시 override 한다. 예: `UserModel`→`user`(단수), `Service`→명시 `services`.
  - 불필요한 잡(noise) diff(모델·DB 불일치로 생긴 무관한 변경) 제거.
- 적용: `make alembic-upgrade-head [ENV=...]` / 다운그레이드: `make alembic-downgrade v=NNNN [ENV=...]`

## 실행·도구 규약 (리포지토리 루트에서)

- 백엔드 모듈/스크립트/테스트 실행: `cd backend/app && PYTHONPATH=. uv run --project .. <cmd>` (uv + `backend/pyproject.toml` 환경).
- 환경 프로파일: `backend/app/config/.env.{local,dev,innogrid}` — `ENV` 로 선택.
- 린트: `make lint` (isort + black + flake8, line-length 120) / 자동수정: `make lint-fix`.
- DB 시드: `make db-seed [MODE=ensure|upsert|sync] [DRY_RUN=1] [ENV=local]` (자세한 옵션은 `make help`).



## 기타 규약
- 모든 결론 문장 끝에 [측정: 스크립트경로] 또는 [기억: 재검증필요] 또는 [추정: 미검증] 중 하나를 붙여라. 태그 없는 단정 금지.
