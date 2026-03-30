"""DB 참조 데이터 시드. 실행 전 `cd backend/app` + `PYTHONPATH=.` 규약을 따른다."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Literal


# settings / rdb_data import 전에 프로파일 적용 (§9.1)
def _apply_env_from_argv() -> None:
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--env" and i + 1 < len(sys.argv):
            os.environ["ENV"] = sys.argv[i + 1].strip()
            break
        i += 1


_apply_env_from_argv()

from config.settings import get_settings
from db.models.experiment import HyperparameterType
from db.models.knowledge_base import ChunkType, Language, SearchMethod
from db.models.model import ModelFormat, ModelProvider, ModelType
from db.models.user import UserModel
from sqlalchemy import create_engine, delete, insert, select, update
from sqlalchemy.orm import sessionmaker

from .rdb_data import (
    CHUNK_TYPE_DATA,
    HYPERPARAMETER_TYPE_DATA,
    LANGUAGE_DATA,
    MODEL_FORMAT_DATA,
    MODEL_PROVIDER_DATA,
    MODEL_TYPE_DATA,
    SEARCH_METHOD_DATA,
    USER_DATA,
)

ResetKind = Literal["user_usernames", "full"]

settings = get_settings()


class _SeedSpec:
    __slots__ = ("model", "rows", "natural_key", "reset_kind")

    def __init__(
        self,
        model: Any,
        rows: list[dict[str, Any]],
        natural_key: str,
        reset_kind: ResetKind = "full",
    ) -> None:
        self.model = model
        self.rows = rows
        self.natural_key = natural_key
        self.reset_kind = reset_kind


# §5.3 실행 순서
SEED_SPECS: tuple[_SeedSpec, ...] = (
    _SeedSpec(UserModel, USER_DATA, "username", "user_usernames"),
    _SeedSpec(ModelFormat, MODEL_FORMAT_DATA, "name"),
    _SeedSpec(ModelProvider, MODEL_PROVIDER_DATA, "name"),
    _SeedSpec(ModelType, MODEL_TYPE_DATA, "name"),
    _SeedSpec(HyperparameterType, HYPERPARAMETER_TYPE_DATA, "param_name"),
    _SeedSpec(ChunkType, CHUNK_TYPE_DATA, "name"),
    _SeedSpec(Language, LANGUAGE_DATA, "name"),
    _SeedSpec(SearchMethod, SEARCH_METHOD_DATA, "name"),
)


def _ensure_spec(session, spec: _SeedSpec) -> int:
    added = 0
    nk = spec.natural_key
    for row in spec.rows:
        key_val = row[nk]
        exists = session.scalar(select(spec.model.id).where(getattr(spec.model, nk) == key_val))
        if exists is None:
            session.execute(insert(spec.model.__table__).values(**row))
            added += 1
    print(f"{spec.model.__tablename__}: ensure — {added}행 삽입 (자연키 {nk})")
    return added


def _upsert_spec(session, spec: _SeedSpec) -> None:
    nk = spec.natural_key
    for row in spec.rows:
        key_val = row[nk]
        pk = session.scalar(select(spec.model.id).where(getattr(spec.model, nk) == key_val))
        payload = {k: v for k, v in row.items() if k != nk}
        if pk is None:
            session.execute(insert(spec.model.__table__).values(**row))
        elif payload:
            session.execute(update(spec.model).where(spec.model.id == pk).values(**payload))
    print(f"{spec.model.__tablename__}: upsert 완료")


def _reset_spec(session, spec: _SeedSpec) -> None:
    if spec.reset_kind == "user_usernames":
        names = [r[spec.natural_key] for r in spec.rows]
        session.execute(delete(spec.model).where(getattr(spec.model, spec.natural_key).in_(names)))
    else:
        session.execute(delete(spec.model))
    if spec.rows:
        session.execute(insert(spec.model.__table__), spec.rows)
    print(f"{spec.model.__tablename__}: reset — 시드 {len(spec.rows)}행 반영")


def apply_mode(session, mode: str) -> None:
    if mode == "ensure":
        for spec in SEED_SPECS:
            _ensure_spec(session, spec)
    elif mode == "upsert":
        for spec in SEED_SPECS:
            _upsert_spec(session, spec)
    elif mode == "reset":
        for spec in SEED_SPECS:
            _reset_spec(session, spec)
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")


def create_db_session():
    engine = create_engine(settings.get_db_uri)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터베이스 참조 데이터 시드")
    parser.add_argument(
        "--mode",
        choices=("ensure", "upsert", "reset"),
        default="ensure",
        help="ensure: 없으면 삽입 | upsert: 없으면 삽입, 있으면 시드 컬럼만 갱신 | reset: 시드 정책에 따라 삭제 후 삽입",
    )
    parser.add_argument(
        "--env",
        metavar="PROFILE",
        help="모듈 import 시 sys.argv에서 읽어 ENV로 설정. 예: staging → config/.env.staging",
    )
    args = parser.parse_args()

    db = create_db_session()
    try:
        print(f"시드 시작 (mode={args.mode})…")
        apply_mode(db, args.mode)
        db.commit()
        print("시드 완료.")
    except Exception as e:
        traceback.print_exc()
        print(f"시드 실패: {e}")
        db.rollback()
        raise SystemExit(1) from e
    finally:
        db.close()


if __name__ == "__main__":
    main()
