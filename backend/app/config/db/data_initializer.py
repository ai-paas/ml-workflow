"""DB 참조 데이터 시드. 실행 전 `cd backend/app` + `PYTHONPATH=.` 규약을 따른다.

사용:
    cd backend/app && PYTHONPATH=. python -m config.db.data_initializer --mode ensure
    cd backend/app && PYTHONPATH=. python -m config.db.data_initializer --mode sync --dry-run
    cd backend/app && ENV=prod PYTHONPATH=. python -m config.db.data_initializer --mode sync --confirm

모드: ensure(추가만) / upsert(추가+수정) / sync(추가+수정+삭제) / reset(전체 재구성)
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from sqlalchemy import and_, create_engine, delete, exists, func, insert, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import ColumnElement


# settings / rdb_data import 전에 프로파일 적용
def _apply_env_from_argv() -> None:
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--env" and i + 1 < len(sys.argv):
            os.environ["ENV"] = sys.argv[i + 1].strip()
            break
        i += 1


_apply_env_from_argv()

from config.settings import get_settings
from db.models.knowledge_base import ChunkType, KnowledgeBase, Language, SearchMethod
from db.models.model import Model, ModelFormat, ModelProvider, ModelType
from db.models.user import UserModel
from utils.crypto import get_sha256_hash

from .rdb_data import (
    CHUNK_TYPE_DATA,
    LANGUAGE_DATA,
    MODEL_FORMAT_DATA,
    MODEL_PROVIDER_DATA,
    MODEL_TYPE_DATA,
    SEARCH_METHOD_DATA,
    USER_DATA,
)

ResetKind = Literal["user_usernames", "full"]


# ---- 적용 시점 transform (settings 의존 컬럼을 채운다) ----
def _user_transform(row: dict[str, Any], settings: Any) -> dict[str, Any]:
    """user 시드의 `_password_env` 키를 settings 값의 sha-256 해시(password)로 변환."""
    out = dict(row)
    env_name = out.pop("_password_env", None)
    if env_name is not None:
        out["password"] = get_sha256_hash(getattr(settings, env_name))
    return out


class _SeedSpec:
    __slots__ = ("model", "rows", "natural_key", "reset_kind", "transform")

    def __init__(
        self,
        model: Any,
        rows: list[dict[str, Any]],
        natural_key: str,
        reset_kind: ResetKind = "full",
        transform: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.rows = rows
        self.natural_key = natural_key
        self.reset_kind = reset_kind
        self.transform = transform


# 시드 적용 실행 순서 — 여기 나열된 순서대로 각 spec에 mode가 적용된다.
SEED_SPECS: tuple[_SeedSpec, ...] = (
    _SeedSpec(UserModel, USER_DATA, "username", "user_usernames", transform=_user_transform),
    _SeedSpec(ModelFormat, MODEL_FORMAT_DATA, "name"),
    _SeedSpec(ModelProvider, MODEL_PROVIDER_DATA, "name"),
    _SeedSpec(ModelType, MODEL_TYPE_DATA, "name"),
    _SeedSpec(ChunkType, CHUNK_TYPE_DATA, "name"),
    _SeedSpec(Language, LANGUAGE_DATA, "name"),
    _SeedSpec(SearchMethod, SEARCH_METHOD_DATA, "name"),
)


# ---- diff 자료구조 (dry-run·리포트 공용) ----
@dataclass
class RowChange:
    """단일 행 변경 기록. column_changes 는 UPDATE 한정 {컬럼: (old, new)}."""

    natural_key: dict[str, Any]
    column_changes: dict[str, tuple[Any, Any]] | None = None


@dataclass
class SeedDiff:
    """단일 테이블 적용 결과(또는 계획)."""

    table: str
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    insert_rows: list[RowChange] = field(default_factory=list)
    update_rows: list[RowChange] = field(default_factory=list)
    delete_rows: list[RowChange] = field(default_factory=list)


def _resolve_row(spec: _SeedSpec, raw: dict[str, Any], settings: Any) -> dict[str, Any]:
    """transform 적용 후, 밑줄(`_`)로 시작하는 런타임 전용 키를 제거한 DB 삽입용 dict."""
    row = dict(raw)
    if spec.transform is not None:
        if settings is None:
            settings = get_settings()
        row = spec.transform(row, settings)
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _values_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) is type(b):
        return a == b
    return str(a) == str(b)


def _column_changes(existing: Any, payload: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    changes: dict[str, tuple[Any, Any]] = {}
    for col, new in payload.items():
        old = getattr(existing, col, None)
        if not _values_equal(old, new):
            changes[col] = (old, new)
    return changes


def _row_referenced_exists(model_cls: type) -> ColumnElement[bool] | None:
    """시드 참조 테이블 한 행이 다른 테이블에서 FK로 쓰이면 True가 되는 EXISTS (상관 서브쿼리)."""
    if model_cls is ModelFormat:
        return exists().where(Model.format_id == ModelFormat.id)
    if model_cls is ModelProvider:
        return exists().where(Model.provider_id == ModelProvider.id)
    if model_cls is ModelType:
        return exists().where(Model.type_id == ModelType.id)
    if model_cls is ChunkType:
        return exists().where(KnowledgeBase.chunk_type_id == ChunkType.id)
    if model_cls is Language:
        return exists().where(KnowledgeBase.language_id == Language.id)
    if model_cls is SearchMethod:
        return exists().where(KnowledgeBase.search_method_id == SearchMethod.id)
    return None


def _ensure_spec(session, spec: _SeedSpec, dry_run: bool, settings: Any) -> SeedDiff:
    diff = SeedDiff(table=spec.model.__tablename__)
    nk = spec.natural_key
    for raw in spec.rows:
        row = _resolve_row(spec, raw, settings)
        key_val = row[nk]
        present = session.scalar(select(spec.model.id).where(getattr(spec.model, nk) == key_val))
        if present is None:
            if not dry_run:
                session.execute(insert(spec.model.__table__).values(**row))
            diff.inserted += 1
            diff.insert_rows.append(RowChange(natural_key={nk: key_val}))
    return diff


def _upsert_spec(session, spec: _SeedSpec, dry_run: bool, settings: Any) -> SeedDiff:
    diff = SeedDiff(table=spec.model.__tablename__)
    nk = spec.natural_key
    for raw in spec.rows:
        row = _resolve_row(spec, raw, settings)
        key_val = row[nk]
        existing = session.execute(select(spec.model).where(getattr(spec.model, nk) == key_val)).scalars().first()
        payload = {k: v for k, v in row.items() if k != nk}
        if existing is None:
            if not dry_run:
                session.execute(insert(spec.model.__table__).values(**row))
            diff.inserted += 1
            diff.insert_rows.append(RowChange(natural_key={nk: key_val}))
        else:
            changes = _column_changes(existing, payload)
            if changes:
                if not dry_run:
                    session.execute(update(spec.model).where(spec.model.id == existing.id).values(**payload))
                diff.updated += 1
                diff.update_rows.append(RowChange(natural_key={nk: key_val}, column_changes=changes))
    return diff


def _sync_orphan_keys(session, spec: _SeedSpec) -> list[Any]:
    """시드 자연키 집합에 없는 행 중, 타 테이블이 참조하지 않는 것의 자연키 목록."""
    nk = spec.natural_key
    seed_keys = {row[nk] for row in spec.rows}
    if not seed_keys:
        return []
    col = getattr(spec.model, nk)
    cond: ColumnElement[bool] = ~col.in_(seed_keys)
    ref = _row_referenced_exists(spec.model)
    if ref is not None:
        cond = and_(cond, ~ref)
    return list(session.execute(select(col).where(cond)).scalars().all())


def _sync_spec(session, spec: _SeedSpec, dry_run: bool, settings: Any) -> SeedDiff:
    """시드와 DB를 맞춤: 시드에 없는 참조 행은(미참조만) 삭제 후 upsert.

    `UserModel`은 `reset_kind=user_usernames`라서 **삭제 없이 upsert만** 한다.
    (시드에 없는 다른 계정을 지우지 않음.)
    """
    if spec.reset_kind == "user_usernames":
        return _upsert_spec(session, spec, dry_run, settings)

    diff = SeedDiff(table=spec.model.__tablename__)
    nk = spec.natural_key

    orphan_keys = _sync_orphan_keys(session, spec)
    for key_val in orphan_keys:
        diff.deleted += 1
        diff.delete_rows.append(RowChange(natural_key={nk: key_val}))
    if orphan_keys and not dry_run:
        col = getattr(spec.model, nk)
        session.execute(delete(spec.model).where(col.in_(orphan_keys)))

    up = _upsert_spec(session, spec, dry_run, settings)
    diff.inserted, diff.updated = up.inserted, up.updated
    diff.insert_rows, diff.update_rows = up.insert_rows, up.update_rows
    return diff


def _reset_spec(session, spec: _SeedSpec, dry_run: bool, settings: Any) -> SeedDiff:
    diff = SeedDiff(table=spec.model.__tablename__)
    nk = spec.natural_key
    col = getattr(spec.model, nk)

    if spec.reset_kind == "user_usernames":
        names = [r[nk] for r in spec.rows]
        diff.deleted = int(session.scalar(select(func.count()).select_from(spec.model).where(col.in_(names))) or 0)
        if names and not dry_run:
            session.execute(delete(spec.model).where(col.in_(names)))
    else:
        diff.deleted = int(session.scalar(select(func.count()).select_from(spec.model)) or 0)
        if not dry_run:
            session.execute(delete(spec.model))

    resolved = [_resolve_row(spec, r, settings) for r in spec.rows]
    if resolved and not dry_run:
        session.execute(insert(spec.model.__table__), resolved)
    for row in resolved:
        diff.inserted += 1
        diff.insert_rows.append(RowChange(natural_key={nk: row[nk]}))
    return diff


_MODE_DISPATCH = {
    "ensure": _ensure_spec,
    "upsert": _upsert_spec,
    "sync": _sync_spec,
    "reset": _reset_spec,
}


def apply_mode(
    session,
    mode: str,
    *,
    dry_run: bool = False,
    settings: Any = None,
    specs: tuple[_SeedSpec, ...] | list[_SeedSpec] | None = None,
) -> list[SeedDiff]:
    """모든 spec에 mode를 적용하고 테이블별 diff 목록을 반환. dry_run이면 DB를 변경하지 않는다."""
    handler = _MODE_DISPATCH.get(mode)
    if handler is None:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    target = SEED_SPECS if specs is None else specs
    return [handler(session, spec, dry_run, settings) for spec in target]


def _check_env_safety(mode: str, confirm: bool, env_value: str) -> tuple[str, int] | None:
    """위험 조합 차단. 위반 시 (메시지, 종료코드) 반환, 안전하면 None."""
    if env_value == "prod":
        if mode == "reset":
            return ("ERROR: prod 에서는 reset 모드를 사용할 수 없습니다.", 3)
        if mode == "sync" and not confirm:
            return ("ERROR: prod 의 sync 는 --confirm 이 필요합니다.", 3)
    if mode == "reset" and not confirm and env_value != "local":
        return ("WARNING: reset 모드는 위험합니다. --confirm 또는 ENV=local 에서만 허용됩니다.", 3)
    return None


def _truncate(value: Any, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_nk(nk: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in nk.items())


def format_diff_report(diffs: list[SeedDiff]) -> str:
    """diff를 사람이 읽기 좋게 직렬화. INSERT/DELETE는 자연키, UPDATE는 컬럼 old→new까지."""
    lines: list[str] = []
    for d in diffs:
        lines.append(f"[{d.table}] INSERT {d.inserted}, UPDATE {d.updated}, DELETE {d.deleted}")
        for rc in d.insert_rows:
            lines.append("  + " + _fmt_nk(rc.natural_key))
        for rc in d.update_rows:
            lines.append("  ~ " + _fmt_nk(rc.natural_key))
            if rc.column_changes:
                for col, (old, new) in rc.column_changes.items():
                    lines.append(f"      {col}: {_truncate(old)} -> {_truncate(new)}")
        for rc in d.delete_rows:
            lines.append("  - " + _fmt_nk(rc.natural_key))
    return "\n".join(lines)


def create_db_session():
    engine = create_engine(get_settings().get_db_uri)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def main() -> int:
    parser = argparse.ArgumentParser(description="데이터베이스 참조 데이터 시드")
    parser.add_argument(
        "--mode",
        choices=("ensure", "upsert", "sync", "reset"),
        default="ensure",
        help="ensure: 없으면 삽입 | upsert: 없으면 삽입·있으면 갱신 | "
        "sync: 시드에 없는 참조행(미참조만) 삭제 후 upsert | "
        "reset: 테이블 비운 뒤 시드만 삽입(user는 시드 username만 대상)",
    )
    parser.add_argument(
        "--env",
        metavar="PROFILE",
        help="모듈 import 시 sys.argv에서 읽어 ENV로 설정. 예: staging → config/.env.staging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB 변동 없이 INSERT/UPDATE/DELETE 계획만 출력",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="prod 의 sync, 그리고 ENV=local 이 아닌 곳의 reset 에 필요",
    )
    args = parser.parse_args()

    env_value = os.environ.get("ENV", "").strip()
    guard = _check_env_safety(args.mode, args.confirm, env_value)
    if guard is not None:
        message, code = guard
        print(message, file=sys.stderr)
        return code

    db = create_db_session()
    try:
        print(f"DB 시드 시작: mode={args.mode}, dry_run={args.dry_run}, ENV={env_value or '(none)'}")
        diffs = apply_mode(db, args.mode, dry_run=args.dry_run, settings=get_settings())
        report = format_diff_report(diffs)
        if report:
            print(report)
        if args.dry_run:
            db.rollback()
            print("\n[dry-run] DB 변동 없음. 적용하려면 --dry-run 없이 다시 실행하세요.")
        else:
            db.commit()
            print("\n시드 완료.")
        return 0
    except Exception as e:
        traceback.print_exc()
        print(f"시드 실패: {e}", file=sys.stderr)
        db.rollback()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
