"""DB 시드 러너(config.db.data_initializer) 단위 테스트.

pytest 미설치 환경을 고려해 stdlib unittest + SQLite in-memory 로 작성한다.
실제 SEED_SPECS/모델 스키마에 의존하지 않도록 합성(synthetic) 모델로 러너 로직만 검증한다.

실행:
    cd backend/app && PYTHONPATH=. python -m unittest tests.test_data_initializer -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from config.db import data_initializer as di
from sqlalchemy import Column, Integer, String, create_engine, func, select
from sqlalchemy.orm import declarative_base, sessionmaker
from utils.crypto import get_sha256_hash

Base = declarative_base()


class Widget(Base):
    __tablename__ = "widget"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    description = Column(String(200))


class Account(Base):
    __tablename__ = "account"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    name = Column(String(50))
    password = Column(String(64))


def _spec(rows, *, model=Widget, nk="name", reset_kind="full", transform=None):
    return di._SeedSpec(model, rows, nk, reset_kind, transform=transform)


class _DBTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _names(self, model=Widget, col="name"):
        return set(self.session.execute(select(getattr(model, col))).scalars().all())

    def _count(self, model=Widget):
        return int(self.session.scalar(select(func.count()).select_from(model)) or 0)


class TestEnsure(_DBTestCase):
    def test_inserts_only_missing(self):
        self.session.execute(Widget.__table__.insert().values(name="a", description="orig"))
        self.session.commit()

        spec = _spec([{"name": "a", "description": "changed"}, {"name": "b", "description": "y"}])
        diffs = di.apply_mode(self.session, "ensure", specs=[spec])
        self.session.commit()

        self.assertEqual(diffs[0].inserted, 1)
        self.assertEqual(self._names(), {"a", "b"})
        # ensure 는 기존 행을 건드리지 않는다
        existing = self.session.execute(select(Widget).where(Widget.name == "a")).scalars().first()
        self.assertEqual(existing.description, "orig")


class TestUpsert(_DBTestCase):
    def test_inserts_and_updates_with_column_diff(self):
        self.session.execute(Widget.__table__.insert().values(name="a", description="orig"))
        self.session.commit()

        spec = _spec([{"name": "a", "description": "new"}, {"name": "b", "description": "y"}])
        diffs = di.apply_mode(self.session, "upsert", specs=[spec])
        self.session.commit()

        d = diffs[0]
        self.assertEqual((d.inserted, d.updated), (1, 1))
        self.assertEqual(d.update_rows[0].column_changes["description"], ("orig", "new"))
        updated = self.session.execute(select(Widget).where(Widget.name == "a")).scalars().first()
        self.assertEqual(updated.description, "new")

    def test_no_change_is_not_reported_as_update(self):
        self.session.execute(Widget.__table__.insert().values(name="a", description="same"))
        self.session.commit()
        spec = _spec([{"name": "a", "description": "same"}])
        diffs = di.apply_mode(self.session, "upsert", specs=[spec])
        self.assertEqual((diffs[0].inserted, diffs[0].updated), (0, 0))


class TestSync(_DBTestCase):
    def test_deletes_orphans_and_upserts(self):
        self.session.execute(Widget.__table__.insert().values(name="a", description="orig"))
        self.session.execute(Widget.__table__.insert().values(name="stale", description="z"))
        self.session.commit()

        spec = _spec([{"name": "a", "description": "new"}, {"name": "b", "description": "y"}])
        diffs = di.apply_mode(self.session, "sync", specs=[spec])
        self.session.commit()

        d = diffs[0]
        self.assertEqual(d.deleted, 1)
        self.assertEqual(d.delete_rows[0].natural_key, {"name": "stale"})
        self.assertEqual(self._names(), {"a", "b"})

    def test_user_kind_does_not_delete_others(self):
        self.session.execute(Account.__table__.insert().values(username="keep", name="k"))
        self.session.commit()
        spec = _spec(
            [{"username": "seed", "name": "s"}],
            model=Account,
            nk="username",
            reset_kind="user_usernames",
        )
        di.apply_mode(self.session, "sync", specs=[spec])
        self.session.commit()
        # 시드에 없는 'keep' 계정은 보존되어야 한다
        self.assertEqual(self._names(model=Account, col="username"), {"keep", "seed"})


class TestReset(_DBTestCase):
    def test_replaces_all(self):
        self.session.execute(Widget.__table__.insert().values(name="old", description="z"))
        self.session.commit()
        spec = _spec([{"name": "a"}, {"name": "b"}])
        diffs = di.apply_mode(self.session, "reset", specs=[spec])
        self.session.commit()
        self.assertEqual(diffs[0].deleted, 1)
        self.assertEqual(diffs[0].inserted, 2)
        self.assertEqual(self._names(), {"a", "b"})


class TestDryRun(_DBTestCase):
    def test_dry_run_makes_no_changes_but_plans(self):
        spec = _spec([{"name": "a", "description": "x"}])
        diffs = di.apply_mode(self.session, "ensure", specs=[spec], dry_run=True)
        self.session.rollback()

        self.assertEqual(diffs[0].inserted, 1)  # 계획은 산출
        self.assertEqual(self._count(), 0)  # 그러나 DB 변동 없음

        # dry-run 없이 다시 실행하면 실제로 반영
        di.apply_mode(self.session, "ensure", specs=[spec])
        self.session.commit()
        self.assertEqual(self._count(), 1)

    def test_dry_run_sync_does_not_delete(self):
        self.session.execute(Widget.__table__.insert().values(name="stale"))
        self.session.commit()
        spec = _spec([{"name": "a"}])
        diffs = di.apply_mode(self.session, "sync", specs=[spec], dry_run=True)
        self.session.rollback()
        self.assertEqual(diffs[0].deleted, 1)  # 삭제 '계획'
        self.assertEqual(self._names(), {"stale"})  # 실제로는 보존


class TestTransform(_DBTestCase):
    def test_user_transform_hashes_password_and_strips_runtime_key(self):
        fake_settings = SimpleNamespace(DEMO_PASSWORD="s3cret")
        spec = _spec(
            [{"username": "u", "name": "n", "_password_env": "DEMO_PASSWORD"}],
            model=Account,
            nk="username",
            transform=di._user_transform,
        )
        di.apply_mode(self.session, "ensure", specs=[spec], settings=fake_settings)
        self.session.commit()

        acc = self.session.execute(select(Account).where(Account.username == "u")).scalars().first()
        self.assertEqual(acc.password, get_sha256_hash("s3cret"))

    def test_resolve_row_strips_underscore_keys(self):
        spec = _spec([{"name": "a", "_runtime": "x"}])
        resolved = di._resolve_row(spec, spec.rows[0], None)
        self.assertNotIn("_runtime", resolved)
        self.assertEqual(resolved["name"], "a")


class TestEnvSafety(unittest.TestCase):
    def test_prod_blocks_reset(self):
        self.assertEqual(di._check_env_safety("reset", False, "prod")[1], 3)
        self.assertEqual(di._check_env_safety("reset", True, "prod")[1], 3)  # prod reset 은 confirm 으로도 불가

    def test_prod_sync_requires_confirm(self):
        self.assertEqual(di._check_env_safety("sync", False, "prod")[1], 3)
        self.assertIsNone(di._check_env_safety("sync", True, "prod"))

    def test_reset_outside_local_requires_confirm(self):
        self.assertEqual(di._check_env_safety("reset", False, "dev")[1], 3)
        self.assertIsNone(di._check_env_safety("reset", False, "local"))
        self.assertIsNone(di._check_env_safety("reset", True, "dev"))

    def test_safe_combinations(self):
        self.assertIsNone(di._check_env_safety("ensure", False, "prod"))
        self.assertIsNone(di._check_env_safety("upsert", False, "prod"))
        self.assertIsNone(di._check_env_safety("sync", False, "dev"))


class TestDiffReport(unittest.TestCase):
    def test_report_includes_column_changes(self):
        diff = di.SeedDiff(table="widget")
        diff.inserted = 1
        diff.insert_rows.append(di.RowChange(natural_key={"name": "b"}))
        diff.updated = 1
        diff.update_rows.append(
            di.RowChange(natural_key={"name": "a"}, column_changes={"description": ("orig", "new")})
        )
        report = di.format_diff_report([diff])
        self.assertIn("[widget] INSERT 1, UPDATE 1, DELETE 0", report)
        self.assertIn("+ name=b", report)
        self.assertIn("description: orig -> new", report)


if __name__ == "__main__":
    unittest.main()
