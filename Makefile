# DB · Alembic · 시드 — docs/db-operations/DB_시드_Alembic_Makefile_설계.md 규약
.DEFAULT_GOAL := help
.PHONY: help alembic-upgrade-head alembic-autogen-file db-seed db-seed-ensure db-seed-upsert db-seed-reset

APP_DIR := backend/app
MODE ?= ensure
msg ?= autogen
PYTHON ?= python3
# ./alembic 은 마이그레이션 디렉터리라 pip 패키지 alembic 과 충돌 → purelib 를 앞에 둔다.
PY_PURELIB := $(shell cd $(APP_DIR) && $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
APP_PYTHONPATH := $(PY_PURELIB):.

help:
	@echo "DB / Alembic (리포지토리 루트에서 실행, APP_DIR=$(APP_DIR))"
	@echo "또는 backend/app 에서: cd backend/app && make <동일타깃>"
	@echo "  make alembic-upgrade-head [ENV=staging]"
	@echo "  make alembic-autogen-file msg=\"...\" [ENV=staging]   # DB 연결·reflection 필요"
	@echo "  make db-seed [MODE=ensure|upsert] [ENV=staging]"
	@echo "  make db-seed MODE=reset CONFIRM=1 [ENV=staging]"
	@echo "별칭: db-seed-ensure, db-seed-upsert, db-seed-reset"

alembic-upgrade-head:
	cd $(APP_DIR) && $(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) alembic upgrade head

alembic-autogen-file:
	cd $(APP_DIR) && \
	REV=$$($(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) scripts/next_alembic_rev.py) && \
	$(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) alembic revision --autogenerate --rev-id=$$REV -m "$(msg)"

db-seed:
	@if [ "$(MODE)" = "reset" ] && [ "$(CONFIRM)" != "1" ]; then \
		echo "MODE=reset 은 데이터 손실이 있을 수 있습니다. CONFIRM=1 을 함께 지정하세요."; \
		echo "예: make db-seed MODE=reset CONFIRM=1"; \
		exit 1; \
	fi
	cd $(APP_DIR) && $(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) -m config.db.data_initializer --mode $(MODE)

db-seed-ensure:
	@$(MAKE) db-seed MODE=ensure

db-seed-upsert:
	@$(MAKE) db-seed MODE=upsert

db-seed-reset:
	@$(MAKE) db-seed MODE=reset CONFIRM=1
