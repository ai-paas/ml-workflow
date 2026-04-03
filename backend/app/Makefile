# backend/app 에서 직접 실행할 때용 (루트 Makefile과 동일 타깃, cd 없음)
# 설계: docs/db-operations/DB_시드_Alembic_Makefile_설계.md
.DEFAULT_GOAL := help
.PHONY: help alembic-upgrade-head alembic-autogen-file db-seed db-seed-ensure db-seed-upsert db-seed-reset

MODE ?= ensure
msg ?= autogen
PYTHON ?= python3
PY_PURELIB := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
APP_PYTHONPATH := $(PY_PURELIB):.

help:
	@echo "DB / Alembic (현재 디렉터리가 backend/app 일 때, PYTHONPATH=.)"
	@echo "  make alembic-upgrade-head [ENV=dev]"
	@echo "  make alembic-autogen-file msg=\"...\" [ENV=dev]"
	@echo "  make db-seed [MODE=ensure|upsert] [ENV=dev]"
	@echo "  make db-seed MODE=reset CONFIRM=1 [ENV=dev]"
	@echo "리포지토리 루트에서는 상위 ../Makefile 사용."

alembic-upgrade-head:
	$(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) alembic upgrade head

alembic-autogen-file:
	REV=$$($(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) scripts/next_alembic_rev.py) && \
	$(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) alembic revision --autogenerate --rev-id=$$REV -m "$(msg)"

db-seed:
	@if [ "$(MODE)" = "reset" ] && [ "$(CONFIRM)" != "1" ]; then \
		echo "MODE=reset 은 데이터 손실이 있을 수 있습니다. CONFIRM=1 을 함께 지정하세요."; \
		echo "예: make db-seed MODE=reset CONFIRM=1"; \
		exit 1; \
	fi
	$(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) -m config.db.data_initializer --mode $(MODE)

db-seed-ensure:
	@$(MAKE) db-seed MODE=ensure

db-seed-upsert:
	@$(MAKE) db-seed MODE=upsert

db-seed-reset:
	@$(MAKE) db-seed MODE=reset CONFIRM=1
