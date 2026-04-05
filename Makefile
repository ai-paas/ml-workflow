# DB · Alembic · 시드 · Harbor
.DEFAULT_GOAL := help
.PHONY: help alembic-upgrade-head alembic-autogen-file db-seed db-seed-ensure db-seed-upsert db-seed-reset \
        harbor-login \
        harbor-build-backend harbor-push-backend harbor-build-push-backend \
        harbor-build-predictor harbor-push-predictor harbor-build-push-predictor \
        harbor-build-train harbor-push-train harbor-build-push-train \
        harbor-build-push-all

APP_DIR := backend/app
MODE ?= ensure
msg ?= autogen
TAG ?= latest
PYTHON ?= python3
# ./alembic 은 마이그레이션 디렉터리라 pip 패키지 alembic 과 충돌 → purelib 를 앞에 둔다.
PY_PURELIB := $(shell cd $(APP_DIR) && $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
APP_PYTHONPATH := $(PY_PURELIB):.
ENV_FILE = $(APP_DIR)/config/.env.$(ENV)

help:
	@echo "DB / Alembic (리포지토리 루트에서 실행, APP_DIR=$(APP_DIR))"
	@echo "또는 backend/app 에서: cd backend/app && make <동일타깃>"
	@echo "  make alembic-upgrade-head [ENV=staging]"
	@echo "  make alembic-autogen-file msg=\"...\" [ENV=staging]   # DB 연결·reflection 필요"
	@echo "  make db-seed [MODE=ensure|upsert] [ENV=staging]"
	@echo "  make db-seed MODE=reset CONFIRM=1 [ENV=staging]"
	@echo "별칭: db-seed-ensure, db-seed-upsert, db-seed-reset"
	@echo ""
	@echo "Harbor · Docker (ENV 필수: dev | innogrid)"
	@echo "  make harbor-login ENV=dev"
	@echo "  make harbor-build-push-backend ENV=dev [TAG=latest]"
	@echo "  make harbor-build-push-predictor ENV=dev [TAG=latest]"
	@echo "  make harbor-build-push-train ENV=dev [TAG=latest]"
	@echo "  make harbor-build-push-all ENV=dev [TAG=latest]"
	@echo "개별: harbor-build-{backend|predictor|train}, harbor-push-{backend|predictor|train}"

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

# ─── Harbor · Docker ──────────────────────────────────────────────────────────
harbor-login:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요. 예: make harbor-login ENV=dev" && exit 1)
	@[ -f "$(ENV_FILE)" ] || (echo "ERROR: 환경 파일이 없습니다: $(ENV_FILE)" && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		echo "$$HARBOR_PASSWORD" | docker login $$HARBOR_URL -u $$HARBOR_USERNAME --password-stdin

harbor-build-backend:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --platform linux/amd64 -f backend/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$BACKEND_PROJECT_NAME:$(TAG) .

harbor-push-backend:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$BACKEND_PROJECT_NAME:$(TAG)

harbor-build-push-backend:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --platform linux/amd64 -f backend/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$BACKEND_PROJECT_NAME:$(TAG) . && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$BACKEND_PROJECT_NAME:$(TAG)

harbor-build-predictor:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --platform linux/amd64 -f predictor/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$INFERENCE_PROJECT_NAME:$(TAG) .

harbor-push-predictor:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$INFERENCE_PROJECT_NAME:$(TAG)

harbor-build-push-predictor:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --platform linux/amd64 -f predictor/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$INFERENCE_PROJECT_NAME:$(TAG) . && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$INFERENCE_PROJECT_NAME:$(TAG)

harbor-build-train:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --platform linux/amd64 -f train_eval/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$TRAIN_PROJECT_NAME:$(TAG) .

harbor-push-train:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$TRAIN_PROJECT_NAME:$(TAG)

harbor-build-push-train:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --platform linux/amd64 -f train_eval/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$TRAIN_PROJECT_NAME:$(TAG) . && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$TRAIN_PROJECT_NAME:$(TAG)

harbor-build-push-all:
	@$(MAKE) harbor-build-push-backend ENV=$(ENV) TAG=$(TAG)
	@$(MAKE) harbor-build-push-predictor ENV=$(ENV) TAG=$(TAG)
	@$(MAKE) harbor-build-push-train ENV=$(ENV) TAG=$(TAG)
