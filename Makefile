# DB · Alembic · 시드 · Harbor · E2E
.DEFAULT_GOAL := help
.PHONY: help alembic-upgrade-head alembic-autogen-file db-seed db-seed-ensure db-seed-upsert db-seed-reset \
        harbor-login \
        harbor-build-backend harbor-push-backend harbor-build-push-backend \
        harbor-build-predictor harbor-push-predictor harbor-build-push-predictor \
        harbor-build-train harbor-push-train harbor-build-push-train \
        harbor-build-push-all \
        harbor-build-push-backend-nc harbor-build-push-predictor-nc harbor-build-push-train-nc \
        harbor-build-push-all-nc \
        flake8 flake8-fix isort black lint lint-fix \
        e2e-workflow-validation \
        e2e-wf-scenario-info e2e-wf-scenario-deploy e2e-wf-scenario-delete e2e-wf-scenario-lifecycle

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
	@echo "  make harbor-build-push-backend-nc ENV=dev [TAG=latest]  (--no-cache)"
	@echo "  make harbor-build-push-all-nc ENV=dev [TAG=latest]      (--no-cache)"
	@echo "개별: harbor-build-{backend|predictor|train}, harbor-push-{backend|predictor|train}"
	@echo ""
	@echo "Lint / Format (.pre-commit-config.yaml 규칙과 동일)"
	@echo "  make flake8    — flake8 린트 검사"
	@echo "  make isort     — import 정렬"
	@echo "  make black     — 코드 포맷팅"
	@echo "  make lint      — isort + black + flake8 전체 검사"
	@echo "  make lint-fix  — isort + black + autopep8 자동 수정"
	@echo "  make flake8-fix — autopep8 자동 수정만 수행"

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

# ─── Harbor Build+Push (--no-cache) ─────────────────────────────────────────
harbor-build-push-backend-nc:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --no-cache --platform linux/amd64 -f backend/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$BACKEND_PROJECT_NAME:$(TAG) . && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$BACKEND_PROJECT_NAME:$(TAG)

harbor-build-push-predictor-nc:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --no-cache --platform linux/amd64 -f predictor/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$INFERENCE_PROJECT_NAME:$(TAG) . && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$INFERENCE_PROJECT_NAME:$(TAG)

harbor-build-push-train-nc:
	@[ -n "$(ENV)" ] || (echo "ERROR: ENV를 지정하세요." && exit 1)
	@set -a && . $(ENV_FILE) && set +a && \
		docker buildx build --no-cache --platform linux/amd64 -f train_eval/Dockerfile \
		-t $$HARBOR_URL/$$HARBOR_REPOSITORY/$$TRAIN_PROJECT_NAME:$(TAG) . && \
		docker push $$HARBOR_URL/$$HARBOR_REPOSITORY/$$TRAIN_PROJECT_NAME:$(TAG)

harbor-build-push-all-nc:
	@$(MAKE) harbor-build-push-backend-nc ENV=$(ENV) TAG=$(TAG)
	@$(MAKE) harbor-build-push-predictor-nc ENV=$(ENV) TAG=$(TAG)
	@$(MAKE) harbor-build-push-train-nc ENV=$(ENV) TAG=$(TAG)

# ─── Lint / Format (.pre-commit-config.yaml 규칙과 동일) ──────────────────
FLAKE8_IGNORE := E203,W503,W605,E712,E266,F401,E402,F821,E711,F403
FLAKE8_EXCLUDE := .venv,*/.venv

flake8:
	uv run --group dev flake8 --max-line-length=120 --ignore=$(FLAKE8_IGNORE) --exclude=$(FLAKE8_EXCLUDE) .

flake8-fix:
	uv run --group dev ruff check --fix .

isort:
	uv run --group dev isort --profile black --line-length 120 .

black:
	uv run --group dev black --line-length 120 .

lint: isort black flake8

lint-fix: isort black flake8-fix

# ─── E2E Tests ─────────────────────────────────────────────────────────────
E2E_DIR := e2e-test

e2e-workflow-validation:
	@echo "▶ E2E: 워크플로우 정의 검증 오류 케이스 테스트"
	uv run --group e2e pytest $(E2E_DIR)/scenarios/test_workflow_validation.py -v -s

# ─── E2E: 워크플로우 시나리오 (9개) ─────────────────────────────────────────────
# SCENARIO=1~9 로 시나리오를 선택한다.
# 1: 단순 LLM  2: 단순 RAG  3~9: 복합 시나리오 (체인/병렬/쿼리정제 등)
#   make e2e-wf-scenario-info                    # 전체 시나리오 목록
#   make e2e-wf-scenario-info SCENARIO=3         # 3번 시나리오 상세
#   make e2e-wf-scenario-deploy SCENARIO=3              # 3번 시나리오 배포
#   make e2e-wf-scenario-delete SCENARIO=3              # 3번 시나리오 저장 건별 삭제 확인(y)
#   make e2e-wf-scenario-lifecycle SCENARIO=3           # 3번 시나리오 전체 생명주기

e2e-wf-scenario-info:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-info SCENARIO=1" && exit 1)
	@uv run --group e2e python $(E2E_DIR)/workflow_scenarios.py $(SCENARIO)

e2e-wf-scenario-deploy:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-deploy SCENARIO=1" && exit 1)
	@echo "▶ E2E: 워크플로우 시나리오 #$(SCENARIO) 배포 테스트"
	E2E_SCENARIO=$(SCENARIO) uv run --group e2e pytest $(E2E_DIR)/scenarios/test_workflow_scenario_deploy.py -v -s

e2e-wf-scenario-delete:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-delete SCENARIO=1" && exit 1)
	@echo "▶ E2E: 워크플로우 시나리오 #$(SCENARIO) 삭제 테스트"
	E2E_SCENARIO=$(SCENARIO) uv run --group e2e pytest $(E2E_DIR)/scenarios/test_workflow_scenario_delete.py -v -s

e2e-wf-scenario-lifecycle:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-lifecycle SCENARIO=1" && exit 1)
	@echo "▶ E2E: 워크플로우 시나리오 #$(SCENARIO) 전체 생명주기 테스트"
	E2E_SCENARIO=$(SCENARIO) uv run --group e2e pytest $(E2E_DIR)/scenarios/test_workflow_scenario_lifecycle.py -v -s
