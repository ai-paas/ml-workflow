# DB · Alembic · 시드 · Harbor · E2E
.DEFAULT_GOAL := help
.PHONY: help alembic-upgrade-head alembic-downgrade alembic-autogen-file db-seed db-seed-ensure db-seed-upsert db-seed-sync db-seed-reset db-seed-dry db-seed-test \
        harbor-login \
        harbor-build-backend harbor-push-backend harbor-build-push-backend \
        harbor-build-predictor harbor-push-predictor harbor-build-push-predictor \
        harbor-build-train harbor-push-train harbor-build-push-train \
        harbor-build-push-all \
        harbor-build-push-backend-nc harbor-build-push-predictor-nc harbor-build-push-train-nc \
        harbor-build-push-all-nc \
        flake8 flake8-fix isort black lint lint-fix \
        e2e-workflow-validation \
        e2e-wf-scenario-info e2e-wf-scenario-deploy e2e-wf-scenario-delete e2e-wf-scenario-lifecycle \
        e2e-wf-template-clone \
        e2e-model-improvement e2e-model-improvement-scenario \
        e2e-training \
        e2e-service-metric

APP_DIR := backend/app
# backend/app 기준 uv 프로젝트 루트(backend/). --project 로 pyproject·venv만 지정 (--directory 는 cwd 가 backend/ 로 바뀌어 alembic.ini·scripts 경로가 깨짐)
UV_PROJECT_REL := ..
MODE ?= ensure
# db-seed 추가 옵션 — DRY_RUN=1 → --dry-run(변경계획만), CONFIRM=1 → --confirm(prod sync·비-local reset)
DB_SEED_OPTS = --mode $(MODE)
DB_SEED_OPTS += $(if $(strip $(DRY_RUN)),--dry-run)
DB_SEED_OPTS += $(if $(strip $(CONFIRM)),--confirm)
msg ?= autogen
TAG ?= latest
# ./alembic 은 마이그레이션 디렉터리라 pip 패키지 alembic 과 충돌 → purelib 를 앞에 둔다.
PY_PURELIB := $(shell cd $(APP_DIR) && uv run --project $(UV_PROJECT_REL) python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
APP_PYTHONPATH := $(PY_PURELIB):.
ENV_FILE = $(APP_DIR)/config/.env.$(ENV)

help:
	@echo "DB / Alembic (리포지토리 루트에서 실행, APP_DIR=$(APP_DIR); uv + backend/pyproject.toml 환경)"
	@echo "  make alembic-upgrade-head [ENV=staging]"
	@echo "  make alembic-downgrade v=0028 [ENV=staging]   # 해당 revision 으로 DB 스키마 다운그레이드"
	@echo "  make alembic-autogen-file msg=\"...\" [ENV=staging]   # DB 연결·reflection 필요"
	@echo "  make db-seed [MODE=ensure|upsert|sync] [DRY_RUN=1] [ENV=local|dev] [CONFIRM=1]"
	@echo "  make db-seed MODE=reset CONFIRM=1 [ENV=local]"
	@echo "  make db-seed-dry           # sync 변경계획만 미리보기(--dry-run)"
	@echo "  make db-seed-test          # 시드 러너 단위 테스트(unittest, ENV 기본 local)"
	@echo "별칭: db-seed-ensure, db-seed-upsert, db-seed-sync, db-seed-reset, db-seed-dry"
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
	cd $(APP_DIR) && $(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) uv run --project $(UV_PROJECT_REL) alembic upgrade head

# v: Alembic revision id (예: 0028, base). 따옴표는 생략해도 됨 — make alembic-downgrade v=0028 ENV=dev
alembic-downgrade:
	@if [ -z "$(strip $(v))" ]; then \
		echo 'ERROR: v 를 지정하세요. 예: make alembic-downgrade v=0028 ENV=dev  또는  v="0028"'; \
		exit 1; \
	fi
	cd $(APP_DIR) && $(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) uv run --project $(UV_PROJECT_REL) alembic downgrade $(strip $(v))

alembic-autogen-file:
	cd $(APP_DIR) && \
	REV=$$($(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) uv run --project $(UV_PROJECT_REL) python scripts/next_alembic_rev.py) && \
	$(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) uv run --project $(UV_PROJECT_REL) alembic revision --autogenerate --rev-id=$$REV -m "$(msg)"

db-seed:
	@if [ "$(MODE)" = "reset" ] && [ "$(CONFIRM)" != "1" ]; then \
		echo "MODE=reset 은 데이터 손실이 있을 수 있습니다. CONFIRM=1 을 함께 지정하세요."; \
		echo "예: make db-seed MODE=reset CONFIRM=1"; \
		exit 1; \
	fi
	cd $(APP_DIR) && $(if $(strip $(ENV)),ENV=$(ENV) )PYTHONPATH=$(APP_PYTHONPATH) uv run --project $(UV_PROJECT_REL) python -m config.db.data_initializer $(DB_SEED_OPTS)

db-seed-ensure:
	@$(MAKE) db-seed MODE=ensure

db-seed-upsert:
	@$(MAKE) db-seed MODE=upsert

db-seed-sync:
	@$(MAKE) db-seed MODE=sync

db-seed-reset:
	@$(MAKE) db-seed MODE=reset CONFIRM=1

db-seed-dry:
	@$(MAKE) db-seed MODE=sync DRY_RUN=1

# 시드 러너 단위 테스트 (pytest 미설치 → stdlib unittest). settings import 위해 ENV 기본 local.
db-seed-test:
	cd $(APP_DIR) && ENV=$(if $(strip $(ENV)),$(ENV),local) PYTHONPATH=$(APP_PYTHONPATH) uv run --project $(UV_PROJECT_REL) python -m unittest tests.test_data_initializer -v

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
# 선택: ENV=dev / ENV=innogrid → e2e-test/.env 를 읽은 뒤 e2e-test/.env.{ENV} 로 덮어씀 (config.py)

e2e-workflow-validation:
	@echo "▶ E2E: 워크플로우 정의 검증 오류 케이스 테스트"
	$(if $(strip $(ENV)),ENV=$(ENV) )uv run --group e2e pytest $(E2E_DIR)/workflow/tests/test_workflow_validation.py -v -s

# ─── E2E: 워크플로우 시나리오 (10개) ─────────────────────────────────────────────
# SCENARIO=1~10 로 시나리오를 선택한다.
# 1: 단순 LLM  2: 단순 RAG  3~9: 복합 시나리오 (체인/병렬/쿼리정제 등)  10: ODM 단순 (START→MODEL→END)
#   make e2e-wf-scenario-info                    # 전체 시나리오 목록
#   make e2e-wf-scenario-info SCENARIO=3         # 3번 시나리오 상세
#   make e2e-wf-scenario-deploy SCENARIO=3 ENV=dev     # 3번 배포 (.env.dev + e2e-test/.state.dev.json)
#   make e2e-wf-scenario-delete SCENARIO=3 ENV=dev     # deploy 와 동일 ENV → 같은 상태 파일에서 삭제
#   make e2e-wf-scenario-lifecycle SCENARIO=3 ENV=dev   # 생명주기(한 번에 생성~삭제, 상태 파일 미사용)

e2e-wf-scenario-info:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-info SCENARIO=1" && exit 1)
	@$(if $(strip $(ENV)),ENV=$(ENV) )uv run --group e2e python $(E2E_DIR)/workflow/definitions.py $(SCENARIO)

e2e-wf-scenario-deploy:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-deploy SCENARIO=1" && exit 1)
	@echo "▶ E2E: 워크플로우 시나리오 #$(SCENARIO) 배포 테스트"
	$(if $(strip $(ENV)),ENV=$(ENV) )E2E_SCENARIO=$(SCENARIO) uv run --group e2e pytest $(E2E_DIR)/workflow/tests/test_workflow_scenario_deploy.py -v -s

e2e-wf-scenario-delete:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-delete SCENARIO=1" && exit 1)
	@echo "▶ E2E: 워크플로우 시나리오 #$(SCENARIO) 삭제 테스트"
	$(if $(strip $(ENV)),ENV=$(ENV) )E2E_SCENARIO=$(SCENARIO) uv run --group e2e pytest $(E2E_DIR)/workflow/tests/test_workflow_scenario_delete.py -v -s

e2e-wf-scenario-lifecycle:
	@[ -n "$(SCENARIO)" ] || (echo "ERROR: SCENARIO를 지정하세요. 예: make e2e-wf-scenario-lifecycle SCENARIO=1" && exit 1)
	@echo "▶ E2E: 워크플로우 시나리오 #$(SCENARIO) 전체 생명주기 테스트"
	$(if $(strip $(ENV)),ENV=$(ENV) )E2E_SCENARIO=$(SCENARIO) uv run --group e2e pytest $(E2E_DIR)/workflow/tests/test_workflow_scenario_lifecycle.py -v -s

# ─── E2E: 워크플로우 템플릿 → 복제 → 실행 → 추론 → 정리 ──────────────────────────
# 가장 단순한 정의(시나리오 #1)로 템플릿 생성 → 복제 → 배포 → 추론 → 삭제 전체를 검증한다.
#   make e2e-wf-template-clone            # 기본
#   make e2e-wf-template-clone ENV=dev    # .env.dev 적용
e2e-wf-template-clone:
	@echo "▶ E2E: 워크플로우 템플릿 생성 → 복제 → 실행 → 추론 → 정리 테스트"
	$(if $(strip $(ENV)),ENV=$(ENV) )uv run --group e2e pytest $(E2E_DIR)/workflow/tests/test_workflow_template_clone.py -v -s

# ─── E2E: 서비스 모니터링 metric ────────────────────────────────────────────────
# 단순 LLM 워크플로우를 서비스에 연결해 배포 → 추론 → GET /services 의 1h/1d/1w metric 검증 → 정리.
# 서버가 기간별 모니터링 코드(정규화 마이그레이션 포함)로 떠 있어야 한다. 기본 SCENARIO=1.
#   make e2e-service-metric ENV=dev               # 시나리오 1 (단순 LLM)
#   make e2e-service-metric SCENARIO=1 ENV=dev
e2e-service-metric:
	@echo "▶ E2E: 서비스 모니터링 metric 기록 검증 (단순 LLM)"
	$(if $(strip $(ENV)),ENV=$(ENV) )E2E_SCENARIO=$(if $(strip $(SCENARIO)),$(SCENARIO),1) uv run --group e2e pytest $(E2E_DIR)/service/tests/test_service_metric_lifecycle.py -v -s

e2e-model-improvement:
	@echo "▶ E2E: 최적화/경량화 (model-improvements) API — 빠른 검증"
	$(if $(strip $(ENV)),ENV=$(ENV) )uv run --group e2e pytest $(E2E_DIR)/model_improvement/test_model_improvement.py -v -s -m model_improvement

e2e-model-improvement-scenario:
	@echo "▶ E2E: 최적화/경량화 시나리오 — 작업 생성 후 SUCCEEDED까지 폴링"
	@echo "    소스: E2E_OPTIMIZATION_SOURCE_MODEL_NAME (E2E_WORKFLOW_TARGET_LLM_MODEL 과 동일 패턴), 선택 E2E_OPTIMIZATION_TASK_TYPE"
	$(if $(strip $(ENV)),ENV=$(ENV) )uv run --group e2e pytest $(E2E_DIR)/model_improvement/test_model_improvement.py -v -s -m model_improvement_scenario

# ─── E2E: 통합 학습→등록 (YOLOX/ESM2 공통, 모델 무관) ─────────────────────────────
# .env.{ENV} 또는 인라인 환경변수로 모델군 선택. 자식 모델 id/name 을 .state.{ENV}.json 에 저장.
#   make e2e-training ENV=dev \
#     E2E_TRAINING_REFERENCE_MODEL_NAME=facebook/esm2_t6_8M_UR50D \
#     E2E_TRAINING_DATASET_FILE=protein_sample.zip E2E_TRAINING_DATASET_KIND=protein-classification \
#     E2E_TRAINING_EXPECT_TYPE=pLM E2E_TRAINING_EXPECT_FORMAT=pytorch
e2e-training:
	@echo "▶ E2E: 통합 학습→등록 (reference=$(if $(strip $(E2E_TRAINING_REFERENCE_MODEL_NAME)),$(E2E_TRAINING_REFERENCE_MODEL_NAME),.env))"
	$(if $(strip $(ENV)),ENV=$(ENV) )uv run --group e2e pytest $(E2E_DIR)/training/test_training_register.py -v -s -m training
