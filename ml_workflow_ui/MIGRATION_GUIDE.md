# Streamlit → Gradio 마이그레이션 가이드

## 개요

기존 Streamlit 기반 UI를 Gradio로 마이그레이션했습니다.

## 주요 변경사항

### 1. 프레임워크 변경

- **이전**: Streamlit
- **이후**: Gradio

### 2. 새로운 파일 구조

```
ml_workflow_ui/
├── app/
│   ├── config/
│   │   └── settings.py          # 설정 (기존과 동일)
│   ├── ui/                       # 🆕 UI 모듈
│   │   ├── __init__.py
│   │   ├── quick_deployment.py  # Quick Deployment UI
│   │   └── playground.py        # Playground UI
│   ├── utils/                    # 🆕 유틸리티
│   │   ├── __init__.py
│   │   └── create_template.py   # 템플릿 생성 스크립트
│   ├── workspaces/               # ⚠️ 기존 Streamlit 코드 (유지)
│   │   ├── inference.py
│   │   └── apis.py
│   ├── api_client.py             # 🆕 통합 API 클라이언트
│   ├── gradio_main.py            # 🆕 Gradio 메인 앱
│   └── main.py                   # 기존 Streamlit 앱
├── run_gradio.sh                 # 🆕 Gradio 실행 스크립트
├── Pipfile                       # gradio, pillow 추가
├── README.md                     # 기본 README
├── GRADIO_GUIDE.md               # 🆕 Gradio 사용 가이드
└── MIGRATION_GUIDE.md            # 🆕 이 문서
```

### 3. 기능 비교

| 기능 | Streamlit | Gradio |
|------|-----------|--------|
| 추론 UI | ✅ | ✅ |
| Quick Deployment | ❌ | ✅ |
| Playground | ❌ | ✅ |
| 템플릿 관리 | ❌ | ✅ |
| 워크플로우 생성 | ❌ | ✅ |
| 워크플로우 실행 | ❌ | ✅ |
| 배포 상태 모니터링 | ❌ | ✅ |

### 4. API 클라이언트 통합

기존에는 `workspaces/apis.py`에 일부 API 호출만 구현되어 있었습니다.
새로운 `api_client.py`는 모든 Backend API 엔드포인트를 지원합니다.

**주요 메서드:**
- `authenticate()`: 로그인
- `get_workflow_templates()`: 템플릿 목록
- `clone_from_template()`: 템플릿에서 워크플로우 생성
- `execute_workflow()`: 워크플로우 실행
- `get_workflow_status()`: 실행 상태 조회
- `sync_workflow_deployments()`: 배포 정보 동기화
- `get_deployed_models()`: 배포된 모델 목록
- `inference()`: 추론 수행

## 마이그레이션 단계

### 1. 패키지 업데이트

```bash
cd ml_workflow_ui
pipenv install
```

새로운 패키지:
- `gradio`: UI 프레임워크
- `pillow`: 이미지 처리
- `requests`: 이미 있음 (명시적으로 추가)

### 2. 환경 설정

`.env` 파일은 기존과 동일하게 사용:

```
REST_API_URL=http://localhost:8000
```

### 3. Gradio UI 실행

```bash
cd ml_workflow_ui
./run_gradio.sh
```

또는

```bash
cd ml_workflow_ui/app
python gradio_main.py
```

### 4. 워크플로우 템플릿 생성 (최초 1회)

```bash
cd ml_workflow_ui/app
python utils/create_template.py --password YOUR_PASSWORD --model-id MODEL_ID
```

## UI 비교

### Streamlit 버전 (기존)

- **메뉴**: Inference만 존재
- **기능**:
  - Inference Service URL, Service Hostname, Model Name 수동 입력
  - 이미지 업로드
  - 텍스트 입력 (동적)
  - 추론 실행

### Gradio 버전 (신규)

- **메뉴**:
  1. 로그인
  2. Quick Service Deployment
  3. Playground
  4. 로그아웃

- **Quick Service Deployment**:
  - 템플릿 목록 조회
  - 워크플로우 생성
  - 워크플로우 실행 (배포)
  - 배포 상태 모니터링
  - 배포 정보 동기화

- **Playground**:
  - 워크플로우 목록 조회
  - 배포된 모델 자동 로드
  - 이미지 업로드
  - 텍스트 레이블 입력
  - 추론 실행
  - 결과 시각화 (이미지 + JSON)

## 장점

### 1. 사용자 경험 개선

- **자동화**: Inference Service URL 등을 수동으로 입력할 필요 없음
- **통합 워크플로우**: 배포부터 추론까지 한 곳에서 처리
- **상태 모니터링**: 배포 진행 상황 확인 가능

### 2. 개발자 경험 개선

- **모듈화**: UI 컴포넌트가 명확히 분리됨
- **API 클라이언트**: 재사용 가능한 통합 클라이언트
- **타입 안정성**: Gradio의 타입 시스템 활용

### 3. 기능 확장

- **템플릿 관리**: 워크플로우 템플릿 생성 및 관리
- **워크플로우 관리**: 전체 생명주기 관리
- **멀티 모델**: 여러 모델을 하나의 워크플로우에서 관리

## 기존 Streamlit UI 유지

기존 Streamlit UI는 `workspaces/` 디렉토리에 그대로 유지됩니다.
필요하면 계속 사용할 수 있습니다:

```bash
cd ml_workflow_ui/app
streamlit run main.py
```

## 향후 계획

1. ✅ Gradio UI 기본 구현
2. ⏳ 워크플로우 템플릿 확장 (다양한 모델)
3. ⏳ 실시간 로그 스트리밍
4. ⏳ 모델 성능 모니터링
5. ⏳ 배포 리소스 관리
6. ⏳ 멀티 유저 지원

## 문의 및 피드백

- Backend API 문서: `/docs` (FastAPI Swagger UI)
- Gradio 문서: https://www.gradio.app/docs
- 이슈 제보: (프로젝트 이슈 트래커)
