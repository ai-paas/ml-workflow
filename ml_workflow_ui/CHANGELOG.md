# Changelog

## [2.0.0] - Gradio UI

### 🎉 Major Changes

#### Replaced Streamlit with Gradio
- ❌ 제거: Streamlit 기반 UI
- ✅ 추가: Gradio 기반 UI

### 🚀 New Features

#### Quick Service Deployment
- 워크플로우 템플릿 목록 조회 및 필터링
- 템플릿으로부터 워크플로우 생성
- 워크플로우 실행 (KServe 배포)
- 실시간 배포 상태 모니터링
- 배포 정보 동기화

#### Playground
- 배포된 워크플로우 목록 조회
- 배포된 모델 자동 로드
- 이미지 업로드 및 추론
- 동적 텍스트 레이블 입력 (최대 5개)
- 결과 시각화 (이미지 + JSON)

#### API Client
- 통합 API 클라이언트 (`api_client.py`)
- 모든 Backend API 엔드포인트 지원
- 타입 힌트 및 docstring 완비

#### Utilities
- 워크플로우 템플릿 생성 스크립트 (`utils/create_template.py`)
- DETR Object Detection 템플릿 지원

### 📁 File Structure Changes

#### Removed
```
app/
  - main.py (Streamlit 메인 앱)
  - workspaces/
    - apis.py
    - inference.py
```

#### Added
```
app/
  - api_client.py (통합 API 클라이언트)
  - gradio_main.py (Gradio 메인 앱)
  - ui/
    - quick_deployment.py
    - playground.py
  - utils/
    - create_template.py
```

### 📦 Dependencies

#### Removed
- `streamlit`

#### Added
- `gradio`
- `pillow`

### 📚 Documentation

#### Added
- `README.md` - 기본 가이드
- `GRADIO_GUIDE.md` - 상세 사용 가이드
- `MIGRATION_GUIDE.md` - 마이그레이션 가이드
- `SUMMARY.md` - 구현 요약
- `CHANGELOG.md` - 변경 이력 (이 파일)

### 🔄 API Changes

#### Old: `/api/v1/inference`
- 수동 입력: `inference_service_url`, `service_hostname`, `model_name`
- 범용 추론 엔드포인트

#### New: `/api/v1/workflows/{workflow_id}/inference`
- 워크플로우 기반: `workflow_id`, `component_id`
- 배포 정보 자동 조회
- 더 풍부한 응답 (model_info 포함)

### 💡 Benefits

1. **사용자 경험 개선**
   - 수동 입력 제거 (자동화)
   - 통합 워크플로우 (배포 → 추론)
   - 실시간 상태 모니터링

2. **개발자 경험 개선**
   - 모듈화된 UI 컴포넌트
   - 통합 API 클라이언트
   - 타입 안정성

3. **기능 확장**
   - 템플릿 관리
   - 워크플로우 생명주기 관리
   - 멀티 모델 지원

### 🚀 Getting Started

```bash
# 패키지 설치
cd ml_workflow_ui
pipenv install

# 환경 설정
echo "REST_API_URL=http://localhost:8000" > app/config/.env

# 템플릿 생성 (최초 1회)
cd app
python utils/create_template.py --password YOUR_PASSWORD --model-id 19

# UI 실행
cd ..
./run_gradio.sh
```

브라우저: `http://localhost:7860`

### 📖 Documentation

- **사용 가이드**: `GRADIO_GUIDE.md`
- **마이그레이션**: `MIGRATION_GUIDE.md`
- **구현 요약**: `SUMMARY.md`

---

## [1.0.0] - Streamlit UI (Legacy)

### Features
- Inference UI
- Manual service configuration
- Image upload
- Dynamic text inputs
- Basic inference

### Dependencies
- `streamlit`
- `pydantic-settings`
