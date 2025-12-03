# ML Workflow UI

Gradio 기반의 ML Workflow 관리 및 추론 인터페이스입니다.

> 🎉 **v2.0.0**: Streamlit에서 Gradio로 완전히 전환되었습니다.

## 목차

- [기능](#기능)
- [사전 준비](#사전-준비)
- [설치 및 실행](#설치-및-실행)
- [상세 사용 가이드](#상세-사용-가이드)
- [프로젝트 구조](#프로젝트-구조)
- [API 클라이언트](#api-클라이언트)
- [트러블슈팅](#트러블슈팅)
- [기술 스택](#기술-스택)

## 기능

### 1. Quick Service Deployment
- 워크플로우 템플릿 목록 조회 및 필터링
- 템플릿으로부터 새로운 워크플로우 생성
- 워크플로우 실행 (KServe 배포)
- 실시간 배포 상태 모니터링
- 배포 정보 동기화

### 2. Playground
- 배포된 워크플로우 목록 조회
- 배포된 모델 자동 로드
- 이미지 업로드 및 추론 실행
- 동적 텍스트 레이블 입력 (최대 5개)
- 추론 결과 시각화 (이미지 + JSON)

## 사전 준비

### 1. Backend API 서버
Backend API 서버가 실행 중이어야 합니다.

### 2. 환경 설정
`app/config/.env` 파일 생성:

```bash
REST_API_URL=http://localhost:8000
```

### 3. 워크플로우 템플릿 생성 (최초 1회)

템플릿을 생성해야 Quick Deployment를 사용할 수 있습니다:

```bash
cd ml_workflow_ui/app
python utils/create_template.py --password YOUR_PASSWORD --model-id MODEL_ID
```

예시:
```bash
python utils/create_template.py --password mypassword --model-id 1
```

## 설치 및 실행

### 설치

```bash
cd ml_workflow_ui
pipenv install
```

### 실행 방법 1: 스크립트 사용

```bash
./run_gradio.sh
```

### 실행 방법 2: 직접 실행

```bash
cd app
pipenv run python gradio_main.py
```

또는

```bash
cd app
python gradio_main.py
```

브라우저에서 `http://localhost:7860` 접속

## 상세 사용 가이드

### 1. 로그인

1. **로그인 탭** 선택
2. 사용자 이름 입력 (기본값: surromind)
3. 비밀번호 입력
4. **로그인** 버튼 클릭
5. 로그인 성공 시 상태가 "✅ 로그인됨: surromind"로 변경됨

### 2. Quick Service Deployment

워크플로우 템플릿을 사용하여 빠르게 모델을 배포합니다.

#### 2.1 템플릿 선택
1. **Quick Service Deployment 탭** 선택
2. 카테고리 선택 (선택사항)
3. **템플릿 목록 새로고침** 버튼 클릭
4. 템플릿 목록에서 원하는 템플릿의 ID 확인
5. "선택한 템플릿 ID" 필드에 ID 입력

#### 2.2 워크플로우 생성
1. "워크플로우 이름" 입력 (예: `my-detr-detection`)
2. **워크플로우 생성** 버튼 클릭
3. 생성 결과 확인
4. "생성된 워크플로우 ID" 필드에 ID가 자동으로 입력됨

#### 2.3 워크플로우 실행 (배포)
1. **워크플로우 실행 (배포)** 버튼 클릭
2. 실행 결과에서 Kubeflow Run ID 확인
3. 배포가 시작됨

#### 2.4 배포 상태 확인
1. **배포 상태 확인** 버튼 클릭
2. 워크플로우 및 배포된 모델의 상태 확인
3. 배포가 완료될 때까지 주기적으로 확인 (약 2-3분 소요)

#### 2.5 배포 정보 동기화
1. Kubeflow Pipeline이 완료된 후 **배포 정보 동기화** 버튼 클릭
2. KServe InferenceService 정보가 DB에 동기화됨
3. 동기화 결과 확인

### 3. Playground

배포된 워크플로우를 사용하여 추론을 수행합니다.

#### 3.1 워크플로우 선택
1. **Playground 탭** 선택
2. **워크플로우 목록 새로고침** 버튼 클릭
3. 배포된 워크플로우 목록 확인
4. "워크플로우 ID" 필드에 ID 입력

#### 3.2 모델 로드
1. **모델 로드** 버튼 클릭
2. 배포된 모델 목록 확인
3. "Component ID" 필드에 추론할 모델의 Component ID 입력

#### 3.3 추론 실행
1. **이미지 업로드** 영역에 이미지 드래그 앤 드롭 또는 클릭하여 업로드
2. "텍스트 1" 필드에 감지할 객체 레이블 입력 (예: "a cat")
3. 추가 레이블이 필요하면 **텍스트 입력 추가** 버튼 클릭
4. "텍스트 2" 등에 추가 레이블 입력 (예: "a dog", "a car")
5. **추론 실행 🚀** 버튼 클릭

#### 3.4 결과 확인
- **실행 상태**: 추론 성공 여부 및 메타 정보
- **결과 이미지**: 객체가 감지된 이미지 (바운딩 박스 포함)
- **상세 결과 (JSON)**: 전체 추론 결과 (좌표, 점수 등)

### 예시: 전체 워크플로우

```
1. 로그인
   └─> 사용자: surromind, 비밀번호 입력

2. Quick Service Deployment
   └─> 템플릿 선택 → 워크플로우 생성 → 실행 → 상태 확인 → 동기화

3. Playground
   └─> 워크플로우 선택 → 모델 로드 → 이미지 업로드 → 추론 실행 → 결과 확인
```

## 프로젝트 구조

```
ml_workflow_ui/
├── app/
│   ├── config/
│   │   └── settings.py               # 환경 설정 (pydantic-settings)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── quick_deployment.py       # Quick Deployment UI
│   │   └── playground.py             # Playground UI
│   ├── utils/
│   │   ├── __init__.py
│   │   └── create_template.py        # 템플릿 생성 스크립트
│   ├── api_client.py                 # 통합 API 클라이언트
│   ├── gradio_main.py                # Gradio 메인 앱
│   └── session_manager.py            # 세션 관리
├── run_gradio.sh                      # 실행 스크립트
├── Pipfile                            # 의존성 관리
├── Dockerfile                         # Docker 이미지 빌드
├── .dockerignore                      # Docker 빌드 제외 파일
└── README.md                          # 이 문서
```

## API 클라이언트

`api_client.py`는 Backend API와 통신하는 통합 클라이언트를 제공합니다.

### 주요 메서드

#### 인증
- `authenticate(username, password)`: 로그인 및 토큰 획득

#### 워크플로우 템플릿
- `get_workflow_templates(category=None, page=None, page_size=None)`: 템플릿 목록 조회
- `get_workflow_template(template_id)`: 템플릿 상세 조회
- `create_workflow_template(name, workflow_definition=None, description=None, category=None)`: 템플릿 생성
- `clone_from_template(template_id, workflow_name, service_id=None)`: 템플릿으로부터 워크플로우 생성

#### 워크플로우
- `get_workflows(status=None, page=None, page_size=None)`: 워크플로우 목록 조회 (템플릿 제외)
- `get_workflow(workflow_id)`: 특정 워크플로우 조회
- `execute_workflow(workflow_id, parameters)`: 워크플로우 실행
- `get_workflow_status(workflow_id)`: 실행 상태 조회
- `sync_workflow_deployments(workflow_id)`: 배포 정보 동기화

#### 배포된 모델
- `get_deployed_models(workflow_id)`: 배포된 모델 목록 조회

#### 추론
- `inference(workflow_id, component_id, image_path, labels)`: 추론 수행

### 사용 예시

```python
from api_client import APIClient

client = APIClient("http://localhost:8000")

# 로그인
token = client.authenticate("surromind", "password")

# 템플릿 조회
templates = client.get_workflow_templates()

# 워크플로우 생성
workflow = client.clone_from_template(template_id=1, workflow_name="my-workflow")

# 추론 수행
result = client.inference(
    workflow_id=workflow["id"],
    component_id="model_component",
    image_path="image.jpg",
    labels=["a cat", "a dog"]
)
```

## 트러블슈팅

### 로그인 실패
- Backend API 서버가 실행 중인지 확인
- `.env` 파일의 `REST_API_URL` 확인
- 사용자 이름과 비밀번호 확인

### 템플릿 목록이 비어있음
- 워크플로우 템플릿을 생성했는지 확인
- `utils/create_template.py` 스크립트 실행
- Backend API에 모델이 등록되어 있는지 확인

### 워크플로우 실행 실패
- 워크플로우 상태가 ACTIVE인지 확인
- Kubeflow 환경이 정상 작동하는지 확인
- Backend 로그 확인

### 배포 정보가 동기화되지 않음
- Kubeflow Pipeline이 완료되었는지 확인
- KServe InferenceService가 생성되었는지 확인 (`kubectl get inferenceservice`)
- 네임스페이스 설정 확인

### 추론 실패
- 모델이 정상적으로 배포되었는지 확인
- 배포 상태가 "deployed"인지 확인
- 이미지 형식 확인 (JPEG/PNG 지원)
- 텍스트 레이블이 입력되었는지 확인
- InferenceService가 Ready 상태인지 확인

### Connection 에러
```bash
# Backend API 서버 상태 확인
curl http://localhost:8000/docs

# .env 파일 확인
cat app/config/.env
```

## 기술 스택

### UI 프레임워크
- **Gradio 4.x**: 웹 UI 프레임워크

### 라이브러리
- **requests**: HTTP 클라이언트
- **Pillow**: 이미지 처리
- **pydantic-settings**: 환경 설정 관리

### Backend 연동
- **FastAPI**: REST API 서버 (Backend)
- **Kubeflow Pipelines**: 워크플로우 오케스트레이션
- **KServe**: 모델 서빙 플랫폼

## 주요 변경사항 (v2.0.0)

### 개선사항
- ✅ Streamlit → Gradio 전환
- ✅ Quick Service Deployment 추가
- ✅ Playground 기능 강화
- ✅ 통합 API 클라이언트
- ✅ 세션 관리 개선
- ✅ 워크플로우 템플릿 관리

### 제거된 기능
- ❌ 수동 Inference Service URL 입력
- ❌ Streamlit 기반 UI

### 장점
1. **사용자 경험 개선**: 자동화된 배포 프로세스, 실시간 상태 모니터링
2. **개발자 경험 개선**: 모듈화된 코드, 타입 안정성
3. **기능 확장**: 템플릿 관리, 워크플로우 생명주기 관리

## 참고 자료

- Backend API 문서: `http://localhost:8000/docs` (FastAPI Swagger UI)
- Gradio 문서: https://www.gradio.app/docs
- KServe 문서: https://kserve.github.io/website/
- Kubeflow 문서: https://www.kubeflow.org/docs/
