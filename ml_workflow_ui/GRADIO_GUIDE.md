# Gradio UI 사용 가이드

## 개요

이 문서는 Gradio 기반 ML Workflow UI의 사용 방법을 설명합니다.

## 사전 준비

### 1. 환경 설정

`app/config/.env` 파일 생성:

```bash
REST_API_URL=http://localhost:8000
```

### 2. 패키지 설치

```bash
cd ml_workflow_ui
pipenv install
```

### 3. Backend API 서버 실행

Backend API 서버가 실행 중이어야 합니다.

### 4. 워크플로우 템플릿 생성 (최초 1회)

```bash
cd ml_workflow_ui/app
python utils/create_template.py --password YOUR_PASSWORD --model-id MODEL_ID
```

예시:
```bash
python utils/create_template.py --password mypassword --model-id 1
```

## UI 실행

### 방법 1: 스크립트 사용

```bash
cd ml_workflow_ui
./run_gradio.sh
```

### 방법 2: 직접 실행

```bash
cd ml_workflow_ui/app
python gradio_main.py
```

브라우저에서 `http://localhost:7860` 접속

## 사용 방법

### 1. 로그인

1. **로그인 탭** 선택
2. 사용자 이름 입력 (기본값: surromind)
3. 비밀번호 입력
4. **로그인** 버튼 클릭
5. 로그인 성공 시 상태 표시가 "✅ 로그인됨: surromind"로 변경됨

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
3. 배포가 완료될 때까지 주기적으로 확인

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
4. "워크플로우 ID" 필드에 ID 입력 (목록에서 복사)

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
6. 실행 상태 및 결과 확인

#### 3.4 결과 확인

- **실행 상태**: 추론 성공 여부 및 메타 정보
- **결과 이미지**: 객체가 감지된 이미지 (바운딩 박스 포함)
- **상세 결과 (JSON)**: 전체 추론 결과 (좌표, 점수 등)

## 워크플로우 템플릿 정보

### DETR Object Detection Template

- **모델**: facebook/detr-resnet-50
- **태스크**: Object Detection
- **프레임워크**: Transformers (Hugging Face)
- **입력**:
  - 이미지 (JPEG/PNG)
  - 텍스트 레이블 리스트 (감지할 객체)
- **출력**:
  - 객체 감지 결과 (바운딩 박스, 점수, 레이블)
  - 시각화된 이미지

## 트러블슈팅

### 로그인 실패

- Backend API 서버가 실행 중인지 확인
- `.env` 파일의 `REST_API_URL` 확인
- 사용자 이름과 비밀번호 확인

### 템플릿 목록이 비어있음

- 워크플로우 템플릿을 생성했는지 확인
- `utils/create_template.py` 스크립트 실행

### 워크플로우 실행 실패

- 워크플로우 상태가 ACTIVE인지 확인
- Kubeflow 환경이 정상 작동하는지 확인
- Backend 로그 확인

### 배포 정보가 동기화되지 않음

- Kubeflow Pipeline이 완료되었는지 확인
- KServe InferenceService가 생성되었는지 확인 (kubectl)
- 네임스페이스 설정 확인

### 추론 실패

- 모델이 정상적으로 배포되었는지 확인
- 배포 상태가 "deployed"인지 확인
- 이미지 형식 확인 (JPEG/PNG)
- 텍스트 레이블이 입력되었는지 확인

## 예시: 전체 워크플로우

1. **로그인**
   - 사용자: surromind
   - 비밀번호: (입력)

2. **Quick Service Deployment**
   - 템플릿 ID: 1 (DETR Object Detection Template)
   - 워크플로우 이름: my-cat-detector
   - 워크플로우 생성 → 실행
   - 배포 상태 확인 (2-3분 소요)
   - 배포 정보 동기화

3. **Playground**
   - 워크플로우 선택: my-cat-detector
   - Component ID: detr_model
   - 이미지: cat.jpg 업로드
   - 텍스트: "a cat", "a remote control"
   - 추론 실행
   - 결과 확인

## 참고

- Backend API 문서: `/docs` (FastAPI Swagger UI)
- Kubeflow Dashboard: Kubeflow UI
- KServe 문서: [KServe Documentation](https://kserve.github.io/website/)
