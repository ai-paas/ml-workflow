# ML Workflow UI

Gradio 기반의 ML Workflow 관리 및 추론 인터페이스입니다.

> 🎉 **v2.0.0**: Streamlit에서 Gradio로 완전히 전환되었습니다. ([변경 이력](CHANGELOG.md))

## 기능

### 1. Quick Service Deployment
- 워크플로우 템플릿 목록 조회
- 템플릿으로부터 새로운 워크플로우 생성
- 워크플로우 실행 (모델 배포)
- 배포 상태 모니터링
- 배포 정보 동기화

### 2. Playground
- 배포된 워크플로우 목록 조회
- 배포된 모델 선택
- 이미지 업로드 및 추론 실행
- 추론 결과 시각화

## 설치

```bash
cd ml_workflow_ui
pipenv install
```

## 실행

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

## 환경 설정

`app/config/.env` 파일 생성:

```
REST_API_URL=http://your-backend-api-url
```

## 사용 방법

### Quick Service Deployment

1. **로그인 탭**에서 인증
2. **Quick Service Deployment 탭** 이동
3. 템플릿 목록에서 원하는 템플릿 선택
4. 워크플로우 이름 입력 후 생성
5. 생성된 워크플로우 실행 (배포)
6. 배포 상태 확인 및 동기화

### Playground

1. **Playground 탭** 이동
2. 워크플로우 목록 새로고침
3. 워크플로우 ID 입력 후 모델 로드
4. 추론할 모델의 Component ID 선택
5. 이미지 업로드
6. 텍스트 레이블 입력 (예: "a cat", "a dog")
7. 추론 실행

## 구조

```
ml_workflow_ui/
├── app/
│   ├── config/
│   │   └── settings.py               # 환경 설정
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── quick_deployment.py       # Quick Deployment UI
│   │   └── playground.py             # Playground UI
│   ├── utils/
│   │   ├── __init__.py
│   │   └── create_template.py        # 템플릿 생성 스크립트
│   ├── api_client.py                 # 통합 API 클라이언트
│   └── gradio_main.py                # Gradio 메인 앱
├── run_gradio.sh                      # 실행 스크립트
├── Pipfile                            # 의존성
├── README.md                          # 이 문서
├── GRADIO_GUIDE.md                    # 상세 사용 가이드
├── MIGRATION_GUIDE.md                 # 마이그레이션 가이드
├── SUMMARY.md                         # 구현 요약
└── CHANGELOG.md                       # 변경 이력
```

## API 클라이언트

`api_client.py`는 Backend API와 통신하는 클라이언트 클래스를 제공합니다:

- `authenticate()`: 로그인
- `get_workflow_templates()`: 템플릿 목록 조회
- `clone_from_template()`: 템플릿으로부터 워크플로우 생성
- `execute_workflow()`: 워크플로우 실행
- `get_deployed_models()`: 배포된 모델 조회
- `inference()`: 추론 수행

## 주의사항

- Backend API 서버가 실행 중이어야 합니다
- Kubeflow 환경이 설정되어 있어야 합니다
- 워크플로우 템플릿이 미리 생성되어 있어야 합니다
