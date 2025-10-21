# KServe Deployment Guide

## 개요
이 가이드는 워크플로우에서 KServe 모델 배포 시 DB 업데이트 프로세스를 설명합니다.

## 아키텍처

```
┌─────────────────────┐
│  Workflow Execute   │
│   (workflow.py)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Kubeflow Pipeline   │
│  (workflow_executor)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Model Deployment    │
│   Component         │
│ (KServe 배포 수행)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Backend API Call   │
│ POST /workflows/    │
│ {id}/components/    │
│ {id}/deployment-    │
│ status              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ KServeDeployment    │
│ Service & Repo      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  PostgreSQL DB      │
│ kserve_deployments  │
│      table          │
└─────────────────────┘
```

## 환경 설정

### 필수 설정

Kubeflow Pipeline 컴포넌트가 Backend API를 호출할 수 있도록 다음 설정이 필요합니다:

#### Settings.py 설정

```python
# backend/app/config/settings.py

# API 및 인증 설정
REST_API_URL: str = "http://backend-service.aipaas-mlops.svc.cluster.local:8000"
DEMO_PASSWORD: str = "your-demo-password"
```

#### 자동 파라미터 전달

`WorkflowExecutor.execute_workflow()` 메서드에서 자동으로 다음 파라미터를 설정합니다:

```python
parameters["rest_api_url"] = settings.REST_API_URL
parameters["restapi_username"] = "surromind"  # 고정 사용자명
parameters["restapi_password"] = settings.DEMO_PASSWORD
```

#### 토큰 발급

Kubeflow Pipeline 컴포넌트 내에서 자동으로 토큰을 발급받습니다:

```python
# model_deployment_component 내부
token_response = requests.post(
    f"{rest_api_url}/api/v1/authentications/token",
    data={"username": restapi_username, "password": restapi_password},
    timeout=10
)
auth_token = token_response.json().get("access_token")
```

## DB 업데이트 프로세스

### 1. 워크플로우 실행 시 (초기 상태)

```python
# workflow_executor.py
KServeDeploymentService.create_deployment(
    db=self.db,
    workflow_id=workflow.id,
    component_id=component.component_id,
    model_name=component.name,
)
```

**DB 레코드 생성:**
- `status`: `DEPLOYING`
- `service_name`: `pending-{workflow_id}-{component_id}`
- `service_hostname`: `pending`

### 2. KServe 배포 완료 후 (성공)

```python
# model_deployment_component 내부
requests.post(
    f"{backend_api_url}/api/v1/workflows/{workflow_id}/components/{component_id}/deployment-status",
    json={
        "service_name": "wf-9e10ae04-m1-abc123",
        "service_hostname": "wf-9e10ae04-m1-abc123.kubeflow-user-example-com.example.com",
        "model_name": "facebook-detr-resnet-50",
        "status": "deployed",
        "internal_url": "http://wf-9e10ae04-m1-abc123.kubeflow-user-example-com.svc.cluster.local",
        "error_message": None
    }
)
```

**DB 레코드 업데이트:**
- `status`: `DEPLOYED`
- `service_name`: 실제 KServe 서비스 이름
- `service_hostname`: 실제 서비스 호스트명
- `deployed_at`: 배포 완료 시간

### 3. KServe 배포 실패 시

```python
requests.post(
    f"{backend_api_url}/api/v1/workflows/{workflow_id}/components/{component_id}/deployment-status",
    json={
        "service_name": "failed-{workflow_id}-{component_id}",
        "service_hostname": "failed",
        "model_name": "facebook-detr-resnet-50",
        "status": "failed",
        "internal_url": None,
        "error_message": "Error message here"
    }
)
```

**DB 레코드 업데이트:**
- `status`: `FAILED`
- `error_message`: 실패 원인

## API 엔드포인트

### POST /api/v1/workflows/{workflow_id}/components/{component_id}/deployment-status

**Request Body:**
```json
{
  "service_name": "wf-9e10ae04-m1-abc123",
  "service_hostname": "wf-9e10ae04-m1-abc123.kubeflow-user-example-com.example.com",
  "model_name": "facebook-detr-resnet-50",
  "status": "deployed",
  "internal_url": "http://...",
  "error_message": null
}
```

**Response:**
```json
{
  "message": "Deployment status updated for component model-1",
  "deployment_info": {
    "service_name": "wf-9e10ae04-m1-abc123",
    "service_hostname": "wf-9e10ae04-m1-abc123.kubeflow-user-example-com.example.com",
    "model_name": "facebook-detr-resnet-50",
    "status": "DEPLOYED",
    "deployed_at": "2025-10-20T05:30:00.000000"
  }
}
```

## 조회 API

### GET /api/v1/workflows/{workflow_id}/models

배포된 모든 모델의 상태를 조회합니다.

**Response:**
```json
{
  "workflow_id": "9e10ae04-1bf0-4047-a4ad-f5099fc368f9",
  "backend_api_url": "http://10.10.30.154:80",
  "deployed_models": [
    {
      "component_id": "model-1",
      "model_id": 123,
      "model_name": "facebook/detr-resnet-50",
      "service_name": "wf-9e10ae04-m1-abc123",
      "gateway_url": "http://10.10.30.154:80",
      "service_hostname": "wf-9e10ae04-m1-abc123.kubeflow-user-example-com.example.com",
      "sanitized_model_name": "facebook-detr-resnet-50",
      "internal_url": "http://...",
      "status": "DEPLOYED",
      "deployed_at": "2025-10-20T05:30:00.000000",
      "error_message": null
    }
  ],
  "total": 1
}
```

## 트러블슈팅

### 1. DB 업데이트가 안 되는 경우

**원인:**
- Backend API URL이 잘못 설정됨
- 네트워크 연결 문제
- API 인증 실패

**해결:**
```bash
# Kubeflow Pipeline 로그 확인
kubectl logs -n kubeflow-user-example-com <pipeline-pod-name>

# Backend 서비스 연결 확인
kubectl run -n kubeflow-user-example-com test-curl --image=curlimages/curl --rm -it -- \
  curl http://backend-service.aipaas-mlops.svc.cluster.local:8000/health
```

### 2. 배포는 성공했지만 DB에 DEPLOYING 상태로 남아있는 경우

**원인:**
- API 호출이 실패했지만 KServe 배포는 성공

**해결:**
```python
# 수동으로 상태 업데이트
curl -X POST http://backend-service:8000/api/v1/workflows/{workflow_id}/components/{component_id}/deployment-status \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "actual-service-name",
    "service_hostname": "actual-hostname",
    "model_name": "model-name",
    "status": "deployed",
    "internal_url": "http://..."
  }'
```

## 보안 고려사항

1. **내부 네트워크 통신**: Backend API는 Kubernetes 내부 서비스 네트워크를 통해서만 접근
2. **인증**: 현재는 인증이 선택적이지만, 프로덕션에서는 Service Account 토큰 사용 권장
3. **네트워크 정책**: Kubeflow namespace에서 Backend namespace로의 통신을 Network Policy로 제어
