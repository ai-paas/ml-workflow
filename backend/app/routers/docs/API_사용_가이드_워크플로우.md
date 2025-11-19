# 워크플로우 관련 API 사용 가이드

UI에서 워크플로우 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 워크플로우 API 활용 방법

### 1. 워크플로우 템플릿에서 워크플로우를 복제하여 배포하는 방법

#### 1-0. 워크플로우 템플릿 생성 (선택사항 - 템플릿이 없는 경우)

**API**: `POST /api/v1/workflows/templates`

- 요청 바디: `WorkflowTemplateCreateRequest`
  - 필수: `name`, `workflow_definition`
  - 선택: `description`, `category`
  - `workflow_definition`:
    - `components`: 컴포넌트 목록 (START, END, MODEL, KNOWLEDGE_BASE)
      - **MODEL 타입**: `model_id` (필수), `prompt_id` (선택)
      - **KNOWLEDGE_BASE 타입**: `knowledge_base_id` (필수)
    - `connections`: 컴포넌트 간 연결 정보
- 응답: 생성된 템플릿 정보 (WorkflowTemplateBriefSchema)
- **주의**:
  - 템플릿은 실행할 수 없고 복사용만 가능
  - `is_template`은 항상 `true`로 자동 설정됨 (요청에서 지정 불가)
  - `service_id`는 템플릿에 포함되지 않음 (템플릿은 서비스에 연결되지 않음)
  - `template_id`는 템플릿에 포함되지 않음

**요청 예시**:
```json
{
  "name": "객체 탐지 템플릿",
  "description": "YOLOX 모델을 사용한 객체 탐지 워크플로우 템플릿",
  "category": "Object Detection",
  "workflow_definition": {
    "components": [
      {
        "name": "시작 노드",
        "type": "START"
      },
      {
        "name": "YOLOX 모델",
        "type": "MODEL",
        "model_id": 1,
        "prompt_id": 1
      },
      {
        "name": "종료 노드",
        "type": "END"
      }
    ],
    "connections": [
      {
        "source_component_type": "START",
        "target_component_type": "MODEL"
      },
      {
        "source_component_type": "MODEL",
        "target_component_type": "END"
      }
    ]
  }
}
```

**Knowledge Base와 프롬프트를 포함한 템플릿 예시**:
```json
{
  "name": "의료 진단 템플릿",
  "description": "Knowledge Base와 LLM을 활용한 의료 진단 워크플로우 템플릿",
  "category": "Medical",
  "workflow_definition": {
    "components": [
      {
        "name": "시작 노드",
        "type": "START"
      },
      {
        "name": "의료 지식 베이스",
        "type": "KNOWLEDGE_BASE",
        "knowledge_base_id": 1
      },
      {
        "name": "의료 LLM 모델",
        "type": "MODEL",
        "model_id": 2,
        "prompt_id": 1
      },
      {
        "name": "종료 노드",
        "type": "END"
      }
    ],
    "connections": [
      {
        "source_component_type": "START",
        "target_component_type": "KNOWLEDGE_BASE"
      },
      {
        "source_component_type": "KNOWLEDGE_BASE",
        "target_component_type": "MODEL"
      },
      {
        "source_component_type": "MODEL",
        "target_component_type": "END"
      }
    ]
  }
}
```

#### 1-1. 템플릿 목록 조회
**API**: `GET /api/v1/workflows/templates`

- 쿼리 파라미터: `page`, `page_size`, `category` (선택사항)
- 생략 시 전체 템플릿 조회 (최대 10000개)
- 응답: 템플릿 목록 (WorkflowTemplateListSchema)

#### 1-2. 템플릿 상세 조회 (선택사항)
**API**: `GET /api/v1/workflows/templates/{template_id}`

- 경로 파라미터: `template_id`
- 응답: 템플릿 상세 정보 (WorkflowTemplateReadSchema)
- 컴포넌트, 연결 등 전체 구조 확인용

#### 1-3. 템플릿 수정
**API**: `PUT /api/v1/workflows/templates/{template_id}`

- 경로 파라미터: `template_id` (템플릿 UUID)
- 요청 바디: `WorkflowTemplateUpdateRequest`
  - `name` (선택): 새 템플릿 이름
  - `description` (선택): 새 설명
  - `category` (선택): 새 카테고리
  - `status` (선택): 새 상태 (DRAFT/ACTIVE/ERROR)
  - `workflow_definition` (선택): 새 템플릿 구조
    - `components`: 컴포넌트 목록 (START, END, MODEL, KNOWLEDGE_BASE)
    - `connections`: 컴포넌트 간 연결 정보
- 응답: 업데이트된 템플릿 정보 (WorkflowTemplateReadSchema)
- **주의**:
  - 제공된 필드만 업데이트됨 (부분 업데이트 가능)
  - `workflow_definition` 제공 시 기존 컴포넌트/연결은 삭제 후 재생성됨
  - `service_id`는 템플릿에 포함되지 않음 (요청에서 제외, 항상 null로 유지)
  - 템플릿은 실행할 수 없고 복사용으로만 사용 가능
  - `usage_count`는 동적으로 계산됨 (파생된 워크플로우 수)

**요청 예시**:
```json
{
  "name": "수정된 객체 탐지 템플릿",
  "description": "업데이트된 YOLOX 모델을 사용한 객체 탐지 워크플로우 템플릿",
  "category": "Object Detection",
  "workflow_definition": {
    "components": [
      {
        "name": "시작 노드",
        "type": "START"
      },
      {
        "name": "YOLOX 모델 v2",
        "type": "MODEL",
        "model_id": 2,
        "prompt_id": 2
      },
      {
        "name": "종료 노드",
        "type": "END"
      }
    ],
    "connections": [
      {
        "source_component_type": "START",
        "target_component_type": "MODEL"
      },
      {
        "source_component_type": "MODEL",
        "target_component_type": "END"
      }
    ]
  }
}
```

#### 1-4. 템플릿으로부터 워크플로우 복제
**API**: `POST /api/v1/workflows/templates/{template_id}/clone`

- 경로 파라미터: `template_id` (템플릿 UUID)
- 쿼리 파라미터:
  - `workflow_name` (필수): 새로 생성할 워크플로우 이름
  - `service_id` (선택): 연결할 서비스 ID
- 응답: 생성된 워크플로우 정보 (WorkflowReadSchema)
- 상태는 **DRAFT로 시작** (실행 후 파이프라인 완료 시 ACTIVE로 자동 변경됨)
- **주의**:
  - 템플릿의 모든 컴포넌트와 연결이 복사됨
  - 생성된 워크플로우는 템플릿과 독립적으로 동작
  - `template_id`가 자동으로 기록됨 (원본 템플릿 추적 가능)
  - DRAFT 상태에서도 바로 실행 가능 (파이프라인 완료 시 자동으로 ACTIVE로 변경됨)

**요청 예시**:
```
POST /api/v1/workflows/templates/123e4567-e89b-12d3-a456-426614174000/clone?workflow_name=My%20Workflow%20from%20Template&service_id=456e7890-e12b-34d5-a678-901234567890
```

#### 1-5. 워크플로우 배포 (실행)
**API**: `POST /api/v1/workflows/{workflow_id}/execute`

- 경로 파라미터: `workflow_id`
- 요청 바디: `{ "parameters": {} }` (선택)
- 응답: `{ "workflow_id": "...", "kubeflow_run_id": "...", "status": "...", "message": "..." }`
- **주의**:
  - DRAFT 상태에서도 실행 가능 (파이프라인 완료 시 자동으로 ACTIVE로 변경됨)
  - ERROR 상태인 경우만 실행 불가
  - 파이프라인 완료 시 워크플로우 상태가 자동으로 ACTIVE로 변경됨

**참고**: 워크플로우를 직접 생성하는 방법은 "2. 워크플로우를 직접 생성하여 배포하는 방법" 섹션을 참조하세요.

### 2. 워크플로우를 직접 생성하여 배포하는 방법

#### 2-1. 컴포넌트 타입 조회 (선택사항)
**API**: `GET /api/v1/workflows/component-types`

- 사용 가능한 컴포넌트 타입 확인용
- 응답: START, END, MODEL, KNOWLEDGE_BASE 타입 정보
- 각 타입별 설명:
  - **START**: 워크플로우 시작점
  - **END**: 워크플로우 종료점
  - **MODEL**: ML 모델 실행 노드 (model_id 필수, prompt_id 선택)
  - **KNOWLEDGE_BASE**: 지식 베이스 검색 노드 (knowledge_base_id 필수)

#### 2-2. 워크플로우 직접 생성
**API**: `POST /api/v1/workflows`

- 요청 바디: `WorkflowCreateRequest`
  - 필수: `name`
  - 선택: `description`, `category`, `service_id`, `workflow_definition`
  - `workflow_definition`:
    - `components`: 컴포넌트 목록 (START, END, MODEL, KNOWLEDGE_BASE)
      - **MODEL 타입**: `model_id` (필수), `prompt_id` (선택)
      - **KNOWLEDGE_BASE 타입**: `knowledge_base_id` (필수)
    - `connections`: 컴포넌트 간 연결 정보
- 응답: 생성된 워크플로우 정보 (WorkflowBaseSchema)
- 상태는 DRAFT로 시작
- **주의**:
  - `is_template`은 항상 `false`로 자동 설정됨 (요청에서 지정 불가, 템플릿 생성은 `/templates` API 사용)
  - 템플릿으로부터 생성하려면 `/workflows/templates/{template_id}/clone` API 사용

**요청 예시**:
```json
{
  "name": "My Object Detection Workflow",
  "description": "객체 탐지 워크플로우",
  "category": "Object Detection",
  "workflow_definition": {
    "components": [
      {
        "name": "시작 노드",
        "type": "START"
      },
      {
        "name": "YOLOX 모델",
        "type": "MODEL",
        "model_id": 1,
        "prompt_id": 1
      },
      {
        "name": "종료 노드",
        "type": "END"
      }
    ],
    "connections": [
      {
        "source_component_type": "START",
        "target_component_type": "MODEL"
      },
      {
        "source_component_type": "MODEL",
        "target_component_type": "END"
      }
    ]
  }
}
```

**Knowledge Base와 프롬프트를 포함한 워크플로우 생성 예시**:
```json
{
  "name": "의료 진단 워크플로우",
  "description": "Knowledge Base 검색 후 LLM으로 진단하는 워크플로우",
  "category": "Medical",
  "workflow_definition": {
    "components": [
      {
        "name": "시작 노드",
        "type": "START"
      },
      {
        "name": "의료 지식 베이스",
        "type": "KNOWLEDGE_BASE",
        "knowledge_base_id": 1
      },
      {
        "name": "의료 LLM 모델",
        "type": "MODEL",
        "model_id": 2,
        "prompt_id": 1
      },
      {
        "name": "종료 노드",
        "type": "END"
      }
    ],
    "connections": [
      {
        "source_component_type": "START",
        "target_component_type": "KNOWLEDGE_BASE"
      },
      {
        "source_component_type": "KNOWLEDGE_BASE",
        "target_component_type": "MODEL"
      },
      {
        "source_component_type": "MODEL",
        "target_component_type": "END"
      }
    ]
  }
}
```

#### 2-3. 워크플로우 배포 (실행)
**API**: `POST /api/v1/workflows/{workflow_id}/execute`

- 경로 파라미터: `workflow_id`
- 요청 바디: `{ "parameters": {} }` (선택)
- 응답: `{ "workflow_id": "...", "kubeflow_run_id": "...", "status": "...", "message": "..." }`
- **주의**: DRAFT 상태에서도 실행 가능 (파이프라인 완료 시 자동으로 ACTIVE로 변경됨)

### 3. 워크플로우의 현재 배포상태를 확인하는 방법

**API**: `GET /api/v1/workflows/{workflow_id}/status`

- 경로 파라미터: `workflow_id`
- 응답:
  - `workflow_id`: 워크플로우 UUID
  - `status`: 워크플로우 상태 (DRAFT/ACTIVE/ERROR)
  - `kubeflow_run_id`: Kubeflow 실행 ID
  - `deployment_status`: 각 모델의 배포 상태 리스트
    - `component_id`: 컴포넌트 ID (추론 API에서 사용 가능)

### 4. 워크플로우의 상세 정보를 조회하는 방법

**API**: `GET /api/v1/workflows/{workflow_id}`

- 경로 파라미터: `workflow_id`
- 응답: 워크플로우 상세 정보 (WorkflowReadSchema)
  - 컴포넌트, 연결, 배포 상태 등 전체 정보 포함
  - 배포된 모델 정보, 엔드포인트 URL 포함
  - **참고**: `components` 배열에서 `type`이 "MODEL"인 컴포넌트의 `id`를 추론 API의 `component_id`로 사용 가능

### 5. 워크플로우의 전체 리스트를 조회하는 방법

**API**: `GET /api/v1/workflows`

- 쿼리 파라미터: `page`, `page_size`, `creator_id`, `service_id`, `status` (선택사항)
- 생략 시 전체 데이터 조회 (최대 10000개)
- 응답: 워크플로우 목록 (WorkflowListSchema)
- **주의**: 템플릿은 제외됨 (템플릿 조회는 `/templates` API 사용)

### 6. 워크플로우 정보 수정하는 방법

**API**: `PUT /api/v1/workflows/{workflow_id}`

- 경로 파라미터: `workflow_id` (워크플로우 UUID)
- 요청 바디: `WorkflowUpdateRequest`
  - `name` (선택): 새 워크플로우 이름
  - `description` (선택): 새 설명
  - `category` (선택): 새 카테고리
  - `status` (선택): 새 상태 (DRAFT/ACTIVE/ERROR)
  - `service_id` (선택): 연결할 서비스 ID (UUID)
    - `workflow_definition` (선택): 새 워크플로우 구조
    - `components`: 컴포넌트 목록 (START, END, MODEL, KNOWLEDGE_BASE)
      - **MODEL 타입**: `model_id` (필수), `prompt_id` (선택)
      - **KNOWLEDGE_BASE 타입**: `knowledge_base_id` (필수)
    - `connections`: 컴포넌트 간 연결 정보
- 응답: 업데이트된 워크플로우 정보 (WorkflowReadSchema)
- **주의**:
  - 제공된 필드만 업데이트됨 (부분 업데이트 가능)
  - `workflow_definition` 제공 시 기존 컴포넌트/연결은 삭제 후 재생성됨
  - 템플릿 수정은 `/workflows/templates/{template_id}` API 사용

### 7. 워크플로우를 아예 삭제하거나, KServe 배포서비스만 삭제하는 방법

#### 7-1. 워크플로우 완전 삭제 (2단계 프로세스)

**단계 1**: 삭제 시작
**API**: `DELETE /api/v1/workflows/{workflow_id}`

- 경로 파라미터: `workflow_id`
- 응답: `{ "cleanup_run_id": "...", "status": "cleanup_in_progress", ... }`
- KServe 리소스 정리 파이프라인 시작

**단계 2**: 삭제 완료 확인
**API**: `POST /api/v1/workflows/{workflow_id}/finalize-deletion`

- 경로 파라미터: `workflow_id`
- 쿼리 파라미터: `run_id` (cleanup_run_id)
- 응답: `{ "status": "completed", "deleted_from_db": true, ... }`
- 파이프라인 완료 확인 후 DB에서 삭제

#### 7-2. KServe 배포서비스만 삭제 (워크플로우는 유지)

**단계 1**: 리소스 정리 시작
**API**: `POST /api/v1/workflows/{workflow_id}/cleanup`

- 경로 파라미터: `workflow_id`
- 응답: `{ "cleanup_run_id": "...", "status": "cleanup_in_progress", ... }`
- KServe InferenceService 삭제 파이프라인 시작

**단계 2**: 정리 완료 확인
**API**: `POST /api/v1/workflows/{workflow_id}/finalize-cleanup`

- 경로 파라미터: `workflow_id`
- 쿼리 파라미터: `run_id` (cleanup_run_id)
- 응답: `{ "status": "completed", "workflow_updated": true, ... }`
- 파이프라인 완료 확인 후 워크플로우 상태를 DRAFT로 변경

### 8. 워크플로우 테스트 방법

#### 8-1. RAG 워크플로우 테스트

**API**: `POST /api/v1/workflows/{workflow_id}/test/rag`

- 경로 파라미터:
  - `workflow_id`: 워크플로우 UUID
- 요청 바디: `multipart/form-data`
  - `text` (required): 검색 쿼리 및 LLM 입력 텍스트
    - Knowledge Base가 있으면 검색 쿼리로 사용
    - LLM 모델의 입력 텍스트로도 사용
- 응답: `WorkflowTestResponse`
  - `workflow_id`: 워크플로우 UUID
  - `execution_order`: 실행된 컴포넌트 ID 순서
  - `results`: 각 컴포넌트 실행 결과 목록
    - Knowledge Base 컴포넌트: 검색 결과 포함
    - LLM 모델 컴포넌트: LLM 응답 포함
  - `final_result`: 마지막 LLM 모델의 결과
- **주의**:
  - 워크플로우는 ACTIVE 상태여야 함 (배포 완료)
  - 워크플로우에 최소 하나의 LLM MODEL 컴포넌트 또는 KNOWLEDGE_BASE 컴포넌트가 있어야 함
  - Knowledge Base 컴포넌트는 선택 사항 (있으면 검색 후 결과를 LLM에 전달)
  - 지식베이스 검색 결과는 자동으로 LLM 모델에 전달됨
  - **프롬프트 처리 방식**:
    - `prompt_id`가 설정된 경우:
      - 프롬프트에 `context` 변수가 있으면: 프롬프트의 `{context}` 또는 `{{context}}` 위치에 자동 치환
      - 프롬프트에 `context` 변수가 없어도: `[참고자료]` 태그와 함께 별도의 system 메시지로 추가
    - `prompt_id`가 없는 경우: `[참고자료]` 태그와 함께 system 메시지로 추가
  - 각 컴포넌트의 실행 결과는 `results` 배열에 순서대로 포함됨

**사용 예시**:
```bash
# RAG 워크플로우 테스트 (Knowledge Base 검색 + LLM 추론 자동 실행)
POST /api/v1/workflows/{workflow_id}/test/rag
Content-Type: multipart/form-data

text=의료진단에 대해 알려주세요
```

#### 8-2. ML 워크플로우 테스트

**API**: `POST /api/v1/workflows/{workflow_id}/test/ml`

- 경로 파라미터:
  - `workflow_id`: 워크플로우 UUID
- 요청 바디: `multipart/form-data`
  - `image` (required): 이미지 파일 (JPEG, PNG, GIF, WebP)
    - ODM 모델 추론용
- 응답: `WorkflowTestResponse`
  - `workflow_id`: 워크플로우 UUID
  - `execution_order`: 실행된 컴포넌트 ID 순서
  - `results`: 각 컴포넌트 실행 결과 목록
    - ODM 모델 컴포넌트: 객체 탐지 결과 포함
  - `final_result`: 마지막 ODM 모델의 결과
- **주의**:
  - 워크플로우는 ACTIVE 상태여야 함 (배포 완료)
  - 워크플로우에 최소 하나의 ODM MODEL 컴포넌트가 있어야 함
  - KNOWLEDGE_BASE 컴포넌트는 포함될 수 없음 (ML 워크플로우는 ODM만 지원)
  - 각 컴포넌트의 실행 결과는 `results` 배열에 순서대로 포함됨
  - 모든 추론 요청은 ServiceMonitoring 테이블에 자동 기록됨 (서비스와 연결된 경우)

**사용 예시**:
```bash
# ML 워크플로우 테스트 (ODM 추론 자동 실행)
POST /api/v1/workflows/{workflow_id}/test/ml
Content-Type: multipart/form-data

image=@test_image.jpg
```

### 9. 워크플로우 템플릿 삭제하는 방법

**API**: `DELETE /api/v1/workflows/templates/{template_id}`

- 경로 파라미터: `template_id`
- 응답: 204 No Content
- **주의**: 파생된 워크플로우가 있으면 삭제 불가 (400 에러)

## 워크플로우 API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 컴포넌트 타입 조회 | `/api/v1/workflows/component-types` | GET | - |
| 템플릿 생성 | `/api/v1/workflows/templates` | POST | `name`, `workflow_definition` (필수), `description`, `category` (선택) |
| 템플릿 목록 조회 | `/api/v1/workflows/templates` | GET | `page`, `page_size`, `category` (선택) |
| 템플릿 상세 조회 | `/api/v1/workflows/templates/{template_id}` | GET | `template_id` (경로) |
| 템플릿 수정 | `/api/v1/workflows/templates/{template_id}` | PUT | `template_id` (경로), `name`, `description`, `category`, `status`, `workflow_definition` (선택) |
| 템플릿으로부터 복제 | `/api/v1/workflows/templates/{template_id}/clone` | POST | `template_id` (경로), `workflow_name` (쿼리) |
| 워크플로우 생성 | `/api/v1/workflows` | POST | `name`, `workflow_definition` |
| 워크플로우 목록 조회 | `/api/v1/workflows` | GET | `page`, `page_size`, `creator_id`, `service_id`, `status` (선택) |
| 워크플로우 상세 조회 | `/api/v1/workflows/{workflow_id}` | GET | `workflow_id` (경로) |
| 워크플로우 수정 | `/api/v1/workflows/{workflow_id}` | PUT | `workflow_id` (경로), `service_id` 등 |
| 워크플로우 실행 (배포) | `/api/v1/workflows/{workflow_id}/execute` | POST | `workflow_id` (경로), `parameters` (선택) |
| 워크플로우 상태 조회 | `/api/v1/workflows/{workflow_id}/status` | GET | `workflow_id` (경로) |
| 배포된 모델 목록 조회 | `/api/v1/workflows/{workflow_id}/models` | GET | `workflow_id` (경로) |
| RAG 워크플로우 테스트 | `/api/v1/workflows/{workflow_id}/test/rag` | POST | `workflow_id` (경로), `text` (필수) |
| ML 워크플로우 테스트 | `/api/v1/workflows/{workflow_id}/test/ml` | POST | `workflow_id` (경로), `image` (파일, 필수) |
| 워크플로우 리소스 정리 | `/api/v1/workflows/{workflow_id}/cleanup` | POST | `workflow_id` (경로) |
| 리소스 정리 완료 확인 | `/api/v1/workflows/{workflow_id}/finalize-cleanup` | POST | `workflow_id` (경로), `run_id` (쿼리) |
| 워크플로우 삭제 시작 | `/api/v1/workflows/{workflow_id}` | DELETE | `workflow_id` (경로) |
| 워크플로우 삭제 완료 확인 | `/api/v1/workflows/{workflow_id}/finalize-deletion` | POST | `workflow_id` (경로), `run_id` (쿼리) |
| 템플릿 삭제 | `/api/v1/workflows/templates/{template_id}` | DELETE | `template_id` (경로) |

## 워크플로우 참고사항

- 모든 API는 Bearer Token 인증 필요
- 워크플로우 상태:
  - `DRAFT`: 임시저장 상태 (실행 가능, 파이프라인 완료 시 ACTIVE로 자동 변경)
  - `ACTIVE`: 활성 상태 (실행 완료, 배포 완료)
  - `ERROR`: 오류 상태 (실행 불가)
- 워크플로우 상태 변경:
  - DRAFT 상태에서도 실행 가능 (파이프라인 완료 시 자동으로 ACTIVE로 변경됨)
  - ERROR 상태인 경우만 실행 불가 (오류 수정 후 실행)
  - 파이프라인 완료 시 워크플로우 상태가 자동으로 ACTIVE로 변경됨
- 배포는 비동기로 진행되므로 상태 조회로 진행 상황 확인
- 삭제와 리소스 정리는 2단계 프로세스 (파이프라인 완료 확인 필요)
- 서비스 연결 방법은 서비스 API 사용 가이드 참조
- 상세한 요청/응답 형식은 각 API의 docstring 참조
- **컴포넌트 타입별 필수 필드**:
  - **MODEL**: `model_id` (필수), `prompt_id` (선택 - Ollama 모델인 경우 프롬프트 적용)
  - **KNOWLEDGE_BASE**: `knowledge_base_id` (필수)
- **프롬프트 사용 방법**:
  - MODEL 컴포넌트에 `prompt_id`를 설정하면 Ollama 모델 추론 시 프롬프트가 자동 적용됨
  - `prompt_id`가 설정된 경우:
    - 프롬프트에 `context` 변수가 있으면: 프롬프트의 `{context}` 또는 `{{context}}` 위치에 자동 치환
    - 프롬프트에 `context` 변수가 없어도: `[참고자료]` 태그와 함께 별도의 system 메시지로 추가
  - `prompt_id`가 없고 `search_text`가 있으면: `[참고자료]` 태그와 함께 system 메시지로 추가됨
- **Knowledge Base 연동 방법**:
  1. `/test/rag` API 호출
  2. Knowledge Base 검색과 LLM 추론이 자동으로 순차 실행됨
  3. 검색 결과가 자동으로 LLM 모델에 전달됨
