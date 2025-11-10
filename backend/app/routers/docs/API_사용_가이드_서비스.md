# 서비스 관련 API 사용 가이드

UI에서 서비스 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 서비스 API 활용 방법

### 1. 서비스 생성

**API**: `POST /api/v1/services`

- 요청 바디: `ServiceCreateRequest`
  - 필수: `name` (1-255자, 고유값)
  - 선택: `description`, `tags` (List[str])
- 응답: 생성된 서비스 정보 (ServiceBriefSchema)
  - `id`: 서비스 UUID
  - `name`: 서비스 이름
  - `description`: 서비스 설명
  - `tags`: 태그 목록
  - `creator_id`: 생성자 ID
  - `workflow_count`: 연결된 워크플로우 수 (초기값: 0)
- **주의**: 서비스 이름은 고유해야 함

### 2. 서비스 목록 조회

**API**: `GET /api/v1/services`

- 쿼리 파라미터: `page`, `page_size`, `creator_id` (선택사항)
- 생략 시 전체 서비스 조회 (최대 10000개)
- 응답: 서비스 목록 (ServiceListResponse)
  - `total`: 전체 서비스 수
  - `items`: 서비스 목록 (각 서비스의 기본 정보와 연결된 워크플로우 수 포함)

### 3. 서비스 상세 조회

**API**: `GET /api/v1/services/{service_id}`

- 경로 파라미터: `service_id` (UUID)
- 응답: 서비스 상세 정보 (ServiceDetailSchema)
  - 기본 정보: id, name, description, tags, creator 등
  - `workflows`: 연결된 워크플로우 목록 (전체 정보 포함)
  - `monitoring_data`: 최근 1시간 모니터링 메트릭
    - `total_metrics`: 전체 서비스 메트릭
      - message_count: 총 메시지 수
      - active_users: 활성 사용자 수
      - token_usage: 토큰 사용량
      - avg_interaction_count: 평균 사용자 상호작용 수
      - response_time_ms: 평균 응답 시간(ms)
      - error_count: 오류 수
      - success_rate: 성공률(%)
    - `workflow_metrics`: 워크플로우별 메트릭 리스트
    - `period_start`, `period_end`: 집계 기간

### 4. 서비스 정보 수정

**API**: `PUT /api/v1/services/{service_id}`

- 경로 파라미터: `service_id` (UUID)
- 요청 바디: `ServiceUpdateRequest`
  - `name` (선택): 새로운 서비스 이름
  - `description` (선택): 새로운 설명 (null로 제거 가능)
  - `tags` (선택): 새로운 태그 목록 (기존 태그 대체)
- 응답: 수정된 서비스 정보 (ServiceBriefSchema)
- **주의**: 서비스 이름 변경 시 중복 검사 수행

### 5. 서비스 삭제

**API**: `DELETE /api/v1/services/{service_id}`

- 경로 파라미터: `service_id` (UUID)
- 응답: 204 No Content
- **주의**:
  - 연결된 워크플로우는 삭제되지 않고 연결만 해제됨 (service_id가 null로 설정)
  - 모니터링 데이터는 보존됨
  - 삭제는 되돌릴 수 없음

### 6. 서비스에 워크플로우 연결하는 방법

**API**: `PUT /api/v1/workflows/{workflow_id}`

- 경로 파라미터: `workflow_id` (워크플로우 UUID)
- 요청 바디: `WorkflowUpdateRequest`
  - `service_id` (str, 선택): 연결할 서비스 ID (UUID)
  - 기타 필드: `name`, `description`, `category`, `status`, `workflow_definition` (선택)
- 응답: 업데이트된 워크플로우 정보 (WorkflowReadSchema)
- **주의**:
  - `service_id`를 null로 설정하면 서비스 연결 해제
  - 여러 워크플로우를 하나의 서비스에 연결 가능
  - 서비스에 연결된 워크플로우는 모니터링 데이터가 자동 기록됨

## 서비스 API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 서비스 생성 | `/api/v1/services` | POST | `name` (필수), `description`, `tags` |
| 서비스 목록 조회 | `/api/v1/services` | GET | `page`, `page_size`, `creator_id` (선택) |
| 서비스 상세 조회 | `/api/v1/services/{service_id}` | GET | `service_id` (경로) |
| 서비스 수정 | `/api/v1/services/{service_id}` | PUT | `service_id` (경로), `name`, `description`, `tags` |
| 서비스 삭제 | `/api/v1/services/{service_id}` | DELETE | `service_id` (경로) |
| 워크플로우에 서비스 연결 | `/api/v1/workflows/{workflow_id}` | PUT | `workflow_id` (경로), `service_id` (바디) |

## 서비스 참고사항

- 모든 API는 Bearer Token 인증 필요
- 서비스는 워크플로우를 그룹화하고 모니터링하는 최상위 단위
- 하나의 서비스에 여러 워크플로우 연결 가능
- 서비스에 연결된 워크플로우는 통합 모니터링 가능
- 서비스 삭제 시 연결된 워크플로우는 자동으로 연결 해제됨 (워크플로우는 유지)
- 상세한 요청/응답 형식은 각 API의 docstring 참조
