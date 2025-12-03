# 모델 및 데이터셋 API 사용 가이드

UI에서 모델과 데이터셋 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

---

# 모델 관련 API 사용 가이드

## 모델 워크플로우

### 1. 모델 타입/포맷/제공자 조회
**API**: `GET /api/v1/models/types`, `/formats`, `/providers`

- 모델 생성 전에 필요한 타입/포맷/제공자 정보 조회용
- 쿼리 파라미터 `type_name`, `format_name`, `provider_name`으로 특정 항목만 조회 가능
- 생략 시 전체 목록 반환

### 2. 모델 생성
**API**: `POST /api/v1/models`

- **Content-Type**: `multipart/form-data`
- 필수 파라미터: `name`, `repo_id`, `provider_id`, `type_id`, `format_id`
- 선택 파라미터: `description`, `task`, `parameter`, `sample_code`
- **HuggingFace 모델**:
  - `provider_id`: huggingface의 ID
  - `repo_id`: HuggingFace repository ID (예: "google/owlv2-base-patch16")
  - `file`: 불필요
- **커스텀 모델**:
  - `provider_id`: custom의 ID
  - `repo_id`: 모델 식별자
  - `file`: 모델 파일 (필수)
- 응답: 생성된 모델 정보 (ModelBriefReadSchema)
- **주의**: `parent_model_id`와 `model_registry_schema`는 내부 시스템용이므로 프론트엔드에서 전달하지 않음

### 3. 모델 목록 조회
**API**: `GET /api/v1/models`

- 쿼리 파라미터: `page`, `page_size` (선택사항)
- 생략 시 전체 데이터 조회 (최대 10000개)
- 응답: 모델 목록 (List[ModelBriefReadSchema])

### 4. 모델 상세 조회
**API**: `GET /api/v1/models/{model_id}`

- 경로 파라미터: `model_id`
- 응답: 모델 상세 정보 (ModelReadSchema)
- 부모/자식 모델 관계 포함

### 5. 모델 삭제
**API**: `DELETE /api/v1/models/{model_id}`

- 경로 파라미터: `model_id`
- 응답: `{ "success": true, "message": "모델이 성공적으로 삭제되었습니다." }`
- **주의**: 다른 엔티티(Experiment, WorkflowComponent, 자식 모델)에서 참조 중이면 삭제 불가 (400 에러)

## 모델 API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 모델 타입 조회 | `/api/v1/models/types` | GET | `type_name` (선택) |
| 모델 포맷 조회 | `/api/v1/models/formats` | GET | `format_name` (선택) |
| 모델 제공자 조회 | `/api/v1/models/providers` | GET | `provider_name` (선택) |
| 모델 생성 | `/api/v1/models` | POST | `name`, `repo_id`, `provider_id`, `type_id`, `format_id`, `file` (커스텀) |
| 모델 목록 조회 | `/api/v1/models` | GET | `page`, `page_size` (선택) |
| 모델 상세 조회 | `/api/v1/models/{model_id}` | GET | `model_id` (경로) |
| 모델 삭제 | `/api/v1/models/{model_id}` | DELETE | `model_id` (경로) |

## 모델 참고사항

- 모든 API는 Bearer Token 인증 필요
- HuggingFace 모델: `provider_id`가 huggingface ID와 일치해야 함
- 커스텀 모델: `provider_id`가 custom ID와 일치하고 `file` 필수
- 모델 이름에 "yolox" 포함 시 자동으로 학습 가능 모델로 설정
- 참조 관계가 있으면 삭제 불가
- 상세한 요청/응답 형식은 각 API의 docstring 참조

---

# 데이터셋 관련 API 사용 가이드

## 데이터셋 워크플로우

### 1. 데이터셋 파일 검증
**API**: `POST /api/v1/datasets/validate`

- **Content-Type**: `multipart/form-data`
- 필수 파라미터: `file` (ZIP 파일)
- 응답: `{ "is_valid": true, "message": "..." }`
- COCO128 형식 검증
- 등록 전에 검증하는 것을 권장

### 2. 데이터셋 생성
**API**: `POST /api/v1/datasets`

- **Content-Type**: `multipart/form-data`
- 필수 파라미터: `name`, `file` (ZIP 파일)
- 선택 파라미터: `description` (데이터셋 설명)
- 응답: 생성된 데이터셋 정보 (DatasetReadSchema)
  - `id`: 데이터셋 고유 ID
  - `name`: 데이터셋 이름
  - `description`: 데이터셋 설명 (optional)
  - `dataset_registry`: 데이터셋 레지스트리 정보
- MLflow에 자동 등록됨

### 3. 데이터셋 목록 조회
**API**: `GET /api/v1/datasets`

- 쿼리 파라미터: `page`, `page_size` (선택사항)
- 생략 시 전체 데이터 조회 (최대 10000개)
- 응답: 데이터셋 목록 (List[DatasetReadSchema])
  - 각 항목은 `id`, `name`, `description` (optional), `dataset_registry` 포함

### 4. 데이터셋 상세 조회
**API**: `GET /api/v1/datasets/{dataset_id}`

- 경로 파라미터: `dataset_id`
- 응답: 데이터셋 상세 정보 (DatasetReadSchema)
  - `id`: 데이터셋 고유 ID
  - `name`: 데이터셋 이름
  - `description`: 데이터셋 설명 (optional)
  - `dataset_registry`: 데이터셋 레지스트리 정보
- 데이터셋 레지스트리 정보 포함

### 5. 데이터셋 수정
**API**: `PUT /api/v1/datasets/{dataset_id}`

- 경로 파라미터: `dataset_id`
- 요청 본문: `name`, `description` (둘 다 선택적)
- 응답: 수정된 데이터셋 정보 (DatasetReadSchema)
- 수정하지 않을 필드는 요청에서 생략 가능

## 데이터셋 API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 데이터셋 파일 검증 | `/api/v1/datasets/validate` | POST | `file` |
| 데이터셋 생성 | `/api/v1/datasets` | POST | `name`, `file` (필수), `description` (선택) |
| 데이터셋 목록 조회 | `/api/v1/datasets` | GET | `page`, `page_size` (선택) |
| 데이터셋 상세 조회 | `/api/v1/datasets/{dataset_id}` | GET | `dataset_id` (경로) |
| 데이터셋 수정 | `/api/v1/datasets/{dataset_id}` | PUT | `name`, `description` (선택) |

## 데이터셋 참고사항

- 모든 API는 Bearer Token 인증 필요
- 파일 형식: COCO128 형식의 ZIP 파일
- 필수 구조: `annotations/instances_train2017.json`, `annotations/instances_val2017.json`, `train2017/`, `val2017/`
- 등록 전 검증 권장 (`/validate` API 사용)
- 데이터셋은 MLflow에 자동 등록됨
- 등록된 데이터셋은 실험(Experiment) 생성 시 사용 가능
- 상세한 요청/응답 형식은 각 API의 docstring 참조
