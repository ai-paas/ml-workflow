# 학습 관련 API 사용 가이드

UI에서 학습 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 학습 워크플로우

### 1. 학습 시작
**API**: `POST /api/v1/pipeline/training`

- 필수 파라미터: `model_id`, `dataset_id`
- 선택 파라미터: `train_name`, `description`, `gpus`, `batch_size`, `epochs`, `save_period`, `weight_decay`, `lr0`, `lrf`
- 응답: `{ "experiment_id": 123 }`
- 반환된 `experiment_id`를 다음 단계에서 사용

### 2. 학습 상태 조회 (폴링)
**API**: `GET /api/v1/pipeline/training/{experiment_id}/status`

- 학습이 완료될 때까지 주기적으로 호출 (예: 5초마다)
- 응답의 `status` 필드 확인:
  - `"RUNNING"`: 계속 폴링
  - `"FINISHED"`: 학습 완료 → 다음 단계로 진행
  - `"FAILED"`: 학습 실패 → 종료

### 3. 학습 완료된 정보 조회 (선택사항)
**API**: `GET /api/v1/experiments/{experiment_id}`

- 학습 완료 후 상세 정보 확인용
- 모델, 데이터셋, 하이퍼파라미터 등 전체 정보 반환

### 4. 학습된 모델 등록
**API**: `POST /api/v1/pipeline/model/registration`

- 필수 파라미터: `model_name`, `description`, `experiment_id`
- 학습이 완료된 실험에 대해서만 호출 가능
- 응답: `true` (성공) 또는 `false` (실패)

### 5. 학습 명과 설명 수정 (선택사항)
**API**: `PATCH /api/v1/experiments/{experiment_id}`

- 수정 가능한 필드: `name`, `description`만 가능
- 다른 필드(model_id, dataset_id, hyperparameters 등)는 수정 불가

## API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 학습 시작 | `/api/v1/pipeline/training` | POST | `model_id`, `dataset_id`, 하이퍼파라미터 |
| 학습 상태 조회 | `/api/v1/pipeline/training/{experiment_id}/status` | GET | `experiment_id` (경로) |
| 학습 완료 정보 조회 | `/api/v1/experiments/{experiment_id}` | GET | `experiment_id` (경로) |
| 학습된 모델 등록 | `/api/v1/pipeline/model/registration` | POST | `model_name`, `description`, `experiment_id` |
| 학습 명/설명 수정 | `/api/v1/experiments/{experiment_id}` | PATCH | `name`, `description` |

## 참고사항

- 모든 API는 Bearer Token 인증 필요
- 학습은 비동기 실행 → 상태 조회로 진행 상황 확인
- 학습 완료 후에만 모델 등록 가능
- 상세한 요청/응답 형식은 각 API의 docstring 참조
