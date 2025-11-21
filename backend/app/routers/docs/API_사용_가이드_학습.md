# 학습 관련 API 사용 가이드

UI에서 학습 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 학습 워크플로우

### 1. 학습 시작
**API**: `POST /api/v1/pipeline/training`

- **Content-Type**: `application/json`
- 필수 파라미터:
  - `model_id` (int): 학습에 사용할 모델 ID
  - `dataset_id` (int): 학습에 사용할 데이터셋 ID
- 선택 파라미터:
  - `train_name` (str): 학습 실험 이름 (기본값: 빈 문자열)
  - `description` (str): 학습 실험 설명 (기본값: 빈 문자열)
  - `gpus` (str): 사용할 GPU 개수 (기본값: "1")
  - `batch_size` (str): 배치 크기 (기본값: "32")
  - `epochs` (str): 학습 에포크 수 (기본값: "5")
  - `save_period` (str): 모델 저장 주기 (기본값: "1")
  - `weight_decay` (str): 가중치 감쇠 계수 (기본값: "5e-4")
  - `lr0` (str): 초기 학습률 (기본값: "0.01")
  - `lrf` (str): 최종 학습률 (기본값: "0.05")
- 응답: `{ "experiment_id": 123 }`
  - `experiment_id`: 생성된 실험(Experiment)의 고유 ID
  - 실패 시 `null` 반환
- **주의**: 학습은 비동기로 실행되며, 즉시 완료되지 않습니다. 반환된 `experiment_id`를 다음 단계에서 사용합니다.

### 2. 학습 상태 조회 (폴링)
**API**: `GET /api/v1/pipeline/training/{experiment_id}/status`

- 경로 파라미터: `experiment_id` (int) - 실험 ID
- 학습이 완료될 때까지 주기적으로 호출 (예: 5초마다)
- 응답의 `status` 필드 확인:
  - `"RUNNING"`: 학습 진행 중 → 계속 폴링
  - `"FINISHED"`: 학습 완료 → 다음 단계로 진행
  - `"FAILED"`: 학습 실패 → 종료
- 응답 필드:
  - `status`: 학습 상태
  - `start_time`: 학습 시작 시각 (밀리초 단위 타임스탬프)
  - `end_time`: 학습 종료 시각 (밀리초 단위 타임스탬프)
  - `max_epoch`: 설정된 최대 에포크 수
  - `current_epoch`: 현재 진행 중인 에포크
  - `loss_history`: 손실(loss) 히스토리
  - `epoch_history`: 에포크 히스토리
  - `average_precision_50_history`: AP@50 히스토리
  - `average_precision_75_history`: AP@75 히스토리
  - `best_average_precision_history`: 최고 평균 정밀도 히스토리
  - `average_precision_50_95_history`: mAP@0.5:0.95 히스토리

### 3. 학습 완료된 정보 조회 (선택사항)
**API**: `GET /api/v1/experiments/{experiment_id}`

- 경로 파라미터: `experiment_id` (int) - 실험 ID
- 학습 완료 후 상세 정보 확인용
- 응답: 실험 상세 정보 (ExperimentReadSchema)
  - `id`: 실험 ID
  - `name`: 실험 이름
  - `description`: 실험 설명
  - `reference_model_id`: 참조 모델 ID
  - `dataset_id`: 데이터셋 ID
  - `kubeflow_run_id`: Kubeflow 파이프라인 실행 ID
  - `mlflow_run_id`: MLflow 실행 ID
  - `status`: 실험 상태
  - `reference_model`: 참조 모델 상세 정보
  - `dataset`: 데이터셋 상세 정보
  - `hyperparameters`: 하이퍼파라미터 목록

### 4. 학습된 모델 등록
**API**: `POST /api/v1/pipeline/model/registration`

- **Content-Type**: `application/json`
- 필수 파라미터:
  - `model_name` (str): 등록할 모델 이름
  - `description` (str): 모델 설명
  - `experiment_id` (int): 학습 실험 ID
- 응답: `true` (성공) 또는 `false` (실패)
- **주의**: 학습이 완료된 실험에 대해서만 호출 가능합니다. 파이프라인 실행은 비동기로 진행되며, 즉시 완료되지 않습니다.

### 5. 학습 명과 설명 수정 (선택사항)
**API**: `PATCH /api/v1/experiments/{experiment_id}`

- 경로 파라미터: `experiment_id` (int) - 실험 ID
- **Content-Type**: `application/json`
- 요청 바디:
  - `name` (str, optional): 새로운 실험 이름
  - `description` (str, optional): 새로운 실험 설명
- 응답: 수정된 실험 정보 (ExperimentReadSchema)
- **주의**: 수정 가능한 필드는 `name`, `description`만 가능합니다. 다른 필드(model_id, dataset_id, hyperparameters 등)는 수정 불가합니다.

### 6. 실험 삭제 (선택사항)
**API**: `DELETE /api/v1/experiments/{experiment_id}`

- 경로 파라미터: `experiment_id` (int) - 실험 ID
- 응답: `{ "message": "실험 {experiment_id}가 성공적으로 삭제되었습니다." }`
- **주의**: 실험 삭제 시 MLflow artifacts와 S3 object도 함께 삭제됩니다. 삭제된 실험은 복구할 수 없으므로 주의하세요.

## API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 학습 시작 | `/api/v1/pipeline/training` | POST | `model_id`, `dataset_id`, 하이퍼파라미터 (요청 바디) |
| 학습 상태 조회 | `/api/v1/pipeline/training/{experiment_id}/status` | GET | `experiment_id` (경로) |
| 학습 완료 정보 조회 | `/api/v1/experiments/{experiment_id}` | GET | `experiment_id` (경로) |
| 학습된 모델 등록 | `/api/v1/pipeline/model/registration` | POST | `model_name`, `description`, `experiment_id` (요청 바디) |
| 학습 명/설명 수정 | `/api/v1/experiments/{experiment_id}` | PATCH | `experiment_id` (경로), `name`, `description` (요청 바디) |
| 실험 삭제 | `/api/v1/experiments/{experiment_id}` | DELETE | `experiment_id` (경로) |

## 참고사항

- 모든 API는 Bearer Token 인증 필요
- 학습은 비동기 실행 → 상태 조회로 진행 상황 확인
- 학습 완료 후에만 모델 등록 가능
- 실험 삭제 시 MLflow artifacts와 S3 object도 함께 삭제됨
- 학습 상태 조회는 주기적으로 폴링하여 진행 상황을 모니터링할 수 있음
- 상세한 요청/응답 형식은 각 API의 docstring 참조
