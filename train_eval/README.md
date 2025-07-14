# YOLOX 학습 시스템

YOLOX 패키지 0.3.0을 사용한 객체 검출 모델 학습 시스템입니다.

## 주요 특징

- **YOLOX 공식 패키지**: YOLOX 0.3.0 패키지의 공식 실험 설정 사용
- **VOC 데이터셋 지원**: Pascal VOC 형식의 데이터셋 학습
- **MLflow 통합**: 실험 추적 및 모델 버전 관리
- **분산 학습**: 다중 GPU 지원
- **Docker 지원**: 컨테이너화된 학습 환경
- **FP16 지원**: 혼합 정밀도 학습으로 메모리 효율성 개선

## 시스템 요구사항

- Python 3.7+
- CUDA 11.0+
- PyTorch 1.9+
- Docker (선택사항)

## 설치 방법

### 1. 환경 설정
```bash
# 프로젝트 디렉토리로 이동
cd train_eval

# 환경 설정 스크립트 실행
chmod +x setup_yolox.sh
./setup_yolox.sh
```

### 2. Docker 사용 (권장)
```bash
# Docker 컨테이너 빌드 및 실행
docker-compose up --build
```

## 데이터 준비

### VOC 형식 데이터셋 구조
```
data/
├── VOC2007/
│   ├── Annotations/
│   │   ├── image1.xml
│   │   └── image2.xml
│   ├── ImageSets/
│   │   └── Main/
│   │       ├── train.txt
│   │       └── val.txt
│   └── JPEGImages/
│       ├── image1.jpg
│       └── image2.jpg
└── VOC2012/
    └── (동일한 구조)
```

## 사용법

### 기본 학습 명령
```bash
python train_eval.py \
  --train_name my_yolox_train \
  --model_name yolox_s \
  --model_uri s3://models/yolox_s \
  --mlflow_tracking_uri http://mlflow:5000 \
  --mlflow_experiment_name yolox_experiments \
  --mlflow_s3_endpoint_url http://minio:9000 \
  --aws_access_key_id minioadmin \
  --aws_secret_access_key minioadmin \
  --dataset_artifact_uri s3://datasets/voc_data \
  --restapi_url http://api:8000 \
  --restapi_username admin \
  --restapi_password password \
  --exp_file yolox.exp.yolox_voc.yolox_voc_s \
  --batch_size 8 \
  --devices 1 \
  --fp16
```

### 실험 파일 옵션

YOLOX 패키지의 공식 실험 파일을 사용할 수 있습니다:

- `yolox.exp.yolox_voc.yolox_voc_s`: VOC 데이터셋용 YOLOX-S 모델
- `yolox.exp.yolox_voc.yolox_voc_m`: VOC 데이터셋용 YOLOX-M 모델  
- `yolox.exp.yolox_voc.yolox_voc_l`: VOC 데이터셋용 YOLOX-L 모델

또는 로컬 실험 파일 경로를 지정할 수 있습니다:
```bash
--exp_file /path/to/custom_exp.py
```

### 다중 GPU 학습
```bash
python train_eval.py \
  [기본 파라미터들...] \
  --devices 4 \
  --batch_size 32 \
  --fp16
```

### 학습 재개
```bash
python train_eval.py \
  [기본 파라미터들...] \
  --resume \
  --ckpt /path/to/checkpoint.pth
```

## 파라미터 설명

### 필수 파라미터
- `--train_name`: 학습 실행명
- `--model_name`: 모델명
- `--model_uri`: 모델 저장 URI
- `--mlflow_tracking_uri`: MLflow 서버 URI
- `--mlflow_experiment_name`: MLflow 실험명
- `--mlflow_s3_endpoint_url`: S3 엔드포인트 URL
- `--aws_access_key_id`: AWS 액세스 키
- `--aws_secret_access_key`: AWS 시크릿 키
- `--dataset_artifact_uri`: 데이터셋 URI
- `--restapi_url`: REST API URL
- `--restapi_username`: API 사용자명
- `--restapi_password`: API 비밀번호
- `--exp_file`: YOLOX 실험 파일 경로 또는 모듈명

### 선택 파라미터
- `--batch_size`: 배치 크기 (기본값: 8)
- `--devices`: 사용할 GPU 수 (기본값: 1)
- `--exp_name`: 실험명 (기본값: yolox_s)
- `--resume`: 학습 재개 여부
- `--ckpt`: 체크포인트 파일 경로
- `--start_epoch`: 시작 에폭 (기본값: 0)
- `--cache`: 이미지 캐싱 사용 여부
- `--fp16`: FP16 혼합 정밀도 사용 여부
- `--logger_type`: 로거 타입 (기본값: tensorboard)

## 출력 결과

### 학습 결과물
- `YOLOX_outputs/`: 모델 체크포인트 및 로그
- `tensorboard_logs/`: TensorBoard 로그
- MLflow: 실험 메트릭 및 모델 아티팩트

### 모델 체크포인트
- `best_ckpt.pth`: 최고 성능 모델
- `last_epoch_ckpt.pth`: 마지막 에폭 모델
- `epoch_*_ckpt.pth`: 각 에폭별 모델

## 모니터링

### TensorBoard 실행
```bash
tensorboard --logdir=YOLOX_outputs
```

### MLflow UI 접속
```
http://localhost:5000
```

## 커스텀 실험 설정

필요에 따라 커스텀 실험 파일을 생성할 수 있습니다:

```python
# custom_exp.py
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 3  # 클래스 수
        self.depth = 0.33     # 모델 깊이
        self.width = 0.50     # 모델 너비
        self.exp_name = "custom_yolox"
        self.data_dir = "data"
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        # 기타 설정...
```

## 문제 해결

### 메모리 부족 오류
- 배치 크기 줄이기: `--batch_size 4`
- FP16 사용: `--fp16`
- 이미지 캐싱 끄기: 캐시 옵션 제거

### 데이터 로딩 오류
- 데이터 경로 확인
- VOC 형식 준수 확인
- 어노테이션 파일 유효성 검사

### Docker 메모리 문제
- `docker-compose.yml`에서 `shm_size: '4g'` 설정 확인
- 호스트 메모리 확인

## 성능 최적화

### 학습 속도 향상
- 다중 GPU 사용: `--devices 4`
- FP16 사용: `--fp16`
- 이미지 캐싱: `--cache`

### 메모리 효율성
- 적절한 배치 크기 설정
- 데이터 로더 워커 수 조정
- 그래디언트 체크포인팅 활용

## 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다. 