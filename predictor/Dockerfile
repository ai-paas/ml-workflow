# Builder stage - uv 로 .venv 구성 + YOLOX editable 설치
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# uv 바이너리
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 의존성 잠금 파일만 먼저 복사해 캐시 활용
COPY predictor/pyproject.toml predictor/uv.lock ./

# .venv 생성 (프로젝트 자체는 설치하지 않음 - 코드는 런타임에 복사)
RUN uv sync --frozen --no-dev --no-install-project --no-cache \
    && find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# YOLOX 는 vendored editable 패키지라 lock 에 넣지 않고 .venv 에 직접 설치
COPY YOLOX ./YOLOX
RUN uv pip install --no-cache setuptools wheel \
    && uv pip install --no-build-isolation --no-cache -e ./YOLOX

# ──────────────────────────────────────────────────────────────────────────────
# Final stage - 실행 환경
# 런타임엔 CUDA 툴킷(nvcc)이 불필요하고 torch(cu128) wheel 이 CUDA 런타임·cudnn·cusparselt 를
# 자체 번들하므로 devel 대신 runtime 베이스를 쓴다(이미지 크기 대폭 절감).
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /app

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# cudnn·cusparselt 는 torch(cu128)가 nvidia-cudnn-cu12 / nvidia-cusparselt-cu12 로 번들하므로
# apt 로 설치하지 않는다. OpenCV 등 런타임에 필요한 시스템 라이브러리만 남긴다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# uv 가 만든 가상환경을 그대로 복사 (YOLOX editable .pth 포함)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 앱 코드 + YOLOX (editable 경로 /app/YOLOX 해석용)
COPY ./predictor/ .
COPY YOLOX ./YOLOX

# app 모듈과 YOLOX 모듈 import 경로
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "app.predictor", "--predictor_protocol", "v2"]
