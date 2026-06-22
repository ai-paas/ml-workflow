# Builder stage - uv 로 .venv 구성 + YOLOX editable 설치
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# uv 바이너리
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 의존성 잠금 파일만 먼저 복사해 캐시 활용
COPY train_eval/pyproject.toml train_eval/uv.lock ./

# .venv 생성 (프로젝트 자체는 설치하지 않음 - 코드는 런타임에 복사)
RUN uv sync --frozen --no-dev --no-install-project --no-cache \
    && find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# YOLOX 는 vendored editable 패키지라 lock 에 넣지 않고 .venv 에 직접 설치
COPY YOLOX ./YOLOX
RUN uv pip install --no-cache setuptools wheel \
    && uv pip install --no-build-isolation --no-cache -e ./YOLOX

# ──────────────────────────────────────────────────────────────────────────────
# Final stage - 실행 환경
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /app

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -f cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# uv 가 만든 가상환경을 그대로 복사 (YOLOX editable .pth 포함)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 앱 코드 + YOLOX (editable 경로 /app/YOLOX 해석용)
COPY ./train_eval/ .
COPY YOLOX ./YOLOX

# app 모듈과 YOLOX 모듈 import 경로
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "app.train_eval"]
