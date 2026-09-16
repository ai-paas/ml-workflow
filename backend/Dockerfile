FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY backend/pyproject.toml backend/uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project --no-cache \
    && find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV TZ=Asia/Seoul

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gnupg wget \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -f cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12 \
        python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY ./backend/ .

EXPOSE 8000

ENV UVICORN_TIMEOUT_KEEP_ALIVE=600

CMD ["fastapi", "run", "app/main.py"]
