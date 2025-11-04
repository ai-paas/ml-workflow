# Builder stage - 컴파일과 패키지 설치
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

WORKDIR /app

# 빌드에 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    gcc \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# pipenv 설치
RUN pip3 install --no-cache-dir pipenv


# Copy only the Pipfile and Pipfile.lock to leverage Docker cache
COPY predictor/Pipfile predictor/Pipfile.lock ./

# dependencies 설치 (LD_LIBRARY_PATH 비워서 PyTorch가 번들된 CUDA 사용)
RUN pipenv install --deploy --system


# Copy YOLOX folder for installation
COPY YOLOX ./YOLOX

# Final stage - 실행 환경
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /app

# 타임존 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies without unattended-upgrades
RUN apt-get update && apt-get install -y wget \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    libcusparselt0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python

# Copy the dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the FastAPI app code into the container
COPY ./predictor/ .
COPY YOLOX ./YOLOX

# YOLOX를 Python 패키지로 설치 (editable install이므로 final stage에서도 필요)
RUN cd YOLOX && pip install --no-build-isolation --use-pep517 -v -e .

# PYTHONPATH 설정 (app 모듈과 YOLOX 모듈 찾기 위해)
ENV PYTHONPATH=/app

# 작업 디렉토리를 app으로 변경하고 python 모듈 실행
WORKDIR /app

ENTRYPOINT ["python", "-m", "app.predictor", "--predictor_protocol", "v2"]
