FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

WORKDIR /app

# # 타임존 설정
# ENV TZ=Asia/Seoul
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    gcc \
    build-essential \
    sudo \
    && pip3 install pipenv


# Copy only the Pipfile and Pipfile.lock to leverage Docker cache
COPY Pipfile Pipfile.lock ./

# Install the dependencies via pipenv
RUN pipenv install --deploy --system
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04



# 타임존 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies without unattended-upgrades
# GPG 키 문제 해결을 위해 ca-certificates와 gnupg 먼저 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# CUDA keyring 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python

# Copy the dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy the FastAPI app code into the container
COPY . .

EXPOSE 8000

# Command to run the FastAPI app
CMD ["fastapi", "run", "app/main.py"]
