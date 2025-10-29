# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pipenv 설치
RUN pip install --no-cache-dir pipenv

# Pipfile과 Pipfile.lock 복사
COPY Pipfile Pipfile.lock ./

# 의존성 설치 (시스템 패키지로 설치하여 가상환경 생성 안 함)
RUN pipenv install --system --deploy --ignore-pipfile

# 애플리케이션 코드 복사
COPY app/ ./app/

# 작업 디렉토리를 app으로 변경
WORKDIR /app/app

# 포트 노출
EXPOSE 7860

# Gradio 애플리케이션 실행
CMD ["python", "gradio_main.py"]
