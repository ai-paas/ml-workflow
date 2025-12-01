# ML-Workflow

ML-Workflow는 kubeflow 기반으로 하는 ML pipeline를 관리하고 다양한 종류의 앱을 손쉽게 빌드할수있는 시스템의 Backend를 담당하고 있습니다. 다양한 모델의 학습/평가/배포 과정을 No-Code 기반으로 수행할 수 있습니다.

## Tech Stack

![Python Icon](https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI Icon](https://img.shields.io/badge/fastapi-009688?style=flat&logo=fastapi&logoColor=white)
![HuggingFace Icon](https://img.shields.io/badge/huggingface-fcbf29?style=flat&logo=huggingface&logoColor=white)
![MLFlow Icon](https://img.shields.io/badge/mlflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![MariaDB Icon](https://img.shields.io/badge/mariadb-003545?style=flat&logo=mariadb&logoColor=white)


# System Architecture and Workflow

ML-Workflow의 시스템 아키텍처 및 흐름도는 아래 링크에서 확인 가능합니다.
https://surromind.atlassian.net/jira/software/c/projects/PAAS/boards/600/backlog?selectedIssue=PAAS-250


# DB Migration

alembic.ini 파일이 존재하는 경로에서 아래 명령어 실행
```shell
alembic upgrade head
```

# DB Data Initialization
root directorty에서 migration할 버전을 확인하고 마이그레이션
`PYTHONPATH=backend/app python -m backend.app.config.db.data_initializer -v v1`
`PYTHONPATH=backend/app python -m backend.app.config.db.data_initializer -v v2`
`PYTHONPATH=backend/app python -m backend.app.config.db.data_initializer -v all`

## 🚀 개발환경 설정 (backend server)
- 대상 디렉토리 : backend

 `setup-dev.sh` 스크립트를 사용하여 개발 환경을 설정할 수 있습니다. (backend folder에 대해 가상환경을 구축하고 pre commit 세팅)
 이 스크립트는 **필수 패키지 설치**와 **pre-commit hook 설정**을 자동으로 처리합니다.

```shell
sh setup-dev.sh
pipenv shell
```

1. **`setup-dev.sh`를 실행**하면, 아래와 같은 작업이 자동으로 처리됩니다:
    - `pipenv`를 사용하여 가상환경을 생성하고, 개발 환경에 필요한 패키지를 설치합니다.
    - `pre-commit`을 설치하고, `.git/hooks` 폴더에 `prepare-commit-msg`를 설정하여 커밋 메시지 형식을 자동으로 검사하도록 설정합니다.

2. `pipenv shell` 명령어를 사용해 가상환경을 활성화시키고 개발을 수행합니다.

3. 이후, **pre-commit hook**이 자동으로 설정되어 커밋 시 코드 스타일과 규칙을 검사하게 됩니다.

#### 기타 디렉토리
- ml_workflow_ui
- predictor
- train_eval
에 대해서는 별도로 pipenv 가상환경을 구축하여 기동테스트 진행요망.
