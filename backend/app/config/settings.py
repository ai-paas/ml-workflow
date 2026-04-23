import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_config_dir = Path(__file__).resolve().parent
_base_env = _config_dir / ".env"
load_dotenv(_base_env)

_profile = os.environ.get("ENV", "").strip()
if _profile:
    _profile_path = _config_dir / f".env.{_profile}"
    if _profile_path.is_file():
        load_dotenv(_profile_path, override=True)
    else:
        logger.warning(
            "ENV=%s 이지만 프로파일 파일이 없습니다: %s — .env 만 사용합니다.",
            _profile,
            _profile_path,
        )


class Settings(BaseSettings):
    """애플리케이션 설정 클래스

    환경 변수를 통해 설정값을 로드하며, 필수 설정값들의 유효성을 검사합니다.
    """

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=None,
        env_file_encoding="utf-8",
    )

    # Kubeflow 설정
    KUBEFLOW_ENDPOINT: str = Field(..., description="Kubeflow API 엔드포인트 URL")
    KUBEFLOW_USERNAME: str = Field(..., description="Kubeflow 사용자명")
    KUBEFLOW_PASSWORD: str = Field(..., description="Kubeflow 비밀번호")
    KUBEFLOW_NAMESPACE: str = Field(..., description="Kubeflow 네임스페이스")
    KUBEFLOW_EXPERIMENT_NAME: str = Field(..., description="Kubeflow 실험명")

    TRAIN_IMAGE_URL: str = Field(..., description="학습 이미지 URL")
    INFER_IMAGE_URL: str = Field(..., description="추론 이미지 URL")

    # 데이터베이스 설정
    DB_TYPE: str = Field(..., description="데이터베이스 타입 (예: postgresql)")
    DB_NAME: str = Field(..., description="데이터베이스 이름")
    DB_USER: str = Field(..., description="데이터베이스 사용자명")
    DB_PASSWORD: str = Field(..., description="데이터베이스 비밀번호")
    DB_HOST: str = Field(..., description="데이터베이스 호스트")
    DB_PORT: str = Field(..., description="데이터베이스 포트")

    # MLflow 설정
    MLFLOW_TRACKING_URI: str = Field(..., description="MLflow 추적 서버 URI")
    MLFLOW_TRACKING_USERNAME: str = Field(..., description="MLflow 추적 서버 사용자명")
    MLFLOW_TRACKING_PASSWORD: str = Field(..., description="MLflow 추적 서버 비밀번호")
    MLFLOW_EXPERIMENT_NAME: str = Field(..., description="MLflow 실험명")

    # MLflow S3 설정
    MLFLOW_S3_ENDPOINT_URL: str = Field(..., description="MLflow S3 엔드포인트 URL")
    MLFLOW_S3_ACCESS_KEY_ID: str = Field(..., description="MLflow S3 액세스 키 ID")
    MLFLOW_S3_SECRET_ACCESS_KEY: str = Field(..., description="MLflow S3 시크릿 액세스 키")
    MLFLOW_S3_BUCKET: str = Field(
        ...,
        description="MLflow 아티팩트 저장용 S3 버킷 이름. \
        모델 파일과 실험 결과를 저장하는 데 사용됩니다.",
    )

    # 데이터셋 업로드 설정
    DATASET_MAX_UPLOAD_SIZE_MB: int = Field(default=500, description="데이터셋 최대 업로드 크기 (MB)")

    # S3 호환 스토리지 설정 (데이터셋 저장용)
    S3_ENDPOINT: str = Field(..., description="S3 호환 스토리지 엔드포인트 URL")
    S3_ACCESS_KEY: str = Field(..., description="S3 액세스 키")
    S3_SECRET_KEY: str = Field(..., description="S3 시크릿 키")
    S3_BUCKET: str = Field(..., description="S3 버킷 이름")

    # 데이터셋 스토리지 설정
    DATASET_STORAGE_TYPE: str = Field(
        default="s3",
        description="데이터셋 저장소 유형 (s3 | hubconnect)",
    )

    # Hub-Connect API 설정 (DATASET_STORAGE_TYPE=hubconnect 시 사용)
    DATALAKE_API_URL: str = Field(default="", description="Hub-Connect API 기본 URL")
    DATALAKE_API_USERNAME: str = Field(default="", description="Hub-Connect API 인증 사용자명")
    DATALAKE_API_PASSWORD: str = Field(default="", description="Hub-Connect API 인증 비밀번호")
    DATALAKE_BUCKET_NAME: str = Field(default="aipaas-datasets", description="데이터레이크 버킷 이름")

    # API 및 인증 설정
    REST_API_URL: str = Field(..., description="REST API 기본 URL")
    DEMO_PASSWORD: str = Field(..., description="데모용 비밀번호")
    LOGIN_SECRET_KEY: str = Field(..., description="로그인 세션 암호화 키")
    INTERNAL_API_KEY: str = Field(
        default="",
        description="내부 전용 API 인증에 사용할 키 (KFP 컴포넌트 콜백용). 서버 시작 시 SHA-256 해시로 검증.",
    )

    # KServe / 워크플로 서빙 공통
    DEFAULT_TRY_GPU: bool = Field(
        default=True,
        description="사전정의 서빙 메타 기준 GPU 경로 우선 시도(임베딩 task는 항상 CPU). "
        "Ollama·KServe(HF 등) 워크플로 서빙 플래너가 동일하게 참고(§7.4).",
    )
    SERVING_DEFAULT_GPU_VRAM_BYTES: int = Field(
        default=16 * 1024 * 1024 * 1024,
        description="노드 라벨·오버라이드 없을 때 GPU 1장당 VRAM 바이트 기본값(§7.5). 인벤토리 기반 k 산정에도 사용.",
    )
    SERVING_NODE_NAMES: str = Field(
        default="",
        description="서빙 스케줄러가 볼 노드 화이트리스트(쉼표 구분 metadata.name). 비면 Ready 노드 전체(§7.3).",
    )
    SERVING_INCLUDE_CONTROL_PLANE_NODES: bool = Field(
        default=False,
        description="True일 때만 인벤토리에 control-plane/master 역할 노드 포함. "
        "기본 False — 서빙 자원 집계·노드 선택에서 제외(단일 노드 클러스터 테스트 시에만 True 권장).",
    )
    SERVING_NODE_VRAM_OVERRIDES_JSON: str = Field(
        default="{}",
        description='노드명→GPU 카드 VRAM GiB 정수 JSON. 예: {"gpu-8g-01":8} (§7.10)',
    )
    SERVING_PIN_SELECTED_NODE: bool = Field(
        default=True,
        description="플래너가 고른 노드를 nodeName/nodeSelector로 고정할지. False면 요청만 두고 스케줄러에 위임(§7.7).",
    )
    KSERVE_GATEWAY_URL: str = Field(
        default="",
        description="KServe Istio Gateway URL(외부 접근용). 비어 있으면 public_url 미제공(§2.6).",
    )
    REMOTE_SERVING_API_URL: str = Field(
        default="",
        description="§6: 원격 LLM 단일 추론 베이스 URL. REMOTE 배포 시 remote_api_url에 저장·추론 시 사용(스펙·인증은 §6.5 미정).",
    )
    REMOTE_SERVING_MODEL_MAP: str = Field(
        default="{}",
        description='§6: 플랫폼 model.repo_id(키) → REMOTE 서버 모델명(값) JSON. 예: {"org/llama3":"gateway-model-a"}',
    )

    KUBEFLOW_IMAGE_PULL_SECRET: str = Field(
        default="harbor",
        description="Kubeflow Pipeline에서 사용할 imagePullSecret 이름 \
(private registry 인증용)",
    )

    # Milvus 설정
    MILVUS_DB_HOST: str = Field(..., description="Milvus 데이터베이스 호스트")
    MILVUS_DB_PORT: str = Field(..., description="Milvus 데이터베이스 포트")
    MILVUS_DB_USERNAME: str = Field(..., description="Milvus 데이터베이스 사용자명")
    MILVUS_DB_PASSWORD: str = Field(..., description="Milvus 데이터베이스 비밀번호")
    MILVUS_DB_NAME: str = Field(..., description="Milvus 데이터베이스 이름")
    MILVUS_ADMIN_PORT: str = Field(default="8000", description="Milvus Admin 포트")

    # Harbor 설정
    HARBOR_URL: str = Field(..., description="Harbor 레지스트리 URL")
    HARBOR_USERNAME: str = Field(..., description="Harbor 사용자명")
    HARBOR_PASSWORD: str = Field(..., description="Harbor 비밀번호")
    HARBOR_REPOSITORY: str = Field(..., description="Harbor 리포지토리명")
    BACKEND_PROJECT_NAME: str = Field(..., description="Harbor 백엔드 프로젝트명")
    TRAIN_PROJECT_NAME: str = Field(..., description="Harbor 학습 프로젝트명")
    INFERENCE_PROJECT_NAME: str = Field(..., description="Harbor 추론 프로젝트명")

    # 기타 설정

    @field_validator("MLFLOW_S3_BUCKET")
    @classmethod
    def validate_mlflow_s3_bucket(cls, v: str) -> str:
        """MLflow S3 버킷 이름 유효성 검사"""
        if not v or not v.strip():
            raise ValueError(
                "MLFLOW_S3_BUCKET은 비어있을 수 없습니다. \
                    MLflow 아티팩트 저장을 위한 S3 버킷 이름을 설정해주세요."
            )

        return v.strip()

    @field_validator("DB_PORT")
    @classmethod
    def validate_db_port(cls, v: str) -> str:
        """데이터베이스 포트 유효성 검사"""
        if not v or not v.strip():
            raise ValueError("DB_PORT는 비어있을 수 없습니다.")

        return v.strip()

    @property
    def get_db_uri(self) -> str:
        """Environment variables로부터 DB 정보를 받아와 URI를 반환."""
        return f"{self.DB_TYPE}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


@lru_cache
def get_settings():
    return Settings()
