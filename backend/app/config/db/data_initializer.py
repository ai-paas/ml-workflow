import argparse
import traceback

from config.settings import get_settings
from db.models.experiment import HyperparameterType
from db.models.model import ModelFormat, ModelProvider, ModelType
from db.models.user import UserModel
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from .rdb_data import (
    HYPERPARAMETER_TYPE_DATA,
    MODEL_FORMAT_DATA,
    MODEL_FORMAT_DATA_2,
    MODEL_FORMAT_DATA_3,
    MODEL_FORMAT_DATA_4,
    MODEL_PROVIDER_DATA,
    MODEL_PROVIDER_DATA_2,
    MODEL_TYPE_DATA,
    MODEL_TYPE_DATA_2,
    MODEL_TYPE_DATA_3,
    USER_DATA,
)

settings = get_settings()


def initialize_user(db) -> None:
    """
    사용자 데이터를 초기화합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(UserModel.__table__).values(USER_DATA))
    print(f"UserModel 테이블에 {len(USER_DATA)}개 데이터 삽입 완료")


def initialize_model_format(db) -> None:
    """
    모델 포맷 데이터를 초기화합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelFormat.__table__).values(MODEL_FORMAT_DATA))
    print(f"ModelFormat 테이블에 {len(MODEL_FORMAT_DATA)}개 데이터 삽입 완료")


def initialize_model_provider(db) -> None:
    """
    모델 제공자 데이터를 초기화합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelProvider.__table__).values(MODEL_PROVIDER_DATA))
    print(f"ModelProvider 테이블에 {len(MODEL_PROVIDER_DATA)}개 데이터 삽입 완료")


def initialize_model_type(db) -> None:
    """
    모델 타입 데이터를 초기화합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelType.__table__).values(MODEL_TYPE_DATA))
    print(f"ModelType 테이블에 {len(MODEL_TYPE_DATA)}개 데이터 삽입 완료")


def add_model_format_2(db) -> None:
    """
    모델 포맷 데이터를 추가합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelFormat.__table__).values(MODEL_FORMAT_DATA_2))
    print(f"ModelFormat 테이블에 {len(MODEL_FORMAT_DATA_2)}개 데이터 삽입 완료")


def add_model_format_3(db) -> None:
    """
    TensorFlow와 YOLOX 모델 포맷 데이터를 추가합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelFormat.__table__).values(MODEL_FORMAT_DATA_3))
    print(f"ModelFormat 테이블에 {len(MODEL_FORMAT_DATA_3)}개 데이터 삽입 완료")


def add_model_format_4(db) -> None:
    """
    GGUF 모델 포맷 데이터를 추가합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelFormat.__table__).values(MODEL_FORMAT_DATA_4))
    print(f"ModelFormat 테이블에 {len(MODEL_FORMAT_DATA_4)}개 데이터 삽입 완료")


def add_model_provider_2(db) -> None:
    """
    Ollama 모델 제공자 데이터를 추가합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelProvider.__table__).values(MODEL_PROVIDER_DATA_2))
    print(f"ModelProvider 테이블에 {len(MODEL_PROVIDER_DATA_2)}개 데이터 삽입 완료")


def add_model_type_2(db) -> None:
    """
    LLM 모델 타입 데이터를 추가합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelType.__table__).values(MODEL_TYPE_DATA_2))
    print(f"ModelType 테이블에 {len(MODEL_TYPE_DATA_2)}개 데이터 삽입 완료")


def add_model_type_3(db) -> None:
    """
    Embedding 모델 타입 데이터를 추가합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(ModelType.__table__).values(MODEL_TYPE_DATA_3))
    print(f"ModelType 테이블에 {len(MODEL_TYPE_DATA_3)}개 데이터 삽입 완료")


def initialize_hyperparameter_type(db) -> None:
    """
    하이퍼파라미터 타입 데이터를 초기화합니다.
    Args:
        db: DB 세션
    """
    db.execute(insert(HyperparameterType.__table__).values(HYPERPARAMETER_TYPE_DATA))
    print(f"HyperparameterType 테이블에 {len(HYPERPARAMETER_TYPE_DATA)}개 데이터 삽입 완료")


def initialize_v1(db) -> None:
    """
    v1 버전의 모든 기본 데이터를 초기화합니다.
    Args:
        db: DB 세션
    """
    initialize_user(db)
    initialize_model_format(db)
    initialize_model_provider(db)
    initialize_model_type(db)


def create_db_session():
    """데이터베이스 세션을 생성합니다."""
    engine = create_engine(settings.get_db_uri)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def main():
    """데이터 초기화를 실행합니다."""
    parser = argparse.ArgumentParser(description="데이터베이스 초기화 스크립트")
    parser.add_argument(
        "--version",
        "-v",
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "all"],
        default="all",
        help="초기화할 데이터 버전을 선택합니다 (v1, v2, v3, v4, v5, v6, all)",
    )

    args = parser.parse_args()

    db = create_db_session()
    try:
        print("데이터 초기화를 시작합니다...")

        if args.version in ["v1", "all"]:
            print("V1 데이터 초기화 중...")
            initialize_v1(db)
            print("V1 데이터 초기화 완료")

        if args.version in ["v2", "all"]:
            print("V2 데이터 초기화 중...")
            add_model_format_2(db)
            print("V2 데이터 초기화 완료")

        if args.version in ["v3", "all"]:
            print("V3 데이터 초기화 중...")
            initialize_hyperparameter_type(db)
            print("V3 데이터 초기화 완료")

        if args.version in ["v4", "all"]:
            print("V4 데이터 초기화 중...")
            add_model_format_3(db)
            print("V4 데이터 초기화 완료")

        if args.version in ["v5", "all"]:
            print("V5 데이터 초기화 중...")
            add_model_format_4(db)
            add_model_provider_2(db)
            add_model_type_2(db)
            print("V5 데이터 초기화 완료")

        if args.version in ["v6", "all"]:
            print("V6 데이터 초기화 중...")
            add_model_type_3(db)
            print("V6 데이터 초기화 완료")

        db.commit()
        print("데이터 초기화가 완료되었습니다.")
    except Exception as e:
        traceback.print_exc()
        print(f"데이터 초기화 중 오류가 발생했습니다: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
