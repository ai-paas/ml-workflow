from config.settings import get_settings
from core.kubeflow.s3.mlflow_s3_manager import MLFlowS3Manager
from db.models.experiment import ExperimentModel
from mlflow import MlflowClient
from repos.experiment import experiment_repository, hyperparameter_repository, hyperparameter_type_repository
from schemas.experiment import (
    ExperimentBaseSchema,
    ExperimentInternalUpdateRequest,
    ExperimentReadSchema,
    ExperimentUpdateRequest,
    HyperparameterBaseSchema,
    HyperparameterReadSchema,
    HyperparameterTypeBaseSchema,
    HyperparameterTypeReadSchema,
)
from sqlalchemy.orm import Session
from utils.model_registry import ModelRegistry


class ExperimentService:
    @staticmethod
    def create(db: Session, *, obj_in: ExperimentBaseSchema):
        experiment_db_obj = experiment_repository.create(db, obj_in=obj_in)
        db.commit()
        return experiment_db_obj

    @staticmethod
    def get(db: Session, pk: int) -> ExperimentReadSchema:
        return experiment_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[ExperimentReadSchema]:
        return experiment_repository.get_multi(db, skip=skip, limit=limit)

    @staticmethod
    def update(db: Session, *, experiment_id: int, obj_in: ExperimentUpdateRequest):
        db_obj = experiment_repository.get(db, experiment_id)
        db_obj = experiment_repository.update(db, db_obj=db_obj, obj_in=obj_in)
        db.commit()
        return db_obj

    @staticmethod
    def update_internal(db: Session, *, experiment_id: int, obj_in: ExperimentInternalUpdateRequest):
        """내부 통신 전용 실험 업데이트 메서드"""
        db_obj = experiment_repository.get(db, experiment_id)
        # 내부 업데이트는 status, mlflow_run_id, kubeflow_run_id만 업데이트 가능
        update_data = {}
        if obj_in.status is not None:
            update_data["status"] = obj_in.status
        if obj_in.mlflow_run_id is not None:
            update_data["mlflow_run_id"] = obj_in.mlflow_run_id
        if obj_in.kubeflow_run_id is not None:
            update_data["kubeflow_run_id"] = obj_in.kubeflow_run_id

        if update_data:
            for key, value in update_data.items():
                setattr(db_obj, key, value)
            db.commit()
            db.refresh(db_obj)
        return db_obj

    @staticmethod
    def delete(db: Session, experiment_id: int):
        """
        실험 삭제 메서드

        MLflow artifacts와 S3 object도 함께 삭제합니다.
        model.py의 delete 메서드를 참고하여 구현했습니다.
        """
        # 1. 실험 조회
        experiment_obj = experiment_repository.get(db, experiment_id)
        if not experiment_obj:
            raise ValueError(f"실험을 찾을 수 없습니다: {experiment_id}")

        # 2. MLflow run_id와 artifact_path 추출
        run_id = experiment_obj.mlflow_run_id if experiment_obj else None
        s3_artifact_path = None

        # 3. 트랜잭션 시작 - MLflow/S3 삭제 후 DB 커밋
        try:
            # 3-1. DB 삭제 준비 (아직 커밋하지 않음)
            experiment_repository.delete(db, pk=experiment_id)

            # 3-2. MLflow/S3 삭제 시도
            if run_id:
                mlflow_deleted = False

                # MLflow artifacts 삭제
                try:
                    ModelRegistry().delete_run_artifacts(run_id)
                    mlflow_deleted = True
                except Exception as mlflow_error:
                    # MLflow 삭제 실패시 DB 롤백
                    db.rollback()
                    raise RuntimeError(f"MLflow 아티팩트 삭제 실패 (DB 변경사항 롤백됨): {str(mlflow_error)}")

                # S3 폴더 삭제 (artifact_path에서 추출)
                # MLflow run의 artifact_uri에서 S3 경로 추출
                try:
                    settings = get_settings()
                    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
                    run_info = client.get_run(run_id)
                    artifact_uri = run_info.info.artifact_uri

                    # artifact_uri에서 S3 경로 추출
                    # 형식 1: mlflow-artifacts:/0/abc123/artifacts
                    # 형식 2: s3://mlflow/8/09efe716fc234f3c87d760c91030b7e6/artifacts/google-owlv2-base-patch16
                    s3_artifact_path = None
                    if artifact_uri.startswith("mlflow-artifacts:/"):
                        s3_artifact_path = artifact_uri.replace("mlflow-artifacts:/", "")
                    elif artifact_uri.startswith("s3://"):
                        # s3://bucket/path 형식에서 버킷 이름 제거
                        # s3://mlflow/8/09efe716fc234f3c87d760c91030b7e6/artifacts/...
                        # -> 8/09efe716fc234f3c87d760c91030b7e6/artifacts/...
                        uri_without_protocol = artifact_uri.replace("s3://", "")
                        # 첫 번째 '/' 이후의 경로만 추출 (버킷 이름 제거)
                        if "/" in uri_without_protocol:
                            s3_artifact_path = uri_without_protocol.split("/", 1)[1]

                    if s3_artifact_path:
                        MLFlowS3Manager.get_instance().delete_folder(s3_artifact_path)
                except Exception as s3_error:
                    # S3 삭제 실패 처리
                    # MLflow가 이미 삭제되었다면 복구 불가능하므로 경고만 하고 진행
                    if mlflow_deleted:
                        import warnings

                        warnings.warn(f"S3 폴더 삭제 실패 (MLflow는 이미 삭제됨): {str(s3_error)}")
                        # S3만 실패한 경우 DB는 커밋 (MLflow는 이미 삭제되었으므로)
                    else:
                        # MLflow도 삭제 안됐고 S3도 실패면 롤백
                        db.rollback()
                        raise RuntimeError(f"S3 폴더 삭제 실패 (DB 변경사항 롤백됨): {str(s3_error)}")

            # 3-3. 모든 삭제가 성공하면 DB 커밋
            db.commit()

        except Exception as e:
            # 이미 처리된 RuntimeError는 그대로 전달
            if isinstance(e, RuntimeError):
                raise
            # 예상치 못한 에러는 롤백 후 전달
            db.rollback()
            raise RuntimeError(f"실험 삭제 중 예상치 못한 오류 발생: {str(e)}")

        return True


class HyperparameterService:
    @staticmethod
    def create(db: Session, *, obj_in: HyperparameterBaseSchema):
        hyperparameter_db_obj = hyperparameter_repository.create(db, obj_in=obj_in)
        db.commit()
        return hyperparameter_db_obj

    @staticmethod
    def get(db: Session, pk: int) -> HyperparameterReadSchema:
        return hyperparameter_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[HyperparameterReadSchema]:
        return hyperparameter_repository.get_multi(db, skip=skip, limit=limit)


class HyperparameterTypeService:
    @staticmethod
    def create(db: Session, *, obj_in: HyperparameterTypeBaseSchema):
        return hyperparameter_type_repository.create(db, obj_in=obj_in)

    @staticmethod
    def get(db: Session, pk: int) -> HyperparameterTypeReadSchema:
        return hyperparameter_type_repository.get(db, pk)

    @staticmethod
    def get_by_param_name(db: Session, param_name: str) -> HyperparameterTypeReadSchema:
        return hyperparameter_type_repository.get_by_param_name(db, param_name)

    @staticmethod
    def update(db: Session, *, obj_in: ExperimentUpdateRequest):
        return experiment_repository.update(db, obj_in=obj_in)
