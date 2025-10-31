import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import UploadFile
from repos.dataset import dataset_registry_repository, dataset_repository
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema, DatasetRegistryBaseSchema, DatasetRegistryReadSchema
from sqlalchemy.orm import Session
from utils.dataset_registry import DatasetRegistry


class DatasetService:
    @staticmethod
    def get(db: Session, pk: int) -> DatasetReadSchema:
        return dataset_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[DatasetReadSchema]:
        return dataset_repository.get_multi(db, skip=skip, limit=limit)

    @staticmethod
    def get_all(db: Session) -> list[DatasetReadSchema]:
        return dataset_repository.get_all(db)

    @staticmethod
    def update(db: Session, db_obj, obj_in):
        return dataset_repository.update(db, db_obj=db_obj, obj_in=obj_in)

    @staticmethod
    def validate_dataset_file(file: UploadFile) -> dict:
        """데이터셋 파일 검증

        Args:
            file: 업로드된 파일

        Returns:
            검증 결과 딕셔너리
            - is_valid: 검증 성공 여부
            - message: 검증 메시지
            - root_dir: 루트 디렉토리 경로 (성공 시)
            - details: 상세 정보 (실패 시)
        """
        temp_dir = None
        try:
            # 업로드된 파일 내용 읽기
            file_content = file.file.read()

            # ZIP 파일 형식 검증 및 압축 해제
            temp_dir = Path(tempfile.mkdtemp())
            try:
                with zipfile.ZipFile(io.BytesIO(file_content)) as zip_ref:
                    zip_ref.extractall(temp_dir)
            except zipfile.BadZipFile:
                return {
                    "is_valid": False,
                    "message": "파일이 유효한 ZIP 형식이 아닙니다.",
                    "root_dir": None,
                    "details": None,
                }

            # 업로드된 파일명 추출
            file_name = Path(file.filename).name
            dataset_name = Path(file_name).stem

            # 압축 해제 후 ZIP 파일명과 동일한 루트 디렉토리 찾기
            root_dir = temp_dir / dataset_name
            if not root_dir.is_dir():
                root_dir = temp_dir  # 동일 이름의 디렉토리가 없으면 temp_dir 자체를 루트로 사용

            # COCO128 데이터셋 구조 검증
            validation_errors = []

            # 1. annotations 폴더 존재 확인
            annotations_dir = root_dir / "annotations"
            if not annotations_dir.is_dir():
                validation_errors.append("annotations 폴더가 존재하지 않습니다.")
            else:
                # 2. instances_train2017.json 파일 존재 확인
                train_json = annotations_dir / "instances_train2017.json"
                if not train_json.is_file():
                    validation_errors.append("annotations/instances_train2017.json 파일이 존재하지 않습니다.")
                else:
                    # JSON 파일 유효성 검증
                    try:
                        with open(train_json, "r", encoding="utf-8") as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        validation_errors.append(
                            "annotations/instances_train2017.json 파일이 유효한 JSON 형식이 아닙니다."
                        )

                # 3. instances_val2017.json 파일 존재 확인
                val_json = annotations_dir / "instances_val2017.json"
                if not val_json.is_file():
                    validation_errors.append("annotations/instances_val2017.json 파일이 존재하지 않습니다.")
                else:
                    # JSON 파일 유효성 검증
                    try:
                        with open(val_json, "r", encoding="utf-8") as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        validation_errors.append(
                            "annotations/instances_val2017.json 파일이 유효한 JSON 형식이 아닙니다."
                        )

            # 4. train 폴더 존재 확인
            train_dir = root_dir / "train2017"
            if not train_dir.is_dir():
                validation_errors.append("train 폴더가 존재하지 않습니다.")

            # 5. val 폴더 존재 확인
            val_dir = root_dir / "val2017"
            if not val_dir.is_dir():
                validation_errors.append("val 폴더가 존재하지 않습니다.")

            # 검증 실패 시 에러 반환
            if validation_errors:
                return {
                    "is_valid": False,
                    "message": "데이터셋 구조 검증 실패",
                    "root_dir": None,
                    "details": {"errors": validation_errors},
                }

            return {
                "is_valid": True,
                "message": "데이터셋 파일이 유효합니다.",
                "root_dir": str(root_dir),
                "details": None,
            }

        except Exception as e:
            return {
                "is_valid": False,
                "message": f"데이터셋 검증 중 오류 발생: {str(e)}",
                "root_dir": None,
                "details": {"error": str(e)},
            }
        finally:
            # 임시 디렉토리 정리
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

    @staticmethod
    def create(db: Session, *, obj_in: DatasetBaseSchema, file: UploadFile):
        # UploadFile의 file 속성은 SpooledTemporaryFile 객체이므로
        # 직접 임시 파일 경로를 사용할 수 있습니다

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file.filename
            temp_file_path.write_bytes(file.file.read())
            run_id, dataset_version, artifact_uri, dataset_uri = DatasetRegistry().log_dataset(temp_dir, obj_in.name)

        # 데이터셋 객체 생성
        dataset_obj = dataset_repository.create(db, obj_in=obj_in)
        dataset_id = dataset_obj.id

        # 데이터셋 레지스트리 정보 생성
        dataset_registry_repository.create(
            db,
            obj_in=DatasetRegistryBaseSchema(
                artifact_path=artifact_uri,
                uri=dataset_uri,
                dataset_id=dataset_id,
            ),
        )

        db.commit()
        return dataset_repository.get(db, dataset_id)


class DatasetRegistryService:
    @staticmethod
    def create(db: Session, *, obj_in: DatasetRegistryBaseSchema):
        return dataset_registry_repository.create(db, obj_in=obj_in)

    @staticmethod
    def get(db: Session, pk: int) -> DatasetRegistryReadSchema:
        return dataset_registry_repository.get(db, pk)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[DatasetRegistryReadSchema]:
        return dataset_registry_repository.get_multi(db, skip=skip, limit=limit)
