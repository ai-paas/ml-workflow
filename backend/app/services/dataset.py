import csv
import io
import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from config.db.enums import DatasetKindEnum
from config.settings import get_settings
from core.storage.factory import get_storage_client
from fastapi import HTTPException, UploadFile
from repos.dataset import dataset_registry_repository, dataset_repository
from schemas.dataset import (
    DatasetBaseSchema,
    DatasetReadSchema,
    DatasetRegistryBaseSchema,
    DatasetRegistryReadSchema,
    DatasetUpdateSchema,
    DatasetValidationResponse,
)
from sqlalchemy.orm import Session

settings = get_settings()

logger = logging.getLogger(__name__)


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
    def update(db: Session, dataset_id: int, obj_in: DatasetUpdateSchema) -> DatasetReadSchema:
        """데이터셋 업데이트 (name, description만 수정)

        Args:
            db: 데이터베이스 세션
            dataset_id: 데이터셋 ID
            obj_in: 업데이트할 데이터

        Returns:
            업데이트된 데이터셋 읽기 스키마

        Raises:
            ValueError: 데이터셋을 찾을 수 없을 때
        """
        dataset_obj = dataset_repository.get(db, dataset_id)
        if not dataset_obj:
            raise ValueError(f"데이터셋 ID {dataset_id}를 찾을 수 없습니다.")

        # 업데이트할 필드가 있는지 확인
        if not obj_in.model_dump(exclude_unset=True):
            # 업데이트할 필드가 없으면 현재 객체 반환
            return dataset_repository.get(db, dataset_id)

        # 업데이트 수행 (CRUDBase.update가 내부에서 model_dump를 수행함)
        dataset_repository.update(db, db_obj=dataset_obj, obj_in=obj_in)
        db.commit()
        logger.info(f"데이터셋 업데이트 성공: {dataset_id}")
        return dataset_repository.get(db, dataset_id)

    @staticmethod
    def validate(file: UploadFile, dataset_kind: DatasetKindEnum) -> DatasetValidationResponse:
        """업로드된 데이터셋 ZIP 을 요청 분류(dataset_kind)의 형식 요구사항으로 검증한다.

        - object-detection: COCO128 구조 (annotations/instances_{train,val}2017.json + train2017/val2017)
          (사용자 친화 ZIP 사양은 §11.1 후속 — 현재는 happy-path 스텁)
        - protein-classification: TCR-Epitope CSV (필수 컬럼 epitope/cdr3b/label, label∈{0,1}, 비어있지 않음, 행≥16)

        출력에는 dataset_kind 가 포함되지 않는다(입력으로 이미 분류가 정해져 있음).
        ZIP 자체가 깨진 경우는 HTTP 400 으로 거부하고(api-spec §3.1),
        분류 형식 요구사항을 충족하지 못한 경우만 200 + is_valid=False 로 응답한다.
        """
        temp_dir = None
        try:
            file_content = file.file.read()
            file.file.seek(0)

            temp_dir = Path(tempfile.mkdtemp())
            try:
                with zipfile.ZipFile(io.BytesIO(file_content)) as zip_ref:
                    zip_ref.extractall(temp_dir)
            except zipfile.BadZipFile as e:
                logger.warning(f"ZIP 파일 형식 검증 실패: {str(e)}")
                raise HTTPException(status_code=400, detail="파일이 유효한 ZIP 형식이 아닙니다.")

            root_dir = DatasetService._resolve_root_dir(temp_dir, file.filename)

            if dataset_kind == DatasetKindEnum.OBJECT_DETECTION:
                ok, errors = DatasetService._check_object_detection(root_dir)
            elif dataset_kind == DatasetKindEnum.PROTEIN_CLASSIFICATION:
                ok, errors = DatasetService._check_protein_classification(root_dir)
            else:
                return DatasetValidationResponse(
                    is_valid=False, message=f"지원하지 않는 데이터셋 분류입니다: {dataset_kind}", details=None
                )

            if ok:
                logger.info(f"데이터셋 파일 검증 성공: {file.filename} (kind={dataset_kind})")
                return DatasetValidationResponse(is_valid=True, message="데이터셋 파일이 유효합니다.", details=None)

            logger.warning(f"데이터셋 구조 검증 실패: {errors}")
            return DatasetValidationResponse(
                is_valid=False, message="데이터셋 구조 검증 실패", details={"errors": errors}
            )

        except HTTPException:
            # 깨진 ZIP(400) 등 명시적 HTTP 오류는 그대로 전달
            raise
        except Exception as e:
            logger.error(f"데이터셋 검증 중 예외 발생: {str(e)}", exc_info=True)
            return DatasetValidationResponse(
                is_valid=False, message=f"데이터셋 검증 중 오류 발생: {str(e)}", details={"error": str(e)}
            )
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

    @staticmethod
    def _resolve_root_dir(temp_dir: Path, filename: str | None) -> Path:
        """ZIP 파일명과 동일한 루트 디렉토리가 있으면 그것을, 없으면 temp_dir 자체를 루트로 사용."""
        if filename:
            stem = Path(Path(filename).name).stem
            candidate = temp_dir / stem
            if candidate.is_dir():
                return candidate
        return temp_dir

    @staticmethod
    def _check_object_detection(root_dir: Path) -> tuple[bool, list[str]]:
        """COCO128 구조 검증 (기존 검증 로직 재사용 — happy-path 스텁, §11.1 후속)."""
        errors: list[str] = []

        annotations_dir = root_dir / "annotations"
        if not annotations_dir.is_dir():
            errors.append("annotations 폴더가 존재하지 않습니다.")
        else:
            for split in ("train", "val"):
                json_path = annotations_dir / f"instances_{split}2017.json"
                if not json_path.is_file():
                    errors.append(f"annotations/instances_{split}2017.json 파일이 존재하지 않습니다.")
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json.load(f)
                except json.JSONDecodeError:
                    errors.append(f"annotations/instances_{split}2017.json 파일이 유효한 JSON 형식이 아닙니다.")

        if not (root_dir / "train2017").is_dir():
            errors.append("train2017 폴더가 존재하지 않습니다.")
        if not (root_dir / "val2017").is_dir():
            errors.append("val2017 폴더가 존재하지 않습니다.")

        return (len(errors) == 0, errors)

    @staticmethod
    def _find_protein_csv(root_dir: Path) -> Path | None:
        """root 또는 1-depth 하위에서 train.csv / train.sample.csv 우선, 없으면 임의 CSV."""
        for name in ("train.csv", "train.sample.csv"):
            preferred = root_dir / name
            if preferred.is_file():
                return preferred
        candidates = sorted(root_dir.glob("*.csv")) + sorted(root_dir.glob("*/*.csv"))
        return candidates[0] if candidates else None

    @staticmethod
    def _check_protein_classification(root_dir: Path) -> tuple[bool, list[str]]:
        """TCR-Epitope CSV 검증. 필수 컬럼 epitope/cdr3b/label, label∈{0,1}, 비어있지 않음, 행≥16.

        (프로즈의 epitope+cdr3b 길이 1~80 상한은 train_eval 이 max_length=80 으로 truncate 하므로 검사하지 않음 — README §5.2)
        """
        csv_path = DatasetService._find_protein_csv(root_dir)
        if csv_path is None:
            return (False, ["CSV 파일(train.csv / train.sample.csv 등)을 찾을 수 없습니다."])

        errors: list[str] = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                missing = {"epitope", "cdr3b", "label"} - fields
                if missing:
                    return (False, [f"필수 컬럼 누락: {', '.join(sorted(missing))}"])

                row_count = 0
                for row in reader:
                    row_count += 1
                    epitope = (row.get("epitope") or "").strip()
                    cdr3b = (row.get("cdr3b") or "").strip()
                    label = (row.get("label") or "").strip()
                    if not epitope or not cdr3b:
                        errors.append(f"{row_count}행: epitope/cdr3b 가 비어 있습니다.")
                    if label not in {"0", "1"}:
                        errors.append(f"{row_count}행: label 은 0 또는 1 이어야 합니다 (현재 값: '{label}').")
                    if len(errors) >= 10:
                        errors.append("... (이하 생략)")
                        break

                if row_count < 16:
                    errors.append(f"학습에 필요한 최소 행 수(16)를 만족하지 않습니다 (현재: {row_count}행).")
        except Exception as e:
            return (False, [f"CSV 파싱 실패: {str(e)}"])

        return (len(errors) == 0, errors)

    @staticmethod
    def create(db: Session, *, obj_in: DatasetBaseSchema, file: UploadFile) -> DatasetReadSchema:
        """데이터셋 생성

        Args:
            db: 데이터베이스 세션
            obj_in: 데이터셋 기본 스키마
            file: 업로드된 파일

        Returns:
            생성된 데이터셋 읽기 스키마

        Raises:
            ValueError: 검증 오류 발생 시
            Exception: 데이터셋 등록 중 오류 발생 시
        """
        try:
            file.file.seek(0)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir) / file.filename
                temp_file_path.write_bytes(file.file.read())

                storage = get_storage_client()
                dataset_uuid = str(uuid.uuid4())
                object_key = f"datasets/{dataset_uuid}/{file.filename}"
                storage.upload_file(str(temp_file_path), object_key)

            dataset_obj = dataset_repository.create(db, obj_in=obj_in)
            dataset_id = dataset_obj.id

            dataset_registry_repository.create(
                db,
                obj_in=DatasetRegistryBaseSchema(
                    artifact_path=object_key,
                    uri=object_key,
                    dataset_id=dataset_id,
                ),
            )

            db.commit()
            logger.info(f"데이터셋 생성 성공: {obj_in.name} (ID: {dataset_id})")
            return dataset_repository.get(db, dataset_id)

        except Exception as e:
            db.rollback()
            logger.error(f"데이터셋 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def delete(db: Session, dataset_id: int):
        """
        데이터셋 삭제 - 오브젝트 스토리지와 DB 레코드를 함께 삭제

        Args:
            db: 데이터베이스 세션
            dataset_id: 삭제할 데이터셋 ID

        Returns:
            bool: 삭제 성공 여부

        Raises:
            ValueError: 데이터셋을 찾을 수 없을 때
            RuntimeError: 삭제 중 오류 발생 시
        """
        dataset_obj = dataset_repository.get(db, dataset_id)
        if not dataset_obj:
            raise ValueError(f"데이터셋 ID {dataset_id}를 찾을 수 없습니다.")

        dataset_registry = dataset_obj.dataset_registry
        if not dataset_registry:
            logger.warning(f"데이터셋 {dataset_id}에 레지스트리 정보가 없습니다.")
            dataset_repository.delete(db, pk=dataset_id)
            db.commit()
            return True

        artifact_path = dataset_registry.artifact_path

        try:
            dataset_registry_repository.delete(db, pk=dataset_registry.id)
            dataset_repository.delete(db, pk=dataset_id)

            folder_path = str(Path(artifact_path).parent)
            get_storage_client().delete_folder(folder_path)

            db.commit()
            return True

        except Exception as e:
            db.rollback()
            raise RuntimeError(f"데이터셋 삭제 중 오류 발생: {str(e)}")


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
