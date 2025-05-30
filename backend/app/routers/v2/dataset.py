import io
import logging
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Annotated

import yaml
from config.db.connect import SessionDepends
from config.settings import get_settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from schemas.dataset import DatasetBaseSchema, DatasetReadSchema
from schemas.user import UserSchema
from services.dataset import DatasetService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

router = APIRouter(prefix="/datasets", tags=["Datasets"])

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: 책임 분리 필요.
@router.post("", response_model=DatasetReadSchema)
def create_dataset(
    *,
    db: Session = SessionDepends,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    file: UploadFile = File(...),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Dataset Registry에 데이터셋을 검증 및 등록하는 API
    """
    # TODO: 데이터셋 유효성 검증

    try:
        # 업로드된 파일 내용 읽기
        file_content = file.file.read()

        # ZIP 파일 형식 검증 및 압축 해제
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with zipfile.ZipFile(io.BytesIO(file_content)) as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            raise ValueError("파일이 유효한 ZIP 형식이 아닙니다.")

        # 업로드된 파일명 추출
        file_name = Path(file.filename).name
        dataset_name = Path(file_name).stem

        # 압축 해제 후 ZIP 파일명과 동일한 루트 디렉토리 찾기
        root_dir = temp_dir / dataset_name
        if not root_dir.is_dir():
            root_dir = temp_dir  # 동일 이름의 디렉토리가 없으면 temp_dir 자체를 루트로 사용

        logger.info(f"데이터셋 루트 디렉토리: {root_dir}")

        # 데이터셋 구조 검증
        yaml_data = validate_dataset_structure(root_dir)

        logger.info(f"데이터셋 '{name}' 검증 완료: 모든 검증 통과")
        logger.info(f"클래스 정보: {yaml_data['names']}")

        # 파일 포인터 초기화 (업로드를 위해)
        file.file.seek(0)

        # 데이터셋 정보 저장
        dataset_data = DatasetBaseSchema(
            name=name,
            description=description,
            version=1,
            subversion=1,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )

        # DatasetService를 통해 DB에 저장
        dataset_service = DatasetService()
        db_dataset = dataset_service.create(db, obj_in=dataset_data, file=file)

        return db_dataset

    except ValueError as e:
        # 검증 오류 발생 시
        logger.error(f"데이터셋 검증 오류: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 기타 오류 발생 시
        traceback.print_exc()
        logger.error(f"데이터셋 등록 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터셋 등록 중 오류가 발생했습니다: {str(e)}")
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)


@router.get("/{dataset_id}", response_model=DatasetReadSchema)
def read_dataset(dataset_id: int, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)):
    db_model = DatasetService().get(db, dataset_id)
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return db_model


@router.get("", response_model=list[DatasetReadSchema])
def read_datasets(
    skip: int = 0, limit: int = 10, db: Session = SessionDepends, current_user: UserSchema = Depends(get_current_user)
):
    datasets = DatasetService().get_multi(db)
    return datasets


def validate_label_files(label_path, class_count):
    """라벨 파일의 내용을 검증합니다."""
    with label_path.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 빈 줄 무시
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"라벨 파일 '{label_path.name}' {line_num}번째 줄에 유효하지 않은 형식이 있습니다. 5개의 값이 있어야 합니다.")

            # 클래스 인덱스 검증
            try:
                class_idx = int(parts[0])
                if class_idx < 0 or class_idx >= class_count:
                    raise ValueError(
                        f"라벨 파일 '{label_path.name}' {line_num}번째 줄에 범위를 벗어난 클래스 인덱스가 있습니다: \
                            {class_idx} (0~{class_count-1} 사이여야 함)"
                    )
            except ValueError:
                raise ValueError(f"라벨 파일 '{label_path.name}' {line_num}번째 줄에 유효하지 않은 클래스 인덱스가 있습니다: {parts[0]}")

            # 바운딩 박스 값 검증 (0~1 사이)
            for i in range(1, 5):
                try:
                    value = float(parts[i])
                    if not 0 <= value <= 1:
                        raise ValueError(
                            f"라벨 파일 '{label_path.name}' {line_num}번째 줄에 범위를 벗어난 바운딩 박스 값이 있습니다: {value} (0~1 사이여야 함)"
                        )
                except ValueError:
                    raise ValueError(f"라벨 파일 '{label_path.name}' {line_num}번째 줄에 유효하지 않은 바운딩 박스 값이 있습니다: {parts[i]}")


def validate_dataset_structure(root_dir):
    """데이터셋 구조를 검증합니다."""
    required_dirs = ["test", "train", "valid"]

    # data.yaml 검증
    yaml_path = root_dir / "data.yaml"
    if not yaml_path.is_file():
        raise ValueError("필수 파일 'data.yaml'이 없습니다.")

    # YAML 내용 로드 및 검증
    with yaml_path.open("r") as f:
        try:
            yaml_data = yaml.safe_load(f)
        except yaml.YAMLError:
            raise ValueError("data.yaml 파일이 유효한 YAML 형식이 아닙니다.")

    # 필수 키 확인
    required_keys = ["train", "val", "test", "nc", "names"]
    missing_keys = [key for key in required_keys if key not in yaml_data]
    if missing_keys:
        raise ValueError(f"data.yaml 파일에 필수 키가 없습니다: {', '.join(missing_keys)}")

    # names 길이와 nc 일치 여부 확인
    if len(yaml_data["names"]) != yaml_data["nc"]:
        raise ValueError(f"data.yaml의 'nc' 값({yaml_data['nc']})과 'names' 배열 길이({len(yaml_data['names'])})가 일치하지 않습니다.")

    class_count = yaml_data["nc"]

    for dir_name in required_dirs:
        dir_path = root_dir / dir_name
        if not dir_path.is_dir():
            raise ValueError(f"필수 폴더 '{dir_name}'이 없습니다.")

        # images와 labels 폴더 확인
        images_dir = dir_path / "images"
        labels_dir = dir_path / "labels"

        if not images_dir.is_dir():
            raise ValueError(f"'{dir_name}' 폴더 내 'images' 폴더가 없습니다.")
        if not labels_dir.is_dir():
            raise ValueError(f"'{dir_name}' 폴더 내 'labels' 폴더가 없습니다.")

        # 각 폴더 내 파일 가져오기
        image_files = list(images_dir.glob("*"))
        label_files = list(labels_dir.glob("*"))

        # 빈 폴더 확인
        if not image_files:
            raise ValueError(f"'{dir_name}/images' 폴더가 비어 있습니다.")
        if not label_files:
            raise ValueError(f"'{dir_name}/labels' 폴더가 비어 있습니다.")

        # 이미지 확장자 검증
        invalid_images = [img for img in image_files if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]]
        if invalid_images:
            raise ValueError(
                f"'{dir_name}/images' 폴더에 지원되지 않는 이미지 형식이 있습니다: {', '.join(f.name for f in invalid_images)}"
            )

        # 라벨 확장자 검증
        invalid_labels = [lbl for lbl in label_files if lbl.suffix.lower() != ".txt"]
        if invalid_labels:
            raise ValueError(
                f"'{dir_name}/labels' 폴더에 지원되지 않는 라벨 형식이 있습니다: {', '.join(f.name for f in invalid_labels)}"
            )

        # 파일 이름 일치 여부 검증 (확장자 제외)
        image_basenames = {img.stem for img in image_files}
        label_basenames = {lbl.stem for lbl in label_files}

        if image_basenames != label_basenames:
            missing_in_images = label_basenames - image_basenames
            missing_in_labels = image_basenames - label_basenames
            error_msgs = []

            if missing_in_images:
                error_msgs.append(f"'{dir_name}/images' 폴더에 없는 파일: {', '.join(missing_in_images)}")
            if missing_in_labels:
                error_msgs.append(f"'{dir_name}/labels' 폴더에 없는 파일: {', '.join(missing_in_labels)}")

            raise ValueError(". ".join(error_msgs))

        # 라벨 파일 내용 검증 (샘플링)
        txt_label_files = [f for f in label_files if f.suffix.lower() == ".txt"]
        sample_size = min(10, len(txt_label_files))

        for label_file in txt_label_files[:sample_size]:
            validate_label_files(label_file, class_count)

    return yaml_data
