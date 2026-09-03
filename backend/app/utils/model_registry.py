import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import mlflow
import torch
from config.settings import get_settings
from fastapi import UploadFile
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

settings = get_settings()

# 환경 변수를 통한 타임아웃 설정
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "14400"  # 5분으로 설정


con = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.9",
        {
            "pip": ["llama-cpp-python"],
        },
    ],
}


class ModelRegistry:
    def __init__(self):
        # mlflow.log_artifacts 등은 자격증명을 boto3 기본 체인(표준 AWS 환경변수)과
        # MLFLOW_S3_ENDPOINT_URL 에서만 읽는다(메소드 인자로 키를 넘길 수 없음). 아티팩트 스토어가
        # 직접 s3:// 인 배포에서는 이 값들이 프로세스 env 에 있어야 업로드가 되므로 설정값에서 채워 준다.
        # 이미 주입돼 있으면 덮지 않으며, 명시 자격증명을 쓰는 S3StorageClient/MLFlowS3Manager 에는 영향이 없다.
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.MLFLOW_S3_ENDPOINT_URL)
        os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.MLFLOW_S3_ACCESS_KEY_ID)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.MLFLOW_S3_SECRET_ACCESS_KEY)
        self._client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        self._experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment == None:
            mlflow.create_experiment(self._experiment_name)

    def log_transformers(self, model: dict[str, Any], model_name: str):
        """
        Private Model을 Model Repository에 저장하는 method

        * Parmas
            * repo: Model 공급 유형에 따라 달라짐
                - Huggingface transfromers : repo_id
                - Huggingface gguf : repo_id, file_name
        """
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            model_name = model_name.replace("/", "-")
            # model_pipeline = pipeline(task="object-detection",
            #                           model=model['model'],
            #                           image_processor=model['image_processor'],
            #                           tokenizer=None
            #                           )
            mlflow.transformers.log_model(
                transformers_model=model,
                artifact_path=model_name,
                registered_model_name=model_name,
            )

            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            model_version = self._client.get_latest_versions(name=model_name, stages=["None"])[0].version
            model_uri = f"models:/{model_name}/{model_version}"
        return run_id, artifact_uri, model_version, model_uri

    def log_sentence_transformers(self, model, model_name: str):
        mlflow.set_experiment(self._experiment_name)

        data = "This is a test data!"
        signature = mlflow.models.infer_signature(
            model_input=data,
            model_output=model.encode(data),
        )
        with mlflow.start_run(run_name=model_name) as run:
            model_name = model_name.replace("/", "-")
            mlflow.sentence_transformers.log_model(
                model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=data,
            )

            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            model_version = self._client.get_latest_versions(name=model_name, stages=["None"])[0].version
            model_uri = f"models:/{model_name}/{model_version}"
        return run_id, artifact_uri, model_version, model_uri

    def log_pyfunc(self, model, model_name: str):
        """
        Private Model을 Model Repository에 저장하는 method

        gguf, BGEMeEmbedding 등 mlflow flavor에 정의되어 있지 않은 것을 등록
        """
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            model_name = model_name.replace("/", "-")
            mlflow.pyfunc.log_model(
                artifact_path=model_name, python_model=PyfuncModelWrapper(model), registered_model_name=model_name
            )
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            model_version = self._client.get_latest_versions(name=model_name, stages=["None"])[0].version
            model_uri = f"models:/{model_name}/{model_version}"
        return run_id, artifact_uri, model_version, model_uri

    def log_pytorch(self, model: torch.nn.Module, model_name: str):
        """
        YOLOX 모델을 Model Repository에 저장하는 method
        * Params
            * model: YOLOX 모델 데이터 (repo_id, local_path, model_files, device, model_state_dict 등)
            * model_name: 저장할 모델 이름
        """
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            model_name = model_name.replace("/", "-")

            mlflow.pytorch.log_model(artifact_path=model_name, pytorch_model=model, registered_model_name=model_name)

            # 메타데이터 저장
            mlflow.log_params(
                {
                    "framework": "pytorch",
                }
            )

            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            model_version = self._client.get_latest_versions(name=model_name, stages=["None"])[0].version
            model_uri = f"models:/{model_name}/{model_version}"
        return run_id, artifact_uri, model_version, model_uri

    def log_llamacpp(self, model, model_name: str):
        """
        Private Model을 Model Repository에 저장하는 method

        gguf, BGEMeEmbedding 등 mlflow flavor에 정의되어 있지 않은 것을 등록
        """
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            model_name = model_name.replace("/", "-")
            mlflow.pyfunc.log_model(
                artifact_path=model_name, python_model=LlamaCppWrapper(model), registered_model_name=model_name
            )
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            model_version = self._client.get_latest_versions(name=model_name, stages=["None"])[0].version
            model_uri = f"models:/{model_name}/{model_version}"
        return run_id, artifact_uri, model_version, model_uri

    def log_artifact(self, model_name: str, file: UploadFile = None, save_dir: str = None):
        if file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir) / file.filename
                temp_file_path.write_bytes(file.file.read())

                with mlflow.start_run(run_name=model_name) as run:
                    mlflow.log_artifacts(local_dir=temp_dir, artifact_path=model_name)
                    artifact_uri = mlflow.get_artifact_uri(model_name)
                    run_id = run.info.run_id
                    return run_id, artifact_uri
        elif save_dir is not None:
            with mlflow.start_run(run_name=model_name) as run:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("framework", "transformers")
                mlflow.log_param("format", "huggingface-bin")
                mlflow.log_artifacts(local_dir=save_dir, artifact_path=model_name)
                artifact_uri = mlflow.get_artifact_uri(model_name)
                run_id = run.info.run_id
                return run_id, artifact_uri
        else:
            raise ValueError("file or save_dir must be provided")

    # 압축 해제 결과에서 무시할 메타(맥 zip 부산물 등)
    _ZIP_JUNK = frozenset({"__MACOSX", ".DS_Store"})

    @classmethod
    def _normalize_model_root(cls, extract_dir: str) -> str:
        """압축 해제 결과에서 실제 모델 루트(모델 파일이 직접 놓인 디렉토리)를 찾는다.

        zip 이 (a) 단일 최상위 폴더로 감싼 형태면 그 폴더로 내려가고(중첩 흡수),
        (b) 파일을 루트에 평탄하게 담은 형태면 extract_dir 을 그대로 쓴다.
        맥 zip 부산물(__MACOSX/.DS_Store)은 무시하고 판정한다. from_pretrained(local_path)
        가 config.json 을 로컬 루트에서 찾도록, 최종적으로 모델 파일이 루트에 놓인 경로를 반환한다.
        """
        root = extract_dir
        while True:
            entries = [e for e in os.listdir(root) if e not in cls._ZIP_JUNK]
            subdirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(root, e))]
            # 폴더 하나만 있고 파일이 없으면(=단일 최상위 폴더로 감싼 zip) 그 안으로 내려간다.
            if len(subdirs) == 1 and not files:
                root = os.path.join(root, subdirs[0])
                continue
            break
        return root

    def log_artifact_from_zip(self, model_name: str, file: UploadFile):
        """커스텀 모델 zip 을 받아 압축 해제 후 모델 폴더 '전체'를 MLflow 아티팩트로 업로드한다.

        단일 파일 업로드(log_artifact(file=...))와 달리, HuggingFace 스냅샷과 동일하게 디렉토리 통째를
        올려(config.json/safetensors/tokenizer 등) 서빙 predictor 가 from_pretrained(local_path) 로
        로드할 수 있게 한다. zip 형태는 (a) 단일 최상위 폴더 감싼 형태와 (b) 루트 평탄 형태 모두 허용한다.

        return: (run_id, artifact_uri)
        """
        filename = (getattr(file, "filename", "") or "").strip()
        if not filename.lower().endswith(".zip"):
            raise ValueError(f"커스텀 모델 파일은 .zip(모델 디렉토리 압축)이어야 합니다. 받은 파일: {filename!r}")

        with tempfile.TemporaryDirectory() as work_dir:
            zip_path = os.path.join(work_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(file.file.read())

            if not zipfile.is_zipfile(zip_path):
                raise ValueError("업로드한 파일이 유효한 zip 아카이브가 아닙니다.")

            # 별도 하위 디렉토리에만 풀어, 현재 경로로 파일이 흩어지는 것을 원천 차단한다.
            extract_dir = os.path.join(work_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            base = os.path.realpath(extract_dir)
            with zipfile.ZipFile(zip_path) as zf:
                # zip-slip 방지: 각 멤버의 대상 경로가 extract_dir 밖으로 나가면 거부.
                for member in zf.namelist():
                    dest = os.path.realpath(os.path.join(extract_dir, member))
                    if dest != base and not dest.startswith(base + os.sep):
                        raise ValueError(f"안전하지 않은 zip 경로 항목이 있어 거부합니다: {member!r}")
                zf.extractall(extract_dir)

            model_root = self._normalize_model_root(extract_dir)

            # 업로드 전 맥 zip 부산물 제거(모델 루트 내부에 남아 있을 수 있음).
            macosx = os.path.join(model_root, "__MACOSX")
            if os.path.isdir(macosx):
                shutil.rmtree(macosx, ignore_errors=True)

            entries = [e for e in os.listdir(model_root) if e not in self._ZIP_JUNK]
            if not entries:
                raise ValueError("zip 압축을 풀었지만 모델 파일이 없습니다(빈 아카이브).")

            with mlflow.start_run(run_name=model_name) as run:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("format", "custom-zip")
                mlflow.log_artifacts(local_dir=model_root, artifact_path=model_name)
                artifact_uri = mlflow.get_artifact_uri(model_name)
                run_id = run.info.run_id
                return run_id, artifact_uri

    def delete_run_artifacts(self, run_id: str):
        """MLflow tracking store 의 run 을 삭제하는 메서드.

        주의: 이것은 run 레코드만 삭제하며 S3 아티팩트 blob 은 남는다(delete_run 은 tracking store 만 건드림).
        S3 까지 지우려면 호출부에서 artifact_uri 를 파싱해 MLFlowS3Manager.delete_folder 를 함께 호출해야 한다.

        Args:
            run_id: 삭제할 run의 ID
        """
        if not run_id:
            return False
        try:
            self._client.delete_run(run_id)
        except Exception as e:
            raise RuntimeError(f"런 아티팩트 삭제 실패: {str(e)}")
        return True


class PyfuncModelWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input):
        if self.model is None:
            raise ValueError("The model has not been loaded.")
        return self.model


class LlamaCppWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input):
        if self.model is None:
            raise ValueError("The model has not been loaded.")
        return self.model
