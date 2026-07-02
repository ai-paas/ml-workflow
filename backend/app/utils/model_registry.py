import os
import tempfile
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

    def delete_run_artifacts(self, run_id: str):
        """
        특정 run의 모든 artifact를 삭제하는 메서드

        Args:
            run_id: 삭제할 run의 ID
        """
        # run을 삭제하면 해당 run의 모든 artifact도 함께 삭제됩니다
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
