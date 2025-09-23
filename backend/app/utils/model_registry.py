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
        model_name = model_name.replace("/", "-")
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


class ModelLoader:
    @staticmethod
    def load_transformers(model_uri: str):
        return mlflow.transformers.load_model(model_uri)

    @staticmethod
    def load_sentence_transformers(model_uri: str):
        return mlflow.sentence_transformers.load_model(model_uri)

    @staticmethod
    def load_pyfunc(model_uri: str):
        return mlflow.pyfunc.load_model(model_uri)

    @staticmethod
    def load_pytorch(model_uri: str):
        return mlflow.pyfunc.load_model(model_uri)  # YOLOX는 pyfunc로 저장되므로 pyfunc로 로드


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
