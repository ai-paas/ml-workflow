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
        실제 torch.nn.Module이 없으므로 pyfunc로 YoloxWrapper 사용

        * Params
            * model: YOLOX 모델 데이터 (repo_id, local_path, model_files, device, model_state_dict 등)
            * model_name: 저장할 모델 이름
        """
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            model_name = model_name.replace("/", "-")

            # YOLOX 모델을 pyfunc로 저장 (YoloxWrapper 사용)
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

    def log_artifact(self, file: UploadFile, model_name: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file.filename
            temp_file_path.write_bytes(file.file.read())

            with mlflow.start_run(run_name=model_name) as run:
                mlflow.log_artifacts(local_dir=temp_dir, artifact_path=model_name)
                artifact_uri = mlflow.get_artifact_uri(model_name)
                run_id = run.info.run_id
                return run_id, artifact_uri


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


class YoloxWrapper(PythonModel):
    def __init__(self, model_data):
        self.model_data = model_data
        self.repo_id = model_data["repo_id"]
        self.local_path = model_data["local_path"]
        self.model_files = model_data["model_files"]
        self.device = model_data["device"]
        self.config = model_data.get("config", {})
        self.model_state_dict = model_data.get("model_state_dict", None)
        self.model_file = model_data.get("model_file", None)
        self.model = None

    def predict(self, model_input):
        """
        YOLOX 모델 추론 수행

        Args:
            model_input: 이미지 경로 또는 이미지 데이터

        Returns:
            Detection 결과
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_context를 먼저 호출하세요.")

        # 실제 추론 로직 구현 필요
        # 현재는 모델 정보만 반환
        return {
            "repo_id": self.repo_id,
            "local_path": self.local_path,
            "model_files": self.model_files,
            "prediction": "YOLOX prediction result",  # 실제 추론 결과로 대체 필요
        }

    def load_context(self, context):
        """
        모델 컨텍스트 로드 (MLflow에서 모델 로드 시 호출)
        """
        try:
            import torch

            # 이미 model_state_dict가 있다면 사용
            if self.model_state_dict:
                print("기존 model_state_dict 사용")
                self.model = self.model_state_dict
                return

            # PyTorch 모델 파일이 있다면 로드
            if self.model_file and os.path.exists(self.model_file):
                print(f"모델 파일 로드 중: {self.model_file}")
                self.model_state_dict = torch.load(self.model_file, map_location=self.device)
                print("모델 로드 완료")

                # 실제 YOLOX 모델 클래스가 필요하면 여기서 초기화
                # 예: self.model = YOLOXModel()
                # self.model.load_state_dict(self.model_state_dict)

                # 현재는 state_dict를 모델로 사용
                self.model = self.model_state_dict
            else:
                print(f"경고: 모델 파일을 찾을 수 없습니다: {self.model_file}")
                # 모델 데이터가 이미 있으면 그것을 사용
                self.model = self.model_data

        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            # 오류가 발생해도 기본 데이터는 사용할 수 있도록
            self.model = self.model_data
