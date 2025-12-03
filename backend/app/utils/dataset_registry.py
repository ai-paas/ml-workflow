import os
from typing import Any

import mlflow
import pandas as pd
from config.settings import get_settings
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

settings = get_settings()

# 환경 변수를 통한 타임아웃 설정
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"  # 5분으로 설정


class DatasetRegistry:
    def __init__(self):
        self._client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        self._experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment == None:
            mlflow.create_experiment(self._experiment_name)

    def log_dataset(
        self,
        dataset_dir: str,
        dataset_name: str,
        # , dataset: dict[str, Any]
    ):
        """
        Private Model을 Model Repository에 저장하는 method

        * Parmas
            * repo: Model 공급 유형에 따라 달라짐
                - Huggingface transfromers : repo_id
                - Huggingface gguf : repo_id, file_name
        """
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=dataset_name) as run:
            # dataset_name = dataset_name.replace("/", "-")
            # model_pipeline = pipeline(task="object-detection",
            #                           model=model['model'],
            #                           image_processor=model['image_processor'],
            #                           tokenizer=None
            #                           )
            mlflow.log_artifacts(
                local_dir=dataset_dir,
                artifact_path=dataset_name,
            )
            # mlflow.pyfunc.log_model(
            #     artifact_path=dataset_name,
            #     python_model=PyfuncDatasetWrapper(dataset), registered_model_name=dataset_name)

            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri(dataset_name)

            # TODO: 이후 버전관리 기능을 적용할시 변경 필요
            dataset_version = 1
            dataset_uri = ""
        return run_id, dataset_version, artifact_uri, dataset_uri

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


# class DatasetLoader:
#     @staticmethod
#     def load_pyfunc(dataset_uri: str):
#         return mlflow.pyfunc.load_model(dataset_uri)

# class PyfuncDatasetWrapper(PythonModel):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     # def load_context(self, context):
#     #     # 저장된 CSV 파일을 불러옵니다
#     #     self.dataset = pd.read_csv(self.dataset_path)

#     def predict(self, model_input):
#         # 데이터셋을 반환하거나 예측 로직을 추가할 수 있습니다
#         return self.dataset
