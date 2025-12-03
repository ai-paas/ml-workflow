import logging
import os
import uuid
from typing import Any

import kfp
from config.settings import get_settings
from kfp import dsl
from kfp.compiler import Compiler
from utils.istio_authentication import get_istio_auth_session

settings = get_settings()

logger = logging.getLogger(__name__)


class KubeflowManager:
    def __init__(self):
        self.endpoint = settings.KUBEFLOW_ENDPOINT
        self.username = settings.KUBEFLOW_USERNAME
        self.password = settings.KUBEFLOW_PASSWORD
        self.namespace = settings.KUBEFLOW_NAMESPACE
        self.auth_session = self._get_istio_auth_session()
        self.kfp_client = self._create_kfp_client()

    def _get_istio_auth_session(self):
        return get_istio_auth_session(url=self.endpoint, username=self.username, password=self.password)

    def _create_kfp_client(self):
        client = kfp.Client(
            host=f"{self.endpoint}/pipeline",
            namespace=self.namespace,
            cookies=self.auth_session.session_cookie,
        )
        # client = kfp.Client(
        #     host=f"{self.endpoint}/pipeline"
        # )
        # client._job_api.api_client.cookie = f"authservice_session={self.auth_session.session_cookie}"
        return client

    # def get_session_cookie(self):
    #     cookie = self.auth_session.session_cookie.split("=")[-1]
    #     return cookie

    def get_kfp_client(self):
        return self.kfp_client

    def compile_pipeline(self, pipeline_func: callable, pipeline_name: str):
        pipeline_filename = f"{pipeline_name}.yaml"
        Compiler().compile(pipeline_func, pipeline_filename)

    def create_pipeline(self, pipeline_func: callable, pipeline_name: str):
        pipeline_filename = f"{pipeline_name}.yaml"
        Compiler().compile(pipeline_func, pipeline_filename)
        self.kfp_client.upload_pipeline(pipeline_package_path=pipeline_filename, pipeline_name=pipeline_name)
        os.remove(pipeline_filename)
        logger.info(f"Pipeline {pipeline_name} created successfully")

    def get_pipeline_by_name(self, pipeline_name: str):
        # pipelines = self.kfp_client.list_pipelines().pipelines
        return next(
            (
                pipeline
                for pipeline in self.kfp_client.list_pipelines().pipelines
                if pipeline.display_name == pipeline_name
            ),
            None,
        )

    def get_pipeline_id(self, pipeline_name: str):
        return next(
            (
                pipeline.id
                for pipeline in self.kfp_client.list_pipelines().pipelines
                if pipeline.display_name == pipeline_name
            ),
            None,
        )

    def create_experiment(self, experiment_name: str):
        return self.kfp_client.create_experiment(name=experiment_name)

    def get_experiment_by_name(self, *, experiment_name: str):
        # experiments = self.kfp_client.list_experiments().experiments

        return next(
            (
                experiment
                for experiment in self.kfp_client.list_experiments().experiments
                if experiment.display_name == experiment_name
            ),
            None,
        )

    # TODO : 현재 pipeline_id에 해당하는 버전ID정보를 확인할수있는 방법을 찾아 다시 시도. client.run_pipeline 메소드 실행방법 파악 불가
    # def run_pipeline(self, pipeline_id: str, experiment_id: str, params: dict[str, Any] = {}):
    #     # version_id = str(uuid.uuid4())
    #     run = self.kfp_client.run_pipeline(
    #         experiment_id=experiment_id,
    #         job_name=f"{pipeline_id}-run",
    #         pipeline_id=pipeline_id,
    #         version_id="1",
    #         params=params,
    #     )
    #     logger.info(f"Pipeline {pipeline_id} started. Run ID: {run.id}")
    #     return run

    def list_pipelines(self):
        return [p.name for p in self.kfp_client.list_pipelines().pipelines]

    def delete_pipeline(self, pipeline_name: str):
        pipeline = self.get_pipeline_by_name(pipeline_name=pipeline_name)
        if pipeline.id:
            self.kfp_client.delete_pipeline(pipeline_id=pipeline.id)
            logger.info(f"Pipeline {pipeline_name} deleted successfully")
        else:
            logger.error(f"Pipeline {pipeline_name} not found")

    def list_runs(self, experiment_name: str):
        experiment = self.get_experiment_by_name(experiment_name=experiment_name)
        runs = self.kfp_client.list_runs(experiment_id=experiment.id)
        return [(run.id, run.name, run.status) for run in runs.runs]

    # TODO: kubeflow run log 가져오는 방법 조사 필요. 현재 파악 불가
    def get_run_logs(self, run_id: str):
        pass

    def get_run_status(self, run_id: str):
        run = self.kfp_client.get_run(run_id)
        return run.status

    def delete_experiment(self, experiment_name: str):
        experiment = self.get_experiment_by_name(experiment_name=experiment_name)
        return self.kfp_client.delete_experiment(experiment_id=experiment.experiment_id)
