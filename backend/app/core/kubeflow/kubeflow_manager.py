import logging
import os
import threading
import time

import kfp
from config.settings import get_settings
from kfp.compiler import Compiler
from utils.istio_authentication import get_istio_auth_session

settings = get_settings()

logger = logging.getLogger(__name__)


class KubeflowManager:
    """Kubeflow(Istio+Dex) 접근 매니저. 프로세스 단위 싱글톤이며 Dex 로그인 세션을 TTL 동안 재사용한다.

    과거처럼 인스턴스 생성(=호출)마다 Dex 로그인하면, 배포 상태 폴링처럼 잦은 호출에서 로그인이 폭주해
    세션 churn 으로 Dex 가 간헐적으로 "No redirect after POST"/403 을 반환한다. 여기서는 세션을
    settings.KUBEFLOW_AUTH_SESSION_TTL_SEC 동안 재사용하고, 만료·강제(refresh) 시에만 재로그인한다.
    kfp_client·auth_session 은 접근 시 세션 유효성을 보장하는 property 라, 기존 호출부는 수정 없이 그대로 쓴다.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._configure()
                    cls._instance = inst
        return cls._instance

    def __init__(self):
        # 초기화는 __new__ 의 _configure 에서 1회만 수행한다(싱글톤 재-init 방지).
        pass

    def _configure(self):
        self.endpoint = settings.KUBEFLOW_ENDPOINT
        self.username = settings.KUBEFLOW_USERNAME
        self.password = settings.KUBEFLOW_PASSWORD
        self.namespace = settings.KUBEFLOW_NAMESPACE
        self._ttl_sec = int(settings.KUBEFLOW_AUTH_SESSION_TTL_SEC)
        self._auth_session = None
        self._kfp_client = None
        self._logged_in_at = 0.0
        self._login_lock = threading.Lock()

    def _login(self):
        logger.info("Kubeflow Dex 로그인(세션 신규/갱신)")
        self._auth_session = get_istio_auth_session(url=self.endpoint, username=self.username, password=self.password)
        self._kfp_client = kfp.Client(
            host=f"{self.endpoint}/pipeline",
            namespace=self.namespace,
            cookies=self._auth_session.session_cookie,
        )
        self._logged_in_at = time.monotonic()

    def _ensure_session(self, force: bool = False):
        def _fresh():
            return self._auth_session is not None and (time.monotonic() - self._logged_in_at) < self._ttl_sec

        if _fresh() and not force:
            return
        with self._login_lock:
            # 락 획득 후 재확인(다른 스레드가 이미 갱신했을 수 있음).
            if force or not _fresh():
                self._login()

    def refresh(self):
        """인증/세션 오류(403·No redirect 등) 발생 시 강제 재로그인한다."""
        self._ensure_session(force=True)

    @property
    def auth_session(self):
        self._ensure_session()
        return self._auth_session

    @property
    def kfp_client(self):
        self._ensure_session()
        return self._kfp_client

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
