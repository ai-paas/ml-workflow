"""워크플로우 실행 관리 모듈"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.kubeflow.kserve_manager import KServeManager
from core.kubeflow.kubeflow_manager import KubeflowManager
from db.models.service import ComponentType, Workflow, WorkflowComponent, WorkflowStatus
from kfp import dsl
from kfp.compiler import Compiler
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()


class WorkflowExecutor:
    """워크플로우 실행기"""

    def __init__(self, db: Session):
        self.db = db
        self.kf_manager = None
        self.kserve_manager = None
        self.deployed_services = {}  # component_id -> inference_service_name

    def execute_workflow(self, workflow: Workflow, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        워크플로우를 실행
        1. MODEL 컴포넌트들을 KServe로 배포
        2. Kubeflow 파이프라인으로 워크플로우 실행

        Args:
            workflow: 실행할 워크플로우
            parameters: 실행 파라미터

        Returns:
            실행 정보 (run_id, deployed_services 등)
        """
        if parameters is None:
            parameters = {}

        # 초기화
        self.kf_manager = KubeflowManager()
        self.kserve_manager = KServeManager()

        # MLflow 설정 가져오기
        mlflow_config = {
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "experiment_name": f"workflow-{workflow.id}",
            "s3_endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL,
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
        }

        try:
            # 1. MODEL 컴포넌트들을 KServe로 배포
            logger.info(f"Deploying model components for workflow {workflow.id}")
            self.deployed_services = self.kserve_manager.deploy_workflow_models(
                workflow_components=workflow.components,
                workflow_id=str(workflow.id),
                db=self.db,
                mlflow_config=mlflow_config,
            )

            # 배포된 서비스 URL을 parameters에 추가
            kserve_urls = []
            for component_id, service_name in self.deployed_services.items():
                service_url = self.kserve_manager.get_inference_service_url(service_name)
                parameters[f"{component_id}_service_url"] = service_url
                kserve_urls.append(service_url)
                logger.info(f"Component {component_id} deployed at {service_url}")

            # 워크플로우에 KServe URL 저장 (첫 번째 모델의 URL을 대표로 사용)
            if kserve_urls:
                workflow.backend_api_url = kserve_urls[0]  # 내부 API URL
                # 외부 URL은 인그레스/로드밸런서 설정에 따라 다를 수 있음
                workflow.public_url = kserve_urls[0]
                self.db.commit()

            # 2. Kubeflow 파이프라인 생성 및 실행
            logger.info(f"Creating and executing Kubeflow pipeline for workflow {workflow.id}")

            # 파이프라인 함수 생성
            pipeline_func = self._create_pipeline_function(workflow, self.deployed_services)

            # 파이프라인 컴파일
            pipeline_name = f"workflow-{workflow.id}"
            pipeline_filename = f"/tmp/{pipeline_name}-{uuid.uuid4().hex[:8]}.yaml"

            Compiler().compile(pipeline_func, pipeline_filename)

            # Kubeflow에 파이프라인 업로드 및 실행
            run_id = self._execute_kubeflow_pipeline(
                workflow=workflow,
                pipeline_filename=pipeline_filename,
                pipeline_name=pipeline_name,
                parameters=parameters,
            )

            # 워크플로우 상태 업데이트
            workflow.kubeflow_run_id = run_id
            workflow.status = WorkflowStatus.ACTIVE
            self.db.commit()

            logger.info(f"Workflow {workflow.id} successfully executed with run ID: {run_id}")

            return {
                "workflow_id": str(workflow.id),
                "kubeflow_run_id": run_id,
                "status": "running",
                "deployed_services": self.deployed_services,
                "backend_api_url": workflow.backend_api_url,
                "public_url": workflow.public_url,
                "message": "Workflow execution initiated successfully",
            }

        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow.id}: {str(e)}")

            # 실패 시 워크플로우 상태 업데이트
            workflow.status = WorkflowStatus.ERROR
            self.db.commit()

            # 배포된 서비스 정리
            self.cleanup_deployed_services(str(workflow.id))

            raise Exception(f"Workflow execution failed: {str(e)}")

    def _create_pipeline_function(self, workflow: Workflow, deployed_services: Dict[str, str]) -> callable:
        """
        워크플로우를 위한 Kubeflow 파이프라인 함수 생성

        Args:
            workflow: 워크플로우
            deployed_services: 배포된 KServe 서비스 정보

        Returns:
            파이프라인 함수
        """

        @dsl.pipeline(
            name=f"workflow-{workflow.id}-{workflow.name}",
            description=workflow.description or "Auto-generated pipeline from workflow",
        )
        def workflow_pipeline(**kwargs):
            tasks = {}

            # 워크플로우 컴포넌트를 순서대로 정렬
            sorted_components = self._sort_components_by_dependencies(workflow)

            for component in sorted_components:
                # 컴포넌트 태스크 생성
                task = self._create_component_task(
                    component=component, workflow=workflow, deployed_services=deployed_services, parameters=kwargs
                )

                if task:
                    tasks[component.component_id] = task

                    # 의존성 설정
                    dependencies = self._get_component_dependencies(component, workflow)
                    for dep_id in dependencies:
                        if dep_id in tasks:
                            task.after(tasks[dep_id])

            return tasks

        return workflow_pipeline

    def _create_component_task(
        self,
        component: WorkflowComponent,
        workflow: Workflow,
        deployed_services: Dict[str, str],
        parameters: Dict[str, Any],
    ) -> Optional[Any]:
        """
        컴포넌트를 위한 Kubeflow 태스크 생성

        Args:
            component: 워크플로우 컴포넌트
            workflow: 워크플로우
            deployed_services: 배포된 서비스 정보
            parameters: 실행 파라미터

        Returns:
            Kubeflow 태스크
        """
        # START 컴포넌트
        if component.type == ComponentType.START:

            @dsl.component(base_image="python:3.10-slim", packages_to_install=["requests", "pandas"])
            def start_component(workflow_id: str, component_id: str, config: str = "{}") -> str:
                import json
                import logging

                logging.info(f"Starting workflow {workflow_id}, component {component_id}")
                config_dict = json.loads(config)

                # 입력 데이터 처리 로직
                result = {
                    "workflow_id": workflow_id,
                    "component_id": component_id,
                    "status": "started",
                    "config": config_dict,
                }

                return json.dumps(result)

            return start_component(
                workflow_id=str(workflow.id),
                component_id=component.component_id,
                config=json.dumps(component.config or {}),
            )

        # MODEL 컴포넌트
        elif component.type == ComponentType.MODEL:
            service_name = deployed_services.get(component.component_id)
            if not service_name:
                logger.warning(f"No deployed service found for model component {component.component_id}")
                return None

            service_url = parameters.get(f"{component.component_id}_service_url", "")

            @dsl.component(base_image="python:3.10-slim", packages_to_install=["requests", "numpy", "pandas"])
            def model_component(
                workflow_id: str, component_id: str, service_url: str, input_data: str = "{}", config: str = "{}"
            ) -> str:
                import json
                import logging

                import requests

                logging.info(f"Calling model service at {service_url}")

                try:
                    # 입력 데이터 파싱
                    input_dict = json.loads(input_data)
                    config_dict = json.loads(config)

                    # KServe 인퍼런스 서비스 호출
                    response = requests.post(
                        f"{service_url}/v1/models/{component_id}:predict",
                        json={"instances": input_dict.get("data", [])},
                    )
                    response.raise_for_status()

                    result = {
                        "workflow_id": workflow_id,
                        "component_id": component_id,
                        "predictions": response.json(),
                        "config": config_dict,
                    }

                    return json.dumps(result)

                except Exception as e:
                    logging.error(f"Model component failed: {str(e)}")
                    return json.dumps({"error": str(e)})

            # 이전 태스크의 출력을 입력으로 연결
            previous_output = "{}"  # 기본값
            # dependencies = self._get_component_dependencies(component, workflow)

            return model_component(
                workflow_id=str(workflow.id),
                component_id=component.component_id,
                service_url=service_url,
                input_data=previous_output,
                config=json.dumps(component.config or {}),
            )

        # END 컴포넌트
        elif component.type == ComponentType.END:

            @dsl.component(base_image="python:3.10-slim", packages_to_install=["requests"])
            def end_component(workflow_id: str, component_id: str, input_data: str = "{}", config: str = "{}") -> str:
                import json
                import logging

                logging.info(f"Ending workflow {workflow_id}, component {component_id}")

                input_dict = json.loads(input_data)
                config_dict = json.loads(config)

                # 결과 처리 로직
                result = {
                    "workflow_id": workflow_id,
                    "component_id": component_id,
                    "status": "completed",
                    "results": input_dict,
                    "config": config_dict,
                }

                return json.dumps(result)

            return end_component(
                workflow_id=str(workflow.id),
                component_id=component.component_id,
                input_data="{}",  # 이전 태스크 출력 연결 필요
                config=json.dumps(component.config or {}),
            )

        return None

    def _sort_components_by_dependencies(self, workflow: Workflow) -> List[WorkflowComponent]:
        """
        컴포넌트를 의존성에 따라 정렬

        Args:
            workflow: 워크플로우

        Returns:
            정렬된 컴포넌트 리스트
        """
        # 간단한 타입 기반 정렬 (실제로는 연결 정보 기반 토폴로지 정렬 필요)
        type_order = {ComponentType.START: 0, ComponentType.MODEL: 1, ComponentType.END: 2}

        return sorted(workflow.components, key=lambda c: type_order.get(c.type, 99))

    def _get_component_dependencies(self, component: WorkflowComponent, workflow: Workflow) -> List[str]:
        """
        컴포넌트의 의존성 컴포넌트 ID 리스트 반환

        Args:
            component: 대상 컴포넌트
            workflow: 워크플로우

        Returns:
            의존성 컴포넌트 ID 리스트
        """
        dependencies = []

        # 컴포넌트 연결 정보에서 의존성 찾기
        for connection in workflow.component_connections:
            if connection.target_component_id == component.id:
                # 소스 컴포넌트의 component_id 찾기
                for comp in workflow.components:
                    if comp.id == connection.source_component_id:
                        dependencies.append(comp.component_id)
                        break

        return dependencies

    def _execute_kubeflow_pipeline(
        self, workflow: Workflow, pipeline_filename: str, pipeline_name: str, parameters: Dict[str, Any]
    ) -> str:
        """
        Kubeflow 파이프라인 실행

        Args:
            workflow: 워크플로우
            pipeline_filename: 컴파일된 파이프라인 파일 경로
            pipeline_name: 파이프라인 이름
            parameters: 실행 파라미터

        Returns:
            실행 ID
        """
        try:
            # 실험 생성 또는 가져오기
            experiment_name = f"workflow-{workflow.id}-experiments"
            experiment = self.kf_manager.get_experiment_by_name(experiment_name)
            if not experiment:
                experiment = self.kf_manager.create_experiment(experiment_name)

            # 파이프라인 실행
            run = self.kf_manager.kfp_client.run_pipeline(
                experiment_id=experiment.id,
                job_name=f"{pipeline_name}-run-{uuid.uuid4().hex[:8]}",
                pipeline_package_path=pipeline_filename,
                params=parameters,
            )

            # 파이프라인 ID 저장
            workflow.kubeflow_pipeline_id = run.pipeline_spec.pipeline_id if run.pipeline_spec else None

            return run.id

        finally:
            # 임시 파일 삭제
            if os.path.exists(pipeline_filename):
                os.remove(pipeline_filename)

    def cleanup_deployed_services(self, workflow_id: str) -> int:
        """
        워크플로우의 배포된 서비스 정리

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            삭제된 서비스 개수
        """
        if self.kserve_manager:
            return self.kserve_manager.cleanup_workflow_services(workflow_id)
        return 0

    def get_workflow_status(self, workflow: Workflow) -> Dict[str, Any]:
        """
        워크플로우 실행 상태 조회

        Args:
            workflow: 워크플로우

        Returns:
            상태 정보
        """
        status = {
            "workflow_id": str(workflow.id),
            "status": workflow.status.value,
            "kubeflow_run_id": workflow.kubeflow_run_id,
            "deployed_services": {},
        }

        # Kubeflow 실행 상태 확인
        if workflow.kubeflow_run_id and self.kf_manager:
            try:
                run = self.kf_manager.kfp_client.get_run(workflow.kubeflow_run_id)
                status["kubeflow_status"] = run.status
            except Exception as e:
                logger.error(f"Failed to get run status: {str(e)}")

        # 배포된 서비스 상태 확인
        if self.kserve_manager:
            for component in workflow.components:
                if component.type == ComponentType.MODEL:
                    service_name = self.deployed_services.get(component.component_id)
                    if service_name:
                        service_url = self.kserve_manager.get_inference_service_url(service_name)
                        status["deployed_services"][component.component_id] = {
                            "service_name": service_name,
                            "url": service_url,
                        }

        return status
