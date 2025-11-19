"""워크플로우 실행 관리 모듈"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from config.settings import get_settings
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
        self.kf_manager = KubeflowManager()
        self.deployed_services = {}  # component_id -> inference_service_name

    def execute_workflow(self, workflow: Workflow, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        워크플로우를 실행
        Kubeflow 파이프라인 내에서 모든 작업 수행 (KServe 배포 포함)

        Args:
            workflow: 실행할 워크플로우
            parameters: 실행 파라미터

        Returns:
            실행 정보 (run_id, deployed_services 등)
        """
        if parameters is None:
            parameters = {}

        # kf_manager는 이미 __init__에서 초기화됨

        # MLflow 설정을 parameters에 추가
        parameters["mlflow_tracking_uri"] = settings.MLFLOW_TRACKING_URI
        # 환경 변수에서 설정된 실제 MLFLOW_EXPERIMENT_NAME을 사용
        parameters["mlflow_experiment_name"] = settings.MLFLOW_EXPERIMENT_NAME
        parameters["mlflow_s3_endpoint_url"] = settings.MLFLOW_S3_ENDPOINT_URL
        parameters["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        parameters["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        parameters["mlflow_s3_bucket"] = settings.MLFLOW_S3_BUCKET

        # REST API 설정을 parameters에 추가 (DB 업데이트용)
        parameters["rest_api_url"] = settings.REST_API_URL
        parameters["restapi_username"] = "surromind"  # 고정 사용자명
        parameters["restapi_password"] = settings.DEMO_PASSWORD

        try:
            # Kubeflow 파이프라인 생성 및 실행 (KServe 배포도 파이프라인 내에서 수행)
            logger.info(f"Creating and executing Kubeflow pipeline for workflow {workflow.id}")

            # 파이프라인 함수 생성 (KServe 배포도 포함)
            pipeline_func = self._create_pipeline_function(workflow, parameters)

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

            # 워크플로우 kubeflow_run_id만 업데이트 (상태는 파이프라인 완료 시 END 컴포넌트에서 업데이트)
            workflow.kubeflow_run_id = run_id

            # KServe 배포 정보 초기화
            from services.kserve_deployment import KServeDeploymentService

            for component in workflow.components:
                if component.type == ComponentType.MODEL:
                    # KServeDeployment 레코드 생성
                    KServeDeploymentService.create_deployment(
                        db=self.db,
                        workflow_id=workflow.id,
                        component_id=component.id,
                        model_name=component.name,
                    )

            self.db.commit()

            logger.info(f"Workflow {workflow.id} successfully executed with run ID: {run_id}")

            # 배포된 모델 정보 조회 (KServeDeployment 테이블에서)
            deployed_models = KServeDeploymentService.get_deployed_models(
                self.db, workflow.id, include_component_info=True
            )

            return {
                "workflow_id": str(workflow.id),
                "kubeflow_run_id": run_id,
                "status": "running",
                "message": "Workflow execution initiated successfully. Model deployments in progress.",
                "deployed_models": deployed_models,
            }

        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow.id}: {str(e)}")

            # 실패 시 워크플로우 상태 업데이트
            workflow.status = WorkflowStatus.ERROR
            self.db.commit()

            raise Exception(f"Workflow execution failed: {str(e)}")

    def _create_pipeline_function(self, workflow: Workflow, parameters: Dict[str, Any]) -> callable:
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
        def workflow_pipeline():
            tasks = {}

            # 워크플로우 컴포넌트를 순서대로 정렬
            sorted_components = self._sort_components_by_dependencies(workflow)

            for component in sorted_components:
                # 컴포넌트 태스크 생성
                task = self._create_component_task(
                    component=component, workflow=workflow, parameters=parameters, db=self.db
                )

                if task:
                    tasks[component.id] = task

                    # 의존성 설정
                    dependencies = self._get_component_dependencies(component, workflow)
                    for dep_id in dependencies:
                        if dep_id in tasks:
                            task.after(tasks[dep_id])

            # KFP 2.0에서는 파이프라인 함수가 None을 반환해야 함

        return workflow_pipeline

    def _create_component_task(
        self,
        component: WorkflowComponent,
        workflow: Workflow,
        parameters: Dict[str, Any],
        db: Optional[Session] = None,
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
                component_id=component.id,
                config=json.dumps(component.config or {}),
            )

        # MODEL 컴포넌트
        elif component.type == ComponentType.MODEL:
            # 모델 정보 조회
            model = None
            model_uri = ""
            run_id = ""
            framework = "pytorch"

            if db and component.model_id:
                from db.models.model import Model

                model = db.query(Model).filter(Model.id == component.model_id).first()
                if model:
                    # Ollama 모델 감지 (provider가 ollama이고 format이 gguf인 경우)
                    if (
                        hasattr(model, "provider_info")
                        and model.provider_info
                        and model.provider_info.name.lower() == "ollama"
                    ):
                        if (
                            hasattr(model, "format_info")
                            and model.format_info
                            and model.format_info.name.lower() == "gguf"
                        ):
                            framework = "ollama"

                    if hasattr(model, "registry") and model.registry:
                        model_uri = model.registry.uri or ""
                        run_id = model.registry.run_id or ""

                    # framework 정보 추론
                    if hasattr(model, "format_info") and model.format_info:
                        format_name = model.format_info.name.lower()
                        if "pytorch" in format_name or "torch" in format_name:
                            framework = "pytorch"
                        elif "tensorflow" in format_name or "tf" in format_name:
                            framework = "tensorflow"
                        elif "onnx" in format_name:
                            framework = "onnx"
                        elif "transformers" in format_name:
                            framework = "transformers"
                        elif "keras" in format_name:
                            framework = "keras"
                        elif "yolox" in format_name:
                            framework = "yolox"

            # KServe 배포만 수행하는 컴포넌트 (추론은 별도로 수행)
            @dsl.component(
                base_image="python:3.10",
                packages_to_install=[
                    "mlflow==2.17.0",
                    "kserve==0.11.2",
                    "kubernetes==28.1.0",
                    "requests==2.31.0",
                ],
            )
            def model_deployment_component(
                workflow_id: str,
                component_id: str,
                model_name: str,
                model_uri: str,
                run_id: str,
                framework: str,
                mlflow_tracking_uri: str,
                mlflow_experiment_name: str,
                mlflow_s3_endpoint_url: str,
                aws_access_key_id: str,
                aws_secret_access_key: str,
                rest_api_url: str,
                restapi_username: str,
                restapi_password: str,
                infer_image_url: str,
                config: str = "{}",
                gpu_enabled: bool = False,
                repo_id: str = "",
            ) -> str:
                import json
                import logging
                import time
                import uuid

                from kserve import (
                    KServeClient,
                    V1beta1InferenceService,
                    V1beta1InferenceServiceSpec,
                    V1beta1PredictorSpec,
                    constants,
                )
                from kubernetes import client
                from kubernetes import config as k8s_config

                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)

                # Kubernetes 설정
                k8s_config.load_incluster_config()
                kserve_client = KServeClient()
                namespace = "kubeflow-user-example-com"

                # 인퍼런스 서비스 이름 생성 (DNS 1035 규칙 준수)
                # DNS 1035 규칙:
                # - 최대 63자
                # - 소문자, 숫자, 하이픈(-)만 사용
                # - 문자나 숫자로 시작하고 끝나야 함
                # - 하이픈은 연속으로 사용 불가

                import re

                # workflow_id와 component_id를 짧게 처리
                wf_hash = workflow_id.split("-")[0][:8].lower()  # 첫 8자만 사용
                comp_hash = component_id.replace("model-", "m")[:8].lower()  # model- 제거하고 축약
                unique_id = uuid.uuid4().hex[:6]  # 6자로 축소

                # 기본 서비스 이름 생성
                service_name = f"wf-{wf_hash}-{comp_hash}-{unique_id}"

                # DNS 1035 규칙에 맞게 정규화
                # 1. 소문자로 변환
                service_name = service_name.lower()

                # 2. 영문자, 숫자, 하이픈만 남기기
                service_name = re.sub(r"[^a-z0-9-]", "-", service_name)

                # 3. 연속된 하이픈 제거
                service_name = re.sub(r"-+", "-", service_name)

                # 4. 시작과 끝의 하이픈 제거
                service_name = service_name.strip("-")

                # 5. 최대 63자 제한
                if len(service_name) > 63:
                    service_name = service_name[:63].rstrip("-")

                # 6. 빈 문자열이거나 숫자로만 시작하는 경우 처리
                if not service_name or service_name[0].isdigit():
                    service_name = f"svc-{service_name}"[:63]

                logger.info(f"Generated service name: {service_name} (length: {len(service_name)})")

                try:
                    # Ollama 모델인 경우 Kubernetes 리소스 직접 생성
                    if framework == "ollama":
                        # Ollama 모델 이름은 repo_id 사용 (model 등록 시 사용한 repo_id)
                        # repo_id가 없으면 pipeline 실패
                        if not repo_id or repo_id == "":
                            error_msg = "repo_id is required for Ollama model deployment"
                            logger.error(error_msg)
                            raise ValueError(error_msg)

                        ollama_model_name = repo_id

                        logger.info(f"Deploying Ollama model: {ollama_model_name} using Kubernetes resources")

                        # Kubernetes API 클라이언트
                        apps_v1 = client.AppsV1Api()
                        core_v1 = client.CoreV1Api()

                        # PVC 이름 생성
                        pvc_name = f"{service_name}-pvc"

                        # 1. PVC 생성
                        pvc = client.V1PersistentVolumeClaim(
                            metadata=client.V1ObjectMeta(
                                name=pvc_name,
                                namespace=namespace,
                                labels={
                                    "workflow-id": workflow_id,
                                    "component-id": component_id,
                                    "app": service_name,
                                },
                            ),
                            spec=client.V1PersistentVolumeClaimSpec(
                                access_modes=["ReadWriteOnce"],
                                resources=client.V1ResourceRequirements(requests={"storage": "30Gi"}),
                            ),
                        )

                        try:
                            core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
                            logger.info(f"Created PVC: {pvc_name}")
                        except Exception as e:
                            # PVC가 이미 존재하는 경우 무시
                            if "already exists" not in str(e).lower():
                                logger.warning(f"PVC creation failed (may already exist): {e}")

                        # Ollama용 리소스 설정
                        ollama_resources = client.V1ResourceRequirements(
                            requests={
                                "memory": "8Gi",
                                "cpu": "500m",
                            },
                            limits={
                                "memory": "16Gi",
                                "cpu": "2000m",
                            },
                        )

                        if gpu_enabled:
                            ollama_resources.requests["nvidia.com/gpu"] = "1"
                            ollama_resources.limits["nvidia.com/gpu"] = "1"

                        # 2. Deployment 생성
                        deployment = client.V1Deployment(
                            metadata=client.V1ObjectMeta(
                                name=service_name,
                                namespace=namespace,
                                labels={
                                    "workflow-id": workflow_id,
                                    "component-id": component_id,
                                    "app": service_name,
                                },
                            ),
                            spec=client.V1DeploymentSpec(
                                replicas=1,
                                selector=client.V1LabelSelector(
                                    match_labels={
                                        "app": service_name,
                                    }
                                ),
                                template=client.V1PodTemplateSpec(
                                    metadata=client.V1ObjectMeta(
                                        labels={
                                            "app": service_name,
                                            "workflow-id": workflow_id,
                                            "component-id": component_id,
                                        },
                                    ),
                                    spec=client.V1PodSpec(
                                        containers=[
                                            client.V1Container(
                                                name="ollama",
                                                image="ollama/ollama:latest",
                                                command=["/bin/sh", "-c"],
                                                args=[
                                                    (
                                                        f"ollama serve & SERVE_PID=$! && "
                                                        f"sleep 10 && ollama pull {ollama_model_name} && "
                                                        f"wait $SERVE_PID"
                                                    )
                                                ],
                                                ports=[
                                                    client.V1ContainerPort(
                                                        container_port=11434, name="http", protocol="TCP"
                                                    )
                                                ],
                                                resources=ollama_resources,
                                                env=[
                                                    client.V1EnvVar(name="WORKFLOW_ID", value=workflow_id),
                                                    client.V1EnvVar(name="COMPONENT_ID", value=component_id),
                                                    client.V1EnvVar(name="OLLAMA_MODEL", value=ollama_model_name),
                                                ],
                                                volume_mounts=[
                                                    client.V1VolumeMount(name="model-data", mount_path="/root/.ollama")
                                                ],
                                                readiness_probe=client.V1Probe(
                                                    http_get=client.V1HTTPGetAction(path="/api/tags", port=11434),
                                                    initial_delay_seconds=30,
                                                    period_seconds=10,
                                                ),
                                            )
                                        ],
                                        volumes=[
                                            client.V1Volume(
                                                name="model-data",
                                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                                    claim_name=pvc_name
                                                ),
                                            )
                                        ],
                                    ),
                                ),
                            ),
                        )

                        # Deployment 생성
                        apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)
                        logger.info(f"Created Deployment: {service_name}")

                        # 3. Service 생성 (ClusterIP 타입)
                        service = client.V1Service(
                            metadata=client.V1ObjectMeta(
                                name=service_name,
                                namespace=namespace,
                                labels={
                                    "workflow-id": workflow_id,
                                    "component-id": component_id,
                                    "app": service_name,
                                },
                            ),
                            spec=client.V1ServiceSpec(
                                type="ClusterIP",
                                selector={"app": service_name},
                                ports=[
                                    client.V1ServicePort(
                                        port=11434,
                                        target_port=11434,
                                        name="http",
                                        protocol="TCP",
                                    )
                                ],
                            ),
                        )

                        # Service 생성
                        created_service = core_v1.create_namespaced_service(namespace=namespace, body=service)
                        logger.info(f"Created Service: {service_name}")

                        # Service의 실제 정보 가져오기
                        service_cluster_ip = created_service.spec.cluster_ip
                        service_port = created_service.spec.ports[0].port

                        # Ollama는 KServe를 사용하지 않으므로 service_hostname 불필요
                        # internal_url만 실제 ClusterIP:Port로 사용
                        service_hostname = ""  # Ollama는 사용하지 않음
                        internal_url = f"http://{service_cluster_ip}:{service_port}"

                        logger.info(f"Service ClusterIP: {service_cluster_ip}, Port: {service_port}")
                        logger.info(f"Internal URL: {internal_url}")

                        # Deployment가 준비될 때까지 대기 (최대 10분)
                        max_wait = 600  # 10분
                        wait_interval = 15  # 15초 간격
                        elapsed = 0
                        deployment_ready = False

                        while elapsed < max_wait:
                            try:
                                deployment_status = apps_v1.read_namespaced_deployment_status(
                                    name=service_name, namespace=namespace
                                )

                                if (
                                    deployment_status.status.ready_replicas
                                    and deployment_status.status.ready_replicas >= 1
                                ):
                                    logger.info(f"Deployment {service_name} is ready")
                                    deployment_ready = True
                                    break

                                time.sleep(wait_interval)
                                elapsed += wait_interval

                            except Exception as e:
                                logger.warning(f"Error checking deployment status: {e}")
                                time.sleep(wait_interval)
                                elapsed += wait_interval

                        if not deployment_ready:
                            logger.warning(f"Deployment {service_name} not ready after {max_wait} seconds")

                        # 배포 상태 결정
                        deployment_status = "deployed" if deployment_ready else "deploying"

                        # DB 업데이트를 위해 Backend API 호출
                        try:
                            import requests

                            if not rest_api_url:
                                logger.warning("REST_API_URL not provided, skipping DB update")
                            else:
                                # 토큰 발급
                                auth_token = None
                                if restapi_username and restapi_password:
                                    try:
                                        token_response = requests.post(
                                            f"{rest_api_url}/api/v1/authentications/token",
                                            data={"username": restapi_username, "password": restapi_password},
                                            timeout=10,
                                        )
                                        if token_response.status_code == 200:
                                            auth_token = token_response.json().get("access_token")
                                            logger.info("Successfully obtained authentication token")
                                        else:
                                            logger.warning(f"Failed to get auth token: {token_response.status_code}")
                                    except Exception as token_error:
                                        logger.warning(f"Failed to obtain auth token: {token_error}")

                                update_url = (
                                    f"{rest_api_url}/api/v1/workflows/{workflow_id}/"
                                    f"components/{component_id}/deployment-status"
                                )

                                update_payload = {
                                    "service_name": service_name,
                                    "service_hostname": service_hostname,  # Ollama는 빈 문자열이지만 형식 유지
                                    "model_name": ollama_model_name,
                                    "status": deployment_status,
                                    "internal_url": internal_url,
                                    "error_message": None,
                                }

                                headers = {"Content-Type": "application/json"}
                                if auth_token:
                                    headers["Authorization"] = f"Bearer {auth_token}"

                                logger.info("Updating Ollama deployment status via API: %s", update_url)
                                response = requests.post(update_url, json=update_payload, headers=headers, timeout=10)

                                if response.status_code == 200:
                                    logger.info("Successfully updated Ollama deployment status in DB")

                                    # deployment-status 업데이트 성공 후 워크플로우 상태를 ACTIVE로 업데이트
                                    if deployment_ready and deployment_status == "deployed":
                                        try:
                                            workflow_update_url = f"{rest_api_url}/api/v1/workflows/{workflow_id}"
                                            workflow_update_data = {"status": "ACTIVE"}
                                            workflow_update_response = requests.put(
                                                workflow_update_url,
                                                json=workflow_update_data,
                                                headers=headers,
                                                timeout=10,
                                            )
                                            if workflow_update_response.status_code == 200:
                                                logger.info(
                                                    f"Successfully updated workflow {workflow_id} status to ACTIVE"
                                                )
                                            else:
                                                status_code = workflow_update_response.status_code
                                                status_text = workflow_update_response.text
                                                logger.warning(
                                                    f"Failed to update workflow status: "
                                                    f"{status_code} - {status_text}"
                                                )
                                        except Exception as workflow_update_error:
                                            logger.error(
                                                f"Error updating workflow status to ACTIVE: {workflow_update_error}"
                                            )
                                else:
                                    status_code = response.status_code
                                    status_text = response.text
                                    logger.warning(
                                        f"Failed to update Ollama deployment status: " f"{status_code} - {status_text}"
                                    )

                        except Exception as e:
                            logger.error(f"Error updating Ollama deployment status in DB: {e}")
                            # DB 업데이트 실패해도 배포 자체는 성공이므로 계속 진행

                        # 배포 상태 업데이트를 위한 정보 반환
                        return json.dumps(
                            {
                                "service_name": service_name,
                                "service_hostname": service_hostname,
                                "model_name": ollama_model_name,
                                "status": deployment_status,
                                "internal_url": internal_url,
                            }
                        )

                    # 기존 MLflow 모델 배포 방식
                    # 컨테이너 args 구성
                    container_args = [
                        f"--model_name={model_name}",
                        f"--model_uri={model_uri}",
                        f"--mlflow_tracking_uri={mlflow_tracking_uri}",
                        f"--mlflow_experiment_name={mlflow_experiment_name}",  # 이미 올바른 값이 전달됨
                        f"--mlflow_s3_endpoint_url={mlflow_s3_endpoint_url}",
                        f"--aws_access_key_id={aws_access_key_id}",
                        f"--aws_secret_access_key={aws_secret_access_key}",
                        f"--framework={framework}",
                    ]

                    if run_id:
                        container_args.append(f"--run_id={run_id}")

                    # 리소스 설정 (ephemeral-storage requests는 제거)
                    resources = client.V1ResourceRequirements(
                        requests={
                            "memory": "2Gi",
                            "cpu": "200m",
                            # ephemeral-storage requests 제거 - 노드 리소스 부족 시 스케줄링 방해 방지
                        },
                        limits={"memory": "4Gi", "cpu": "500m", "ephemeral-storage": "1Gi"},  # 폭주 방지용 제한만 설정
                    )

                    if gpu_enabled:
                        resources.requests["nvidia.com/gpu"] = "1"
                        resources.limits["nvidia.com/gpu"] = "1"

                    # Predictor 스펙 생성
                    predictor_spec = V1beta1PredictorSpec(
                        min_replicas=1,
                        containers=[
                            client.V1Container(
                                name="kserve-container",
                                image=infer_image_url,
                                args=container_args,
                                resources=resources,
                                env=[
                                    client.V1EnvVar(name="WORKFLOW_ID", value=workflow_id),
                                    client.V1EnvVar(name="COMPONENT_ID", value=component_id),
                                ],
                            )
                        ],
                    )

                    # InferenceService 생성
                    inference_service = V1beta1InferenceService(
                        api_version=constants.KSERVE_V1BETA1,
                        kind=constants.KSERVE_KIND,
                        metadata=client.V1ObjectMeta(
                            name=service_name,
                            namespace=namespace,
                            labels={
                                "workflow-id": workflow_id,
                                "component-id": component_id,
                            },
                            annotations={
                                "serving.kserve.io/enable-metric-aggregation": "true",
                                "serving.kserve.io/enable-prometheus-scraping": "true",
                            },
                        ),
                        spec=V1beta1InferenceServiceSpec(predictor=predictor_spec),
                    )

                    # KServe에 배포
                    logger.info(f"Deploying model service {service_name}")
                    kserve_client.create(inference_service, namespace=namespace)

                    # 서비스가 준비될 때까지 대기 (최대 10분, ephemeral-storage 문제 대응)
                    max_wait = 600  # 10분으로 증가
                    wait_interval = 15  # 15초 간격
                    elapsed = 0
                    service_ready = False
                    evicted_count = 0
                    last_status = None

                    # Pod 상태 확인을 위한 k8s client
                    k8s_v1 = client.CoreV1Api()

                    while elapsed < max_wait:
                        try:
                            # InferenceService 상태 확인
                            status = kserve_client.get(service_name, namespace=namespace)
                            conditions = status.get("status", {}).get("conditions", [])

                            # Ready 상태 확인
                            ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

                            if ready:
                                logger.info(f"Service {service_name} is ready")
                                service_ready = True
                                break

                            # 실패 상태 확인
                            failed = any(c.get("type") == "Ready" and c.get("status") == "False" for c in conditions)

                            if failed:
                                # 실패 이유 확인
                                for c in conditions:
                                    if c.get("type") == "Ready" and c.get("status") == "False":
                                        reason = c.get("reason", "")
                                        message = c.get("message", "")
                                        logger.warning(f"Service not ready: {reason} - {message}")

                                        # RevisionMissing은 Pod가 생성 중이거나 Evicted된 경우
                                        if "RevisionMissing" in reason or "Evicted" in message:
                                            evicted_count += 1
                                            logger.info(
                                                f"Pod may have been evicted (count: {evicted_count}), "
                                                f"waiting for retry..."
                                            )

                            # Pod 직접 확인 (Evicted 상태 감지)
                            try:
                                pods = k8s_v1.list_namespaced_pod(
                                    namespace=namespace,
                                    label_selector=f"serving.kserve.io/inferenceservice={service_name}",
                                )
                                for pod in pods.items:
                                    if pod.status.phase == "Failed" and pod.status.reason == "Evicted":
                                        logger.warning(f"Pod {pod.metadata.name} was evicted: {pod.status.message}")
                                        # Evicted 메시지에서 노드 리소스 부족 정보 추출
                                        if "ephemeral-storage" in str(pod.status.message):
                                            logger.info("  → Node-level ephemeral-storage shortage detected")
                                        evicted_count += 1
                            except Exception as pod_error:
                                logger.debug(f"Could not check pod status: {pod_error}")

                            # 진행 상황 로그 (30초마다)
                            if elapsed % 30 == 0:
                                logger.info(f"Waiting for service {service_name}... ({elapsed}s elapsed)")
                                if evicted_count > 0:
                                    logger.info(
                                        f"  Eviction detected {evicted_count} time(s), KServe will retry automatically"
                                    )

                        except Exception as e:
                            logger.warning(f"Error checking service status: {e}")

                        time.sleep(wait_interval)
                        elapsed += wait_interval

                    if not service_ready:
                        if evicted_count > 0:
                            logger.warning(
                                f"Service {service_name} not ready after {max_wait}s, "
                                f"but detected {evicted_count} eviction(s)"
                            )
                            logger.info("KServe will continue retrying in the background")
                        else:
                            logger.warning(f"Service {service_name} may not be fully ready after {max_wait} seconds")

                    # 서비스 URL 정보 생성
                    internal_url = f"http://{service_name}.{namespace}.svc.cluster.local"

                    # 외부 접근을 위한 정보 (Istio Gateway 경유)
                    gateway_url = "http://10.10.30.154:80"  # Istio Gateway External IP
                    service_hostname = f"{service_name}.{namespace}.example.com"

                    config_dict = json.loads(config)

                    # 상태 결정 (evicted 포함)
                    if service_ready:
                        deployment_status = "deployed"
                    elif evicted_count > 0:
                        deployment_status = "deploying"  # Evicted 후 재시도 중
                        logger.info(f"Setting status to 'deploying' due to {evicted_count} eviction(s)")
                    else:
                        deployment_status = "deploying"  # 아직 배포 중

                    # DB 업데이트를 위해 Backend API 호출
                    try:
                        import requests

                        if not rest_api_url:
                            logger.warning("REST_API_URL not provided, skipping DB update")
                        else:
                            # 토큰 발급
                            auth_token = None
                            if restapi_username and restapi_password:
                                try:
                                    token_response = requests.post(
                                        f"{rest_api_url}/api/v1/authentications/token",
                                        data={"username": restapi_username, "password": restapi_password},
                                        timeout=10,
                                    )
                                    if token_response.status_code == 200:
                                        auth_token = token_response.json().get("access_token")
                                        logger.info("Successfully obtained authentication token")
                                    else:
                                        logger.warning(f"Failed to get auth token: {token_response.status_code}")
                                except Exception as token_error:
                                    logger.warning(f"Failed to obtain auth token: {token_error}")

                            update_url = (
                                f"{rest_api_url}/api/v1/workflows/{workflow_id}/"
                                f"components/{component_id}/deployment-status"
                            )

                            # error_message 설정
                            error_msg = None
                            if evicted_count > 0:
                                error_msg = (
                                    f"Pod evicted {evicted_count} time(s) due to node resource shortage, "
                                    f"KServe retrying..."
                                )

                            update_payload = {
                                "service_name": service_name,
                                "service_hostname": service_hostname,
                                "model_name": model_name,
                                "status": deployment_status,
                                "internal_url": internal_url,
                                "error_message": error_msg,
                            }

                            headers = {"Content-Type": "application/json"}
                            if auth_token:
                                headers["Authorization"] = f"Bearer {auth_token}"

                            logger.info("Updating deployment status via API: %s", update_url)
                            response = requests.post(update_url, json=update_payload, headers=headers, timeout=10)

                            if response.status_code == 200:
                                logger.info("Successfully updated deployment status in DB")

                                # deployment-status 업데이트 성공 후 워크플로우 상태를 ACTIVE로 업데이트
                                if service_ready and deployment_status == "deployed":
                                    try:
                                        workflow_update_url = f"{rest_api_url}/api/v1/workflows/{workflow_id}"
                                        workflow_update_data = {"status": "ACTIVE"}
                                        workflow_update_response = requests.put(
                                            workflow_update_url,
                                            json=workflow_update_data,
                                            headers=headers,
                                            timeout=10,
                                        )
                                        if workflow_update_response.status_code == 200:
                                            logger.info(f"Successfully updated workflow {workflow_id} status to ACTIVE")
                                        else:
                                            logger.warning(
                                                f"Failed to update workflow status: "
                                                f"{workflow_update_response.status_code} \
                                                - {workflow_update_response.text}"
                                            )
                                    except Exception as workflow_update_error:
                                        logger.error(
                                            f"Error updating workflow status to ACTIVE: {workflow_update_error}"
                                        )
                            else:
                                logger.warning(
                                    f"Failed to update deployment status: {response.status_code} - {response.text}"
                                )

                    except Exception as e:
                        logger.error(f"Error updating deployment status in DB: {e}")
                        # DB 업데이트 실패해도 배포 자체는 성공이므로 계속 진행

                    # 메시지 생성
                    if service_ready:
                        message = (
                            f"Model deployed successfully. Access via gateway: {gateway_url} "
                            f"with Host: {service_hostname}"
                        )
                    elif evicted_count > 0:
                        message = (
                            f"Model deployment in progress. Pod evicted {evicted_count} time(s) due to "
                            f"node resource shortage. KServe is retrying automatically. "
                            f"Service: {service_name}"
                        )
                    else:
                        message = (
                            f"Model deployment in progress. Service: {service_name}. "
                            f"Check status at gateway: {gateway_url} with Host: {service_hostname}"
                        )

                    result = {
                        "workflow_id": workflow_id,
                        "component_id": component_id,
                        "service_name": service_name,
                        "internal_url": internal_url,
                        "gateway_url": gateway_url,
                        "service_hostname": service_hostname,
                        "model_name": model_name,
                        "status": deployment_status,
                        "config": config_dict,
                        "message": message,
                        "evicted_count": evicted_count,  # 추가 정보
                    }

                    logger.info(f"Model deployment completed: {json.dumps(result)}")
                    return json.dumps(result)

                except Exception as e:
                    logger.error(f"Model deployment failed: {str(e)}")

                    # 실패한 경우에도 DB 업데이트
                    try:
                        import requests

                        if rest_api_url:
                            # 토큰 발급
                            auth_token = None
                            if restapi_username and restapi_password:
                                try:
                                    token_response = requests.post(
                                        f"{rest_api_url}/api/v1/authentications/token",
                                        data={"username": restapi_username, "password": restapi_password},
                                        timeout=10,
                                    )
                                    if token_response.status_code == 200:
                                        auth_token = token_response.json().get("access_token")
                                except Exception as token_error:
                                    logger.warning(f"Failed to obtain auth token for failure update: {token_error}")

                            update_url = (
                                f"{rest_api_url}/api/v1/workflows/{workflow_id}/"
                                f"components/{component_id}/deployment-status"
                            )

                            update_payload = {
                                "service_name": (
                                    service_name
                                    if "service_name" in locals()
                                    else f"failed-{workflow_id[:8]}-{component_id[:8]}"
                                ),
                                "service_hostname": "failed",
                                "model_name": model_name,
                                "status": "failed",
                                "internal_url": None,
                                "error_message": str(e),
                            }

                            headers = {"Content-Type": "application/json"}
                            if auth_token:
                                headers["Authorization"] = f"Bearer {auth_token}"

                            requests.post(update_url, json=update_payload, headers=headers, timeout=10)
                    except Exception as db_error:
                        logger.error(f"Failed to update DB with failure status: {db_error}")

                    return json.dumps(
                        {
                            "error": str(e),
                            "service_name": service_name if "service_name" in locals() else None,
                            "status": "failed",
                        }
                    )

            # 컴포넌트 실행 (배포만 수행)
            # model_name에 슬래시(/)가 있으면 하이픈(-)으로 변경 (Kubernetes 리소스 이름 규칙)
            sanitized_model_name = model.name.replace("/", "-") if model else "model"

            # repo_id 추출 (Ollama 모델용)
            repo_id_value = model.repo_id if model and model.repo_id else ""

            return model_deployment_component(
                workflow_id=str(workflow.id),
                component_id=component.id,
                model_name=sanitized_model_name,
                model_uri=model_uri,
                run_id=run_id,
                framework=framework,
                mlflow_tracking_uri=parameters.get("mlflow_tracking_uri", ""),
                mlflow_experiment_name=parameters.get("mlflow_experiment_name", ""),
                mlflow_s3_endpoint_url=parameters.get("mlflow_s3_endpoint_url", ""),
                aws_access_key_id=parameters.get("aws_access_key_id", ""),
                aws_secret_access_key=parameters.get("aws_secret_access_key", ""),
                rest_api_url=parameters.get("rest_api_url", ""),
                restapi_username=parameters.get("restapi_username", ""),
                restapi_password=parameters.get("restapi_password", ""),
                infer_image_url=settings.INFER_IMAGE_URL,
                config=json.dumps(component.config or {}),
                gpu_enabled=component.config.get("gpu_enabled", False) if component.config else False,
                repo_id=repo_id_value,
            )

        # END 컴포넌트
        elif component.type == ComponentType.END:

            @dsl.component(base_image="python:3.10-slim", packages_to_install=["requests"])
            def end_component(
                workflow_id: str,
                component_id: str,
                input_data: str = "{}",
                config: str = "{}",
                rest_api_url: str = "",
                restapi_username: str = "",
                restapi_password: str = "",
            ) -> str:
                import json
                import logging

                import requests

                logging.info(f"Ending workflow {workflow_id}, component {component_id}")

                input_dict = json.loads(input_data)
                config_dict = json.loads(config)

                # 결과 처리 로직
                # 참고: 워크플로우 상태는 MODEL component에서 배포 완료 시 ACTIVE로 업데이트됨
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
                component_id=component.id,
                input_data="{}",  # 이전 태스크 출력 연결 필요
                config=json.dumps(component.config or {}),
                rest_api_url=parameters.get("rest_api_url", ""),
                restapi_username=parameters.get("restapi_username", ""),
                restapi_password=parameters.get("restapi_password", ""),
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
                # 소스 컴포넌트의 id 찾기
                for comp in workflow.components:
                    if comp.id == connection.source_component_id:
                        dependencies.append(comp.id)
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
            # 실험 생성 또는 가져오기 (환경변수의 KUBEFLOW_EXPERIMENT_NAME 사용)
            experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
            experiment = self.kf_manager.get_experiment_by_name(experiment_name=experiment_name)
            if not experiment:
                experiment = self.kf_manager.create_experiment(experiment_name)

            # 파이프라인 실행
            # KFP 2.0에서는 experiment.experiment_id 사용
            exp_id = experiment.experiment_id if hasattr(experiment, "experiment_id") else experiment.id
            # 파이프라인 함수가 파라미터를 받지 않으므로 params를 전달하지 않음
            run = self.kf_manager.kfp_client.run_pipeline(
                experiment_id=exp_id,
                job_name=f"{pipeline_name}-run-{uuid.uuid4().hex[:8]}",
                pipeline_package_path=pipeline_filename,
                enable_caching=False,  # 캐시 비활성화 - 매번 새로 실행
                # params=parameters,  # 파이프라인 내부에서 클로저로 접근하므로 제거
            )

            # 파이프라인 ID 저장
            # run 객체 타입 확인을 위한 로깅
            logger.info(f"Run object type: {type(run)}")

            # V2beta1Run 객체 처리
            if hasattr(run, "__class__") and "V2beta1Run" in str(type(run)):
                # KFP v2 API의 V2beta1Run 객체
                logger.info("Processing V2beta1Run object")

                # run_id 추출
                if hasattr(run, "run_id"):
                    run_id = run.run_id
                elif hasattr(run, "id"):
                    run_id = run.id
                else:
                    # 속성을 dict로 변환해서 확인
                    run_dict = run.to_dict() if hasattr(run, "to_dict") else {}
                    run_id = run_dict.get("run_id") or run_dict.get("id")

            elif isinstance(run, dict):
                # dict 형태인 경우
                logger.info(f"Run dict keys: {run.keys()}")
                run_id = run.get("run_id") or run.get("id")
            else:
                # 기타 객체 형태인 경우
                logger.info(f"Run object attributes: {dir(run)}")
                # ApiRunDetail 객체의 경우
                if hasattr(run, "run"):
                    actual_run = run.run
                    run_id = actual_run.id if hasattr(actual_run, "id") else None
                else:
                    run_id = run.run_id if hasattr(run, "run_id") else (run.id if hasattr(run, "id") else None)

            if not run_id:
                logger.warning("Could not extract run_id from response")
                run_id = f"unknown-{uuid.uuid4().hex[:8]}"

            logger.info(f"Extracted run_id: {run_id}")
            return run_id

        finally:
            # 임시 파일 삭제
            if os.path.exists(pipeline_filename):
                os.remove(pipeline_filename)

    def cleanup_deployed_services(self, workflow_id: str) -> Dict[str, Any]:
        """
        워크플로우의 배포된 서비스 정리
        Kubeflow Pipeline을 통해 배포된 KServe 서비스들을 정리

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            삭제 결과 정보
        """
        try:
            # kf_manager는 이미 __init__에서 초기화됨

            logger.info(f"Creating cleanup pipeline for workflow {workflow_id}")

            # 파이프라인 함수 생성
            pipeline_func = self._create_cleanup_pipeline_function(workflow_id)

            # 파이프라인 컴파일
            pipeline_name = f"cleanup-workflow-{workflow_id}"
            pipeline_filename = f"/tmp/{pipeline_name}-{uuid.uuid4().hex[:8]}.yaml"

            Compiler().compile(pipeline_func, pipeline_filename)

            # 파이프라인 실행 (환경변수의 KUBEFLOW_EXPERIMENT_NAME 사용)
            experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
            experiment = self.kf_manager.get_experiment_by_name(experiment_name=experiment_name)
            if not experiment:
                experiment = self.kf_manager.create_experiment(experiment_name)

            exp_id = experiment.experiment_id if hasattr(experiment, "experiment_id") else experiment.id

            run = self.kf_manager.kfp_client.run_pipeline(
                experiment_id=exp_id,
                job_name=f"{pipeline_name}-run-{uuid.uuid4().hex[:8]}",
                pipeline_package_path=pipeline_filename,
                enable_caching=False,
            )

            # run_id 추출
            if hasattr(run, "run_id"):
                run_id = run.run_id
            elif hasattr(run, "id"):
                run_id = run.id
            else:
                run_dict = run.to_dict() if hasattr(run, "to_dict") else {}
                run_id = run_dict.get("run_id") or run_dict.get("id") or f"unknown-{uuid.uuid4().hex[:8]}"

            logger.info(f"Cleanup pipeline started with run_id: {run_id}")

            # 임시 파일 삭제
            if os.path.exists(pipeline_filename):
                os.remove(pipeline_filename)

            return {
                "workflow_id": workflow_id,
                "cleanup_run_id": run_id,
                "status": "cleanup_initiated",
                "message": "Cleanup pipeline started successfully",
            }

        except Exception as e:
            logger.error(f"Failed to start cleanup pipeline: {e}")
            raise Exception(f"Failed to start cleanup pipeline: {str(e)}")

    def _create_cleanup_pipeline_function(self, workflow_id: str) -> callable:
        """
        InferenceService 삭제를 위한 Kubeflow 파이프라인 함수 생성

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            파이프라인 함수
        """

        @dsl.pipeline(
            name=f"cleanup-workflow-{workflow_id}",
            description=f"Cleanup InferenceServices for workflow {workflow_id}",
        )
        def cleanup_pipeline():
            self._create_cleanup_component_task(workflow_id)

        return cleanup_pipeline

    def _create_cleanup_component_task(self, workflow_id: str) -> Any:
        """
        InferenceService 삭제 컴포넌트 태스크 생성

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            Kubeflow 태스크
        """

        @dsl.component(
            base_image="python:3.10",
            packages_to_install=[
                "kubernetes==28.1.0",
            ],
        )
        def cleanup_inference_services(workflow_id: str) -> str:
            import json
            import logging

            from kubernetes import client
            from kubernetes import config as k8s_config

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                # Kubernetes 설정
                k8s_config.load_incluster_config()

                namespace = "kubeflow-user-example-com"
                label_selector = f"workflow-id={workflow_id}"

                deleted_count = 0
                failed_count = 0
                total_count = 0

                # 1. KServe InferenceService 삭제
                try:
                    api = client.CustomObjectsApi()
                    logger.info(f"Querying InferenceServices for workflow: {workflow_id}")

                    result = api.list_namespaced_custom_object(
                        group="serving.kserve.io",
                        version="v1beta1",
                        namespace=namespace,
                        plural="inferenceservices",
                        label_selector=label_selector,
                    )

                    services = result.get("items", [])
                    logger.info(f"Found {len(services)} InferenceServices for workflow {workflow_id}")
                    total_count += len(services)

                    # CustomObjectsApi로 각 서비스 삭제
                    for service in services:
                        service_name = service.get("metadata", {}).get("name")
                        if service_name:
                            try:
                                logger.info(f"Deleting InferenceService: {service_name} in namespace: {namespace}")

                                # InferenceService 삭제
                                api.delete_namespaced_custom_object(
                                    group="serving.kserve.io",
                                    version="v1beta1",
                                    namespace=namespace,
                                    plural="inferenceservices",
                                    name=service_name,
                                )

                                deleted_count += 1
                                logger.info(f"Successfully deleted InferenceService: {service_name}")

                            except client.exceptions.ApiException as e:
                                if e.status == 404:
                                    logger.warning(f"InferenceService not found: {service_name}")
                                    # 404는 이미 없는 것이므로 failed에 카운트하지 않음
                                else:
                                    logger.error(
                                        f"Failed to delete InferenceService {service_name}: {e.status} - {e.reason}"
                                    )
                                    failed_count += 1
                            except Exception as e:
                                logger.error(f"Unexpected error deleting InferenceService {service_name}: {e}")
                                failed_count += 1
                        else:
                            logger.warning("Service name not found in service metadata")
                            failed_count += 1
                except Exception as e:
                    logger.warning(f"Error cleaning up InferenceServices: {e}")

                # 2. Ollama 리소스 삭제 (Deployment, Service, PVC)
                try:
                    apps_v1 = client.AppsV1Api()
                    core_v1 = client.CoreV1Api()

                    # Deployment 삭제
                    logger.info(f"Querying Deployments for workflow: {workflow_id}")
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    deployment_items = deployments.items
                    logger.info(f"Found {len(deployment_items)} Deployments for workflow {workflow_id}")
                    total_count += len(deployment_items)

                    for deployment in deployment_items:
                        deployment_name = deployment.metadata.name
                        try:
                            logger.info(f"Deleting Deployment: {deployment_name} in namespace: {namespace}")
                            apps_v1.delete_namespaced_deployment(
                                name=deployment_name,
                                namespace=namespace,
                            )
                            deleted_count += 1
                            logger.info(f"Successfully deleted Deployment: {deployment_name}")
                        except client.exceptions.ApiException as e:
                            if e.status == 404:
                                logger.warning(f"Deployment not found: {deployment_name}")
                            else:
                                logger.error(f"Failed to delete Deployment {deployment_name}: {e.status} - {e.reason}")
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Unexpected error deleting Deployment {deployment_name}: {e}")
                            failed_count += 1

                    # Service 삭제
                    logger.info(f"Querying Services for workflow: {workflow_id}")
                    services = core_v1.list_namespaced_service(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    service_items = services.items
                    logger.info(f"Found {len(service_items)} Services for workflow {workflow_id}")
                    total_count += len(service_items)

                    for service in service_items:
                        service_name = service.metadata.name
                        try:
                            logger.info(f"Deleting Service: {service_name} in namespace: {namespace}")
                            core_v1.delete_namespaced_service(
                                name=service_name,
                                namespace=namespace,
                            )
                            deleted_count += 1
                            logger.info(f"Successfully deleted Service: {service_name}")
                        except client.exceptions.ApiException as e:
                            if e.status == 404:
                                logger.warning(f"Service not found: {service_name}")
                            else:
                                logger.error(f"Failed to delete Service {service_name}: {e.status} - {e.reason}")
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Unexpected error deleting Service {service_name}: {e}")
                            failed_count += 1

                    # PVC 삭제
                    logger.info(f"Querying PVCs for workflow: {workflow_id}")
                    pvcs = core_v1.list_namespaced_persistent_volume_claim(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    pvc_items = pvcs.items
                    logger.info(f"Found {len(pvc_items)} PVCs for workflow {workflow_id}")
                    total_count += len(pvc_items)

                    for pvc in pvc_items:
                        pvc_name = pvc.metadata.name
                        try:
                            logger.info(f"Deleting PVC: {pvc_name} in namespace: {namespace}")
                            core_v1.delete_namespaced_persistent_volume_claim(
                                name=pvc_name,
                                namespace=namespace,
                            )
                            deleted_count += 1
                            logger.info(f"Successfully deleted PVC: {pvc_name}")
                        except client.exceptions.ApiException as e:
                            if e.status == 404:
                                logger.warning(f"PVC not found: {pvc_name}")
                            else:
                                logger.error(f"Failed to delete PVC {pvc_name}: {e.status} - {e.reason}")
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Unexpected error deleting PVC {pvc_name}: {e}")
                            failed_count += 1

                except Exception as e:
                    logger.warning(f"Error cleaning up Ollama resources: {e}")

                result_data = {
                    "workflow_id": workflow_id,
                    "deleted": deleted_count,
                    "failed": failed_count,
                    "total": total_count,
                    "status": "completed",
                }

                logger.info(f"Cleanup completed: {json.dumps(result_data)}")
                return json.dumps(result_data)

            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")
                error_result = {
                    "workflow_id": workflow_id,
                    "deleted": 0,
                    "failed": 0,
                    "total": 0,
                    "status": "failed",
                    "error": str(e),
                }
                return json.dumps(error_result)

        return cleanup_inference_services(workflow_id=workflow_id)

    def get_workflow_status(self, workflow: Workflow) -> Dict[str, Any]:
        """
        워크플로우 실행 상태 조회 (DB 기반)

        워크플로우의 기본 상태와 배포된 모델들의 상태를 조회합니다.
        kserve_deployments 테이블의 정보를 기반으로 상태를 반환하며,
        Kubernetes나 Kubeflow를 직접 조회하지 않습니다.

        Args:
            workflow (Workflow): 상태를 조회할 워크플로우 객체

        Returns:
            Dict[str, Any]: 워크플로우 상태 정보
                - workflow_id (str): 워크플로우 UUID
                - status (str): 워크플로우 상태 (DRAFT/ACTIVE/ERROR)
                - kubeflow_run_id (str, optional): Kubeflow 파이프라인 실행 ID
                - deployed_models (List[Dict]): 배포된 모델 목록
                    - 각 항목은 KServeDeploymentService.get_deployed_models()의 반환 형식과 동일

        Process:
            1. 워크플로우 기본 정보 수집 (id, status, kubeflow_run_id)
            2. KServeDeploymentService.get_deployed_models()로 배포 정보 조회
            3. 결과 반환

        Note:
            - 모든 조회는 DB 기반으로 수행됩니다
            - kubeflow_run_id는 참조용으로만 포함되며, 실제 파이프라인 상태는 조회하지 않습니다
            - 배포 상태는 kserve_deployments 테이블의 정보를 기반으로 합니다
        """
        from services.kserve_deployment import KServeDeploymentService

        status = {
            "workflow_id": str(workflow.id),
            "status": workflow.status.value,
            "kubeflow_run_id": workflow.kubeflow_run_id,
            "deployed_models": [],
        }

        # DB 기반 배포 상태 조회 (kserve_deployments 테이블)
        if self.db:
            try:
                deployed_models = KServeDeploymentService.get_deployed_models(
                    self.db, str(workflow.id), include_component_info=True
                )

                status["deployed_models"] = deployed_models

            except Exception as e:
                logger.error(f"Failed to get deployed models from DB: {e}")

        return status
