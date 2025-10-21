"""KServe 배포 관리 모듈"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from db.models.model import Model, ModelRegistry
from db.models.service import ComponentType, WorkflowComponent
from kserve import KServeClient, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec, constants
from kubernetes import client, config
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class KServeManager:
    """KServe 배포 관리자"""

    def __init__(self, namespace: str = "kubeflow-user-example-com"):
        """
        KServe 매니저 초기화

        Args:
            namespace: Kubernetes 네임스페이스 (기본값: kubeflow-user-example-com)
        """
        try:
            config.load_incluster_config()
        except Exception as e:
            logger.error(f"Failed to load KServe client: {str(e)}")
            config.load_kube_config()

        self.kserve_client = KServeClient()
        self.namespace = namespace
        self.deployed_services = {}  # component_id -> inference_service_name 매핑

    def deploy_model_component(
        self,
        component: WorkflowComponent,
        model: Model,
        workflow_id: str,
        mlflow_config: Dict[str, str],
        gpu_enabled: bool = False,
        resources: Optional[Dict[str, str]] = None,
        db: Optional[Session] = None,
    ) -> str:
        """
        모델 컴포넌트를 KServe로 배포

        Args:
            component: 워크플로우 컴포넌트
            model: 모델 정보
            workflow_id: 워크플로우 ID
            mlflow_config: MLflow 설정 정보
            gpu_enabled: GPU 사용 여부
            resources: 리소스 설정 (cpu, memory, gpu)
            db: 데이터베이스 세션 (ModelRegistry 조회용)

        Returns:
            배포된 인퍼런스 서비스 이름
        """
        # 인퍼런스 서비스 이름 생성
        service_name = f"workflow-{workflow_id}-{component.component_id}-{uuid.uuid4().hex[:8]}"

        # 기본 리소스 설정
        if resources is None:
            resources = {
                "request_cpu": "200m",
                "request_memory": "2Gi",
                "limit_cpu": "500m",
                "limit_memory": "4Gi",
                "request_gpu": "1",
                "limit_gpu": "1",
            }

        # ModelRegistry에서 model_uri와 run_id 가져오기
        model_uri = ""
        run_id = ""
        framework = "pytorch"  # 기본값

        if db and hasattr(model, "registry") and model.registry:
            # artifact_path를 model_uri로 사용
            model_uri = model.registry.artifact_path or ""
            run_id = model.registry.run_id or ""

        # framework 정보는 model의 type_info에서 가져올 수도 있음
        if hasattr(model, "type_info") and model.type_info:
            # framework를 type_info.name으로부터 추론 (예: "PyTorch", "TensorFlow" 등)
            type_name = model.type_info.name.lower()
            if "pytorch" in type_name or "torch" in type_name:
                framework = "pytorch"
            elif "tensorflow" in type_name or "tf" in type_name:
                framework = "tensorflow"
            elif "onnx" in type_name:
                framework = "onnx"
            else:
                framework = "pytorch"  # 기본값

        # 컨테이너 args 구성
        from config.settings import get_settings

        settings = get_settings()

        # mlflow_config에 experiment_name이 있어도 무시하고 항상 settings의 값을 사용
        # 이렇게 하면 일관된 experiment name을 보장할 수 있음
        container_args = [
            f"--model_name={model.name}",
            f"--model_uri={model_uri}",
            f"--mlflow_tracking_uri={mlflow_config.get('tracking_uri', settings.MLFLOW_TRACKING_URI)}",
            # 환경 변수에서 설정된 실제 MLFLOW_EXPERIMENT_NAME을 항상 사용
            f"--mlflow_experiment_name={mlflow_config.get('experiment_name', settings.MLFLOW_EXPERIMENT_NAME)}",
            f"--mlflow_s3_endpoint_url={mlflow_config.get('s3_endpoint_url', settings.MLFLOW_S3_ENDPOINT_URL)}",
            f"--aws_access_key_id={mlflow_config.get('aws_access_key_id', settings.AWS_ACCESS_KEY_ID)}",
            f"--aws_secret_access_key={mlflow_config.get('aws_secret_access_key', settings.AWS_SECRET_ACCESS_KEY)}",
            f"--framework={framework}",
        ]

        # run_id가 있으면 추가
        if run_id:
            container_args.append(f"--run_id={run_id}")

        # 컴포넌트 config에서 추가 설정 가져오기
        if component.config:
            for key, value in component.config.items():
                if key not in ["model_id"]:  # model_id는 제외
                    container_args.append(f"--{key}={value}")

        # 리소스 요구사항 설정
        resource_requirements = client.V1ResourceRequirements(
            requests={"memory": resources["request_memory"], "cpu": resources["request_cpu"]},
            limits={"memory": resources["limit_memory"], "cpu": resources["limit_cpu"]},
        )

        # GPU 사용 시 추가
        if gpu_enabled:
            resource_requirements.requests["nvidia.com/gpu"] = resources["request_gpu"]
            resource_requirements.limits["nvidia.com/gpu"] = resources["limit_gpu"]

        # Predictor 스펙 생성
        predictor_spec = V1beta1PredictorSpec(
            min_replicas=1,
            containers=[
                client.V1Container(
                    name="kserve-container",
                    image="aipaas-harbor.surromind.ai/ml-workflow/inference:latest",
                    args=container_args,
                    resources=resource_requirements,
                    env=[
                        client.V1EnvVar(name="WORKFLOW_ID", value=workflow_id),
                        client.V1EnvVar(name="COMPONENT_ID", value=component.component_id),
                    ],
                )
            ],
        )

        # InferenceService 스펙 생성
        inference_service_spec = V1beta1InferenceServiceSpec(predictor=predictor_spec)

        # InferenceService 생성
        inference_service = V1beta1InferenceService(
            api_version=constants.KSERVE_V1BETA1,
            kind=constants.KSERVE_KIND,
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=self.namespace,
                labels={"workflow-id": workflow_id, "component-id": component.component_id, "model-id": str(model.id)},
                annotations={
                    "serving.kserve.io/enable-metric-aggregation": "true",
                    "serving.kserve.io/enable-prometheus-scraping": "true",
                },
            ),
            spec=inference_service_spec,
        )

        try:
            # KServe에 배포 (namespace 명시적 지정)
            self.kserve_client.create(inference_service, namespace=self.namespace)
            logger.info(f"Successfully deployed model component {component.component_id} as {service_name}")

            # 배포된 서비스 정보 저장
            self.deployed_services[component.component_id] = service_name

            return service_name

        except Exception as e:
            logger.error(f"Failed to deploy model component {component.component_id}: {str(e)}")
            raise

    def deploy_workflow_models(
        self, workflow_components: List[WorkflowComponent], workflow_id: str, db: Session, mlflow_config: Dict[str, str]
    ) -> Dict[str, str]:
        """
        워크플로우의 모든 모델 컴포넌트를 KServe로 배포

        Args:
            workflow_components: 워크플로우 컴포넌트 리스트
            workflow_id: 워크플로우 ID
            db: 데이터베이스 세션
            mlflow_config: MLflow 설정

        Returns:
            컴포넌트 ID -> 인퍼런스 서비스 이름 매핑
        """
        deployed_services = {}

        for component in workflow_components:
            # MODEL 타입 컴포넌트만 처리
            if component.type != ComponentType.MODEL:
                continue

            # 모델 ID가 없으면 건너뛰기
            if not component.model_id:
                logger.warning(f"Model component {component.component_id} has no model_id")
                continue

            # 모델 정보 조회
            model = db.query(Model).filter(Model.id == component.model_id).first()
            if not model:
                logger.error(f"Model with id {component.model_id} not found")
                continue

            try:
                # GPU 사용 여부 확인 (컴포넌트 config에서 가져오기)
                gpu_enabled = False
                if component.config and component.config.get("gpu_enabled"):
                    gpu_enabled = True

                # 리소스 설정 (컴포넌트 config에서 가져오기)
                resources = None
                if component.config and "resources" in component.config:
                    resources = component.config["resources"]

                # 모델 배포
                service_name = self.deploy_model_component(
                    component=component,
                    model=model,
                    workflow_id=workflow_id,
                    mlflow_config=mlflow_config,
                    gpu_enabled=gpu_enabled,
                    resources=resources,
                    db=db,
                )

                deployed_services[component.component_id] = service_name

            except Exception as e:
                logger.error(f"Failed to deploy model component {component.component_id}: {str(e)}")
                # 실패한 컴포넌트가 있어도 다른 컴포넌트는 계속 배포 시도
                continue

        return deployed_services

    def get_inference_service_url(self, service_name: str) -> str:
        """
        배포된 인퍼런스 서비스의 URL 가져오기

        Args:
            service_name: 인퍼런스 서비스 이름

        Returns:
            인퍼런스 서비스 URL
        """
        try:
            service = self.kserve_client.get(service_name, namespace=self.namespace)
            if service.get("status", {}).get("url"):
                return service["status"]["url"]
            else:
                # 기본 URL 형식 반환
                return f"http://{service_name}.{self.namespace}.svc.cluster.local/v1/models/{service_name}"
        except Exception as e:
            logger.error(f"Failed to get inference service URL for {service_name}: {str(e)}")
            return ""

    def delete_inference_service(self, service_name: str) -> bool:
        """
        인퍼런스 서비스 삭제

        Args:
            service_name: 삭제할 인퍼런스 서비스 이름

        Returns:
            삭제 성공 여부
        """
        try:
            self.kserve_client.delete(service_name, namespace=self.namespace)
            logger.info(f"Successfully deleted inference service {service_name}")

            # 배포된 서비스 목록에서 제거
            for comp_id, svc_name in list(self.deployed_services.items()):
                if svc_name == service_name:
                    del self.deployed_services[comp_id]

            return True

        except Exception as e:
            logger.error(f"Failed to delete inference service {service_name}: {str(e)}")
            return False

    def cleanup_workflow_services(self, workflow_id: str) -> int:
        """
        워크플로우의 모든 인퍼런스 서비스 정리

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            삭제된 서비스 개수
        """
        deleted_count = 0

        try:
            # KServe API를 통해 모든 인퍼런스 서비스 가져오기
            # get 메서드를 사용하여 namespace의 모든 서비스 조회
            services = self.kserve_client.get(namespace=self.namespace)

            for service in services:
                # service는 V1beta1InferenceService 객체
                if hasattr(service, "metadata") and service.metadata:
                    metadata = service.metadata

                    # 해당 워크플로우의 서비스인지 확인
                    # 워크플로우 ID가 서비스 이름에 포함되어 있는지 확인
                    service_name = metadata.name
                    if service_name and f"workflow-{workflow_id}" in service_name:
                        if self.delete_inference_service(service_name):
                            deleted_count += 1
                            logger.info(f"Deleted service {service_name} for workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup workflow services for {workflow_id}: {str(e)}")
            # 대체 방법: deployed_services 딕셔너리를 사용하여 정리
            try:
                for comp_id, service_name in list(self.deployed_services.items()):
                    if f"workflow-{workflow_id}" in service_name:
                        if self.delete_inference_service(service_name):
                            deleted_count += 1
            except Exception as inner_e:
                logger.error(f"Alternative cleanup also failed: {str(inner_e)}")

        return deleted_count
