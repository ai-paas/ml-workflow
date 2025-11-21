"""모델 기본 배포 관리 Service"""

import logging
import os
import re
import uuid
from typing import Any

from config.settings import get_settings
from core.kubeflow.kubeflow_manager import KubeflowManager
from db.models.model_base_deployment import BaseDeploymentStatus, ModelBaseDeployment
from kfp import dsl
from kfp.compiler import Compiler
from repos.model_base_deployment import model_base_deployment_repository
from schemas.model_base_deployment import ModelBaseDeploymentBaseSchema
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelBaseDeploymentService:
    """모델 기본 배포 관리 Service"""

    @staticmethod
    def deploy_ollama_embedding_model(
        db: Session,
        model_id: int,
        model_name: str,
        repo_id: str,
        gpu_enabled: bool = False,
    ) -> ModelBaseDeployment:
        """
        Ollama Embedding 모델을 Kubernetes에 배포 (Kubeflow 파이프라인 사용)

        Args:
            db: DB 세션
            model_id: 모델 ID
            model_name: 모델 이름 (정제된 이름)
            repo_id: Ollama 모델 repo_id
            gpu_enabled: GPU 사용 여부

        Returns:
            ModelBaseDeployment: 배포 정보
        """
        try:
            namespace = settings.KUBEFLOW_NAMESPACE
            ollama_model_name = repo_id

            # 모델 레지스트리에서 PVC 정보 가져오기
            from db.models.model import Model

            model_obj = db.query(Model).filter(Model.id == model_id).first()
            if not model_obj:
                raise ValueError(f"Model not found for model_id: {model_id}")

            if not model_obj.registry:
                raise ValueError(f"Model registry not found for model_id: {model_id}")

            pvc_name = model_obj.registry.pvc
            if not pvc_name:
                raise ValueError(
                    f"PVC not found for model_id: {model_id}. "
                    f"Please ensure model download pipeline completed successfully."
                )

            logger.info(f"Using existing PVC: {pvc_name} for model_id: {model_id}")

            # 서비스 이름 생성 (DNS 1035 규칙 준수)
            model_hash = model_name.replace("/", "-")[:20].lower()
            unique_id = uuid.uuid4().hex[:6]
            service_name = f"embedding-{model_hash}-{unique_id}"

            # DNS 1035 규칙에 맞게 정규화
            service_name = service_name.lower()
            service_name = re.sub(r"[^a-z0-9-]", "-", service_name)
            service_name = re.sub(r"-+", "-", service_name)
            service_name = service_name.strip("-")
            if len(service_name) > 63:
                service_name = service_name[:63].rstrip("-")
            if not service_name or service_name[0].isdigit():
                service_name = f"svc-{service_name}"[:63]

            logger.info(
                f"Deploying Ollama embedding model: {ollama_model_name} "
                f"as service: {service_name} using PVC: {pvc_name}"
            )

            # 배포 정보를 DB에 저장 (DEPLOYING 상태로)
            deployment_data = ModelBaseDeploymentBaseSchema(
                model_id=model_id,
                service_name=service_name,
                service_hostname="",  # Ollama는 사용하지 않음
                model_name=ollama_model_name,
                internal_url=None,  # 파이프라인 완료 후 업데이트
                status=BaseDeploymentStatus.DEPLOYING,
            )

            deployment_obj = model_base_deployment_repository.create(db, obj_in=deployment_data)
            db.commit()

            # Kubeflow 파이프라인 생성 및 실행
            kf_manager = KubeflowManager()

            # 파이프라인 함수 생성
            pipeline_func = ModelBaseDeploymentService._create_deployment_pipeline_function(
                model_id=model_id,
                model_name=model_name,
                service_name=service_name,
                repo_id=ollama_model_name,
                pvc_name=pvc_name,
                gpu_enabled=gpu_enabled,
            )

            # 파이프라인 컴파일
            pipeline_name = f"embedding-deploy-{model_id}-{unique_id}"
            pipeline_filename = f"/tmp/{pipeline_name}-{uuid.uuid4().hex[:8]}.yaml"

            try:
                Compiler().compile(pipeline_func, pipeline_filename)

                # Kubeflow에 파이프라인 실행
                experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
                experiment = kf_manager.get_experiment_by_name(experiment_name=experiment_name)
                if not experiment:
                    experiment = kf_manager.create_experiment(experiment_name)

                exp_id = experiment.experiment_id if hasattr(experiment, "experiment_id") else experiment.id

                run = kf_manager.kfp_client.run_pipeline(
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

                logger.info(f"Deployment pipeline started with run_id: {run_id}")

            finally:
                # 임시 파일 삭제
                if os.path.exists(pipeline_filename):
                    os.remove(pipeline_filename)

            # 배포는 비동기로 진행되므로 DEPLOYING 상태로 반환
            db.refresh(deployment_obj)
            return deployment_obj

        except Exception as e:
            logger.error(f"Failed to deploy Ollama embedding model: {e}")
            # 실패한 경우 DB에 기록
            try:
                deployment_data = ModelBaseDeploymentBaseSchema(
                    model_id=model_id,
                    service_name=service_name if "service_name" in locals() else f"failed-{model_id}",
                    service_hostname="",
                    model_name=model_name,
                    internal_url=None,
                    status=BaseDeploymentStatus.FAILED,
                    error_message=str(e),
                )
                deployment_obj = model_base_deployment_repository.create(db, obj_in=deployment_data)
                db.commit()
                return deployment_obj
            except Exception as db_error:
                logger.error(f"Failed to save deployment failure to DB: {db_error}")
                raise

    @staticmethod
    def _create_deployment_pipeline_function(
        model_id: int,
        model_name: str,
        service_name: str,
        repo_id: str,
        pvc_name: str,
        gpu_enabled: bool,
    ) -> callable:
        """
        Ollama Embedding 모델 배포를 위한 Kubeflow 파이프라인 함수 생성

        Args:
            model_id: 모델 ID
            model_name: 모델 이름
            service_name: 서비스 이름
            repo_id: Ollama 모델 repo_id
            pvc_name: PVC 이름
            gpu_enabled: GPU 사용 여부

        Returns:
            파이프라인 함수
        """

        @dsl.pipeline(
            name=f"embedding-deploy-{model_id}",
            description=f"Deploy Ollama embedding model: {model_name}",
        )
        def deployment_pipeline():
            ModelBaseDeploymentService._create_deployment_component_task(
                model_id=model_id,
                model_name=model_name,
                service_name=service_name,
                repo_id=repo_id,
                pvc_name=pvc_name,
                gpu_enabled=gpu_enabled,
            )

        return deployment_pipeline

    @staticmethod
    def _create_deployment_component_task(
        model_id: int,
        model_name: str,
        service_name: str,
        repo_id: str,
        pvc_name: str,
        gpu_enabled: bool,
    ) -> Any:
        """
        Ollama Embedding 모델 배포 컴포넌트 태스크 생성

        Args:
            model_id: 모델 ID
            model_name: 모델 이름
            service_name: 서비스 이름
            repo_id: Ollama 모델 repo_id
            pvc_name: PVC 이름 (기존에 생성된 PVC)
            gpu_enabled: GPU 사용 여부

        Returns:
            Kubeflow 태스크
        """

        @dsl.component(
            base_image="python:3.10",
            packages_to_install=[
                "kubernetes==28.1.0",
                "requests==2.31.0",
            ],
        )
        def deploy_ollama_embedding(
            model_id: int,
            model_name: str,
            service_name: str,
            repo_id: str,
            pvc_name: str,
            gpu_enabled: bool,
            namespace: str,
            rest_api_url: str,
            restapi_username: str,
            restapi_password: str,
        ) -> str:
            import json  # noqa: F811
            import logging
            import time

            from kubernetes import client
            from kubernetes import config as k8s_config

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                # Kubernetes 설정
                k8s_config.load_incluster_config()

                ollama_model_name = repo_id

                # Kubernetes API 클라이언트
                apps_v1 = client.AppsV1Api()
                core_v1 = client.CoreV1Api()

                logger.info(
                    f"Deploying Ollama embedding model: {ollama_model_name} "
                    f"as service: {service_name} using PVC: {pvc_name}"
                )

                # PVC가 생성될 때까지 대기 (최대 10분)
                max_wait = 600  # 10분
                wait_interval = 5  # 5초 간격
                elapsed = 0
                pvc_found = False

                logger.info(f"Waiting for PVC {pvc_name} to be created...")
                while elapsed < max_wait:
                    try:
                        pvc = core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
                        # PVC가 존재하고 Bound 상태인지 확인
                        if pvc.status.phase == "Bound":
                            logger.info(f"PVC {pvc_name} is ready (Bound)")
                            pvc_found = True
                            break
                        else:
                            logger.info(
                                f"PVC {pvc_name} exists but not Bound yet (phase: {pvc.status.phase}), waiting..."
                            )
                    except client.exceptions.ApiException as e:
                        if e.status == 404:
                            logger.info(f"PVC {pvc_name} not found yet, waiting... (elapsed: {elapsed}s)")
                        else:
                            logger.warning(f"Error checking PVC status: {e.status} - {e.reason}")
                    except Exception as e:
                        logger.warning(f"Error checking PVC: {e}")

                    time.sleep(wait_interval)
                    elapsed += wait_interval

                if not pvc_found:
                    raise RuntimeError(
                        f"PVC {pvc_name} not found or not Bound after {max_wait} seconds. "
                        f"Please ensure model download pipeline completed successfully."
                    )

                # Ollama용 리소스 설정
                ollama_resources = client.V1ResourceRequirements(
                    requests={
                        "memory": "4Gi",
                        "cpu": "500m",
                    },
                    limits={
                        "memory": "8Gi",
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
                            "model-id": str(model_id),
                            "model-type": "embedding",
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
                                    "model-id": str(model_id),
                                    "model-type": "embedding",
                                },
                            ),
                            spec=client.V1PodSpec(
                                containers=[
                                    client.V1Container(
                                        name="ollama",
                                        image="ollama/ollama:latest",
                                        command=["/bin/sh", "-c"],
                                        args=["ollama serve & SERVE_PID=$! && wait $SERVE_PID"],
                                        ports=[
                                            client.V1ContainerPort(container_port=11434, name="http", protocol="TCP")
                                        ],
                                        resources=ollama_resources,
                                        env=[
                                            client.V1EnvVar(name="MODEL_ID", value=str(model_id)),
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
                            "model-id": str(model_id),
                            "model-type": "embedding",
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

                        if deployment_status.status.ready_replicas and deployment_status.status.ready_replicas >= 1:
                            logger.info(f"Deployment {service_name} is ready")
                            deployment_ready = True
                            break

                        time.sleep(wait_interval)
                        elapsed += wait_interval

                    except Exception as e:
                        logger.warning(f"Error checking deployment status: {e}")
                        time.sleep(wait_interval)
                        elapsed += wait_interval

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

                        update_url = f"{rest_api_url}/api/v1/models/base-deployments/{model_id}/status"

                        update_payload = {
                            "service_name": service_name,
                            "service_hostname": service_hostname,
                            "internal_url": internal_url,
                            "status": deployment_status,
                            "error_message": None if deployment_ready else "Deployment not ready after timeout",
                        }

                        headers = {"Content-Type": "application/json"}
                        if auth_token:
                            headers["Authorization"] = f"Bearer {auth_token}"

                        logger.info("Updating deployment status via API: %s", update_url)
                        response = requests.put(update_url, json=update_payload, headers=headers, timeout=10)

                        if response.status_code == 200:
                            logger.info("Successfully updated deployment status in DB")
                        else:
                            logger.warning(
                                f"Failed to update deployment status: {response.status_code} - {response.text}"
                            )

                except Exception as e:
                    logger.error(f"Error updating deployment status in DB: {e}")
                    # DB 업데이트 실패해도 배포 자체는 성공이므로 계속 진행

                # 배포 상태 업데이트를 위한 정보 반환
                return json.dumps(
                    {
                        "service_name": service_name,
                        "service_hostname": service_hostname,
                        "model_name": model_name,
                        "status": deployment_status,
                        "internal_url": internal_url,
                    }
                )

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

                        update_url = f"{rest_api_url}/api/v1/models/base-deployments/{model_id}/status"

                        update_payload = {
                            "service_name": service_name,
                            "service_hostname": "",
                            "internal_url": None,
                            "status": "failed",
                            "error_message": str(e),
                        }

                        headers = {"Content-Type": "application/json"}
                        if auth_token:
                            headers["Authorization"] = f"Bearer {auth_token}"

                        requests.put(update_url, json=update_payload, headers=headers, timeout=10)
                except Exception as db_error:
                    logger.error(f"Failed to update DB with failure status: {db_error}")

                return json.dumps(
                    {
                        "error": str(e),
                        "service_name": service_name,
                        "status": "failed",
                    }
                )

        return deploy_ollama_embedding(
            model_id=model_id,
            model_name=model_name,
            service_name=service_name,
            repo_id=repo_id,
            pvc_name=pvc_name,
            gpu_enabled=gpu_enabled,
            namespace=settings.KUBEFLOW_NAMESPACE,
            rest_api_url=settings.REST_API_URL,
            restapi_username="surromind",  # 고정 사용자명
            restapi_password=settings.DEMO_PASSWORD,
        )

    @staticmethod
    def cleanup_model_deployment(db: Session, model_id: int) -> dict:
        """
        모델의 배포된 Kubernetes 리소스 정리 (Kubeflow 파이프라인 사용)
        파이프라인을 시작만 하고 상태 체크하지 않음 (백그라운드 실행)

        Args:
            db: DB 세션
            model_id: 모델 ID

        Returns:
            dict: 삭제 결과 정보
        """
        try:
            # 배포 정보 조회
            deployment_info = model_base_deployment_repository.get_by_model_id(db, model_id)
            if not deployment_info:
                logger.info(f"No deployment found for model_id: {model_id}")
                return {
                    "model_id": model_id,
                    "deleted": 0,
                    "failed": 0,
                    "total": 0,
                    "status": "no_deployment",
                }

            # Kubeflow 파이프라인 생성 및 실행
            kf_manager = KubeflowManager()

            # 파이프라인 함수 생성
            pipeline_func = ModelBaseDeploymentService._create_cleanup_pipeline_function(model_id=model_id)

            # 파이프라인 컴파일
            pipeline_name = f"cleanup-embedding-{model_id}"
            pipeline_filename = f"/tmp/{pipeline_name}-{uuid.uuid4().hex[:8]}.yaml"

            try:
                Compiler().compile(pipeline_func, pipeline_filename)

                # Kubeflow에 파이프라인 실행
                experiment_name = settings.KUBEFLOW_EXPERIMENT_NAME
                experiment = kf_manager.get_experiment_by_name(experiment_name=experiment_name)
                if not experiment:
                    experiment = kf_manager.create_experiment(experiment_name)

                exp_id = experiment.experiment_id if hasattr(experiment, "experiment_id") else experiment.id

                run = kf_manager.kfp_client.run_pipeline(
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
                # 파이프라인은 백그라운드에서 실행되며, 상태 체크하지 않음

            finally:
                # 임시 파일 삭제
                if os.path.exists(pipeline_filename):
                    os.remove(pipeline_filename)

            # 파이프라인 시작 후 바로 DB 레코드 삭제
            # (외래키 제약 조건 때문에 모델 삭제 전에 삭제해야 함)
            # 파이프라인은 백그라운드에서 실행되므로 상태 체크하지 않음
            try:
                model_base_deployment_repository.delete(db, pk=deployment_info.id)
                db.commit()
                logger.info(f"Deleted model_base_deployment record for model_id: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to delete model_base_deployment record: {e}")
                # DB 삭제 실패 시 예외 발생 (외래키 제약 조건 위반 방지)
                raise RuntimeError(f"모델 기본 배포 정보 삭제 실패: {str(e)}")

            return {
                "model_id": model_id,
                "cleanup_run_id": run_id,
                "status": "started",
                "message": "Cleanup pipeline started (running in background)",
            }

        except Exception as e:
            logger.error(f"Failed to cleanup model deployment for model_id {model_id}: {e}")
            raise RuntimeError(f"모델 배포 정리 실패: {str(e)}")

    @staticmethod
    def _create_cleanup_pipeline_function(model_id: int) -> callable:
        """
        모델 배포 정리를 위한 Kubeflow 파이프라인 함수 생성

        Args:
            model_id: 모델 ID

        Returns:
            파이프라인 함수
        """

        @dsl.pipeline(
            name=f"cleanup-embedding-{model_id}",
            description=f"Cleanup Kubernetes resources for model {model_id}",
        )
        def cleanup_pipeline():
            ModelBaseDeploymentService._create_cleanup_component_task(model_id=model_id)

        return cleanup_pipeline

    @staticmethod
    def _create_cleanup_component_task(model_id: int) -> Any:
        """
        모델 배포 정리 컴포넌트 태스크 생성

        Args:
            model_id: 모델 ID

        Returns:
            Kubeflow 태스크
        """

        @dsl.component(
            base_image="python:3.10",
            packages_to_install=[
                "kubernetes==28.1.0",
            ],
        )
        def cleanup_model_resources(model_id: int, namespace: str) -> str:
            import json  # noqa: F811
            import logging

            from kubernetes import client
            from kubernetes import config as k8s_config

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                # Kubernetes 설정
                k8s_config.load_incluster_config()

                label_selector = f"model-id={model_id}"

                deleted_count = 0
                failed_count = 0
                total_count = 0

                apps_v1 = client.AppsV1Api()
                core_v1 = client.CoreV1Api()

                # 1. Deployment 삭제
                try:
                    logger.info(f"Querying Deployments for model_id: {model_id}")
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    deployment_items = deployments.items
                    logger.info(f"Found {len(deployment_items)} Deployments for model_id {model_id}")
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
                except Exception as e:
                    logger.warning(f"Error cleaning up Deployments: {e}")

                # 2. Service 삭제
                try:
                    logger.info(f"Querying Services for model_id: {model_id}")
                    services = core_v1.list_namespaced_service(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    service_items = services.items
                    logger.info(f"Found {len(service_items)} Services for model_id {model_id}")
                    total_count += len(service_items)

                    for service in service_items:
                        service_name_k8s = service.metadata.name
                        try:
                            logger.info(f"Deleting Service: {service_name_k8s} in namespace: {namespace}")
                            core_v1.delete_namespaced_service(
                                name=service_name_k8s,
                                namespace=namespace,
                            )
                            deleted_count += 1
                            logger.info(f"Successfully deleted Service: {service_name_k8s}")
                        except client.exceptions.ApiException as e:
                            if e.status == 404:
                                logger.warning(f"Service not found: {service_name_k8s}")
                            else:
                                logger.error(f"Failed to delete Service {service_name_k8s}: {e.status} - {e.reason}")
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Unexpected error deleting Service {service_name_k8s}: {e}")
                            failed_count += 1
                except Exception as e:
                    logger.warning(f"Error cleaning up Services: {e}")

                # 3. PVC 삭제
                try:
                    logger.info(f"Querying PVCs for model_id: {model_id}")
                    pvcs = core_v1.list_namespaced_persistent_volume_claim(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    pvc_items = pvcs.items
                    logger.info(f"Found {len(pvc_items)} PVCs for model_id {model_id}")
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
                    logger.warning(f"Error cleaning up PVCs: {e}")

                result_data = {
                    "model_id": model_id,
                    "deleted": deleted_count,
                    "failed": failed_count,
                    "total": total_count,
                    "status": "completed" if failed_count == 0 else "partial",
                }

                logger.info(f"Cleanup completed: {json.dumps(result_data)}")
                return json.dumps(result_data)

            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")
                error_result = {
                    "model_id": model_id,
                    "deleted": 0,
                    "failed": 0,
                    "total": 0,
                    "status": "failed",
                    "error": str(e),
                }
                return json.dumps(error_result)

        return cleanup_model_resources(
            model_id=model_id,
            namespace=settings.KUBEFLOW_NAMESPACE,
        )
