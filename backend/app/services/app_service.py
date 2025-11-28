"""Application Service 비즈니스 로직"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

from config.settings import get_settings
from db.models.kserve_deployment import DeploymentStatus
from db.models.service import Service, ServiceMonitoring, Workflow, WorkflowStatus
from repos.app_service import service_monitoring_repository, service_repository
from repos.kserve_deployment import kserve_deployment_repository
from repos.workflow import workflow_repository
from schemas.app_service import (
    DeploymentResourceUsage,
    MonitoringMetrics,
    PodResourceUsage,
    ResourceUsage,
    ServiceCreateInternal,
    ServiceCreateRequest,
    ServiceMonitoringData,
    ServiceResourceUsageResponse,
    ServiceUpdateRequest,
    WorkflowMonitoring,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()


class AppServiceService:
    """서비스 관련 비즈니스 로직"""

    @staticmethod
    def create_service(db: Session, service_data: ServiceCreateRequest, creator_id: int) -> Service:
        """새로운 서비스 생성"""
        try:
            # 서비스 이름 중복 체크
            existing = service_repository.get_by_name(db, service_data.name)
            if existing:
                raise ValueError(f"Service with name '{service_data.name}' already exists")

            # ServiceCreateInternal 사용 (creator_id 포함)
            service_internal = ServiceCreateInternal(**service_data.model_dump(), creator_id=creator_id)

            # base의 create 메서드 사용 (DB 작업 포함)
            service = service_repository.create(db, obj_in=service_internal)

            db.commit()

            logger.info(f"Service created successfully: {service.id}")
            return service

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create service: {str(e)}")
            raise

    @staticmethod
    def get_service_by_id(db: Session, service_id: str) -> Optional[Service]:
        """ID로 서비스 조회"""
        return service_repository.get_with_relations(db, service_id)

    @staticmethod
    def get_services(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        creator_id: Optional[int] = None,
    ) -> List[Service]:
        """서비스 목록 조회"""
        return service_repository.get_multi_with_filters(db, skip=skip, limit=limit, creator_id=creator_id)

    @staticmethod
    def count_services(db: Session, *, creator_id: Optional[int] = None) -> int:
        """필터 조건에 맞는 서비스 개수 조회"""
        return service_repository.count(db, creator_id=creator_id)

    @staticmethod
    def update_service(db: Session, service_id: str, service_data: ServiceUpdateRequest) -> Optional[Service]:
        """서비스 정보 수정"""
        service = service_repository.get(db, service_id)
        if not service:
            return None

        try:
            # base의 update 메서드 직접 사용
            updated_service = service_repository.update(db, db_obj=service, obj_in=service_data)

            db.commit()
            logger.info(f"Service updated successfully: {service.id}")
            return updated_service

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update service: {str(e)}")
            raise

    @staticmethod
    def delete_service(db: Session, service_id: str) -> bool:
        """서비스 삭제"""
        try:
            # Repository를 통한 삭제 (워크플로우 연결 해제 포함)
            success = service_repository.delete_with_workflow_unlink(db, service_id)

            if success:
                db.commit()
                logger.info(f"Service deleted successfully: {service_id}")

            return success

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete service: {str(e)}")
            raise

    @staticmethod
    def get_service_monitoring_data(db: Session, service_id: str, hours: int = 1) -> Optional[ServiceMonitoringData]:
        """서비스 모니터링 데이터 조회"""
        service = service_repository.get(db, service_id)
        if not service:
            return None

        # 시간 범위 설정
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Repository를 통한 메트릭 집계 조회
        total_metrics_query = service_monitoring_repository.get_metrics_aggregate(
            db, service_id=service_id, start_time=start_time, end_time=end_time
        )

        # 워크플로우별 메트릭 집계
        workflow_metrics = []
        for workflow in service.workflows:
            wf_metrics_query = service_monitoring_repository.get_metrics_aggregate(
                db, service_id=service_id, start_time=start_time, end_time=end_time, workflow_id=workflow.id
            )

            if wf_metrics_query and wf_metrics_query.message_count:
                workflow_metrics.append(
                    WorkflowMonitoring(
                        workflow_id=workflow.id,
                        workflow_name=workflow.name,
                        metrics=MonitoringMetrics(
                            message_count=wf_metrics_query.message_count or 0,
                            active_users=wf_metrics_query.active_users or 0,
                            token_usage=wf_metrics_query.token_usage or 0,
                            avg_interaction_count=float(wf_metrics_query.avg_interaction_count or 0),
                            response_time_ms=(
                                float(wf_metrics_query.response_time_ms or 0)
                                if wf_metrics_query.response_time_ms
                                else None
                            ),
                            error_count=wf_metrics_query.error_count or 0,
                            success_rate=float(wf_metrics_query.success_rate or 100.0),
                        ),
                        last_updated=end_time,
                    )
                )

        # 전체 메트릭 구성
        total_metrics = MonitoringMetrics(
            message_count=total_metrics_query.message_count or 0,
            active_users=total_metrics_query.active_users or 0,
            token_usage=total_metrics_query.token_usage or 0,
            avg_interaction_count=float(total_metrics_query.avg_interaction_count or 0),
            response_time_ms=(
                float(total_metrics_query.response_time_ms or 0) if total_metrics_query.response_time_ms else None
            ),
            error_count=total_metrics_query.error_count or 0,
            success_rate=float(total_metrics_query.success_rate or 100.0),
        )

        return ServiceMonitoringData(
            total_metrics=total_metrics, workflow_metrics=workflow_metrics, period_start=start_time, period_end=end_time
        )

    @staticmethod
    def get_service_resource_usages(db: Session, service_id: str) -> Optional[ServiceResourceUsageResponse]:
        """서비스 리소스 사용량 조회

        서비스에 속한 워크플로우의 배포된 모델들의 리소스 사용량을 조회합니다.
        k8s metrics API를 사용하여 CPU, Memory, GPU 사용량을 가져옵니다.

        Args:
            db: 데이터베이스 세션
            service_id: 서비스 ID

        Returns:
            ServiceResourceUsageResponse 또는 None (서비스가 없는 경우)
        """
        service = service_repository.get(db, service_id)
        if not service:
            return None

        try:
            from kubernetes import client
            from kubernetes import config as k8s_config
            from kubernetes.client.rest import ApiException

            # Kubernetes 설정
            try:
                k8s_config.load_incluster_config()
            except Exception:
                logger.warning("Failed to load in-cluster config, trying kubeconfig")
                try:
                    k8s_config.load_kube_config()
                except Exception as e:
                    logger.error(f"Failed to load kubeconfig: {str(e)}")
                    raise

            core_v1 = client.CoreV1Api()
            metrics_v1beta1 = client.CustomObjectsApi()
            namespace = settings.KUBEFLOW_NAMESPACE

            # Metrics Server 사용 가능 여부 플래그 (첫 번째 실패 시 이후 Pod들도 건너뛰기)
            metrics_server_available = True  # 초기값은 True, 실패 시 False로 설정

            deployments: List[DeploymentResourceUsage] = []
            total_cpu_usage = 0.0
            total_memory_usage = 0
            total_gpu_usage = 0.0

            # 서비스에 속한 모든 워크플로우 조회
            for workflow in service.workflows:
                # 워크플로우의 배포된 모델 조회
                kserve_deployments = kserve_deployment_repository.get_by_workflow(
                    db, workflow.id, DeploymentStatus.DEPLOYED
                )

                for kserve_deployment in kserve_deployments:
                    pods: List[PodResourceUsage] = []
                    deployment_cpu_usage = 0.0
                    deployment_memory_usage = 0
                    deployment_gpu_usage = 0.0

                    # KServe InferenceService의 경우 pod 이름 패턴: {service_name}-predictor-{revision}-{random}
                    # 일반 Service의 경우: {service_name}-{random}
                    service_name = kserve_deployment.service_name

                    try:
                        # InferenceService인지 확인
                        try:
                            # KServe InferenceService 조회 시도
                            from kserve import KServeClient

                            kserve_client = KServeClient()
                            kserve_client.get(service_name, namespace=namespace)  # InferenceService 존재 확인
                            deployment_type = "inferenceservice"

                            # InferenceService의 pod 찾기
                            label_selector = f"serving.kserve.io/inferenceservice={service_name}"
                            pod_list = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
                        except Exception:
                            # 일반 Service로 처리
                            deployment_type = "service"
                            # Service를 통해 pod 찾기
                            try:
                                service_obj = core_v1.read_namespaced_service(service_name, namespace=namespace)
                                if service_obj.spec.selector:
                                    label_selector = ",".join(
                                        [f"{k}={v}" for k, v in service_obj.spec.selector.items()]
                                    )
                                    pod_list = core_v1.list_namespaced_pod(
                                        namespace=namespace, label_selector=label_selector
                                    )
                                else:
                                    pod_list = client.V1PodList(items=[])
                            except ApiException as e:
                                if e.status == 404:
                                    logger.warning(f"Service {service_name} not found in namespace {namespace}")
                                    pod_list = client.V1PodList(items=[])
                                else:
                                    raise

                        # 각 pod의 리소스 사용량 조회
                        for pod in pod_list.items:
                            pod_name = pod.metadata.name
                            pod_status = pod.status.phase

                            # Pod의 리소스 요청/제한 정보 가져오기
                            cpu_request = None
                            cpu_limit = None
                            memory_request = None
                            memory_limit = None

                            if pod.spec.containers:
                                container = pod.spec.containers[0]  # 첫 번째 컨테이너
                                if container.resources:
                                    if container.resources.requests:
                                        if "cpu" in container.resources.requests:
                                            cpu_request_str = container.resources.requests["cpu"]
                                            cpu_request = _parse_cpu_to_millicores(cpu_request_str)
                                        if "memory" in container.resources.requests:
                                            memory_request_str = container.resources.requests["memory"]
                                            memory_request = _parse_memory_to_bytes(memory_request_str)
                                    if container.resources.limits:
                                        if "cpu" in container.resources.limits:
                                            cpu_limit_str = container.resources.limits["cpu"]
                                            cpu_limit = _parse_cpu_to_millicores(cpu_limit_str)
                                        if "memory" in container.resources.limits:
                                            memory_limit_str = container.resources.limits["memory"]
                                            memory_limit = _parse_memory_to_bytes(memory_limit_str)

                            # Metrics API를 통해 실제 사용량 조회
                            cpu_usage = None
                            memory_usage = None
                            gpu_usage = None
                            gpu_memory_usage = None

                            # Metrics API를 통해 실제 사용량 조회
                            # Metrics Server가 없으면 자동으로 건너뜀
                            if metrics_server_available:
                                try:
                                    # Pod metrics 조회
                                    logger.debug(
                                        f"Attempting to get metrics for pod {pod_name} in namespace {namespace}"
                                    )
                                    metrics = metrics_v1beta1.get_namespaced_custom_object(
                                        group="metrics.k8s.io",
                                        version="v1beta1",
                                        namespace=namespace,
                                        plural="pods",
                                        name=pod_name,
                                    )

                                    logger.debug(
                                        f"Metrics response for pod {pod_name}: {json.dumps(metrics, default=str)}"
                                    )

                                    if "containers" in metrics and metrics["containers"]:
                                        # 모든 컨테이너의 메트릭 합산
                                        total_cpu_nanocores = 0
                                        total_memory_bytes = 0

                                        for container_metrics in metrics["containers"]:
                                            if "usage" in container_metrics:
                                                usage = container_metrics["usage"]
                                                if "cpu" in usage:
                                                    cpu_str = usage["cpu"]
                                                    # CPU는 보통 "123456n" (나노코어) 형식
                                                    total_cpu_nanocores += _parse_cpu_to_nanocores(cpu_str)
                                                if "memory" in usage:
                                                    memory_str = usage["memory"]
                                                    total_memory_bytes += _parse_memory_to_bytes(memory_str)

                                        # 나노코어를 밀리코어로 변환
                                        if total_cpu_nanocores > 0:
                                            cpu_usage = total_cpu_nanocores / 1_000_000.0  # 나노코어 -> 밀리코어
                                            logger.debug(f"Pod {pod_name} CPU usage: {cpu_usage} millicores")

                                        if total_memory_bytes > 0:
                                            memory_usage = total_memory_bytes
                                            logger.debug(f"Pod {pod_name} Memory usage: {memory_usage} bytes")
                                    else:
                                        logger.warning(f"No container metrics found in response for pod {pod_name}")

                                except ApiException as e:
                                    error_body = str(e.body) if e.body else ""
                                    if e.status == 403:
                                        logger.warning(
                                            f"Permission denied (403) when getting metrics for pod {pod_name}. "
                                            f"Metrics API may require additional RBAC permissions."
                                        )
                                    elif e.status == 404:
                                        logger.debug(
                                            f"Metrics not available for pod {pod_name} (404). "
                                            f"Pod may not have metrics yet."
                                        )
                                    elif (
                                        "doesn't have a resource type" in error_body.lower()
                                        or "doesn't have a resource type" in str(e.reason).lower()
                                    ):
                                        # Metrics Server가 설치되지 않은 경우
                                        logger.info(
                                            "Metrics Server is not installed. "
                                            "Resource usage will be null. "
                                            "Only resource requests/limits are available."
                                        )
                                        metrics_server_available = False  # 이후 Pod들도 건너뛰기 위해 플래그 설정
                                    else:
                                        logger.warning(
                                            f"Failed to get metrics for pod {pod_name}: {e.status} - {e.reason}"
                                        )
                                except Exception as e:
                                    error_msg = str(e).lower()
                                    if "doesn't have a resource type" in error_msg or "metrics.k8s.io" in error_msg:
                                        logger.info(
                                            "Metrics Server is not installed. "
                                            "Resource usage will be null. "
                                            "Only resource requests/limits are available."
                                        )
                                        metrics_server_available = False
                                    else:
                                        logger.warning(f"Error getting metrics for pod {pod_name}: {str(e)}")
                                        import traceback

                                        logger.debug(traceback.format_exc())
                            else:
                                logger.debug(
                                    f"Metrics Server not available. Skipping metrics collection for pod {pod_name}"
                                )

                            # GPU 사용량 조회 (nvidia-smi를 pod 내에서 실행하거나 dcgm-exporter 사용)
                            # 여기서는 기본적으로 None으로 설정하고, 필요시 별도 구현
                            # GPU는 일반적으로 node-exporter나 dcgm-exporter를 통해 수집됨

                            resource_usage = ResourceUsage(
                                cpu_usage_millicores=cpu_usage,
                                cpu_request_millicores=cpu_request,
                                cpu_limit_millicores=cpu_limit,
                                memory_usage_bytes=memory_usage,
                                memory_request_bytes=memory_request,
                                memory_limit_bytes=memory_limit,
                                gpu_usage_percent=gpu_usage,
                                gpu_memory_usage_bytes=gpu_memory_usage,
                            )

                            pods.append(
                                PodResourceUsage(
                                    pod_name=pod_name,
                                    namespace=namespace,
                                    deployment_type=deployment_type,
                                    resource_usage=resource_usage,
                                    status=pod_status,
                                )
                            )

                            # 총 사용량 누적
                            if cpu_usage:
                                deployment_cpu_usage += cpu_usage
                            if memory_usage:
                                deployment_memory_usage += memory_usage
                            if gpu_usage:
                                deployment_gpu_usage += gpu_usage

                    except ApiException as e:
                        logger.error(f"Failed to get pods for deployment {service_name}: {e.status} - {e.reason}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing deployment {service_name}: {str(e)}")
                        continue

                    if pods:
                        deployments.append(
                            DeploymentResourceUsage(
                                deployment_id=kserve_deployment.id,
                                service_name=service_name,
                                workflow_id=workflow.id,
                                component_id=kserve_deployment.component_id,
                                model_name=kserve_deployment.model_name,
                                pods=pods,
                            )
                        )

                        total_cpu_usage += deployment_cpu_usage
                        total_memory_usage += deployment_memory_usage
                        total_gpu_usage += deployment_gpu_usage

            return ServiceResourceUsageResponse(
                service_id=service.id,
                service_name=service.name,
                deployments=deployments,
                total_cpu_usage_millicores=total_cpu_usage if total_cpu_usage > 0 else None,
                total_memory_usage_bytes=total_memory_usage if total_memory_usage > 0 else None,
                total_gpu_usage_percent=total_gpu_usage if total_gpu_usage > 0 else None,
            )

        except Exception as e:
            logger.error(f"Failed to get resource usages for service {service_id}: {str(e)}")
            raise


def _parse_cpu_to_nanocores(cpu_str: str) -> int:
    """CPU 문자열을 나노코어 단위로 변환

    예: "100m" -> 100000000, "1" -> 1000000000, "123456n" -> 123456
    """
    if not cpu_str:
        return 0

    cpu_str = cpu_str.strip()

    # 나노코어 단위인 경우 (Metrics API가 반환하는 형식)
    if cpu_str.endswith("n"):
        try:
            return int(cpu_str[:-1])
        except ValueError:
            logger.warning(f"Invalid CPU nanocores value: {cpu_str}")
            return 0

    # 밀리코어 단위인 경우
    if cpu_str.endswith("m"):
        try:
            millicores = float(cpu_str[:-1])
            return int(millicores * 1_000_000)  # 밀리코어 -> 나노코어
        except ValueError:
            logger.warning(f"Invalid CPU millicores value: {cpu_str}")
            return 0

    # 코어 단위인 경우 (나노코어로 변환)
    try:
        cores = float(cpu_str)
        return int(cores * 1_000_000_000)  # 코어 -> 나노코어
    except ValueError:
        logger.warning(f"Invalid CPU value: {cpu_str}")
        return 0


def _parse_cpu_to_millicores(cpu_str: str) -> float:
    """CPU 문자열을 밀리코어 단위로 변환

    예: "100m" -> 100.0, "1" -> 1000.0, "0.5" -> 500.0, "123456n" -> 0.123456
    """
    if not cpu_str:
        return 0.0

    cpu_str = cpu_str.strip()

    # 나노코어 단위인 경우 (Metrics API가 반환하는 형식)
    if cpu_str.endswith("n"):
        try:
            nanocores = int(cpu_str[:-1])
            return nanocores / 1_000_000.0  # 나노코어 -> 밀리코어
        except ValueError:
            logger.warning(f"Invalid CPU nanocores value: {cpu_str}")
            return 0.0

    # 밀리코어 단위인 경우
    if cpu_str.endswith("m"):
        try:
            return float(cpu_str[:-1])
        except ValueError:
            logger.warning(f"Invalid CPU millicores value: {cpu_str}")
            return 0.0

    # 코어 단위인 경우 (밀리코어로 변환)
    try:
        cores = float(cpu_str)
        return cores * 1000.0
    except ValueError:
        logger.warning(f"Invalid CPU value: {cpu_str}")
        return 0.0


def _parse_memory_to_bytes(memory_str: str) -> int:
    """메모리 문자열을 바이트 단위로 변환

    예: "100Mi" -> 104857600, "1Gi" -> 1073741824, "500M" -> 500000000
    """
    if not memory_str:
        return 0

    memory_str = memory_str.strip().upper()

    # 단위 매핑
    units = {
        "KI": 1024,
        "MI": 1024**2,
        "GI": 1024**3,
        "TI": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
    }

    # 숫자와 단위 분리
    match = re.match(r"^(\d+(?:\.\d+)?)([A-Z]+)?$", memory_str)
    if not match:
        # 바이트 단위로 직접 지정된 경우
        try:
            return int(memory_str)
        except ValueError:
            logger.warning(f"Invalid memory value: {memory_str}")
            return 0

    value_str, unit = match.groups()
    value = float(value_str)

    if unit:
        multiplier = units.get(unit, 1)
        return int(value * multiplier)

    # 단위가 없으면 바이트로 간주
    return int(value)


class ServiceMonitoringService:
    """서비스 모니터링 관련 비즈니스 로직"""

    @staticmethod
    def record_inference_request(
        db: Session,
        service_id: str,
        workflow_id: str,
        user_id: int,
        response_time_ms: float,
        is_success: bool,
        is_object_detection: bool = True,
    ) -> ServiceMonitoring:
        """추론 요청 기록

        Args:
            db: 데이터베이스 세션
            service_id: 서비스 ID
            workflow_id: 워크플로우 ID
            user_id: 사용자 ID
            response_time_ms: 응답 시간 (밀리초)
            is_success: 성공 여부
            is_object_detection: Object Detection 여부 (토큰 사용량 0으로 설정)

        Returns:
            생성된 ServiceMonitoring 레코드
        """
        try:
            # 메트릭 계산
            message_count = 1  # 요청당 1건
            active_users = 1  # 해당 사용자 1명
            token_usage = 0 if is_object_detection else 0  # Object Detection은 토큰 사용량 없음
            avg_interaction_count = 1.0  # 사용자당 평균 요청 수 (개별 레코드는 1)
            error_count = 0 if is_success else 1
            success_rate = 100.0 if is_success else 0.0

            # ServiceMonitoring 레코드 생성
            monitoring_record = ServiceMonitoring(
                service_id=service_id,
                workflow_id=workflow_id,
                timestamp=datetime.utcnow(),
                message_count=message_count,
                active_users=active_users,
                token_usage=token_usage,
                avg_interaction_count=avg_interaction_count,
                response_time_ms=response_time_ms,
                error_count=error_count,
                success_rate=success_rate,
            )

            db.add(monitoring_record)
            db.flush()

            logger.info(
                f"Monitoring record created: service_id={service_id}, workflow_id={workflow_id}, "
                f"user_id={user_id}, response_time={response_time_ms}ms, success={is_success}"
            )

            return monitoring_record

        except Exception as e:
            logger.error(f"Failed to record inference request: {str(e)}")
            db.rollback()
            raise
