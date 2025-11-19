"""KServe 배포 관리 Service"""

from datetime import datetime
from typing import Dict, List, Optional

from config.settings import get_settings
from db.models.kserve_deployment import DeploymentStatus, KServeDeployment
from repos.kserve_deployment import kserve_deployment_repository
from schemas.kserve_deployment import KServeDeploymentBaseSchema
from services.workflow import WorkflowService
from sqlalchemy.orm import Session

settings = get_settings()


class KServeDeploymentService:
    """KServe 배포 관리 Service"""

    @staticmethod
    def create_deployment(
        db: Session,
        workflow_id: str,
        component_id: str,
        model_name: str,
        service_name: Optional[str] = None,
        service_hostname: Optional[str] = None,
    ) -> KServeDeployment:
        """배포 정보 생성"""
        # 기존 배포가 있는지 확인
        existing = kserve_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)
        if existing:
            return existing

        # 임시 서비스 이름 생성
        if not service_name:
            service_name = f"pending-{workflow_id[:8]}-{component_id[:8]}"
        if not service_hostname:
            service_hostname = "pending"

        deployment_data = KServeDeploymentBaseSchema(
            workflow_id=workflow_id,
            component_id=component_id,
            service_name=service_name,
            service_hostname=service_hostname,
            model_name=model_name.replace("/", "-"),  # 모델 이름 정제
            status=DeploymentStatus.DEPLOYING,
        )

        return kserve_deployment_repository.create(db, obj_in=deployment_data)

    @staticmethod
    def update_deployment_status(
        db: Session,
        workflow_id: str,
        component_id: str,
        service_name: str,
        service_hostname: str,
        model_name: str,
        status: str,
        internal_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> KServeDeployment:
        """배포 상태 업데이트"""
        # 기존 배포 정보 조회 또는 생성
        deployment = kserve_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)

        if deployment:
            # 기존 레코드 업데이트
            deployment.service_name = service_name
            deployment.service_hostname = service_hostname
            deployment.model_name = model_name
            deployment.internal_url = internal_url

            # 상태 업데이트
            if status == "deployed":
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.deployed_at = datetime.utcnow()
                deployment.error_message = None
            elif status == "failed":
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = error_message
            elif status == "deleted":
                deployment.status = DeploymentStatus.DELETED
                deployment.deleted_at = datetime.utcnow()

            db.commit()
            db.refresh(deployment)
        else:
            # 새 레코드 생성
            deployment_data = KServeDeploymentBaseSchema(
                workflow_id=workflow_id,
                component_id=component_id,
                service_name=service_name,
                service_hostname=service_hostname,
                model_name=model_name,
                internal_url=internal_url,
                status=DeploymentStatus.DEPLOYED if status == "deployed" else DeploymentStatus.DEPLOYING,
                deployed_at=datetime.utcnow() if status == "deployed" else None,
                error_message=error_message if status == "failed" else None,
            )
            deployment = kserve_deployment_repository.create(db, obj_in=deployment_data)

        return deployment

    @staticmethod
    def get_deployment_info(db: Session, workflow_id: str, component_id: str) -> Optional[Dict]:
        """배포 정보 조회"""
        deployment = kserve_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)

        if not deployment:
            return None

        return {
            "service_name": deployment.service_name,
            "service_hostname": deployment.service_hostname,
            "model_name": deployment.model_name,
            "internal_url": deployment.internal_url,
            "gateway_url": settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80",
            "status": deployment.status.value,
            "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
            "error_message": deployment.error_message,
        }

    @staticmethod
    def get_deployed_models(db: Session, workflow_id: str, include_component_info: bool = True) -> List[Dict]:
        """워크플로우의 배포된 모델 목록 조회"""
        deployments = kserve_deployment_repository.get_by_workflow(db, workflow_id)

        deployed_models = []
        for deployment in deployments:
            model_info = {
                "component_id": deployment.component_id,
                "service_name": deployment.service_name,
                "service_hostname": deployment.service_hostname,
                "model_name": deployment.model_name,
                "sanitized_model_name": deployment.model_name,
                "internal_url": deployment.internal_url,
                "gateway_url": settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80",
                "status": deployment.status.value,
                "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
                "error_message": deployment.error_message,
            }

            # 컴포넌트 정보 추가
            if include_component_info:
                component = WorkflowService.get_component_by_id_and_workflow_id(
                    db, deployment.component_id, workflow_id
                )

                if component:
                    model_info.update(
                        {
                            "model_id": component.model_id,
                            "model_name": component.name,
                        }
                    )

            deployed_models.append(model_info)

        return deployed_models

    @staticmethod
    def validate_deployment_ready(
        db: Session, workflow_id: str, component_id: str
    ) -> tuple[bool, str, Optional[KServeDeployment]]:
        """배포가 준비되었는지 검증

        Returns:
            (is_ready, error_message, deployment)
        """
        deployment = kserve_deployment_repository.get_by_workflow_component(db, workflow_id, component_id)

        if not deployment:
            return False, f"Model component {component_id} not deployed", None

        if deployment.status != DeploymentStatus.DEPLOYED:
            return False, f"Model is in {deployment.status.value} state", deployment

        return True, "", deployment

    @staticmethod
    def cleanup_workflow_deployments(db: Session, workflow_id: str) -> int:
        """워크플로우의 모든 배포 정보 정리"""
        # 상태를 DELETED로 변경
        deployments = kserve_deployment_repository.get_by_workflow(db, workflow_id)
        for deployment in deployments:
            kserve_deployment_repository.update_status(db, deployment, DeploymentStatus.DELETED)

        return len(deployments)
