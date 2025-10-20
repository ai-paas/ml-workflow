"""Service 및 관련 엔티티 Repository"""

from typing import Any, List, Optional

from db.models.service import (
    ComponentConnection,
    Service,
    ServiceMonitoring,
    ServiceStatus,
    Workflow,
    WorkflowComponent,
    WorkflowStatus,
)
from repos.base import CRUDBase
from schemas.app_service import ServiceCreateRequest, ServiceUpdateRequest
from sqlalchemy import and_, func
from sqlalchemy.orm import Session, joinedload


class ServiceRepository(CRUDBase[Service, ServiceCreateRequest, ServiceUpdateRequest]):
    """Service Repository"""

    # create_service 제거 - base의 create 메서드로 충분

    def get_by_name(self, db: Session, name: str) -> Optional[Service]:
        """이름으로 서비스 조회 - base의 filter 활용"""
        results = self.filter(db, {"name": name})
        return results[0] if results else None

    def get_with_relations(self, db: Session, service_id: str) -> Optional[Service]:
        """관계 포함 서비스 조회"""
        return (
            db.query(Service)
            .options(joinedload(Service.creator), joinedload(Service.workflows).joinedload(Workflow.creator))
            .filter(Service.id == service_id)
            .first()
        )

    def get_multi_with_filters(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        creator_id: Optional[int] = None,
        status: Optional[ServiceStatus] = None
    ) -> List[Service]:
        """필터링된 서비스 목록 조회"""
        query = db.query(Service).options(joinedload(Service.creator), joinedload(Service.workflows))

        if creator_id:
            query = query.filter(Service.creator_id == creator_id)

        if status:
            query = query.filter(Service.status == status)

        return query.offset(skip).limit(limit).all()

    # update_service 메서드 제거 - base의 update() 메서드 사용

    def delete_with_workflow_unlink(self, db: Session, service_id: str) -> bool:
        """워크플로우 연결 해제 후 서비스 삭제"""
        service = db.query(Service).filter(Service.id == service_id).first()
        if not service:
            return False

        # 연결된 워크플로우의 service_id를 null로 설정
        if service.workflows:
            for workflow in service.workflows:
                workflow.service_id = None

        db.delete(service)
        db.flush()
        return True


class ServiceMonitoringRepository(CRUDBase[ServiceMonitoring, Any, Any]):
    """ServiceMonitoring Repository"""

    def get_metrics_aggregate(
        self, db: Session, service_id: str, start_time, end_time, workflow_id: Optional[int] = None
    ):
        """메트릭 집계 조회"""
        query = db.query(
            func.sum(ServiceMonitoring.message_count).label("message_count"),
            func.sum(ServiceMonitoring.active_users).label("active_users"),
            func.sum(ServiceMonitoring.token_usage).label("token_usage"),
            func.avg(ServiceMonitoring.avg_interaction_count).label("avg_interaction_count"),
            func.avg(ServiceMonitoring.response_time_ms).label("response_time_ms"),
            func.sum(ServiceMonitoring.error_count).label("error_count"),
            func.avg(ServiceMonitoring.success_rate).label("success_rate"),
        ).filter(
            and_(
                ServiceMonitoring.service_id == service_id
                if not workflow_id
                else ServiceMonitoring.workflow_id == workflow_id,
                ServiceMonitoring.timestamp >= start_time,
                ServiceMonitoring.timestamp <= end_time,
            )
        )

        return query.first()

    # create_monitoring_record 메서드 제거 - 직접 생성 또는 base 메서드 사용


# Repository 인스턴스
service_repository = ServiceRepository(Service)
service_monitoring_repository = ServiceMonitoringRepository(ServiceMonitoring)
