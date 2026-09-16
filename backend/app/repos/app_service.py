"""Service 및 관련 엔티티 Repository"""

from datetime import timedelta
from typing import Any, List, Optional

from db.models.service import (
    ComponentConnection,
    Service,
    ServiceMonitoring,
    Workflow,
    WorkflowComponent,
    WorkflowStatus,
)
from repos.base import CRUDBase
from schemas.app_service import ServiceCreateRequest, ServiceUpdateRequest
from sqlalchemy import and_, case, distinct, func
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
        self, db: Session, *, skip: int = 0, limit: int = 100, creator_id: Optional[int] = None
    ) -> List[Service]:
        """필터링된 서비스 목록 조회"""
        query = db.query(Service).options(joinedload(Service.creator), joinedload(Service.workflows))

        if creator_id:
            query = query.filter(Service.creator_id == creator_id)

        return query.offset(skip).limit(limit).all()

    def count(self, db: Session, *, creator_id: Optional[int] = None) -> int:
        """필터 조건에 맞는 서비스 개수 조회"""
        query = db.query(Service)

        if creator_id is not None:
            query = query.filter(Service.creator_id == creator_id)

        return query.count()

    # update_service 메서드 제거 - base의 update() 메서드 사용

    def delete_with_workflow_unlink(self, db: Session, service_id: str) -> bool:
        """서비스 삭제.

        DB FK 제약에 위임한다: 연결 워크플로우는 ON DELETE SET NULL 로 보존(연결만 해제),
        서비스의 모니터링 행은 ON DELETE CASCADE 로 함께 삭제된다.
        """
        service = db.query(Service).filter(Service.id == service_id).first()
        if not service:
            return False

        db.delete(service)
        db.flush()
        return True


class ServiceMonitoringRepository(CRUDBase[ServiceMonitoring, Any, Any]):
    """ServiceMonitoring Repository"""

    # 기간 접두사 → 윈도우 길이 (가장 넓은 창은 1w)
    PERIODS = {
        "h1": timedelta(hours=1),
        "d1": timedelta(days=1),
        "w1": timedelta(weeks=1),
    }

    def _period_columns(self, now):
        """기간별(1h/1d/1w) 조건부 원시 집계 컬럼 생성.

        message_count·success_count·active_users·token_usage·response_time_ms 만 뽑고,
        error_count/success_rate/avg_interaction_count 는 서비스 계층에서 유도한다.
        """
        columns = []
        for prefix, delta in self.PERIODS.items():
            in_window = ServiceMonitoring.timestamp >= (now - delta)
            columns.extend(
                [
                    # 전체 행 수 → message_count
                    func.coalesce(func.sum(case((in_window, 1), else_=0)), 0).label(f"{prefix}_message_count"),
                    # 성공 행 수 → error_count/success_rate 유도용
                    func.coalesce(
                        func.sum(case((and_(in_window, ServiceMonitoring.success.is_(True)), 1), else_=0)), 0
                    ).label(f"{prefix}_success_count"),
                    # 고유 사용자 수 → active_users (창 밖/NULL은 제외)
                    func.count(distinct(case((in_window, ServiceMonitoring.user_id)))).label(f"{prefix}_active_users"),
                    # 토큰 합산
                    func.coalesce(func.sum(case((in_window, ServiceMonitoring.token_usage), else_=0)), 0).label(
                        f"{prefix}_token_usage"
                    ),
                    # 응답 시간 평균 (창 밖은 NULL → AVG가 자동 제외)
                    func.avg(case((in_window, ServiceMonitoring.response_time_ms))).label(f"{prefix}_response_time_ms"),
                ]
            )
        return columns

    def get_metrics_multi_period(self, db: Session, service_id: str, now):
        """전체 서비스 기간별(1h/1d/1w) 집계 — 단일 행 반환"""
        widest = now - self.PERIODS["w1"]
        return (
            db.query(*self._period_columns(now))
            .filter(and_(ServiceMonitoring.service_id == service_id, ServiceMonitoring.timestamp >= widest))
            .first()
        )

    def get_workflow_metrics_multi_period(self, db: Session, service_id: str, now):
        """워크플로우별 기간별 집계 — workflow_id로 GROUP BY (서비스당 1쿼리)"""
        widest = now - self.PERIODS["w1"]
        return (
            db.query(ServiceMonitoring.workflow_id.label("workflow_id"), *self._period_columns(now))
            .filter(
                and_(
                    ServiceMonitoring.service_id == service_id,
                    ServiceMonitoring.workflow_id.isnot(None),
                    ServiceMonitoring.timestamp >= widest,
                )
            )
            .group_by(ServiceMonitoring.workflow_id)
            .all()
        )

    def delete_older_than(self, db: Session, cutoff) -> int:
        """retention: cutoff(datetime) 이전 레코드 일괄 삭제. 삭제 건수 반환."""
        deleted = (
            db.query(ServiceMonitoring).filter(ServiceMonitoring.timestamp < cutoff).delete(synchronize_session=False)
        )
        db.commit()
        return deleted


# Repository 인스턴스
service_repository = ServiceRepository(Service)
service_monitoring_repository = ServiceMonitoringRepository(ServiceMonitoring)
