"""Application Service 비즈니스 로직"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from config.settings import get_settings
from db.models.service import Service, ServiceMonitoring, Workflow, WorkflowStatus
from repos.app_service import service_monitoring_repository, service_repository
from repos.workflow import workflow_repository
from schemas.app_service import (
    MonitoringMetrics,
    ServiceCreateInternal,
    ServiceCreateRequest,
    ServiceMonitoringData,
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
