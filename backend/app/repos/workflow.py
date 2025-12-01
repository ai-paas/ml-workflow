"""Workflow 및 관련 엔티티 Repository"""

from typing import List, Optional

from db.models.service import ComponentConnection, ComponentType, Workflow, WorkflowComponent, WorkflowStatus
from repos.base import CRUDBase
from schemas.workflow import (
    ComponentCreateRequest,
    ConnectionCreateRequest,
    WorkflowCreateInternal,
    WorkflowCreateRequest,
    WorkflowUpdateInternal,
    WorkflowUpdateRequest,
)
from sqlalchemy.orm import Session, joinedload


class WorkflowRepository(CRUDBase[Workflow, WorkflowCreateInternal, WorkflowUpdateInternal]):
    """Workflow Repository"""

    def get_with_relations(self, db: Session, workflow_id: str) -> Optional[Workflow]:
        """관계 포함 워크플로우 조회"""
        return (
            db.query(Workflow)
            .options(
                joinedload(Workflow.creator),
                joinedload(Workflow.service),
                joinedload(Workflow.template),
                joinedload(Workflow.components).joinedload(WorkflowComponent.model),
                joinedload(Workflow.component_connections),
                joinedload(Workflow.kserve_deployments),
            )
            .filter(Workflow.id == workflow_id)
            .first()
        )

    def get_multi_with_filters(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        creator_id: Optional[int] = None,
        service_id: Optional[int] = None,
        is_template: Optional[bool] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[Workflow]:
        """필터링된 워크플로우 목록 조회"""
        query = db.query(Workflow).options(joinedload(Workflow.creator), joinedload(Workflow.service))

        if creator_id is not None:
            query = query.filter(Workflow.creator_id == creator_id)

        if service_id is not None:
            query = query.filter(Workflow.service_id == service_id)

        if is_template is not None:
            query = query.filter(Workflow.is_template == is_template)

        if status is not None:
            query = query.filter(Workflow.status == status)

        return query.offset(skip).limit(limit).all()

    def get_templates(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        creator_id: Optional[int] = None,
        category: Optional[str] = None
    ) -> List[Workflow]:
        """워크플로우 템플릿 목록 조회"""
        query = db.query(Workflow).filter(Workflow.is_template == True).options(joinedload(Workflow.creator))

        if creator_id:
            query = query.filter(Workflow.creator_id == creator_id)

        if category:
            query = query.filter(Workflow.category == category)

        return query.offset(skip).limit(limit).all()

    def get_derived_workflows_count(self, db: Session, template_id: int) -> int:
        """템플릿에서 파생된 워크플로우 개수 조회"""
        # base에 count 메서드가 없으므로 유지
        return db.query(Workflow).filter(Workflow.template_id == template_id).count()

    # create_workflow, update_workflow_fields 제거 - base의 create, update 메서드로 충분

    def update_workflow_status(
        self,
        db: Session,
        workflow_id: str,
        status: WorkflowStatus,
        kubeflow_run_id: Optional[str] = None,
    ) -> Optional[Workflow]:
        """워크플로우 상태 업데이트"""
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            return None

        workflow.status = status
        if kubeflow_run_id:
            workflow.kubeflow_run_id = kubeflow_run_id

        db.add(workflow)
        db.flush()
        db.refresh(workflow)
        return workflow

    def get_by_service_id(self, db: Session, service_id: str) -> List[Workflow]:
        """서비스 ID로 워크플로우 목록 조회 - base의 filter 활용"""
        return self.filter(db, {"service_id": service_id})


class WorkflowComponentRepository(CRUDBase[WorkflowComponent, ComponentCreateRequest, ComponentCreateRequest]):
    """WorkflowComponent Repository"""

    def create_component(
        self, db: Session, workflow_id: str, component_data: ComponentCreateRequest
    ) -> WorkflowComponent:
        """워크플로우 컴포넌트 생성 - base의 create 활용 가능하지만 특별한 로직 있어 유지"""
        component = WorkflowComponent(
            workflow_id=workflow_id,
            name=component_data.name,
            type=ComponentType(component_data.type),
            config=None,  # config는 사용하지 않음
            model_id=component_data.model_id,
        )
        db.add(component)
        db.flush()
        return component

    def get_by_workflow_id(self, db: Session, workflow_id: str) -> List[WorkflowComponent]:
        """워크플로우 ID로 컴포넌트 목록 조회 - base의 filter 활용"""
        return self.filter(db, {"workflow_id": workflow_id})

    def delete_by_workflow_id(self, db: Session, workflow_id: str) -> int:
        """워크플로우의 모든 컴포넌트 삭제"""
        deleted_count = db.query(WorkflowComponent).filter(WorkflowComponent.workflow_id == workflow_id).delete()
        db.flush()
        return deleted_count


class ComponentConnectionRepository(CRUDBase[ComponentConnection, ConnectionCreateRequest, ConnectionCreateRequest]):
    """ComponentConnection Repository"""

    def create_connection(
        self,
        db: Session,
        workflow_id: str,
        source_component_id: str,
        target_component_id: str,
        connection_type: str = "DATA",
        config: Optional[dict] = None,
    ) -> ComponentConnection:
        """컴포넌트 연결 생성"""
        connection = ComponentConnection(
            workflow_id=workflow_id,
            source_component_id=source_component_id,
            target_component_id=target_component_id,
            connection_type=connection_type,
            config=config,
        )
        db.add(connection)
        db.flush()
        return connection

    def get_by_workflow_id(self, db: Session, workflow_id: str) -> List[ComponentConnection]:
        """워크플로우 ID로 연결 목록 조회 - base의 filter 활용"""
        return self.filter(db, {"workflow_id": workflow_id})

    def delete_by_workflow_id(self, db: Session, workflow_id: str) -> int:
        """워크플로우의 모든 연결 삭제"""
        deleted_count = db.query(ComponentConnection).filter(ComponentConnection.workflow_id == workflow_id).delete()
        db.flush()
        return deleted_count


# Repository 인스턴스
workflow_repository = WorkflowRepository(Workflow)
workflow_component_repository = WorkflowComponentRepository(WorkflowComponent)
component_connection_repository = ComponentConnectionRepository(ComponentConnection)
