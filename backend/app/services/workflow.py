"""Workflow 관련 비즈니스 로직"""

import json
import logging
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from db.models.service import ComponentConnection, ComponentType, Workflow, WorkflowComponent, WorkflowStatus
from repos.workflow import component_connection_repository, workflow_component_repository, workflow_repository
from schemas.workflow import (
    ComponentCreateRequest,
    ConnectionCreateRequest,
    WorkflowCreateInternal,
    WorkflowCreateRequest,
    WorkflowDefinition,
    WorkflowUpdateRequest,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()


class WorkflowService:
    """워크플로우 관련 비즈니스 로직"""

    @staticmethod
    def create_workflow(db: Session, workflow_data: WorkflowCreateRequest, creator_id: int) -> Workflow:
        """새로운 워크플로우 생성"""
        try:
            # 템플릿으로부터 생성하는 경우
            if workflow_data.template_id:
                template = workflow_repository.get_with_relations(db, workflow_data.template_id)

                if not template or not template.is_template:
                    raise ValueError(f"Template {workflow_data.template_id} not found")

                # 템플릿의 정의를 복사
                if not workflow_data.workflow_definition and template.workflow_definition:
                    workflow_data.workflow_definition = WorkflowDefinition(
                        **json.loads(template.workflow_definition)
                        if isinstance(template.workflow_definition, str)
                        else template.workflow_definition
                    )

            # WorkflowCreateInternal 사용 (status와 creator_id 포함)
            workflow_internal = WorkflowCreateInternal(
                **workflow_data.model_dump(), status=WorkflowStatus.DRAFT, creator_id=creator_id
            )

            # workflow_definition을 dict로 변환 (필요시)
            if workflow_internal.workflow_definition:
                workflow_internal.workflow_definition = workflow_internal.workflow_definition.model_dump()

            # base의 create 메서드 사용 (DB 작업 포함)
            workflow = workflow_repository.create(db, obj_in=workflow_internal)

            # 워크플로우 정의가 있으면 컴포넌트와 연결 생성
            if workflow_data.workflow_definition:
                WorkflowService._create_components_and_connections(db, workflow.id, workflow_data.workflow_definition)

            db.commit()

            logger.info(f"Workflow created successfully: {workflow.id}")
            return workflow

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create workflow: {str(e)}")
            raise

    @staticmethod
    def _create_components_and_connections(db: Session, workflow_id: str, definition: WorkflowDefinition):
        """컴포넌트와 연결 정보 생성"""
        component_map = {}  # component_id -> db_component_id 매핑

        # 컴포넌트 생성
        for comp_data in definition.components:
            component = workflow_component_repository.create_component(
                db, workflow_id=workflow_id, component_data=comp_data
            )
            component_map[comp_data.component_id] = component.id

        # 연결 정보 생성
        for conn_data in definition.connections:
            if conn_data.source_component_id not in component_map:
                logger.warning(f"Source component {conn_data.source_component_id} not found")
                continue
            if conn_data.target_component_id not in component_map:
                logger.warning(f"Target component {conn_data.target_component_id} not found")
                continue

            component_connection_repository.create_connection(
                db,
                workflow_id=workflow_id,
                source_component_id=component_map[conn_data.source_component_id],
                target_component_id=component_map[conn_data.target_component_id],
                connection_type=conn_data.connection_type,
                config=conn_data.config,
            )

    @staticmethod
    def get_workflow_by_id(db: Session, workflow_id: str) -> Optional[Workflow]:
        """ID로 워크플로우 조회"""
        return workflow_repository.get_with_relations(db, workflow_id)

    @staticmethod
    def get_workflows(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        creator_id: Optional[int] = None,
        service_id: Optional[int] = None,
        is_template: Optional[bool] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> List[Workflow]:
        """워크플로우 목록 조회"""
        return workflow_repository.get_multi_with_filters(
            db,
            skip=skip,
            limit=limit,
            creator_id=creator_id,
            service_id=service_id,
            is_template=is_template,
            status=status,
        )

    @staticmethod
    def update_workflow(db: Session, workflow_id: str, workflow_data: WorkflowUpdateRequest) -> Optional[Workflow]:
        """워크플로우 정보 수정"""
        workflow = workflow_repository.get(db, workflow_id)
        if not workflow:
            return None

        try:
            # 업데이트할 필드만 수정
            update_data = workflow_data.dict(exclude_unset=True)

            # workflow_definition이 업데이트되면 컴포넌트와 연결도 업데이트
            if "workflow_definition" in update_data and update_data["workflow_definition"]:
                # 기존 컴포넌트와 연결 삭제
                component_connection_repository.delete_by_workflow_id(db, workflow_id)
                workflow_component_repository.delete_by_workflow_id(db, workflow_id)

                # 새로운 컴포넌트와 연결 생성
                definition = WorkflowDefinition(**update_data["workflow_definition"])
                WorkflowService._create_components_and_connections(db, workflow_id, definition)

                # JSON으로 저장
                update_data["workflow_definition"] = definition.dict()

            # base의 update 메서드 사용
            # UpdateSchemaType을 받으므로 WorkflowUpdateRequest 생성
            update_request = WorkflowUpdateRequest(**update_data)
            workflow = workflow_repository.update(db, db_obj=workflow, obj_in=update_request)

            db.commit()

            logger.info(f"Workflow updated successfully: {workflow.id}")
            return workflow

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update workflow: {str(e)}")
            raise

    @staticmethod
    def delete_workflow(db: Session, workflow_id: str) -> bool:
        """워크플로우 삭제"""
        workflow = workflow_repository.get(db, workflow_id)
        if not workflow:
            return False

        try:
            # 템플릿인 경우 파생된 워크플로우가 있는지 확인
            if workflow.is_template:
                derived_count = workflow_repository.get_derived_workflows_count(db, workflow_id)

                if derived_count > 0:
                    logger.warning(f"Cannot delete template {workflow_id}: {derived_count} derived workflows exist")
                    raise ValueError(f"Template has {derived_count} derived workflows")

            workflow_repository.delete(db, pk=workflow_id)
            db.commit()

            logger.info(f"Workflow deleted successfully: {workflow_id}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete workflow: {str(e)}")
            raise

    @staticmethod
    def create_workflow_template(db: Session, template_data: WorkflowCreateRequest, creator_id: int) -> Workflow:
        """워크플로우 템플릿 생성"""
        template_data.is_template = True
        template_data.service_id = None  # 템플릿은 서비스에 직접 연결되지 않음
        return WorkflowService.create_workflow(db, template_data, creator_id)

    @staticmethod
    def get_workflow_templates(
        db: Session, skip: int = 0, limit: int = 100, creator_id: Optional[int] = None, category: Optional[str] = None
    ) -> List[Workflow]:
        """워크플로우 템플릿 목록 조회"""
        return workflow_repository.get_templates(db, skip=skip, limit=limit, creator_id=creator_id, category=category)

    @staticmethod
    def clone_from_template(
        db: Session, template_id: int, workflow_name: str, service_id: Optional[int], creator_id: int
    ) -> Workflow:
        """템플릿으로부터 워크플로우 생성"""
        template = workflow_repository.get_with_relations(db, template_id)

        if not template or not template.is_template:
            raise ValueError(f"Template {template_id} not found")

        # 템플릿 정의를 복사하여 새 워크플로우 생성
        workflow_data = WorkflowCreateRequest(
            name=workflow_name,
            description=f"Created from template: {template.name}",
            category=template.category,
            service_id=service_id,
            is_template=False,
            template_id=template_id,
            workflow_definition=WorkflowDefinition(
                **json.loads(template.workflow_definition)
                if isinstance(template.workflow_definition, str)
                else template.workflow_definition
            )
            if template.workflow_definition
            else None,
        )

        return WorkflowService.create_workflow(db, workflow_data, creator_id)
