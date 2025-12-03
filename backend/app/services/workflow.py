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
    WorkflowTemplateCreateRequest,
    WorkflowUpdateDefinition,
    WorkflowUpdateInternal,
    WorkflowUpdateRequest,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()


class WorkflowService:
    """워크플로우 관련 비즈니스 로직"""

    @staticmethod
    def create_workflow(
        db: Session,
        workflow_data: WorkflowCreateRequest,
        creator_id: int,
        initial_status: WorkflowStatus = WorkflowStatus.DRAFT,
    ) -> Workflow:
        """새로운 워크플로우 생성 (직접 생성만 지원, 템플릿으로부터 생성은 clone_from_template 사용)"""
        try:
            # workflow_definition은 DB에 저장하지 않고 컴포넌트/연결 생성에만 사용
            workflow_definition_dict = None
            workflow_data_dict = workflow_data.model_dump()
            if workflow_data_dict.get("workflow_definition"):
                workflow_definition_dict = workflow_data_dict.pop("workflow_definition")

            # WorkflowCreateInternal 사용 (status와 creator_id 포함, workflow_definition 제외)
            # is_template은 항상 False (워크플로우 생성용)
            # template_id는 None으로 설정 (템플릿으로부터 생성은 clone_from_template 사용)
            workflow_data_dict["template_id"] = None
            workflow_internal = WorkflowCreateInternal(
                **workflow_data_dict, is_template=False, status=initial_status, creator_id=creator_id
            )

            # base의 create 메서드 사용 (DB 작업 포함)
            workflow = workflow_repository.create(db, obj_in=workflow_internal)

            # 워크플로우 정의가 있으면 컴포넌트와 연결 생성
            if workflow_definition_dict:
                definition = WorkflowDefinition(**workflow_definition_dict)
                WorkflowService._create_components_and_connections(db, workflow.id, definition)

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
        # 타입별 컴포넌트 매핑 (같은 타입이 여러 개일 수 있으므로 리스트로 관리)
        component_type_map: Dict[ComponentType, List[str]] = {}  # type -> [component_ids]

        # 컴포넌트 생성
        for comp_data in definition.components:
            component = workflow_component_repository.create_component(
                db, workflow_id=workflow_id, component_data=comp_data
            )
            # 타입별로 컴포넌트 ID 저장
            if comp_data.type not in component_type_map:
                component_type_map[comp_data.type] = []
            component_type_map[comp_data.type].append(component.id)

        # 연결 정보 생성
        for conn_data in definition.connections:
            # 타입으로 소스/타겟 컴포넌트 찾기
            source_components = component_type_map.get(conn_data.source_component_type, [])
            target_components = component_type_map.get(conn_data.target_component_type, [])

            if not source_components:
                logger.warning(f"Source component type {conn_data.source_component_type} not found")
                continue
            if not target_components:
                logger.warning(f"Target component type {conn_data.target_component_type} not found")
                continue

            # 첫 번째 매칭되는 컴포넌트 사용 (같은 타입이 여러 개인 경우 첫 번째)
            source_component_id = source_components[0]
            target_component_id = target_components[0]

            component_connection_repository.create_connection(
                db,
                workflow_id=workflow_id,
                source_component_id=source_component_id,
                target_component_id=target_component_id,
                connection_type="DATA",  # 기본값 사용
                config=None,  # config는 사용하지 않음
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
    def count_workflows(
        db: Session,
        *,
        creator_id: Optional[int] = None,
        service_id: Optional[int] = None,
        is_template: Optional[bool] = None,
        status: Optional[WorkflowStatus] = None,
        category: Optional[str] = None,
    ) -> int:
        """필터 조건에 맞는 워크플로우 개수 조회"""
        return workflow_repository.count(
            db,
            creator_id=creator_id,
            service_id=service_id,
            is_template=is_template,
            status=status,
            category=category,
        )

    @staticmethod
    def get_template_usage_count(db: Session, template_id: str) -> int:
        """템플릿 사용 횟수 조회"""
        return workflow_repository.get_derived_workflows_count(db, template_id)

    @staticmethod
    def get_component_by_id_and_workflow_id(
        db: Session, component_id: str, workflow_id: str
    ) -> Optional[WorkflowComponent]:
        """컴포넌트 ID와 워크플로우 ID로 컴포넌트 조회"""
        return workflow_component_repository.get_by_id_and_workflow_id(db, component_id, workflow_id)

    @staticmethod
    def update_workflow(db: Session, workflow_id: str, workflow_data: WorkflowUpdateRequest) -> Optional[Workflow]:
        """워크플로우 정보 수정"""
        workflow = workflow_repository.get(db, workflow_id)
        if not workflow:
            return None

        try:
            # 업데이트할 필드만 수정
            update_data = workflow_data.dict(exclude_unset=True)

            # 템플릿인 경우 service_id는 수정할 수 없음
            if workflow.is_template and "service_id" in update_data:
                update_data.pop("service_id")

            # workflow_definition이 업데이트되면 컴포넌트와 연결도 업데이트
            if "workflow_definition" in update_data and update_data["workflow_definition"]:
                # 기존 컴포넌트와 연결 삭제
                component_connection_repository.delete_by_workflow_id(db, workflow_id)
                workflow_component_repository.delete_by_workflow_id(db, workflow_id)

                # WorkflowUpdateDefinition을 WorkflowDefinition으로 변환 (기본값 사용)
                update_def = WorkflowUpdateDefinition(**update_data["workflow_definition"])

                # ComponentUpdateRequest를 ComponentCreateRequest로 변환 (기본값 사용)
                components = []
                for comp_update in update_def.components:
                    components.append(
                        ComponentCreateRequest(
                            name=comp_update.name,
                            type=comp_update.type,
                            model_id=comp_update.model_id,
                            knowledge_base_id=comp_update.knowledge_base_id,
                            prompt_id=comp_update.prompt_id,
                        )
                    )

                # ConnectionUpdateRequest를 ConnectionCreateRequest로 변환 (기본값 사용)
                connections = []
                for conn_update in update_def.connections:
                    connections.append(
                        ConnectionCreateRequest(
                            source_component_type=conn_update.source_component_type,
                            target_component_type=conn_update.target_component_type,
                        )
                    )

                definition = WorkflowDefinition(components=components, connections=connections)
                WorkflowService._create_components_and_connections(db, workflow_id, definition)

            # workflow_definition은 DB에 저장하지 않으므로 update_data에서 무조건 제거
            update_data.pop("workflow_definition", None)

            # base의 update 메서드 사용
            # UpdateSchemaType을 받으므로 WorkflowUpdateInternal 생성
            update_request = WorkflowUpdateInternal(**update_data)
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
    def create_workflow_template(
        db: Session, template_data: WorkflowTemplateCreateRequest, creator_id: int
    ) -> Workflow:
        """워크플로우 템플릿 생성"""
        try:
            # workflow_definition은 DB에 저장하지 않고 컴포넌트/연결 생성에만 사용
            workflow_definition_dict = None
            template_data_dict = template_data.model_dump()
            if template_data_dict.get("workflow_definition"):
                workflow_definition_dict = template_data_dict.pop("workflow_definition")

            # 템플릿은 서비스에 직접 연결되지 않음 (service_id와 template_id는 스키마에 포함되지 않음)
            template_data_dict["service_id"] = None
            template_data_dict["template_id"] = None

            # WorkflowCreateInternal 사용 (status와 creator_id 포함, workflow_definition 제외)
            # is_template은 True로 설정
            template_internal = WorkflowCreateInternal(
                **template_data_dict, is_template=True, status=WorkflowStatus.DRAFT, creator_id=creator_id
            )

            # base의 create 메서드 사용 (DB 작업 포함)
            template = workflow_repository.create(db, obj_in=template_internal)

            # 워크플로우 정의가 있으면 컴포넌트와 연결 생성
            if workflow_definition_dict:
                definition = WorkflowDefinition(**workflow_definition_dict)
                WorkflowService._create_components_and_connections(db, template.id, definition)

            db.commit()

            logger.info(f"Template created successfully: {template.id}")
            return template

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create template: {str(e)}")
            raise

    @staticmethod
    def get_workflow_templates(
        db: Session, skip: int = 0, limit: int = 100, creator_id: Optional[int] = None, category: Optional[str] = None
    ) -> List[Workflow]:
        """워크플로우 템플릿 목록 조회"""
        return workflow_repository.get_templates(db, skip=skip, limit=limit, creator_id=creator_id, category=category)

    @staticmethod
    def get_workflow_template_by_id(db: Session, template_id: str) -> Optional[Workflow]:
        """워크플로우 템플릿 ID로 조회"""
        template = workflow_repository.get_with_relations(db, template_id)
        if template and template.is_template:
            return template
        return None

    @staticmethod
    def clone_from_template(
        db: Session, template_id: str, workflow_name: str, service_id: Optional[int], creator_id: int
    ) -> Workflow:
        """템플릿으로부터 워크플로우 생성"""
        template = workflow_repository.get_with_relations(db, template_id)

        if not template or not template.is_template:
            raise ValueError(f"Template {template_id} not found")

        # 새 워크플로우 생성 (workflow_definition 없이)
        workflow_data = WorkflowCreateRequest(
            name=workflow_name,
            description=f"Created from template: {template.name}",
            category=template.category,
            service_id=service_id,
            template_id=template_id,
            workflow_definition=None,  # workflow_definition은 사용하지 않음
        )

        # workflow_definition은 DB에 저장하지 않으므로 제거
        workflow_data_dict = workflow_data.model_dump()
        workflow_data_dict.pop("workflow_definition", None)

        # WorkflowCreateInternal 사용 (status와 creator_id 포함, workflow_definition 제외)
        # is_template은 False로 설정 (템플릿으로부터 생성된 워크플로우)
        # 상태는 DRAFT로 시작 (실행 후 파이프라인 완료 시 ACTIVE로 변경됨)
        workflow_internal = WorkflowCreateInternal(
            **workflow_data_dict, is_template=False, status=WorkflowStatus.DRAFT, creator_id=creator_id
        )

        # base의 create 메서드 사용 (DB 작업 포함)
        workflow = workflow_repository.create(db, obj_in=workflow_internal)

        # 템플릿의 컴포넌트와 연결을 직접 복사
        component_map = {}  # template_component_id -> new_component_id 매핑

        # 컴포넌트 복사
        for template_component in template.components:
            new_component = WorkflowComponent(
                workflow_id=workflow.id,
                name=template_component.name,
                type=template_component.type,
                config=None,  # config는 사용하지 않음
                model_id=template_component.model_id,
            )
            db.add(new_component)
            db.flush()
            component_map[template_component.id] = new_component.id

        # 연결 복사
        for template_connection in template.component_connections:
            # source_component_id와 target_component_id를 새 컴포넌트 ID로 매핑
            source_new_id = component_map.get(template_connection.source_component_id)
            target_new_id = component_map.get(template_connection.target_component_id)

            if source_new_id and target_new_id:
                new_connection = ComponentConnection(
                    workflow_id=workflow.id,
                    source_component_id=source_new_id,
                    target_component_id=target_new_id,
                    connection_type="DATA",  # 기본값 사용
                    config=None,  # config는 사용하지 않음
                )
                db.add(new_connection)

        db.commit()

        logger.info(f"Workflow cloned from template successfully: {workflow.id}")
        return workflow

    @staticmethod
    def _validate_knowledge_base_before_model(workflow: Workflow) -> None:
        """
        워크플로우에서 지식베이스가 모델 앞에 있는지 검증

        Args:
            workflow: 검증할 워크플로우

        Raises:
            ValueError: 지식베이스가 모델 뒤에 있는 경우
        """
        # START 컴포넌트 찾기
        start_components = [c for c in workflow.components if c.type == ComponentType.START]
        if not start_components:
            return  # START가 없으면 검증 스킵

        # 컴포넌트 ID -> 컴포넌트 매핑
        component_map = {c.id: c for c in workflow.components}

        # 연결 정보로 그래프 구성 (source -> target)
        graph = {}
        for conn in workflow.component_connections:
            if conn.source_component_id not in graph:
                graph[conn.source_component_id] = []
            graph[conn.source_component_id].append(conn.target_component_id)

        # BFS로 워크플로우 순회하면서 KNOWLEDGE_BASE와 MODEL의 순서 확인
        visited = set()
        knowledge_base_found = False

        def traverse(component_id: str):
            nonlocal knowledge_base_found
            if component_id in visited:
                return
            visited.add(component_id)

            component = component_map.get(component_id)
            if not component:
                return

            # KNOWLEDGE_BASE 발견
            if component.type == ComponentType.KNOWLEDGE_BASE:
                knowledge_base_found = True

            # MODEL 발견 - 이전에 KNOWLEDGE_BASE가 없었으면 에러
            if component.type == ComponentType.MODEL:
                if not knowledge_base_found:
                    # 하지만 KNOWLEDGE_BASE가 워크플로우에 없으면 괜찮음
                    kb_components = [c for c in workflow.components if c.type == ComponentType.KNOWLEDGE_BASE]
                    if kb_components:
                        raise ValueError(
                            "Knowledge Base components must come before Model components in the workflow. "
                            f"Found Model component '{component.name}' before any Knowledge Base component."
                        )

            # 다음 컴포넌트로 이동
            for next_id in graph.get(component_id, []):
                traverse(next_id)

        # 모든 START 컴포넌트에서 시작
        for start_component in start_components:
            traverse(start_component.id)
