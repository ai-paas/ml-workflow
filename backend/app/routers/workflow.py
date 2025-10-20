"""Workflow API 라우터"""

import logging
from typing import List, Optional

from config.db.connect import SessionDepends
from core.kubeflow.kubeflow_manager import KubeflowManager
from core.kubeflow.workflow_executor import WorkflowExecutor
from db.models.service import Workflow, WorkflowStatus
from fastapi import APIRouter, Depends, HTTPException, Query, status
from repos.workflow import workflow_repository
from schemas.user import UserSchema
from schemas.workflow import (
    WorkflowBaseSchema,
    WorkflowCreateRequest,
    WorkflowExecuteRequest,
    WorkflowExecuteResponse,
    WorkflowListSchema,
    WorkflowReadSchema,
    WorkflowTemplateCreateRequest,
    WorkflowTemplateReadSchema,
    WorkflowUpdateRequest,
)
from services.workflow import WorkflowService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ============= Workflow CRUD =============


@router.post("", response_model=WorkflowReadSchema, status_code=status.HTTP_201_CREATED)
def create_workflow(
    *,
    db: Session = SessionDepends,
    workflow_data: WorkflowCreateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    새로운 워크플로우 생성

    - 직접 생성: workflow_definition 제공
    - 템플릿으로부터 생성: template_id 제공
    """
    try:
        workflow = WorkflowService.create_workflow(db=db, workflow_data=workflow_data, creator_id=current_user.id)

        return WorkflowReadSchema.model_validate(workflow)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create workflow")


@router.get("", response_model=WorkflowListSchema)
def list_workflows(
    *,
    db: Session = SessionDepends,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    creator_id: Optional[int] = Query(None),
    service_id: Optional[int] = Query(None),
    is_template: Optional[bool] = Query(None),
    status: Optional[str] = Query(None),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 목록 조회

    - **is_template**: true면 템플릿만, false면 일반 워크플로우만 조회
    """
    workflows = WorkflowService.get_workflows(
        db=db,
        skip=skip,
        limit=limit,
        creator_id=creator_id,
        service_id=service_id,
        is_template=is_template,
        status=status,
    )

    items = [WorkflowBaseSchema.model_validate(w) for w in workflows]

    return WorkflowListSchema(total=len(items), items=items)


@router.get("/{workflow_id}", response_model=WorkflowReadSchema)
def get_workflow(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """워크플로우 상세정보 조회"""
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    result = WorkflowReadSchema.from_orm(workflow)

    # 추가 정보 설정
    if workflow.service:
        result.service_name = workflow.service.name
    if workflow.template:
        result.template_name = workflow.template.name

    return result


@router.put("/{workflow_id}", response_model=WorkflowReadSchema)
def update_workflow(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    workflow_data: WorkflowUpdateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 수정

    workflow_definition이 제공되면 컴포넌트와 연결도 업데이트됨
    """
    workflow = WorkflowService.update_workflow(db=db, workflow_id=workflow_id, workflow_data=workflow_data)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    return WorkflowReadSchema.from_orm(workflow)


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workflow(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 삭제

    템플릿의 경우 파생된 워크플로우가 있으면 삭제 불가
    """
    try:
        success = WorkflowService.delete_workflow(db, workflow_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return None


# ============= Template Management =============


@router.post("/templates", response_model=WorkflowTemplateReadSchema, status_code=status.HTTP_201_CREATED)
def create_workflow_template(
    *,
    db: Session = SessionDepends,
    template_data: WorkflowTemplateCreateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """워크플로우 템플릿 생성"""
    try:
        template = WorkflowService.create_workflow_template(
            db=db, template_data=template_data, creator_id=current_user.id
        )

        result = WorkflowTemplateReadSchema.model_validate(template)

        # 사용 횟수 계산
        usage_count = db.query(Workflow).filter(Workflow.template_id == template.id).count()
        result.usage_count = usage_count

        return result

    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create template")


@router.get("/templates", response_model=List[WorkflowTemplateReadSchema])
def list_workflow_templates(
    *,
    db: Session = SessionDepends,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    current_user: UserSchema = Depends(get_current_user),
):
    """워크플로우 템플릿 목록 조회"""
    templates = WorkflowService.get_workflow_templates(
        db=db, skip=skip, limit=limit, creator_id=None, category=category  # 모든 사용자의 템플릿 조회 가능
    )

    results = []
    for template in templates:
        result = WorkflowTemplateReadSchema.model_validate(template)

        # 사용 횟수 계산
        usage_count = db.query(Workflow).filter(Workflow.template_id == template.id).count()
        result.usage_count = usage_count

        results.append(result)

    return results


@router.post("/templates/{template_id}/clone", response_model=WorkflowReadSchema)
def clone_from_template(
    *,
    db: Session = SessionDepends,
    template_id: int,
    workflow_name: str = Query(..., description="새 워크플로우 이름"),
    service_id: Optional[int] = Query(None, description="연결할 서비스 ID"),
    current_user: UserSchema = Depends(get_current_user),
):
    """템플릿으로부터 워크플로우 생성"""
    try:
        workflow = WorkflowService.clone_from_template(
            db=db,
            template_id=template_id,
            workflow_name=workflow_name,
            service_id=service_id,
            creator_id=current_user.id,
        )

        return WorkflowReadSchema.from_orm(workflow)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to clone from template: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clone from template")


@router.put("/templates/{template_id}", response_model=WorkflowTemplateReadSchema)
def update_workflow_template(
    *,
    db: Session = SessionDepends,
    template_id: int,
    template_data: WorkflowUpdateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """워크플로우 템플릿 수정"""
    # 템플릿인지 확인
    template = WorkflowService.get_workflow_by_id(db, template_id)
    if not template or not template.is_template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

    updated_template = WorkflowService.update_workflow(db=db, workflow_id=template_id, workflow_data=template_data)

    result = WorkflowTemplateReadSchema.from_orm(updated_template)

    # 사용 횟수 계산
    usage_count = db.query(Workflow).filter(Workflow.template_id == template_id).count()
    result.usage_count = usage_count

    return result


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workflow_template(
    *, db: Session = SessionDepends, template_id: int, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 템플릿 삭제

    파생된 워크플로우가 있으면 삭제 불가
    """
    # 템플릿인지 확인
    template = WorkflowService.get_workflow_by_id(db, template_id)
    if not template or not template.is_template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

    try:
        success = WorkflowService.delete_workflow(db, template_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return None


# ============= Workflow Execution =============


@router.post("/{workflow_id}/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    execute_data: WorkflowExecuteRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 실행 (KServe 배포 + Kubeflow 파이프라인 실행)

    1. MODEL 컴포넌트들을 KServe로 배포
    2. 워크플로우를 Kubeflow 파이프라인으로 변환하여 실행
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    if workflow.status != WorkflowStatus.ACTIVE:
        if workflow.status == WorkflowStatus.DRAFT:
            detail = "Workflow is in draft status. Please activate it before execution."
        elif workflow.status == WorkflowStatus.ERROR:
            detail = "Workflow has errors. Please fix the errors before execution."
        else:
            detail = f"Workflow is not active. Current status: {workflow.status}"

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    try:
        # WorkflowExecutor를 사용하여 워크플로우 실행
        executor = WorkflowExecutor(db)
        execution_result = executor.execute_workflow(workflow=workflow, parameters=execute_data.parameters)

        return WorkflowExecuteResponse(
            workflow_id=execution_result["workflow_id"],
            kubeflow_run_id=execution_result["kubeflow_run_id"],
            status=execution_result["status"],
            message=execution_result["message"],
        )

    except Exception as e:
        logger.error(f"Failed to execute workflow {workflow_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to execute workflow: {str(e)}"
        )


@router.get("/{workflow_id}/status")
def get_workflow_execution_status(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 실행 상태 조회

    KServe 배포 상태와 Kubeflow 파이프라인 실행 상태를 조회
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    try:
        executor = WorkflowExecutor(db)
        workflow_status = executor.get_workflow_status(workflow)

        return workflow_status

    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        return {
            "workflow_id": workflow_id,
            "status": workflow.status.value,
            "kubeflow_run_id": workflow.kubeflow_run_id,
            "error": str(e),
        }


@router.post("/{workflow_id}/cleanup")
def cleanup_workflow_resources(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 리소스 정리

    배포된 KServe 인퍼런스 서비스들을 삭제
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    try:
        executor = WorkflowExecutor(db)
        deleted_count = executor.cleanup_deployed_services(str(workflow_id))

        # 워크플로우 상태를 DRAFT로 변경 (재실행 가능하도록)
        if workflow.status == WorkflowStatus.ERROR:
            workflow.status = WorkflowStatus.DRAFT
            db.commit()

        return {
            "workflow_id": workflow_id,
            "deleted_services": deleted_count,
            "message": f"Successfully cleaned up {deleted_count} services",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup workflow resources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to cleanup resources: {str(e)}"
        )
