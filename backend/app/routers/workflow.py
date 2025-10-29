"""Workflow API 라우터"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.db.connect import SessionDepends
from config.settings import get_settings
from core.kubeflow.kubeflow_manager import KubeflowManager
from core.kubeflow.workflow_executor import WorkflowExecutor
from db.models.kserve_deployment import DeploymentStatus, KServeDeployment
from db.models.model import Model
from db.models.service import ComponentType, Workflow, WorkflowComponent, WorkflowStatus
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from repos.workflow import workflow_repository
from schemas.user import UserSchema
from schemas.workflow import (
    ComponentTypeInfo,
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
from services.kserve_deployment import KServeDeploymentService
from services.workflow import WorkflowService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ============= Component Types =============


@router.get("/component-types", response_model=List[ComponentTypeInfo])
def get_component_types():
    """
    사용 가능한 컴포넌트 타입 및 component_id 조회

    워크플로우 템플릿/워크플로우 생성 시 사용할 수 있는 컴포넌트 타입과 component_id를 반환합니다.

    **사용 방법:**
    1. 이 API를 호출하여 사용 가능한 component_id 확인
    2. 워크플로우 정의 시 확인한 component_id를 명시적으로 입력

    **필드 설명:**
    - `type`: 컴포넌트 타입 (START, END, MODEL)
    - `component_id`: 해당 타입에서 사용해야 하는 ID (일반적으로 type과 동일)
    - `name`: 타입의 한글 표시명
    - `description`: 타입 설명
    """
    return [
        ComponentTypeInfo(
            type=ComponentType.START.value,
            component_id=ComponentType.START.value,
            name="시작 노드",
            description="워크플로우의 시작점",
        ),
        ComponentTypeInfo(
            type=ComponentType.END.value,
            component_id=ComponentType.END.value,
            name="종료 노드",
            description="워크플로우의 종료점",
        ),
        ComponentTypeInfo(
            type=ComponentType.MODEL.value,
            component_id=ComponentType.MODEL.value,
            name="모델 노드",
            description="ML 모델을 실행하는 노드",
        ),
    ]


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


# ============= Template Management =============
# NOTE: 템플릿 라우트는 /{workflow_id} 보다 먼저 정의되어야 합니다.
# FastAPI는 위에서 아래로 순서대로 라우트를 매칭하므로,
# /templates가 {workflow_id}로 잘못 매칭되는 것을 방지합니다.


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


@router.get("/templates/{template_id}", response_model=WorkflowTemplateReadSchema)
def get_workflow_template(
    *,
    db: Session = SessionDepends,
    template_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """워크플로우 템플릿 상세 조회"""
    template = WorkflowService.get_workflow_template_by_id(db=db, template_id=template_id)

    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

    result = WorkflowTemplateReadSchema.model_validate(template)

    # 사용 횟수 계산
    usage_count = db.query(Workflow).filter(Workflow.template_id == template.id).count()
    result.usage_count = usage_count

    return result


@router.post("/templates/{template_id}/clone", response_model=WorkflowReadSchema)
def clone_from_template(
    *,
    db: Session = SessionDepends,
    template_id: str,  # UUID 문자열로 변경
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

    템플릿은 배포된 KServe InferenceService가 없으므로 즉시 DB에서 삭제됩니다.
    파생된 워크플로우가 있으면 삭제 불가
    """
    # 템플릿인지 확인
    template = WorkflowService.get_workflow_by_id(db, template_id)
    if not template or not template.is_template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

    try:
        # DB에서 템플릿 삭제
        success = WorkflowService.delete_workflow(db, template_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return None


# ============= Workflow CRUD =============
# NOTE: 이 섹션은 템플릿 라우트 아래에 위치해야 합니다.
# /{workflow_id} 패턴이 /templates를 가로채지 않도록 합니다.


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


async def _wait_for_pipeline_completion(run_id: str, max_wait_seconds: int = 300) -> bool:
    """
    Kubeflow Pipeline 실행 완료를 대기

    Args:
        run_id: Pipeline run ID
        max_wait_seconds: 최대 대기 시간 (초)

    Returns:
        성공 여부
    """
    import asyncio

    kf_manager = KubeflowManager()
    elapsed = 0
    check_interval = 3  # 3초마다 확인

    logger.info(f"Waiting for pipeline run {run_id} to complete (max {max_wait_seconds}s)")

    while elapsed < max_wait_seconds:
        try:
            run = kf_manager.kfp_client.get_run(run_id)

            # Run 객체 구조 디버깅
            logger.info(f"Run object type: {type(run)}")
            logger.info(f"Run object attributes: {dir(run)}")

            # 다양한 경로로 status 추출 시도
            status = None

            # V2beta1Run 객체 처리
            if hasattr(run, "state"):
                status = run.state
                logger.info(f"Found status via run.state: {status}")
            elif hasattr(run, "status"):
                status = run.status
                logger.info(f"Found status via run.status: {status}")
            elif hasattr(run, "run"):
                if hasattr(run.run, "status"):
                    status = run.run.status
                    logger.info(f"Found status via run.run.status: {status}")
                elif hasattr(run.run, "state"):
                    status = run.run.state
                    logger.info(f"Found status via run.run.state: {status}")

            # dict로 변환 가능한 경우
            if not status and hasattr(run, "to_dict"):
                run_dict = run.to_dict()
                logger.info(f"Run dict keys: {run_dict.keys()}")
                status = run_dict.get("state") or run_dict.get("status")
                if status:
                    logger.info(f"Found status via to_dict: {status}")

            if status:
                logger.info(f"Pipeline run {run_id} current status: {status}")

                # 완료 상태 확인 (다양한 상태 문자열 지원)
                status_upper = str(status).upper()

                if status_upper in ["SUCCEEDED", "SUCCESS", "COMPLETED"]:
                    logger.info(f"Pipeline run {run_id} completed successfully")
                    return True
                elif status_upper in ["FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"]:
                    logger.error(f"Pipeline run {run_id} failed with status: {status}")
                    return False
                else:
                    logger.info(f"Pipeline run {run_id} is still running: {status}")
            else:
                logger.warning(f"Could not extract status from run object for {run_id}")

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        except Exception as e:
            logger.warning(f"Error checking pipeline status for {run_id}: {e}")
            await asyncio.sleep(check_interval)
            elapsed += check_interval

    logger.warning(f"Pipeline run {run_id} did not complete within {max_wait_seconds} seconds")
    return False


@router.delete("/{workflow_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_workflow(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 삭제 시작

    Kubeflow Pipeline을 통해 KServe InferenceService 리소스 삭제를 시작하고 run_id를 반환합니다.
    실제 DB 삭제는 /workflows/{workflow_id}/finalize-deletion API를 통해 완료 확인 후 수행됩니다.

    Returns:
        202 Accepted: cleanup_run_id를 포함한 응답
    """
    try:
        # 워크플로우 존재 여부 확인
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

        # Kubeflow Pipeline을 통해 KServe InferenceService 리소스 삭제 시작
        cleanup_run_id = None
        try:
            executor = WorkflowExecutor(db)
            cleanup_result = executor.cleanup_deployed_services(workflow_id)
            cleanup_run_id = cleanup_result.get("cleanup_run_id")

            logger.info(f"Cleanup pipeline started for workflow {workflow_id}: run_id={cleanup_run_id}")

        except Exception as e:
            logger.error(f"Failed to start cleanup pipeline for workflow {workflow_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start cleanup pipeline: {str(e)}"
            )

        return {
            "message": "Workflow deletion started",
            "workflow_id": workflow_id,
            "cleanup_run_id": cleanup_run_id,
            "status": "cleanup_in_progress",
            "next_step": (
                f"Call /workflows/{workflow_id}/finalize-deletion?" f"run_id={cleanup_run_id} to complete deletion"
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete workflow: {str(e)}"
        )


@router.post("/{workflow_id}/finalize-deletion")
async def finalize_workflow_deletion(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    run_id: str = Query(..., description="Kubeflow Pipeline cleanup run ID"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 삭제 완료 처리

    Kubeflow Pipeline cleanup이 완료되었는지 확인하고, 완료되었다면 DB에서 워크플로우를 삭제합니다.

    Args:
        workflow_id: 삭제할 워크플로우 ID
        run_id: Cleanup pipeline run ID

    Returns:
        - status: "completed" (완료), "in_progress" (진행중), "failed" (실패)
        - deleted_from_db: DB 삭제 여부 (completed인 경우에만 true)
    """
    try:
        # 워크플로우 존재 여부 확인
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        if not workflow:
            # 이미 삭제된 경우
            return {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "status": "completed",
                "deleted_from_db": True,
                "message": "Workflow already deleted",
            }

        # Pipeline 완료 확인
        success = await _wait_for_pipeline_completion(run_id, max_wait_seconds=5)  # 짧은 timeout으로 즉시 확인

        if success:
            # Pipeline 완료됨 - DB에서 삭제
            logger.info(f"Cleanup pipeline completed for workflow {workflow_id}, deleting from DB")

            delete_success = WorkflowService.delete_workflow(db, workflow_id)

            if delete_success:
                return {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "status": "completed",
                    "deleted_from_db": True,
                    "message": "Workflow deleted successfully",
                }
            else:
                return {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "status": "completed",
                    "deleted_from_db": False,
                    "message": "Pipeline completed but DB deletion failed",
                }
        else:
            # 아직 진행중이거나 실패
            # 실제 상태 확인
            try:
                from ..core.kubeflow.kubeflow_manager import KubeflowManager

                kf_manager = KubeflowManager()
                run = kf_manager.kfp_client.get_run(run_id)

                # 상태 추출
                status_value = None
                if hasattr(run, "state"):
                    status_value = run.state
                elif hasattr(run, "status"):
                    status_value = run.status
                elif hasattr(run, "run"):
                    if hasattr(run.run, "status"):
                        status_value = run.run.status
                    elif hasattr(run.run, "state"):
                        status_value = run.run.state

                if status_value:
                    status_upper = str(status_value).upper()
                    if status_upper in ["FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"]:
                        return {
                            "workflow_id": workflow_id,
                            "run_id": run_id,
                            "status": "failed",
                            "deleted_from_db": False,
                            "message": f"Cleanup pipeline failed with status: {status_value}",
                        }

                # 진행중
                return {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "status": "in_progress",
                    "deleted_from_db": False,
                    "message": "Cleanup pipeline still in progress",
                }

            except Exception as e:
                logger.error(f"Failed to check pipeline status: {e}")
                return {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "status": "unknown",
                    "deleted_from_db": False,
                    "message": f"Failed to check pipeline status: {str(e)}",
                }

    except Exception as e:
        logger.error(f"Failed to finalize workflow deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to finalize deletion: {str(e)}"
        )


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


@router.post("/{workflow_id}/components/{component_id}/deployment-status")
async def update_component_deployment_status(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    component_id: str,
    service_name: str = Body(...),
    service_hostname: str = Body(...),
    model_name: str = Body(...),
    status: str = Body(...),
    internal_url: Optional[str] = Body(None),
    error_message: Optional[str] = Body(None),
):
    """
    컴포넌트의 KServe 배포 상태를 업데이트합니다.
    Kubeflow Pipeline 내에서 배포 완료 후 호출됩니다.
    """
    try:
        # Service를 통한 배포 상태 업데이트
        deployment = KServeDeploymentService.update_deployment_status(
            db=db,
            workflow_id=workflow_id,
            component_id=component_id,
            service_name=service_name,
            service_hostname=service_hostname,
            model_name=model_name,
            status=status,
            internal_url=internal_url,
            error_message=error_message,
        )

        return {
            "message": f"Deployment status updated for component {component_id}",
            "deployment_info": {
                "service_name": deployment.service_name,
                "service_hostname": deployment.service_hostname,
                "model_name": deployment.model_name,
                "status": deployment.status.value,
                "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
            },
        }

    except Exception as e:
        logger.error(f"Failed to update deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/inference")
async def inference_workflow_model(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    component_id: str,
    image: UploadFile = File(...),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    배포된 모델에 추론 요청 (KServe V2 Protocol)

    워크플로우에서 배포된 특정 모델 컴포넌트에 추론을 수행합니다.
    """
    import base64  # noqa: F401, F811

    import requests  # noqa: F401, F811

    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # Service를 통한 배포 검증
    is_ready, error_msg, deployment = KServeDeploymentService.validate_deployment_ready(db, workflow_id, component_id)

    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE if deployment else status.HTTP_404_NOT_FOUND,
            detail=error_msg,
        )

    # 이미지 읽기 및 base64 인코딩
    try:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to read image file")

    # 배포 정보에서 필요한 값들 추출
    infer_svc_url = settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80"  # settings에서 가져오기
    service_hostname = deployment.service_hostname
    model_name = deployment.model_name  # 이미 정제된 이름 (슬래시가 하이픈으로 변경됨)
    service_name = deployment.service_name

    # 모델 정보 가져오기 (추가 메타데이터용)
    model_info = {"component_id": component_id, "service_name": service_name, "sanitized_model_name": model_name}

    # 컴포넌트의 모델 정보 추가
    component = (
        db.query(WorkflowComponent)
        .filter(WorkflowComponent.workflow_id == workflow_id, WorkflowComponent.component_id == component_id)
        .first()
    )

    if component and component.model_id:
        model = db.query(Model).filter(Model.id == component.model_id).first()
        if model:
            model_info.update(
                {
                    "model_id": model.id,
                    "original_model_name": model.name,
                    "model_type": model.type_info.name if model.type_info else None,
                    "model_format": model.format_info.name if model.format_info else None,
                }
            )

    # KServe V2 Protocol 형식으로 요청 데이터 구성
    payload = {"image": image_base64}

    # V2 프로토콜 요청 형식
    data = {"inputs": [{"name": "INPUT_1", "shape": [1], "datatype": "BYTES", "data": [payload]}]}

    # 헤더 설정 (Istio Virtual Service routing을 위한 Host 헤더)
    headers = {"Content-Type": "application/json", "Host": service_hostname}  # Istio가 라우팅하는데 사용

    # Kubeflow 인증이 필요한 경우
    kf_manager = KubeflowManager()
    cookies = kf_manager.auth_session.session_cookie_dict if hasattr(kf_manager, "auth_session") else {}

    try:
        # V2 프로토콜 엔드포인트로 요청 (Istio Gateway 경유)
        url = f"{infer_svc_url}/v2/models/{model_name}/infer"

        logger.info(f"Sending inference request to {url}")

        response = requests.post(url, json=data, headers=headers, cookies=cookies, timeout=30)
        response.raise_for_status()

        result = response.json()

        # V2 프로토콜 응답 파싱
        outputs = result.get("outputs", [])
        if outputs and len(outputs) > 0:
            prediction_data = outputs[0].get("data", [])
            if prediction_data and len(prediction_data) > 0:
                # 첫 번째 출력 데이터 반환
                response_data = prediction_data[0]

                # JSON 문자열인 경우 파싱
                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except Exception:
                        pass

                # response_data가 dict이고 predictions와 image_info를 포함하는 경우
                if isinstance(response_data, dict):
                    predictions = response_data.get("predictions", response_data)
                    image_info = response_data.get("image_info", {})

                    logger.info(f"Parsed predictions with image_info: {image_info}")

                    return {
                        "workflow_id": workflow_id,
                        "component_id": component_id,
                        "predictions": predictions,
                        "image_info": image_info,
                        "model_info": model_info,
                    }
                else:
                    # 하위 호환성: response_data가 dict가 아닌 경우
                    return {
                        "workflow_id": workflow_id,
                        "component_id": component_id,
                        "predictions": response_data,
                        "model_info": model_info,
                    }

        # 예상치 못한 응답 형식
        return {
            "workflow_id": workflow_id,
            "component_id": component_id,
            "raw_response": result,
            "model_info": model_info,
        }

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        logger.error(f"Response content: {http_err.response.text if hasattr(http_err, 'response') else 'N/A'}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model service returned error: {str(http_err)}"
        )
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error: {conn_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to model service. Service may not be ready.",
        )
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Model inference request timed out")
    except Exception as e:
        logger.error(f"Unexpected error during inference: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(e)}")


@router.get("/{workflow_id}/models")
def get_deployed_models(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우에 배포된 모델 목록 조회

    DB의 kserve_deployments 테이블에서 배포된 모델 정보를 조회합니다.
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # DB에서 배포된 모델 목록 조회
    deployed_models = KServeDeploymentService.get_deployed_models(db, workflow_id, include_component_info=True)

    return {
        "workflow_id": workflow_id,
        "backend_api_url": workflow.backend_api_url,
        "deployed_models": deployed_models,
        "total": len(deployed_models),
    }


@router.get("/{workflow_id}/models/{component_id}/status")
def check_model_status(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    component_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    배포된 모델의 상태 확인 (KServe V2 Protocol)
    """
    import requests  # noqa: F401, F811

    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # 배포된 모델 정보 확인
    deployed_models = workflow.workflow_definition.get("deployed_models", []) if workflow.workflow_definition else []

    model_info = None
    for model in deployed_models:
        if model.get("component_id") == component_id:
            model_info = model
            break

    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model component {component_id} not found in workflow {workflow_id}",
        )

    # KServe 서비스 접근 설정
    namespace = settings.KUBEFLOW_NAMESPACE or "kubeflow-user-example-com"
    service_name = f"workflow-{workflow_id}-{component_id}"

    # Istio Gateway URL
    infer_svc_url = settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80"

    # Virtual Service hostname
    service_hostname = f"{service_name}.{namespace}.example.com"

    # 모델 이름 조회
    model_name = model_info.get("model_name", component_id)
    if model_info.get("model_id"):
        model = db.query(Model).filter(Model.id == model_info["model_id"]).first()
        if model:
            model_name = model.name

    # V2 프로토콜 ready 엔드포인트
    url = f"{infer_svc_url}/v2/models/{model_name}/ready"

    # 헤더 설정 (Istio routing용)
    headers = {"Host": service_hostname}

    # Kubeflow 인증
    kf_manager = KubeflowManager()
    cookies = kf_manager.auth_session.session_cookie_dict if hasattr(kf_manager, "auth_session") else {}

    try:
        logger.info(f"Checking model status at {url}")

        response = requests.get(url, headers=headers, cookies=cookies, timeout=10)

        if response.status_code == 200:
            result = response.json()
            return {
                "workflow_id": workflow_id,
                "component_id": component_id,
                "ready": result.get("ready", False),
                "model_info": model_info,
                "service_name": service_name,
                "status": "ready" if result.get("ready", False) else "not_ready",
            }
        else:
            return {
                "workflow_id": workflow_id,
                "component_id": component_id,
                "ready": False,
                "model_info": model_info,
                "service_name": service_name,
                "status": "not_ready",
                "error": f"Service returned status code: {response.status_code}",
            }

    except requests.exceptions.ConnectionError:
        return {
            "workflow_id": workflow_id,
            "component_id": component_id,
            "ready": False,
            "model_info": model_info,
            "service_name": service_name,
            "status": "not_deployed",
            "error": "Unable to connect to model service",
        }
    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return {
            "workflow_id": workflow_id,
            "component_id": component_id,
            "ready": False,
            "model_info": model_info,
            "service_name": service_name,
            "status": "error",
            "error": str(e),
        }


@router.post("/{workflow_id}/cleanup", status_code=status.HTTP_202_ACCEPTED)
async def cleanup_workflow_resources(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 리소스 정리 시작

    Kubeflow Pipeline을 통해 배포된 KServe InferenceService들을 삭제하고 run_id를 반환합니다.
    워크플로우 자체는 유지하되, 배포된 서비스만 정리

    완료 확인은 /workflows/{workflow_id}/finalize-cleanup API를 사용하세요.

    Returns:
        202 Accepted: cleanup_run_id를 포함한 응답
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    try:
        executor = WorkflowExecutor(db)
        cleanup_result = executor.cleanup_deployed_services(str(workflow_id))
        cleanup_run_id = cleanup_result.get("cleanup_run_id")

        logger.info(f"Cleanup pipeline started for workflow {workflow_id}: run_id={cleanup_run_id}")

        return {
            "message": "Cleanup started",
            "workflow_id": workflow_id,
            "cleanup_run_id": cleanup_run_id,
            "status": "cleanup_in_progress",
            "next_step": f"Call /workflows/{workflow_id}/finalize-cleanup?run_id={cleanup_run_id} to check completion",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup workflow resources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to cleanup resources: {str(e)}"
        )


@router.post("/{workflow_id}/finalize-cleanup")
async def finalize_cleanup(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    run_id: str = Query(..., description="Kubeflow Pipeline cleanup run ID"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 정리 완료 처리

    Kubeflow Pipeline cleanup이 완료되었는지 확인하고, 완료되었다면 워크플로우 상태를 업데이트합니다.

    Args:
        workflow_id: 워크플로우 ID
        run_id: Cleanup pipeline run ID

    Returns:
        - status: "completed" (완료), "in_progress" (진행중), "failed" (실패)
        - workflow_updated: 워크플로우 상태 업데이트 여부
    """
    try:
        # 워크플로우 존재 여부 확인
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

        # Pipeline 완료 확인
        success = await _wait_for_pipeline_completion(run_id, max_wait_seconds=5)  # 짧은 timeout으로 즉시 확인

        if success:
            # Pipeline 완료됨 - 워크플로우 상태 업데이트
            logger.info(f"Cleanup pipeline completed for workflow {workflow_id}, updating workflow state")

            # 워크플로우 상태를 DRAFT로 변경 (재실행 가능하도록)
            if workflow.status == WorkflowStatus.ERROR:
                workflow.status = WorkflowStatus.DRAFT
                db.commit()

            # 배포된 모델 정보 초기화
            if workflow.workflow_definition:
                workflow.workflow_definition["deployed_models"] = []
                db.commit()

            return {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "status": "completed",
                "workflow_updated": True,
                "message": "Cleanup completed and workflow state updated",
            }
        else:
            # 아직 진행중이거나 실패
            # 실제 상태 확인
            try:
                from ..core.kubeflow.kubeflow_manager import KubeflowManager

                kf_manager = KubeflowManager()
                run = kf_manager.kfp_client.get_run(run_id)

                # 상태 추출
                status_value = None
                if hasattr(run, "state"):
                    status_value = run.state
                elif hasattr(run, "status"):
                    status_value = run.status
                elif hasattr(run, "run"):
                    if hasattr(run.run, "status"):
                        status_value = run.run.status
                    elif hasattr(run.run, "state"):
                        status_value = run.run.state

                if status_value:
                    status_upper = str(status_value).upper()
                    if status_upper in ["FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"]:
                        return {
                            "workflow_id": workflow_id,
                            "run_id": run_id,
                            "status": "failed",
                            "workflow_updated": False,
                            "message": f"Cleanup pipeline failed with status: {status_value}",
                        }

                # 진행중
                return {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "status": "in_progress",
                    "workflow_updated": False,
                    "message": "Cleanup pipeline still in progress",
                }

            except Exception as e:
                logger.error(f"Failed to check pipeline status: {e}")
                return {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "status": "unknown",
                    "workflow_updated": False,
                    "message": f"Failed to check pipeline status: {str(e)}",
                }

    except Exception as e:
        logger.error(f"Failed to finalize cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to finalize cleanup: {str(e)}"
        )
