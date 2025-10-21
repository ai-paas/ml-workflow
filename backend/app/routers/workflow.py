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


@router.post("/{workflow_id}/sync-deployments")
async def sync_workflow_deployments(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우의 Kubeflow Pipeline 상태를 확인하고 KServe 배포 정보를 동기화합니다.

    1. Kubeflow Pipeline 실행 상태 확인
    2. Pipeline이 완료되었으면 KServe InferenceService 조회
    3. DB에 배포 정보 업데이트
    """
    from core.kubeflow.kubeflow_manager import KubeflowManager
    from kserve import KServeClient

    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if not workflow.kubeflow_run_id:
        raise HTTPException(status_code=400, detail="Workflow has no Kubeflow run ID")

    try:
        # 1. Kubeflow Pipeline 상태 확인
        kf_manager = KubeflowManager()
        run = kf_manager.kfp_client.get_run(workflow.kubeflow_run_id)

        pipeline_status = run.state if hasattr(run, "state") else run.run.state if hasattr(run, "run") else "UNKNOWN"

        logger.info(f"Pipeline {workflow.kubeflow_run_id} status: {pipeline_status}")

        # 2. Pipeline이 완료되었으면 KServe 서비스 조회
        if pipeline_status in ["SUCCEEDED", "COMPLETED"]:
            kserve_client = KServeClient()
            namespace = settings.KUBEFLOW_NAMESPACE or "kubeflow-user-example-com"

            synced_count = 0
            failed_count = 0

            # 워크플로우의 모든 MODEL 컴포넌트 처리
            for component in workflow.components:
                if component.type != ComponentType.MODEL:
                    continue

                try:
                    # KServe InferenceService 조회 (workflow-id 레이블로 필터링)
                    services = kserve_client.get(namespace=namespace)

                    # 해당 컴포넌트의 서비스 찾기
                    matching_service = None
                    if hasattr(services, "items"):
                        for svc in services.get("items", []):
                            labels = svc.get("metadata", {}).get("labels", {})
                            if (
                                labels.get("workflow-id") == workflow_id
                                and labels.get("component-id") == component.component_id
                            ):
                                matching_service = svc
                                break

                    if matching_service:
                        # 서비스 정보 추출
                        metadata = matching_service.get("metadata", {})
                        status_info = matching_service.get("status", {})

                        service_name = metadata.get("name", "")

                        # URL 정보 추출
                        internal_url = status_info.get("address", {}).get("url", "")

                        # Hostname 생성
                        service_hostname = f"{service_name}.{namespace}.example.com"

                        # 모델 이름 정제
                        model = db.query(Model).filter(Model.id == component.model_id).first()
                        model_name = model.name.replace("/", "-") if model else component.name

                        # Ready 상태 확인
                        conditions = status_info.get("conditions", [])
                        is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

                        deployment_status = "deployed" if is_ready else "deploying"

                        # DB 업데이트
                        KServeDeploymentService.update_deployment_status(
                            db=db,
                            workflow_id=workflow_id,
                            component_id=component.component_id,
                            service_name=service_name,
                            service_hostname=service_hostname,
                            model_name=model_name,
                            status=deployment_status,
                            internal_url=internal_url or f"http://{service_name}.{namespace}.svc.cluster.local",
                            error_message=None,
                        )

                        synced_count += 1
                        logger.info(f"Synced deployment info for component {component.component_id}: {service_name}")

                    else:
                        # 서비스를 찾지 못한 경우
                        logger.warning(f"No KServe service found for component {component.component_id}")
                        failed_count += 1

                except Exception as e:
                    logger.error(f"Failed to sync component {component.component_id}: {str(e)}")
                    failed_count += 1

            return {
                "workflow_id": workflow_id,
                "pipeline_status": pipeline_status,
                "synced_count": synced_count,
                "failed_count": failed_count,
                "message": f"Synced {synced_count} deployments, {failed_count} failed",
            }

        else:
            return {
                "workflow_id": workflow_id,
                "pipeline_status": pipeline_status,
                "message": f"Pipeline is still running or not completed yet (status: {pipeline_status})",
            }

    except Exception as e:
        logger.error(f"Failed to sync workflow deployments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/inference")
async def inference_workflow_model(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    component_id: str,
    image: UploadFile = File(...),
    labels: List[str] = Query(...),
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
        from db.models.model import Model

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
    payload = {"image": image_base64, "text": labels}  # 레이블 리스트

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
        logger.info(f"Labels: {labels}")

        response = requests.post(url, json=data, headers=headers, cookies=cookies, timeout=30)
        response.raise_for_status()

        result = response.json()

        # V2 프로토콜 응답 파싱
        outputs = result.get("outputs", [])
        if outputs and len(outputs) > 0:
            prediction_data = outputs[0].get("data", [])
            if prediction_data and len(prediction_data) > 0:
                # 첫 번째 출력 데이터 반환
                predictions = prediction_data[0]

                # JSON 문자열인 경우 파싱
                if isinstance(predictions, str):
                    try:
                        predictions = json.loads(predictions)
                    except Exception:
                        pass

                return {
                    "workflow_id": workflow_id,
                    "component_id": component_id,
                    "predictions": predictions,
                    "model_info": model_info,
                    "labels": labels,
                }

        # 예상치 못한 응답 형식
        return {
            "workflow_id": workflow_id,
            "component_id": component_id,
            "raw_response": result,
            "model_info": model_info,
            "labels": labels,
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
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우에 배포된 모델 목록 조회
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # Service를 통한 배포된 모델 목록 조회
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
        from db.models.model import Model

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

        # 배포된 모델 정보 초기화
        if workflow.workflow_definition:
            workflow.workflow_definition["deployed_models"] = []
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
