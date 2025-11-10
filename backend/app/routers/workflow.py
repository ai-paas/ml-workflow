"""Workflow API 라우터"""

import json
import logging
import math
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
    WorkflowTemplateBriefSchema,
    WorkflowTemplateCreateRequest,
    WorkflowTemplateListSchema,
    WorkflowTemplateReadSchema,
    WorkflowTemplateUpdateRequest,
    WorkflowUpdateRequest,
)
from services.kserve_deployment import KServeDeploymentService
from services.workflow import WorkflowService
from sqlalchemy import func
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ============= Component Types =============


@router.get("/component-types", response_model=List[ComponentTypeInfo])
def get_component_types():
    """
    사용 가능한 컴포넌트 타입 조회

    워크플로우 구성에 사용할 수 있는 컴포넌트 타입 목록을 조회합니다.
    각 타입별로 고유한 component_id와 설명을 제공하여 워크플로우 정의 시 활용할 수 있습니다.

    ## Response (List[ComponentTypeInfo])
    각 항목은 다음 필드를 포함:
    - **type** (str): 컴포넌트 타입
        - "START": 워크플로우 시작점
        - "END": 워크플로우 종료점
        - "MODEL": ML 모델 실행 노드
    - **component_id** (str): 컴포넌트 식별자
        - 워크플로우 정의 시 사용할 고유 ID
        - 일반적으로 type과 동일 (예: "START", "END", "MODEL")
    - **name** (str): 타입 표시명 (한글)
        - "시작 노드", "종료 노드", "모델 노드" 등
    - **description** (str): 타입 설명
        - 각 컴포넌트 타입의 역할과 용도 설명

    ## Usage Example
    1. 이 API로 사용 가능한 컴포넌트 타입 확인
    2. workflow_definition 작성 시 component_id 사용
    3. 각 컴포넌트 타입에 맞는 설정 적용

    ## Notes
    - 고정된 타입 목록 반환 (동적 변경 없음)
    - 워크플로우는 반드시 START로 시작하고 END로 종료
    - MODEL 타입은 model_id 필수
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


@router.post("", response_model=WorkflowBaseSchema, status_code=status.HTTP_201_CREATED)
def create_workflow(
    *,
    db: Session = SessionDepends,
    workflow_data: WorkflowCreateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    새로운 워크플로우 생성 (직접 생성)

    워크플로우를 직접 정의하여 생성합니다.
    템플릿으로부터 생성하려면 `/workflows/templates/{template_id}/clone` API를 사용하세요.
    생성된 워크플로우는 DRAFT 상태로 시작하며, execute API를 통해 실행할 수 있습니다.

    ## Request Body (WorkflowCreateRequest)
    - **name** (str, required): 워크플로우 이름
    - **description** (str, optional): 워크플로우 설명
    - **category** (str, optional): 카테고리 (분류용)
    - **service_id** (str, optional): 연결할 서비스 ID
    - **workflow_definition** (WorkflowDefinition, optional): 워크플로우 정의
        - components (List[ComponentCreateRequest]): 컴포넌트 목록
            - name (str): 컴포넌트 이름
            - type (ComponentType): 타입 (START/END/MODEL)
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
        - connections (List[ConnectionCreateRequest]): 연결 목록
            - source_component_type (ComponentType): 소스 컴포넌트 타입
            - target_component_type (ComponentType): 타겟 컴포넌트 타입

    ## Response (WorkflowBaseSchema)
    - **id** (str): 워크플로우 UUID
    - **name** (str): 워크플로우 이름
    - **description** (str): 워크플로우 설명
    - **category** (str): 워크플로우 카테고리
    - **status** (str): 워크플로우 상태
        - "DRAFT": 임시저장 상태 (아직 실행되지 않음)
        - "ACTIVE": 활성 상태 (배포 완료, 실행 가능)
        - "ERROR": 오류 발생 상태 (실행 실패 또는 배포 오류)
    - **service_id** (str): 연결된 서비스 ID
        - 모니터링 및 서비스 관리용 서비스 ID
        - null 가능 (서비스 연결 없이도 워크플로우 생성 가능)
    - **creator_id** (int): 생성자 ID
        - 워크플로우를 생성한 사용자의 ID
    - **is_template** (bool): 템플릿 여부
        - false: 일반 워크플로우
        - true: 템플릿 (템플릿 조회 API 사용 권장)
    - **template_id** (str): 원본 템플릿 ID
        - 직접 생성한 경우 항상 null
        - 템플릿으로부터 생성된 경우 `/workflows/templates/{template_id}/clone` API 사용
    - **created_at** (datetime): 워크플로우 생성 시각
    - **updated_at** (datetime): 워크플로우 수정 시각

    ## Notes
    - 템플릿으로부터 생성하려면 `/workflows/templates/{template_id}/clone` API 사용
    - MODEL 컴포넌트는 유효한 model_id 필요
    - 생성 직후 상태는 DRAFT
    - is_template은 항상 false로 설정됨 (템플릿 생성은 /workflows/templates API 사용)
    - 상세 정보(components, connections, creator 등)는 GET /workflows/{workflow_id}로 조회 가능

    ## Errors
    - 400: 잘못된 요청 (정의 오류 등)
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    try:
        # create_workflow는 항상 is_template=False로 생성됨
        workflow = WorkflowService.create_workflow(db=db, workflow_data=workflow_data, creator_id=current_user.id)

        # 기본 스키마로 변환
        return WorkflowBaseSchema.model_validate(workflow)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create workflow")


@router.get("", response_model=WorkflowListSchema)
def list_workflows(
    *,
    db: Session = SessionDepends,
    page_size: Optional[int] = Query(
        default=None,
        description="페이지 사이즈",
        examples=[10, 20, 30],
        ge=1,
        le=1000,
    ),
    page: Optional[int] = Query(
        default=None,
        description="페이지 번호",
        examples=[1, 2, 3],
        ge=1,
    ),
    creator_id: Optional[int] = Query(None),
    service_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 목록 조회 (템플릿 제외)

    생성된 워크플로우 목록을 조회합니다. 템플릿은 포함되지 않으며,
    페이지네이션과 다양한 필터 옵션을 제공합니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
    - **page_size** (int, optional): 페이지당 항목 수 (1-1000)
        - 페이지 파라미터 생략 시 전체 데이터 반환 (최대 10000개)
    - **creator_id** (int, optional): 특정 사용자가 생성한 워크플로우만 필터
    - **service_id** (int, optional): 특정 서비스에 연결된 워크플로우만 필터
    - **status** (str, optional): 워크플로우 상태 필터
        - "DRAFT": 임시저장 상태
        - "ACTIVE": 활성 상태 (배포됨)
        - "ERROR": 오류 상태

    ## Response (WorkflowListSchema)
    - **total** (int): 필터 조건에 맞는 전체 워크플로우 수
    - **items** (List[WorkflowBaseSchema]): 워크플로우 목록
        - id (str): 워크플로우 UUID
        - name (str): 워크플로우 이름
        - description (str): 설명
        - category (str): 카테고리
        - status (str): 상태 (DRAFT/ACTIVE/ERROR)
        - service_id (str): 연결된 서비스 ID
        - creator_id (int): 생성자 ID
        - is_template (bool): 템플릿 여부 (항상 false)
        - template_id (str): 원본 템플릿 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각

    ## Notes
    - 템플릿을 조회하려면 /workflows/templates API 사용
    - 페이지네이션 생략 시 최대 10000개까지 반환

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    from db.models.service import Workflow, WorkflowStatus

    # 전체 개수 조회를 위한 쿼리 구성 (템플릿 제외)
    count_query = db.query(func.count(Workflow.id)).filter(Workflow.is_template == False)
    if creator_id is not None:
        count_query = count_query.filter(Workflow.creator_id == creator_id)
    if service_id is not None:
        count_query = count_query.filter(Workflow.service_id == service_id)
    if status is not None:
        try:
            status_enum = WorkflowStatus(status)
            count_query = count_query.filter(Workflow.status == status_enum)
        except ValueError:
            pass

    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        workflows = WorkflowService.get_workflows(
            db=db,
            skip=0,
            limit=10000,
            creator_id=creator_id,
            service_id=service_id,
            is_template=False,
            status=status,
        )
        items = [WorkflowBaseSchema.model_validate(w) for w in workflows]
        return WorkflowListSchema(total=len(items), items=items)

    # 페이지네이션 적용
    total_count = count_query.scalar()
    skip = page_size * (page - 1)

    workflows = WorkflowService.get_workflows(
        db=db,
        skip=skip,
        limit=page_size,
        creator_id=creator_id,
        service_id=service_id,
        is_template=False,
        status=status,
    )

    items = [WorkflowBaseSchema.model_validate(w) for w in workflows]

    return WorkflowListSchema(total=total_count, items=items)


# ============= Template Management =============
# NOTE: 템플릿 라우트는 /{workflow_id} 보다 먼저 정의되어야 합니다.
# FastAPI는 위에서 아래로 순서대로 라우트를 매칭하므로,
# /templates가 {workflow_id}로 잘못 매칭되는 것을 방지합니다.


@router.post("/templates", response_model=WorkflowTemplateBriefSchema, status_code=status.HTTP_201_CREATED)
def create_workflow_template(
    *,
    db: Session = SessionDepends,
    template_data: WorkflowTemplateCreateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 템플릿 생성

    재사용 가능한 워크플로우 템플릿을 생성합니다.
    템플릿은 다른 사용자들이 복사하여 사용할 수 있는 기본 워크플로우 구조입니다.

    ## Request Body (WorkflowTemplateCreateRequest)
    - **name** (str, required): 템플릿 이름
    - **description** (str, optional): 템플릿 설명
    - **category** (str, optional): 템플릿 카테고리
    - **workflow_definition** (WorkflowDefinition, required): 템플릿 구조
        - components (List[ComponentCreateRequest]): 컴포넌트 정의
            - name (str): 컴포넌트 이름
            - type (str): 타입 (START/END/MODEL)
            - model_id (int): MODEL 타입의 모델 ID
        - connections (List[ConnectionCreateRequest]): 연결 정의
            - source_component_id (str): 소스 컴포넌트 ID
            - target_component_id (str): 타겟 컴포넌트 ID

    ## Response (WorkflowTemplateBriefSchema)
    - **id** (str): 템플릿 UUID
    - **name** (str): 템플릿 이름
    - **description** (str): 템플릿 설명
    - **category** (str): 템플릿 카테고리
    - **status** (str): 템플릿 상태 (DRAFT)
    - **service_id** (str): 기본 서비스 ID
    - **creator_id** (int): 템플릿 생성자 ID
    - **creator** (UserSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **is_template** (bool): 템플릿 여부 (항상 true)
    - **template_id** (str): 원본 템플릿 ID (null)
    - **usage_count** (int): 템플릿 사용 횟수
        - 템플릿을 복사하여 생성된 워크플로우의 총 개수
        - 동적으로 계산됨 (실시간 반영)
        - 생성 직후는 0
    - **created_at** (datetime): 생성 시각
    - **updated_at** (datetime): 수정 시각

    ## Notes
    - 템플릿은 실행할 수 없고 복사용만 가능
    - 모든 사용자가 템플릿을 볼 수 있음
    - is_template은 항상 true로 설정됨
    - service_id는 항상 null로 설정됨 (템플릿은 서비스에 연결되지 않음)
    - usage_count는 0으로 시작
    - 상세 정보(components, connections 등)는 GET /workflows/templates/{template_id}로 조회 가능

    ## Errors
    - 400: 잘못된 워크플로우 정의
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    try:
        # is_template은 WorkflowService.create_workflow_template 내부에서 true로 설정됨
        template = WorkflowService.create_workflow_template(
            db=db, template_data=template_data, creator_id=current_user.id
        )

        result = WorkflowTemplateBriefSchema.model_validate(template)

        # 사용 횟수 계산
        usage_count = db.query(Workflow).filter(Workflow.template_id == template.id).count()
        result.usage_count = usage_count

        return result

    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create template")


@router.get("/templates", response_model=WorkflowTemplateListSchema)
def list_workflow_templates(
    *,
    db: Session = SessionDepends,
    page_size: Optional[int] = Query(
        default=None,
        description="페이지 사이즈",
        examples=[10, 20, 30],
        ge=1,
        le=1000,
    ),
    page: Optional[int] = Query(
        default=None,
        description="페이지 번호",
        examples=[1, 2, 3],
        ge=1,
    ),
    category: Optional[str] = Query(None),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 템플릿 목록 조회

    사용 가능한 모든 워크플로우 템플릿을 조회합니다.
    템플릿은 모든 사용자가 확인하고 복사하여 사용할 수 있습니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
    - **page_size** (int, optional): 페이지당 항목 수 (1-1000)
        - 페이지 파라미터 생략 시 전체 데이터 반환
    - **category** (str, optional): 템플릿 카테고리 필터
        - 특정 카테고리의 템플릿만 필터링

    ## Response (WorkflowTemplateListSchema)
    - **total** (int): 필터 조건에 맞는 전체 템플릿 수
    - **items** (List[WorkflowTemplateBriefSchema]): 템플릿 목록
        - id (str): 템플릿 UUID
        - name (str): 템플릿 이름
        - description (str): 템플릿 설명
        - category (str): 템플릿 카테고리
        - status (str): 템플릿 상태 (DRAFT)
        - service_id (str): 기본 서비스 ID
        - creator_id (int): 템플릿 생성자 ID
        - creator (UserSchema): 생성자 정보
            - id (int): 사용자 ID
            - username (str): 사용자명
            - name (str): 사용자 이름
            - password (str): 비밀번호 (해시된 값)
            - created_at (datetime): 계정 생성 시각
            - updated_at (datetime): 계정 정보 수정 시각
            - created_by (str, optional): 계정 생성자
            - updated_by (str, optional): 계정 정보 수정자
        - is_template (bool): 템플릿 여부 (항상 true)
        - template_id (str): 원본 템플릿 ID (null)
        - usage_count (int): 해당 템플릿으로 생성된 워크플로우 수
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각

    ## Notes
    - 모든 사용자의 템플릿이 표시됨 (creator_id 필터 없음)
    - usage_count는 동적으로 계산됨
    - 페이지네이션 생략 시 최대 10000개까지 반환

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    from db.models.service import Workflow

    # 전체 개수 조회를 위한 쿼리 구성
    count_query = db.query(func.count(Workflow.id)).filter(Workflow.is_template == True)
    if category:
        count_query = count_query.filter(Workflow.category == category)

    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        templates = WorkflowService.get_workflow_templates(
            db=db, skip=0, limit=10000, creator_id=None, category=category  # 모든 사용자의 템플릿 조회 가능
        )
        results = []
        for template in templates:
            result = WorkflowTemplateBriefSchema.model_validate(template)
            # 사용 횟수 계산
            usage_count = db.query(Workflow).filter(Workflow.template_id == template.id).count()
            result.usage_count = usage_count
            results.append(result)
        return WorkflowTemplateListSchema(total=len(results), items=results)

    # 페이지네이션 적용
    total_count = count_query.scalar()
    skip = page_size * (page - 1)

    templates = WorkflowService.get_workflow_templates(
        db=db, skip=skip, limit=page_size, creator_id=None, category=category  # 모든 사용자의 템플릿 조회 가능
    )

    results = []
    for template in templates:
        result = WorkflowTemplateBriefSchema.model_validate(template)

        # 사용 횟수 계산
        usage_count = db.query(Workflow).filter(Workflow.template_id == template.id).count()
        result.usage_count = usage_count

        results.append(result)

    return WorkflowTemplateListSchema(total=total_count, items=results)


@router.get("/templates/{template_id}", response_model=WorkflowTemplateReadSchema)
def get_workflow_template(
    *,
    db: Session = SessionDepends,
    template_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 템플릿 상세 조회

    특정 템플릿의 상세 정보를 조회합니다.
    템플릿의 전체 구조와 컴포넌트, 연결 정보를 포함합니다.
    템플릿은 다른 사용자들이 복사하여 사용할 수 있는 재사용 가능한 워크플로우 구조입니다.

    ## Path Parameters
    - **template_id** (str): 조회할 템플릿 UUID
        - 템플릿 목록 조회 API(/workflows/templates)에서 확인 가능

    ## Response (WorkflowTemplateReadSchema)
    - **id** (str): 템플릿 UUID
    - **name** (str): 템플릿 이름
    - **description** (str): 템플릿 설명
        - 템플릿의 용도와 사용 방법에 대한 설명
    - **category** (str): 템플릿 카테고리
        - 템플릿 분류를 위한 카테고리 (예: "Object Detection", "Classification")
    - **status** (str): 템플릿 상태
        - "DRAFT": 템플릿은 항상 DRAFT 상태 (실행 불가)
    - **service_id** (str): 기본 서비스 ID
        - 템플릿으로부터 워크플로우 생성 시 기본으로 연결될 서비스 ID
        - null 가능 (서비스 연결 없이 생성 가능)
    - **creator_id** (int): 템플릿 생성자 ID
    - **creator** (UserSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **is_template** (bool): 템플릿 여부
        - 항상 true (템플릿 조회 API이므로)
    - **components** (List[ComponentReadSchema]): 컴포넌트 상세 정보
        - id (str): 컴포넌트 UUID (workflow_component 테이블의 PK)
        - workflow_id (str): 소속 워크플로우 ID (템플릿 ID)
        - component_id (str): 컴포넌트 식별자
            - 워크플로우 내에서 고유한 식별자 (예: "START", "END", "MODEL-1")
        - name (str): 컴포넌트 이름
            - 사용자가 지정한 컴포넌트 표시명
        - type (ComponentType): 컴포넌트 타입
            - "START": 워크플로우 시작점
            - "END": 워크플로우 종료점
            - "MODEL": ML 모델 실행 노드
        - model_id (int, optional): 연결된 모델 ID
            - MODEL 타입인 경우 필수, 다른 타입은 null
        - model (ModelBriefReadSchema, optional): 모델 상세 정보
            - MODEL 타입인 경우에만 포함
            - id (int): 모델 ID
            - name (str): 모델 이름
            - description (str): 모델 설명
            - provider_info (ModelProviderReadSchema): 모델 제공자 정보
                - id (int): 제공자 ID
                - name (str): 제공자 이름
                - description (str): 제공자 설명
            - type_info (ModelTypeReadSchema): 모델 타입 정보
                - id (int): 타입 ID
                - name (str): 타입 이름
                - description (str): 타입 설명
            - format_info (ModelFormatReadSchema): 모델 포맷 정보
                - id (int): 포맷 ID
                - name (str): 포맷 이름
                - description (str): 포맷 설명
            - parent_model_id (int, optional): 부모 모델 ID
                - 파인튜닝된 모델인 경우 원본 모델 ID
            - registry (ModelRegistryReadSchema): 모델 레지스트리 정보
                - id (int): 레지스트리 ID
                - artifact_path (str): 아티팩트 경로
                - uri (str): 모델 URI
                - run_id (str, optional): MLflow 실행 ID
                - reference_model_id (int): 참조 모델 ID
                - created_at (datetime): 생성 시각
                - updated_at (datetime): 수정 시각
            - created_at (datetime): 모델 생성 시각
            - updated_at (datetime): 모델 수정 시각
        - created_at (datetime): 컴포넌트 생성 시각
        - updated_at (datetime): 컴포넌트 수정 시각
    - **component_connections** (List[ConnectionReadSchema]): 연결 정보
        - id (str): 연결 UUID (workflow_component_connection 테이블의 PK)
        - workflow_id (str): 소속 워크플로우 ID (템플릿 ID)
        - source_component_id (str): 소스 컴포넌트 ID
            - workflow_component 테이블의 PK (출발점 컴포넌트)
        - target_component_id (str): 타겟 컴포넌트 ID
            - workflow_component 테이블의 PK (도착점 컴포넌트)
        - source_component (ComponentReadSchema): 소스 컴포넌트 상세 정보
            - 위의 ComponentReadSchema 구조와 동일한 전체 정보 포함
        - target_component (ComponentReadSchema): 타겟 컴포넌트 상세 정보
            - 위의 ComponentReadSchema 구조와 동일한 전체 정보 포함
        - created_at (datetime): 연결 생성 시각
    - **usage_count** (int): 해당 템플릿으로 생성된 워크플로우 수
        - 템플릿을 복사하여 생성된 워크플로우의 총 개수
        - 동적으로 계산됨 (실시간 반영)
    - **created_at** (datetime): 템플릿 생성 시각
    - **updated_at** (datetime): 템플릿 수정 시각

    ## Notes
    - 템플릿은 실행할 수 없고 복사용으로만 사용 가능
    - 모든 사용자가 템플릿을 조회할 수 있음 (공개)
    - usage_count는 템플릿 복사 시 자동 증가
    - 템플릿으로부터 워크플로우 생성 시 /workflows/templates/{template_id}/clone API 사용

    ## Usage Example
    1. 템플릿 목록에서 원하는 템플릿 ID 확인
    2. 이 API로 템플릿 상세 정보 조회
    3. 템플릿 구조 확인 후 clone API로 워크플로우 생성

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 템플릿을 찾을 수 없음
        - template_id가 존재하지 않거나 삭제된 경우
    - 500: 서버 내부 오류
    """
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
    """
    템플릿으로부터 워크플로우 생성

    기존 템플릿을 복사하여 새로운 워크플로우를 생성합니다.
    템플릿의 모든 구조가 복사되며, 생성된 워크플로우는 즉시 실행 가능합니다.

    ## Path Parameters
    - **template_id** (str): 복사할 템플릿 UUID

    ## Query Parameters
    - **workflow_name** (str, required): 새로 생성할 워크플로우 이름
    - **service_id** (int, optional): 연결할 서비스 ID
        - 서비스와 연결시 모니터링 가능

    ## Response (WorkflowReadSchema)
    - **id** (str): 생성된 워크플로우 UUID
    - **name** (str): 워크플로우 이름
    - **description** (str): 워크플로우 설명 (템플릿에서 복사)
    - **category** (str): 카테고리 (템플릿에서 복사)
    - **status** (str): 상태 (DRAFT로 시작)
    - **service_id** (str): 연결된 서비스 ID
    - **service_name** (str): 연결된 서비스 이름
    - **creator_id** (int): 생성자 ID (현재 사용자)
    - **creator** (UserSchema): 생성자 정보 (현재 사용자)
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **is_template** (bool): 템플릿 여부 (false)
    - **template_id** (str): 원본 템플릿 ID
    - **template_name** (str): 원본 템플릿 이름
    - **kubeflow_run_id** (str): Kubeflow 실행 ID (null)
    - **components** (List[ComponentReadSchema]): 복사된 컴포넌트
    - **component_connections** (List[ConnectionReadSchema]): 복사된 연결
        - id (str): 연결 UUID
        - workflow_id (str): 소속 워크플로우 ID
        - source_component_id (str): 소스 컴포넌트 ID
        - target_component_id (str): 타겟 컴포넌트 ID
        - source_component (ComponentReadSchema): 소스 컴포넌트 상세
        - target_component (ComponentReadSchema): 타겟 컴포넌트 상세
        - created_at (datetime): 생성 시각
    - **created_at** (datetime): 생성 시각
    - **updated_at** (datetime): 수정 시각

    ## Notes
    - 템플릿의 모든 컴포넌트와 연결이 복사됨
    - 생성된 워크플로우는 템플릿과 독립적으로 동작
    - template_id가 자동으로 기록됨

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 템플릿을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        workflow = WorkflowService.clone_from_template(
            db=db,
            template_id=template_id,
            workflow_name=workflow_name,
            service_id=service_id,
            creator_id=current_user.id,
        )

        # 관계를 다시 로드하여 스키마로 변환
        workflow = WorkflowService.get_workflow_by_id(db, workflow.id)
        return WorkflowReadSchema.model_validate(workflow)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to clone from template: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clone from template")


@router.put("/templates/{template_id}", response_model=WorkflowTemplateReadSchema)
def update_workflow_template(
    *,
    db: Session = SessionDepends,
    template_id: str,
    template_data: WorkflowTemplateUpdateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 템플릿 수정

    기존 워크플로우 템플릿의 정보를 수정합니다.
    workflow_definition이 제공되면 컴포넌트와 연결도 함께 업데이트됩니다.
    템플릿은 서비스에 연결되지 않으므로 service_id는 수정할 수 없습니다.

    ## Path Parameters
    - **template_id** (str): 수정할 템플릿 UUID
        - 템플릿 목록 조회 API(/workflows/templates)에서 확인 가능

    ## Request Body (WorkflowTemplateUpdateRequest)
    - **name** (str, optional): 새 템플릿 이름
    - **description** (str, optional): 새 설명
    - **category** (str, optional): 새 카테고리
    - **status** (str, optional): 새 상태 (DRAFT/ACTIVE/ERROR)
        - 템플릿은 일반적으로 DRAFT 상태 유지 (실행 불가)
    - **workflow_definition** (WorkflowUpdateDefinition, optional): 새 템플릿 구조
        - components (List[ComponentUpdateRequest]): 컴포넌트 목록
            - name (str): 컴포넌트 이름
            - type (ComponentType): 타입 (START/END/MODEL)
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
        - connections (List[ConnectionUpdateRequest]): 연결 목록
            - source_component_type (ComponentType): 소스 컴포넌트 타입
            - target_component_type (ComponentType): 타겟 컴포넌트 타입

    ## Response (WorkflowTemplateReadSchema)
    - **id** (str): 템플릿 UUID
    - **name** (str): 템플릿 이름
    - **description** (str): 템플릿 설명
    - **category** (str): 템플릿 카테고리
    - **status** (str): 템플릿 상태 (DRAFT)
    - **service_id** (str): 기본 서비스 ID (항상 null)
    - **creator_id** (int): 템플릿 생성자 ID
    - **creator** (UserSchema): 생성자 정보
    - **is_template** (bool): 템플릿 여부 (항상 true)
    - **template_id** (str): 원본 템플릿 ID (항상 null)
    - **components** (List[ComponentReadSchema]): 컴포넌트 상세 정보
    - **component_connections** (List[ConnectionReadSchema]): 연결 정보
    - **usage_count** (int): 해당 템플릿으로 생성된 워크플로우 수
        - 템플릿을 복사하여 생성된 워크플로우의 총 개수
        - 동적으로 계산됨 (실시간 반영)
    - **created_at** (datetime): 템플릿 생성 시각
    - **updated_at** (datetime): 템플릿 수정 시각

    ## Notes
    - 제공된 필드만 업데이트됨 (부분 업데이트 가능)
    - workflow_definition 제공 시 기존 컴포넌트/연결은 삭제 후 재생성됨
    - service_id는 템플릿에 포함되지 않음 (요청에서 제외, 항상 null로 유지)
    - 템플릿은 실행할 수 없고 복사용으로만 사용 가능
    - usage_count는 동적으로 계산됨 (파생된 워크플로우 수)
    - 일반 워크플로우 수정은 /workflows/{workflow_id} API 사용

    ## Usage Example
    1. 템플릿 목록에서 수정할 템플릿 ID 확인
    2. 이 API로 템플릿 정보 수정
    3. 수정된 템플릿으로부터 워크플로우 생성 가능

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 템플릿을 찾을 수 없음
        - template_id가 존재하지 않거나 템플릿이 아닌 경우
    - 500: 서버 내부 오류
    """
    # 템플릿인지 확인
    template = WorkflowService.get_workflow_by_id(db, template_id)
    if not template or not template.is_template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

    # WorkflowTemplateUpdateRequest를 WorkflowUpdateRequest로 변환 (service_id는 None으로 설정)
    workflow_update_data = WorkflowUpdateRequest(
        name=template_data.name,
        description=template_data.description,
        category=template_data.category,
        status=template_data.status,
        service_id=None,  # 템플릿은 service_id를 수정할 수 없음
        workflow_definition=template_data.workflow_definition,
    )

    updated_template = WorkflowService.update_workflow(
        db=db, workflow_id=template_id, workflow_data=workflow_update_data
    )

    result = WorkflowTemplateReadSchema.from_orm(updated_template)

    # 사용 횟수 계산
    usage_count = db.query(Workflow).filter(Workflow.template_id == template_id).count()
    result.usage_count = usage_count

    return result


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workflow_template(
    *, db: Session = SessionDepends, template_id: str, current_user: UserSchema = Depends(get_current_user)
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
    """
    워크플로우 상세정보 조회

    특정 워크플로우의 상세 정보를 조회합니다.
    컴포넌트, 연결, 배포 상태 등 모든 정보를 포함합니다.
    워크플로우 실행 상태, 배포된 모델 정보, Kubeflow 파이프라인 실행 정보 등을 확인할 수 있습니다.

    ## Path Parameters
    - **workflow_id** (str): 조회할 워크플로우 UUID
        - 워크플로우 목록 조회 API(/workflows)에서 확인 가능

    ## Response (WorkflowReadSchema)
    - **id** (str): 워크플로우 UUID
    - **name** (str): 워크플로우 이름
    - **description** (str): 워크플로우 설명
        - 워크플로우의 용도와 목적에 대한 설명
    - **category** (str): 워크플로우 카테고리
        - 워크플로우 분류를 위한 카테고리 (예: "Object Detection", "Classification")
    - **status** (str): 워크플로우 상태
        - "DRAFT": 임시저장 상태 (아직 실행되지 않음)
        - "ACTIVE": 활성 상태 (배포 완료, 실행 가능)
        - "ERROR": 오류 발생 상태 (실행 실패 또는 배포 오류)
    - **service_id** (str): 연결된 서비스 ID
        - 모니터링 및 서비스 관리용 서비스 ID
        - null 가능 (서비스 연결 없이도 워크플로우 생성 가능)
    - **service_name** (str): 연결된 서비스 이름
        - service_id로부터 동적으로 조회된 서비스 이름
        - service_id가 null이면 null
    - **creator_id** (int): 생성자 ID
        - 워크플로우를 생성한 사용자의 ID
    - **creator** (UserSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **is_template** (bool): 템플릿 여부
        - false: 일반 워크플로우
        - true: 템플릿 (템플릿 조회 API 사용 권장)
    - **template_id** (str): 원본 템플릿 ID
        - 템플릿으로부터 생성된 경우 원본 템플릿 ID
        - 직접 생성한 경우 null
    - **template_name** (str): 원본 템플릿 이름
        - template_id로부터 동적으로 조회된 템플릿 이름
        - template_id가 null이면 null
    - **kubeflow_run_id** (str): Kubeflow 파이프라인 실행 ID
        - 워크플로우 실행 시 생성된 Kubeflow Pipeline 실행 ID
        - 실행 전이면 null
        - Kubeflow UI에서 파이프라인 실행 상태 확인 가능
    - **components** (List[ComponentReadSchema]): 컴포넌트 목록
        - id (str): 컴포넌트 UUID (workflow_component 테이블의 PK)
        - workflow_id (str): 소속 워크플로우 ID
        - component_id (str): 컴포넌트 식별자
            - 워크플로우 내에서 고유한 식별자 (예: "START", "END", "MODEL-1")
        - name (str): 컴포넌트 이름
            - 사용자가 지정한 컴포넌트 표시명
        - type (ComponentType): 컴포넌트 타입
            - "START": 워크플로우 시작점
            - "END": 워크플로우 종료점
            - "MODEL": ML 모델 실행 노드
        - model_id (int, optional): 모델 ID
            - MODEL 타입인 경우 필수, 다른 타입은 null
        - model (ModelBriefReadSchema, optional): 모델 상세 정보
            - MODEL 타입인 경우에만 포함
            - id (int): 모델 ID
            - name (str): 모델 이름
            - description (str): 모델 설명
            - provider_info (ModelProviderReadSchema): 모델 제공자 정보
                - id (int): 제공자 ID
                - name (str): 제공자 이름
                - description (str): 제공자 설명
            - type_info (ModelTypeReadSchema): 모델 타입 정보
                - id (int): 타입 ID
                - name (str): 타입 이름
                - description (str): 타입 설명
            - format_info (ModelFormatReadSchema): 모델 포맷 정보
                - id (int): 포맷 ID
                - name (str): 포맷 이름
                - description (str): 포맷 설명
            - parent_model_id (int, optional): 부모 모델 ID
                - 파인튜닝된 모델인 경우 원본 모델 ID
            - registry (ModelRegistryReadSchema): 모델 레지스트리 정보
                - id (int): 레지스트리 ID
                - artifact_path (str): 아티팩트 경로
                - uri (str): 모델 URI
                - run_id (str, optional): MLflow 실행 ID
                - reference_model_id (int): 참조 모델 ID
                - created_at (datetime): 생성 시각
                - updated_at (datetime): 수정 시각
            - created_at (datetime): 모델 생성 시각
            - updated_at (datetime): 모델 수정 시각
        - created_at (datetime): 컴포넌트 생성 시각
        - updated_at (datetime): 컴포넌트 수정 시각
    - **component_connections** (List[ConnectionReadSchema]): 연결 정보
        - id (str): 연결 UUID (workflow_component_connection 테이블의 PK)
        - workflow_id (str): 소속 워크플로우 ID
        - source_component_id (str): 소스 컴포넌트 ID
            - workflow_component 테이블의 PK (출발점 컴포넌트)
        - target_component_id (str): 타겟 컴포넌트 ID
            - workflow_component 테이블의 PK (도착점 컴포넌트)
        - source_component (ComponentReadSchema): 소스 컴포넌트 상세 정보
            - 위의 ComponentReadSchema 구조와 동일한 전체 정보 포함
        - target_component (ComponentReadSchema): 타겟 컴포넌트 상세 정보
            - 위의 ComponentReadSchema 구조와 동일한 전체 정보 포함
        - created_at (datetime): 연결 생성 시각
    - **public_url** (str): KServe 공개 엔드포인트 URL
        - 배포 후 동적으로 생성되는 공개 접근 URL
        - 배포 전이면 null
        - 형식: {gateway_url}/v2/models/{model_name}/infer
    - **backend_api_url** (str): 백엔드 API URL
        - 배포 후 동적으로 생성되는 백엔드 API URL
        - 배포 전이면 null
        - 형식: {gateway_url}/v2/models/{model_name}/infer
    - **created_at** (datetime): 워크플로우 생성 시각
    - **updated_at** (datetime): 워크플로우 수정 시각

    ## Notes
    - public_url과 backend_api_url은 워크플로우 실행 후 배포가 완료되면 동적으로 생성됨
    - 템플릿인 경우 is_template=true (템플릿 조회 API 사용 권장)
    - kubeflow_run_id가 있으면 /workflows/{workflow_id}/status로 실행 상태 확인 가능
    - 배포된 모델 정보는 /workflows/{workflow_id}/models로 확인 가능
    - 워크플로우 실행은 /workflows/{workflow_id}/execute API 사용

    ## Usage Example
    1. 워크플로우 목록에서 원하는 워크플로우 ID 확인
    2. 이 API로 워크플로우 상세 정보 조회
    3. 상태가 ACTIVE면 execute API로 실행
    4. 실행 후 status API로 모니터링

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
        - workflow_id가 존재하지 않거나 삭제된 경우
    - 500: 서버 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    return WorkflowReadSchema.model_validate(workflow)


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

    기존 워크플로우의 정보를 수정합니다.
    workflow_definition이 제공되면 컴포넌트와 연결도 함께 업데이트됩니다.

    ## Path Parameters
    - **workflow_id** (str): 수정할 워크플로우 UUID

    ## Request Body (WorkflowUpdateRequest)
    - **name** (str, optional): 새 워크플로우 이름
    - **description** (str, optional): 새 설명
    - **category** (str, optional): 새 카테고리
    - **status** (str, optional): 새 상태 (DRAFT/ACTIVE/ERROR)
        - "DRAFT": 임시저장 상태 (아직 실행되지 않음)
        - "ACTIVE": 활성 상태 (배포 완료, 실행 가능)
        - "ERROR": 오류 발생 상태 (실행 실패 또는 배포 오류)
    - **service_id** (str, optional): 연결할 서비스 ID
        - 모니터링 및 서비스 관리용 서비스 ID
        - null로 설정 시 서비스 연결 해제
    - **workflow_definition** (WorkflowUpdateDefinition, optional): 새 워크플로우 구조
        - components (List[ComponentUpdateRequest]): 컴포넌트 목록
            - name (str): 컴포넌트 이름
            - type (ComponentType): 타입 (START/END/MODEL)
                - "START": 워크플로우 시작점
                - "END": 워크플로우 종료점
                - "MODEL": ML 모델 실행 노드
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
                - MODEL 타입인 경우 필수, 다른 타입은 null
        - connections (List[ConnectionUpdateRequest]): 연결 목록
            - source_component_type (ComponentType): 소스 컴포넌트 타입
            - target_component_type (ComponentType): 타겟 컴포넌트 타입

    ## Response (WorkflowReadSchema)
    - **id** (str): 워크플로우 UUID
    - **name** (str): 워크플로우 이름
    - **description** (str): 워크플로우 설명
    - **category** (str): 워크플로우 카테고리
    - **status** (str): 워크플로우 상태 (DRAFT/ACTIVE/ERROR)
    - **service_id** (str): 연결된 서비스 ID
    - **service_name** (str): 연결된 서비스 이름
    - **creator_id** (int): 생성자 ID
    - **creator** (UserSchema): 생성자 정보
    - **is_template** (bool): 템플릿 여부 (false)
    - **template_id** (str): 원본 템플릿 ID
    - **template_name** (str): 원본 템플릿 이름
    - **kubeflow_run_id** (str): Kubeflow 파이프라인 실행 ID
    - **components** (List[ComponentReadSchema]): 컴포넌트 목록
    - **component_connections** (List[ConnectionReadSchema]): 연결 정보
    - **public_url** (str): KServe 공개 엔드포인트 URL
    - **backend_api_url** (str): 백엔드 API URL
    - **created_at** (datetime): 워크플로우 생성 시각
    - **updated_at** (datetime): 워크플로우 수정 시각

    ## Notes
    - 제공된 필드만 업데이트됨 (부분 업데이트 가능)
    - workflow_definition 제공 시 기존 컴포넌트/연결은 삭제 후 재생성됨
    - status를 ACTIVE로 변경해도 자동 배포되지 않음 (execute API 사용 필요)
    - service_id를 null로 설정하면 서비스 연결이 해제됨
    - 템플릿 수정은 /workflows/templates/{template_id} API 사용
    - 배포된 워크플로우의 구조 변경 시 재배포 필요

    ## Usage Example
    1. 워크플로우 목록에서 수정할 워크플로우 ID 확인
    2. 이 API로 워크플로우 정보 수정
    3. 구조 변경 시 execute API로 재배포

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    workflow = WorkflowService.update_workflow(db=db, workflow_id=workflow_id, workflow_data=workflow_data)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # 관계를 다시 로드하여 스키마로 변환
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
    return WorkflowReadSchema.model_validate(workflow)


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
    워크플로우 삭제 시작 (2단계 프로세스)

    워크플로우 삭제를 시작합니다. KServe InferenceService를 정리하는
    Kubeflow Pipeline을 실행하고 cleanup_run_id를 반환합니다.
    실제 DB 삭제는 finalize-deletion API를 통해 완료 확인 후 수행됩니다.

    ## Path Parameters
    - **workflow_id** (str): 삭제할 워크플로우 UUID

    ## Response (202 Accepted)
    - **message** (str): 상태 메시지 "Workflow deletion started"
    - **workflow_id** (str): 워크플로우 UUID
    - **cleanup_run_id** (str): 정리 파이프라인 실행 ID
    - **status** (str): 현재 상태 "cleanup_in_progress"
    - **next_step** (str): 다음 단계 API 안내
        - 형식: "Call /workflows/{workflow_id}/finalize-deletion?run_id={cleanup_run_id} to complete deletion"

    ## Deletion Process
    1. 현재 API 호출: 정리 파이프라인 시작
    2. Kubeflow Pipeline: KServe InferenceService 삭제
    3. finalize-deletion API: 완료 확인 및 DB 삭제

    ## Notes
    - 비동기 프로세스로 진행됨 (202 Accepted)
    - KServe 리소스 정리에 시간이 걸릴 수 있음
    - 템플릿은 바로 DB에서 삭제됨 (배포 리소스 없음)

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 정리 파이프라인 시작 실패
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

    KServe 리소스 정리가 완료되었는지 확인하고,
    완료된 경우 DB에서 워크플로우를 삭제합니다.

    ## Path Parameters
    - **workflow_id** (str): 삭제할 워크플로우 UUID

    ## Query Parameters
    - **run_id** (str, required): Kubeflow Pipeline cleanup run ID
        - delete API에서 반환된 cleanup_run_id 사용

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **run_id** (str): 정리 파이프라인 실행 ID
    - **status** (str): 삭제 상태
        - "completed": 삭제 완료
        - "in_progress": 아직 진행중
        - "failed": 삭제 실패
        - "unknown": 상태 확인 불가
    - **deleted_from_db** (bool): DB에서 삭제 여부
        - true: 완전히 삭제됨
        - false: 아직 삭제되지 않음
    - **message** (str): 상태 메시지

    ## Process
    1. Pipeline 상태 확인 (5초 타임아웃)
    2. 완료 시: DB에서 워크플로우 삭제
    3. 진행중: 진행 상태 반환
    4. 실패: 오류 메시지 반환

    ## Notes
    - 이미 삭제된 워크플로우 호출 시 "already deleted" 반환
    - Pipeline 상태 확인에 실패해도 DB 조회 시도
    - 삭제는 되돌릴 수 없는 작업

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 삭제 처리 중 오류 발생
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

    워크플로우를 실행하여 ML 모델을 배포합니다.
    Kubeflow 파이프라인을 통해 KServe InferenceService를 생성하고,
    모델 서빙 엔드포인트를 활성화합니다.

    ## Path Parameters
    - **workflow_id** (str): 실행할 워크플로우 UUID

    ## Request Body (WorkflowExecuteRequest)
    - **parameters** (Dict[str, Any], optional): 실행 파라미터
        - 커스텀 설정 값들을 전달할 수 있음
        - 예: {"gpu_enabled": true, "replicas": 2}

    ## Response (WorkflowExecuteResponse)
    - **workflow_id** (str): 실행된 워크플로우 UUID
    - **kubeflow_run_id** (str): Kubeflow 파이프라인 실행 ID
    - **status** (str): 실행 상태
        - "PENDING": 대기중
        - "RUNNING": 실행중
        - "SUCCEEDED": 성공
        - "FAILED": 실패
    - **message** (str): 상태 메시지

    ## Process
    1. MODEL 컴포넌트를 KServe InferenceService로 배포
    2. 워크플로우를 Kubeflow 파이프라인으로 변환
    3. 파이프라인 실행 및 모니터링 시작
    4. KServeDeployment 테이블에 배포 정보 기록

    ## Notes
    - 워크플로우 상태가 ERROR인 경우만 실행 불가
    - DRAFT 상태에서도 실행 가능 (파이프라인 완료 시 자동으로 ACTIVE로 변경됨)
    - 배포된 모델은 /workflows/{workflow_id}/models로 확인
    - 실행 상태는 /workflows/{workflow_id}/status로 모니터링
    - 파이프라인 완료 시 워크플로우 상태가 자동으로 ACTIVE로 변경됨

    ## Errors
    - 400: 워크플로우가 ERROR 상태임
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 실행 중 오류 발생
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # ERROR 상태인 경우만 실행 불가
    if workflow.status == WorkflowStatus.ERROR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow has errors. Please fix the errors before execution.",
        )

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

    워크플로우의 실행 상태와 배포된 모델들의 상태를 종합적으로 조회합니다.
    KServe 배포 상태와 Kubeflow 파이프라인 실행 상태를 모두 포함합니다.

    ## Path Parameters
    - **workflow_id** (str): 조회할 워크플로우 UUID

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
        - `str(workflow.id)`로 변환된 값
    - **status** (str): 워크플로우 상태
        - "DRAFT": 임시저장 상태 (아직 실행되지 않음)
        - "ACTIVE": 활성 상태 (배포 완료, 실행 가능)
        - "ERROR": 오류 발생 상태 (실행 실패 또는 배포 오류)
    - **kubeflow_run_id** (str, optional): Kubeflow 파이프라인 실행 ID
        - 워크플로우가 실행된 경우에만 포함
        - 실행 전이면 null
        - 참조용으로만 포함되며, 실제 파이프라인 상태는 조회하지 않음
    - **deployed_models** (List[dict]): 배포된 모델 목록
        - 각 항목은 다음 필드를 포함:
        - **component_id** (str): 컴포넌트 UUID
            - 워크플로우 컴포넌트의 고유 ID
        - **service_name** (str): KServe InferenceService 이름
            - Kubernetes 리소스 이름 (DNS 1035 규칙 준수)
        - **service_hostname** (str): KServe 서비스 호스트명
            - Istio Virtual Service 라우팅에 사용
            - 형식: `{service_name}.{namespace}.example.com`
        - **model_name** (str): 컴포넌트 이름
            - 워크플로우 컴포넌트의 표시명 (사용자가 지정한 이름)
        - **sanitized_model_name** (str): 정제된 모델 이름
            - DNS 규칙에 맞게 변환된 모델 이름 (슬래시가 하이픈으로 변경됨)
            - KServe 엔드포인트에서 실제로 사용되는 이름
            - `model_name`과는 다를 수 있음 (model_name은 컴포넌트 이름, sanitized_model_name은 실제 배포된 모델 이름)
        - **model_id** (int, optional): 모델 ID
            - 컴포넌트에 연결된 모델의 ID
            - MODEL 타입 컴포넌트인 경우에만 포함
        - **internal_url** (str, optional): 내부 접근 URL
            - 클러스터 내부에서 접근 가능한 URL
            - 형식: `http://{service_name}.{namespace}.svc.cluster.local`
        - **gateway_url** (str): 게이트웨이 URL
            - 외부에서 접근 가능한 KServe Gateway 엔드포인트 URL
        - **status** (str): 배포 상태
            - 가능한 값:
                - "DEPLOYING": 배포 중
                - "DEPLOYED": 배포 완료
                - "FAILED": 배포 실패
                - "DELETED": 삭제됨
        - **deployed_at** (str, optional): 배포 시각
            - ISO 8601 형식의 문자열
            - 배포 완료 시 기록됨 (DEPLOYED 상태인 경우)
        - **error_message** (str, optional): 오류 메시지
            - 배포 실패 시 오류 내용
    - **error** (str, optional): 상태 조회 실패 시 오류 메시지
        - 정상 조회 시에는 포함되지 않음

    ## Notes
    - 워크플로우가 실행되지 않았다면 `kubeflow_run_id`는 null
    - `deployed_models`는 MODEL 타입 컴포넌트가 있는 경우만 포함
    - 모든 조회는 DB 기반으로 수행되며, Kubernetes나 Kubeflow를 직접 조회하지 않음
    - 배포 상태는 `kserve_deployments` 테이블의 정보를 기반으로 함
    - `deployed_models` 조회 실패 시에도 에러를 발생시키지 않고 빈 리스트로 처리됨

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 상태 조회 중 오류 발생
        - 이 경우 `error` 필드가 포함된 응답이 반환될 수 있음
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

    **중요**: 이 API는 Kubeflow Pipeline 내부에서만 호출되는 내부 API입니다.
    프론트엔드나 외부 클라이언트에서는 사용하지 않아야 합니다.

    Kubeflow Pipeline 실행 중 컴포넌트의 KServe 배포가 완료되면,
    Pipeline 내부에서 자동으로 이 API를 호출하여 배포 상태를 업데이트합니다.

    ## Path Parameters
    - **workflow_id** (str): 워크플로우 ID
    - **component_id** (str): 컴포넌트 UUID

    ## Request Body
    - **service_name** (str, required): KServe 서비스 이름
    - **service_hostname** (str, required): KServe 서비스 호스트명
    - **model_name** (str, required): 배포된 모델 이름
    - **status** (str, required): 배포 상태 (예: "ready", "failed")
    - **internal_url** (str, optional): 내부 서비스 URL
    - **error_message** (str, optional): 배포 실패 시 에러 메시지

    ## Response
    - **message** (str): 업데이트 결과 메시지
    - **deployment_info** (dict): 배포 정보
        - service_name (str): 서비스 이름
        - service_hostname (str): 서비스 호스트명
        - model_name (str): 모델 이름
        - status (str): 배포 상태
        - internal_url (str, optional): 내부 서비스 URL

    ## Notes
    - 이 API는 Kubeflow Pipeline의 컴포넌트 내부에서만 호출됩니다
    - 프론트엔드나 사용자 애플리케이션에서는 직접 호출하지 않아야 합니다
    - 배포 상태는 Pipeline 실행 중 자동으로 업데이트됩니다

    ## Errors
    - 404: 워크플로우를 찾을 수 없음
    - 500: 배포 상태 업데이트 중 서버 내부 오류
    """
    try:
        # 워크플로우 존재 여부 확인
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        if not workflow:
            logger.warning(f"Workflow {workflow_id} not found when updating deployment status")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found",
            )

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/models/{component_id}/inference")
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
    KServe V2 프로토콜을 사용하며, Object Detection 모델을 지원합니다.

    ## Path Parameters
    - **workflow_id** (str): 워크플로우 UUID
    - **component_id** (str): 컴포넌트 UUID (WorkflowComponent.id)
        - 컴포넌트 ID 조회 방법:
          1. 워크플로우 상세 조회: `GET /api/v1/workflows/{workflow_id}`
             - 응답의 `components` 배열에서 `id` 필드 확인
             - `type`이 "MODEL"인 컴포넌트의 `id` 사용
          2. 배포된 모델 목록 조회: `GET /api/v1/workflows/{workflow_id}/models`
             - 응답의 `deployed_models` 배열에서 `component_id` 필드 확인
             - 배포된 모델만 조회 가능 (DEPLOYED 상태)

    ## Request Body (Form Data)
    - **image** (file, required): 분석할 이미지 파일
        - 지원 형식: JPEG, PNG, GIF, WebP
        - Base64로 인코딩되어 서버로 전송

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **component_id** (str): 컴포넌트 UUID
    - **predictions** (List[dict]): 추론 결과 목록
        - Object Detection의 경우 각 항목은 다음 필드를 포함:
            - **score** (float): 객체 감지 신뢰도 점수 (0.0 ~ 1.0)
            - **label** (str): 감지된 객체의 레이블 (예: "person", "laptop")
            - **box** (List[float]): 바운딩 박스 좌표 [x1, y1, x2, y2]
                - x1, y1: 좌상단 좌표
                - x2, y2: 우하단 좌표
    - **image_info** (dict): 이미지 메타데이터
        - **original_size** (dict): 원본 이미지 크기
            - width (int): 원본 이미지 넓이
            - height (int): 원본 이미지 높이
        - **model_input_size** (dict): 모델 입력 크기
            - width (int): 모델에 입력된 이미지 넓이
            - height (int): 모델에 입력된 이미지 높이
    - **model_info** (dict): 모델 정보
        - component_id (str): 컴포넌트 ID
        - service_name (str): KServe 서비스 이름
        - sanitized_model_name (str): 정제된 모델 이름 (DNS 규칙 준수)
        - model_id (int, optional): 모델 ID
        - original_model_name (str, optional): 원본 모델 이름
        - model_type (str, optional): 모델 타입 (예: "ODM")
        - model_format (str, optional): 모델 포맷 (예: "pytorch")

    ## Monitoring
    - 모든 추론 요청은 ServiceMonitoring 테이블에 자동 기록
    - 응답 시간, 성공/실패 여부, 사용자 정보 포함
    - 서비스와 연결된 경우만 모니터링 데이터 저장

    ## Notes
    - Istio Gateway를 통해 KServe InferenceService에 접근
    - V2 프로토콜 엔드포인트: /v2/models/{model_name}/infer
    - Host 헤더로 Istio 라우팅 제어

    ## Errors
    - 401: 인증되지 않은 사용자
    - 400: 잘못된 이미지 파일
    - 404: 워크플로우나 컴포넌트를 찾을 수 없음
    - 503: 모델 서비스가 준비되지 않음
    - 504: 추론 요청 타임아웃
    """
    import base64  # noqa: F401, F811
    import time  # noqa: F401, F811

    import requests  # noqa: F401, F811
    from services.app_service import ServiceMonitoringService

    # 요청 시작 시간 기록
    start_time = time.time()

    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # Service ID 확인 (워크플로우가 서비스에 연결되어 있는지)
    service_id = workflow.service_id
    if not service_id:
        logger.warning(f"Workflow {workflow_id} is not associated with a service. Monitoring will be skipped.")
        # 서비스가 없어도 추론은 진행 (하위 호환성)

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
        .filter(WorkflowComponent.workflow_id == workflow_id, WorkflowComponent.id == component_id)
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

    response_time_ms = 0.0

    try:
        # V2 프로토콜 엔드포인트로 요청 (Istio Gateway 경유)
        url = f"{infer_svc_url}/v2/models/{model_name}/infer"

        logger.info(f"Sending inference request to {url}")

        response = requests.post(url, json=data, headers=headers, cookies=cookies, timeout=30)
        response.raise_for_status()

        # 응답 시간 계산
        response_time_ms = (time.time() - start_time) * 1000  # 밀리초로 변환

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

                    # 모니터링 데이터 기록 (성공)
                    if service_id:
                        try:
                            ServiceMonitoringService.record_inference_request(
                                db=db,
                                service_id=service_id,
                                workflow_id=workflow_id,
                                user_id=current_user.id,
                                response_time_ms=response_time_ms,
                                is_success=True,
                                is_object_detection=True,  # 현재는 Object Detection만 지원
                            )
                            db.commit()
                        except Exception as e:
                            logger.error(f"Failed to record monitoring data: {e}")
                            # 모니터링 실패해도 추론 결과는 반환
                            db.rollback()

                    return {
                        "workflow_id": workflow_id,
                        "component_id": component_id,
                        "predictions": predictions,
                        "image_info": image_info,
                        "model_info": model_info,
                    }
                else:
                    # 하위 호환성: response_data가 dict가 아닌 경우
                    # 모니터링 데이터 기록 (성공)
                    if service_id:
                        try:
                            ServiceMonitoringService.record_inference_request(
                                db=db,
                                service_id=service_id,
                                workflow_id=workflow_id,
                                user_id=current_user.id,
                                response_time_ms=response_time_ms,
                                is_success=True,
                                is_object_detection=True,
                            )
                            db.commit()
                        except Exception as e:
                            logger.error(f"Failed to record monitoring data: {e}")
                            db.rollback()

                    return {
                        "workflow_id": workflow_id,
                        "component_id": component_id,
                        "predictions": response_data,
                        "model_info": model_info,
                    }

        # 예상치 못한 응답 형식
        # 모니터링 데이터 기록 (성공 - 응답은 받았지만 형식이 예상과 다름)
        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.id,
                    response_time_ms=response_time_ms,
                    is_success=True,
                    is_object_detection=True,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        return {
            "workflow_id": workflow_id,
            "component_id": component_id,
            "raw_response": result,
            "model_info": model_info,
        }

    except requests.exceptions.HTTPError as http_err:
        # 응답 시간 계산
        response_time_ms = (time.time() - start_time) * 1000

        logger.error(f"HTTP error occurred: {http_err}")
        logger.error(f"Response content: {http_err.response.text if hasattr(http_err, 'response') else 'N/A'}")

        # 모니터링 데이터 기록 (실패)
        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.id,
                    response_time_ms=response_time_ms,
                    is_success=False,
                    is_object_detection=True,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model service returned error: {str(http_err)}"
        )
    except requests.exceptions.ConnectionError as conn_err:
        # 응답 시간 계산
        response_time_ms = (time.time() - start_time) * 1000

        logger.error(f"Connection error: {conn_err}")

        # 모니터링 데이터 기록 (실패)
        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.id,
                    response_time_ms=response_time_ms,
                    is_success=False,
                    is_object_detection=True,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to model service. Service may not be ready.",
        )
    except requests.exceptions.Timeout as timeout_err:
        # 응답 시간 계산
        response_time_ms = (time.time() - start_time) * 1000

        logger.error(f"Request timeout: {timeout_err}")

        # 모니터링 데이터 기록 (실패 - 타임아웃)
        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.id,
                    response_time_ms=response_time_ms,
                    is_success=False,
                    is_object_detection=True,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Model inference request timed out")
    except Exception as e:
        # 응답 시간 계산
        response_time_ms = (time.time() - start_time) * 1000

        logger.error(f"Unexpected error occurred: {e}")

        # 모니터링 데이터 기록 (실패)
        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.id,
                    response_time_ms=response_time_ms,
                    is_success=False,
                    is_object_detection=True,
                )
                db.commit()
            except Exception as mon_error:
                logger.error(f"Failed to record monitoring data: {mon_error}")
                db.rollback()

        raise


@router.get("/{workflow_id}/models")
def get_deployed_models(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우에 배포된 모델 목록 조회

    워크플로우에서 배포된 모든 ML 모델의 상세 정보를 조회합니다.
    KServe InferenceService로 배포된 모델들의 엔드포인트와 상태를 포함합니다.

    ## Path Parameters
    - **workflow_id** (str): 조회할 워크플로우 UUID

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **backend_api_url** (str): 추론 API URL (첫 번째 모델 기준)
        - 형식: {gateway_url}/v2/models/{model_name}/infer
    - **deployed_models** (List[dict]): 배포된 모델 목록
        - workflow_id (str): 소속 워크플로우 ID
        - component_id (str): 컴포넌트 ID
        - component_name (str): 컴포넌트 이름
        - model_id (int): 모델 ID
        - model_name (str): 원본 모델 이름
        - sanitized_model_name (str): DNS 규칙에 맞게 변환된 모델 이름
        - service_name (str): KServe 서비스 이름
        - service_hostname (str): KServe 서비스 호스트명
        - status (str): 배포 상태
            - "PENDING": 배포 대기중
            - "DEPLOYED": 배포 완료
            - "FAILED": 배포 실패
            - "DELETED": 삭제됨
        - internal_url (str): 내부 접근 URL
        - gateway_url (str): 외부 게이트웨이 URL
        - deployed_at (datetime): 배포 시각
        - deleted_at (datetime): 삭제 시각 (삭제된 경우)
        - error_message (str): 오류 메시지 (실패 시)
        - created_at (datetime): 레코드 생성 시각
        - updated_at (datetime): 레코드 업데이트 시각
    - **total** (int): 배포된 모델 총 개수

    ## Notes
    - backend_api_url은 첫 번째 배포된 모델 기준으로 생성
    - 각 모델마다 고유한 service_name과 hostname을 가짐
    - 배포 상태가 DEPLOYED인 모델만 추론 가능

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # DB에서 배포된 모델 목록 조회
    deployed_models = KServeDeploymentService.get_deployed_models(db, workflow_id, include_component_info=True)

    # backend_api_url 동적 생성
    backend_api_url = None
    if deployed_models:
        first_deployment = deployed_models[0]
        gateway_url = first_deployment.get("gateway_url") or settings.KSERVE_GATEWAY_URL or "http://10.10.30.154:80"
        model_name = first_deployment.get("sanitized_model_name") or first_deployment.get("model_name")
        backend_api_url = f"{gateway_url}/v2/models/{model_name}/infer"

    return {
        "workflow_id": workflow_id,
        "backend_api_url": backend_api_url,
        "deployed_models": deployed_models,
        "total": len(deployed_models),
    }


@router.post("/{workflow_id}/cleanup", status_code=status.HTTP_202_ACCEPTED)
async def cleanup_workflow_resources(
    *, db: Session = SessionDepends, workflow_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 리소스 정리 시작

    배포된 KServe InferenceService들을 정리합니다.
    워크플로우 자체는 유지하면서 배포된 리소스만 제거합니다.

    ## Path Parameters
    - **workflow_id** (str): 정리할 워크플로우 UUID

    ## Response (202 Accepted)
    - **message** (str): 상태 메시지 "Cleanup started"
    - **workflow_id** (str): 워크플로우 UUID
    - **cleanup_run_id** (str): 정리 파이프라인 실행 ID
    - **status** (str): 현재 상태 "cleanup_in_progress"
    - **next_step** (str): 다음 단계 API 안내
        - 형식: "Call /workflows/{workflow_id}/finalize-cleanup?run_id={cleanup_run_id} to check completion"

    ## Use Cases
    - 비용 절감을 위해 배포된 리소스 정리
    - 오류 발생 후 재배포 준비
    - 워크플로우 구조 변경 전 리소스 정리

    ## Process
    1. KServe InferenceService 삭제 파이프라인 시작
    2. cleanup_run_id 반환
    3. finalize-cleanup API로 완료 확인
    4. 워크플로우 상태를 DRAFT로 변경 (재실행 가능)

    ## Notes
    - 워크플로우는 삭제되지 않고 리소스만 정리
    - 비동기 프로세스 (202 Accepted)
    - 정리 후 워크플로우 재실행 가능

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 정리 파이프라인 시작 실패
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

    KServe 리소스 정리가 완료되었는지 확인하고,
    완료된 경우 워크플로우 상태를 업데이트합니다.
    워크플로우는 삭제되지 않고 리소스만 정리되며, 정리 후 재실행이 가능합니다.

    ## Path Parameters
    - **workflow_id** (str): 정리할 워크플로우 UUID
        - 워크플로우 목록 조회 API(/workflows)에서 확인 가능

    ## Query Parameters
    - **run_id** (str, required): Kubeflow Pipeline cleanup run ID
        - cleanup API에서 반환된 cleanup_run_id 사용
        - 형식: Kubeflow Pipeline 실행 UUID

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **run_id** (str): 정리 파이프라인 실행 ID
    - **status** (str): 정리 상태
        - "completed": 정리 완료
            - Pipeline이 성공적으로 완료되어 리소스가 정리됨
            - 워크플로우 상태가 업데이트됨 (ERROR → DRAFT)
        - "in_progress": 아직 진행중
            - Pipeline이 아직 실행 중임
            - 완료될 때까지 대기 후 재호출 필요
        - "failed": 정리 실패
            - Pipeline 실행이 실패했거나 오류 발생
            - error_message에 상세 오류 정보 포함
        - "unknown": 상태 확인 불가
            - Pipeline 상태 조회에 실패한 경우
            - Kubeflow 연결 문제 또는 run_id 오류 가능
    - **workflow_updated** (bool): 워크플로우 상태 업데이트 여부
        - true: 워크플로우 상태가 성공적으로 업데이트됨
            - ERROR 상태였던 경우 DRAFT로 변경됨
            - 재실행 가능한 상태로 변경됨
        - false: 워크플로우 상태가 업데이트되지 않음
            - Pipeline이 아직 진행중이거나 실패한 경우
            - 또는 워크플로우가 이미 DRAFT 상태인 경우
    - **message** (str): 상태 메시지
        - 정리 완료: "Cleanup completed and workflow state updated"
        - 진행중: "Cleanup pipeline still in progress"
        - 실패: "Cleanup pipeline failed with status: {status_value}"
        - 상태 확인 불가: "Failed to check pipeline status: {error}"

    ## Process
    1. 워크플로우 존재 여부 확인
    2. Pipeline 상태 확인 (5초 타임아웃)
    3. 완료 시:
       - 워크플로우 상태가 ERROR인 경우 DRAFT로 변경
       - 재실행 가능한 상태로 업데이트
    4. 진행중: 진행 상태 반환 (재호출 필요)
    5. 실패: 오류 메시지 반환

    ## Notes
    - 워크플로우는 삭제되지 않고 리소스만 정리됨
    - 정리 완료 후 워크플로우를 재실행할 수 있음
    - Pipeline 상태 확인은 짧은 타임아웃(5초)으로 즉시 확인
    - Pipeline이 아직 진행중이면 재호출하여 완료 확인 필요
    - ERROR 상태의 워크플로우는 정리 완료 시 DRAFT로 변경됨
    - 이미 DRAFT 상태인 워크플로우는 상태 변경 없음
    - cleanup API 호출 후 이 API를 호출하여 완료 확인 필요

    ## Usage Example
    1. cleanup API 호출하여 정리 파이프라인 시작
    2. cleanup_run_id 받기
    3. 이 API를 호출하여 완료 확인
    4. status가 "completed"이고 workflow_updated가 true면 정리 완료
    5. status가 "in_progress"면 잠시 후 재호출

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
        - workflow_id가 존재하지 않거나 삭제된 경우
    - 500: 정리 처리 중 오류 발생
        - Pipeline 상태 확인 실패 또는 워크플로우 상태 업데이트 실패
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
