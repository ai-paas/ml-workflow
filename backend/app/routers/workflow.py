"""Workflow API 라우터"""

import base64
import io
import json
import logging
import math
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import requests
from config.db.connect import SessionDepends
from config.db.enums import ModelFormatEnum, ModelProviderEnum, ModelTypeEnum
from config.settings import get_settings
from core.kubeflow.kubeflow_manager import KubeflowManager
from core.kubeflow.workflow_executor import WorkflowExecutor
from core.serving.remote_workflow_serving import remote_chat_completion_async
from core.serving.serving_resource_meta import serving_meta_validation_error
from db.models.model import Model, ModelTaskType
from db.models.model_workflow_deployment import WorkflowServingDeploymentType
from db.models.prompt import PromptVariableType
from db.models.service import ComponentType, Workflow, WorkflowComponent, WorkflowStatus
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image, ImageDraw, ImageFont
from repos.prompt import prompt_repository
from repos.workflow import workflow_repository
from schemas.user import UserSchema
from schemas.workflow import (
    ComponentCreateRequest,
    ComponentTestErrorResult,
    ComponentTestResult,
    ComponentTypeInfo,
    KnowledgeBaseComponentTestResult,
    KnowledgeBaseTestResult,
    ModelComponentTestResult,
    ModelFillMaskTestResult,
    ModelLLMTestResult,
    ModelODMTestResult,
    ModelProteinClassificationTestResult,
    ValidationCheckResponse,
    WorkflowBaseSchema,
    WorkflowCreateRequest,
    WorkflowDefinition,
    WorkflowExecuteResponse,
    WorkflowFillMaskTestResponse,
    WorkflowListSchema,
    WorkflowMLTestResponse,
    WorkflowProteinClassificationTestResponse,
    WorkflowRAGTestResponse,
    WorkflowReadSchema,
    WorkflowTemplateBriefSchema,
    WorkflowTemplateCreateRequest,
    WorkflowTemplateListSchema,
    WorkflowTemplateReadSchema,
    WorkflowTemplateUpdateRequest,
    WorkflowUpdateRequest,
    WorkflowValidateRequest,
    WorkflowValidateResponse,
)
from services.app_service import ServiceMonitoringService
from services.knowledge_base import KnowledgeBaseService
from services.model import PREDEFINED_MODEL_CONFIGS, ModelService
from services.model_workflow_deployment import ModelWorkflowDeploymentService
from services.workflow import WorkflowService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user, get_current_user_or_internal, verify_internal_api_key

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
        - "KNOWLEDGE_BASE": 지식 베이스 검색 노드
    - **component_id** (str): 컴포넌트 식별자
        - 워크플로우 정의 시 사용할 고유 ID
        - 일반적으로 type과 동일 (예: "START", "END", "MODEL", "KNOWLEDGE_BASE")
    - **name** (str): 타입 표시명 (한글)
        - "시작 노드", "종료 노드", "모델 노드", "지식 베이스 노드" 등
    - **description** (str): 타입 설명
        - 각 컴포넌트 타입의 역할과 용도 설명

    ## Usage Example
    1. 이 API로 사용 가능한 컴포넌트 타입 확인
    2. workflow_definition 작성 시 component_id 사용
    3. 각 컴포넌트 타입에 맞는 설정 적용

    ## Notes
    - 고정된 타입 목록 반환 (동적 변경 없음)
    - 워크플로우는 반드시 START로 시작하고 END로 종료
    - MODEL 타입은 model_id 필수, prompt_id 선택
    - KNOWLEDGE_BASE 타입은 knowledge_base_id 필수
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
        ComponentTypeInfo(
            type=ComponentType.KNOWLEDGE_BASE.value,
            component_id=ComponentType.KNOWLEDGE_BASE.value,
            name="지식 베이스 노드",
            description="Knowledge Base에서 정보를 검색하는 노드",
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
            - ref_id (str): 프론트 생성 임시 참조 ID
            - name (str): 컴포넌트 이름
            - type (ComponentType): 타입 (START/END/MODEL/KNOWLEDGE_BASE)
            - description (str, optional): 설명
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
            - knowledge_base_id (int, optional): KNOWLEDGE_BASE 타입인 경우 Knowledge Base ID
            - prompt_id (int, optional): MODEL 타입인 경우 프롬프트 ID
            - config (dict, optional): 컴포넌트별 세부 설정
            - x (int, optional): 프론트 캔버스 x 좌표 (음수 허용)
            - y (int, optional): 프론트 캔버스 y 좌표 (음수 허용)
        - connections (List[ConnectionCreateRequest]): 연결 목록
            - source_ref_id (str): 소스 컴포넌트 ref_id
            - target_ref_id (str): 타겟 컴포넌트 ref_id

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
    - MODEL 컴포넌트는 유효한 model_id 필요, prompt_id는 선택
    - KNOWLEDGE_BASE 컴포넌트는 유효한 knowledge_base_id 필요
    - 생성 직후 상태는 DRAFT
    - is_template은 항상 false로 설정됨 (템플릿 생성은 /workflows/templates API 사용)
    - 상세 정보(components, connections, creator 등)는 GET /workflows/{workflow_id}로 조회 가능

    ## Errors
    - 400: 잘못된 요청 (정의 오류 등)
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    try:
        if workflow_data.workflow_definition:
            _validate_workflow_definition_or_raise(db, workflow_data.workflow_definition)

        workflow = WorkflowService.create_workflow(db=db, workflow_data=workflow_data, creator_id=current_user.id)

        return WorkflowBaseSchema.model_validate(workflow)

    except HTTPException:
        raise
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
    service_id: Optional[str] = Query(None, description="서비스 ID (UUID)"),
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
    - **service_id** (str, optional): 특정 서비스에 연결된 워크플로우만 필터 (UUID)
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
    total_count = WorkflowService.count_workflows(
        db=db,
        creator_id=creator_id,
        service_id=service_id,
        is_template=False,
        status=status,
    )
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


# ============= Workflow Validation Endpoint =============


@router.post("/validate", response_model=WorkflowValidateResponse)
def validate_workflow_definition_endpoint(
    *,
    db: Session = SessionDepends,
    body: WorkflowValidateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 정의 사전 검증

    워크플로우 생성 전에 workflow_definition이 유효한지 사전 체크합니다.
    DB에 아무것도 생성하지 않고 항목별 결과를 반환합니다.

    ## Request Body (WorkflowValidateRequest)
    - **workflow_definition** (WorkflowDefinition): 검증할 워크플로우 정의

    ## Response (WorkflowValidateResponse)
    - **valid** (bool): 전체 검증 통과 여부
    - **checks** (List[ValidationCheckResponse]): 검증 항목별 결과 리스트
    """
    checks = _validate_workflow_definition_checks(db, body.workflow_definition)
    return WorkflowValidateResponse(
        valid=all(c.passed for c in checks),
        checks=[ValidationCheckResponse(rule=c.rule, passed=c.passed, message=c.message) for c in checks],
    )


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
    - **workflow_definition** (WorkflowDefinition, optional): 템플릿 구조 (생성 시 검증·저장에 사용)
        - components (List[ComponentCreateRequest]): 컴포넌트 정의
            - ref_id (str): 프론트·스크립트용 임시 참조 ID (연결의 source_ref_id/target_ref_id와 매칭)
            - name (str): 컴포넌트 이름
            - type (str): 타입 (START/END/MODEL/KNOWLEDGE_BASE)
            - description (str, optional): 설명
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
            - knowledge_base_id (int, optional): KNOWLEDGE_BASE 타입인 경우 Knowledge Base ID
            - prompt_id (int, optional): MODEL 타입인 경우 프롬프트 ID
            - config (dict, optional): 타입별 세부 설정 (MODEL/KB 등, START/END는 미사용)
            - x (int, optional): 프론트 캔버스 x 좌표 (음수 허용)
            - y (int, optional): 프론트 캔버스 y 좌표 (음수 허용)
        - connections (List[ConnectionCreateRequest]): 연결 정의 (요청 시 ref_id 기준)
            - source_ref_id (str): 소스 컴포넌트의 ref_id
            - target_ref_id (str): 타겟 컴포넌트의 ref_id

    ## Response (WorkflowTemplateBriefSchema)
    - **id** (str): 템플릿 UUID
    - **name** (str): 템플릿 이름
    - **description** (str): 템플릿 설명
    - **category** (str): 템플릿 카테고리
    - **status** (str): 템플릿 상태 (DRAFT)
    - **service_id** (str): 기본 서비스 ID
    - **creator_id** (int): 템플릿 생성자 ID
    - **creator** (UserBriefSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
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
        if template_data.workflow_definition:
            _validate_workflow_definition_or_raise(db, template_data.workflow_definition)

        template = WorkflowService.create_workflow_template(
            db=db, template_data=template_data, creator_id=current_user.id
        )

        result = WorkflowTemplateBriefSchema.model_validate(template)

        # 사용 횟수 계산
        usage_count = WorkflowService.get_template_usage_count(db, template.id)
        result.usage_count = usage_count

        return result

    except HTTPException:
        raise
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
        - creator (UserBriefSchema): 생성자 정보
            - id (int): 사용자 ID
            - username (str): 사용자명
            - name (str): 사용자 이름
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
    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        templates = WorkflowService.get_workflow_templates(
            db=db, skip=0, limit=10000, creator_id=None, category=category  # 모든 사용자의 템플릿 조회 가능
        )
        results = []
        for template in templates:
            result = WorkflowTemplateBriefSchema.model_validate(template)
            # 사용 횟수 계산
            usage_count = WorkflowService.get_template_usage_count(db, template.id)
            result.usage_count = usage_count
            results.append(result)
        return WorkflowTemplateListSchema(total=len(results), items=results)

    # 페이지네이션 적용
    total_count = WorkflowService.count_workflows(
        db=db,
        is_template=True,
        category=category,
    )
    skip = page_size * (page - 1)

    templates = WorkflowService.get_workflow_templates(
        db=db, skip=skip, limit=page_size, creator_id=None, category=category  # 모든 사용자의 템플릿 조회 가능
    )

    results = []
    for template in templates:
        result = WorkflowTemplateBriefSchema.model_validate(template)

        # 사용 횟수 계산
        usage_count = WorkflowService.get_template_usage_count(db, template.id)
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
    - **creator** (UserBriefSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
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
            - "KNOWLEDGE_BASE": 지식 베이스 검색 노드
        - model_id (int, optional): 연결된 모델 ID
            - MODEL 타입인 경우 필수, 다른 타입은 null
        - knowledge_base_id (int, optional): 연결된 Knowledge Base ID
            - KNOWLEDGE_BASE 타입인 경우 필수, 다른 타입은 null
        - prompt_id (int, optional): 연결된 프롬프트 ID
            - MODEL 타입인 경우 선택, 다른 타입은 null
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
        - x (int, optional): 프론트 캔버스 x 좌표 (음수 허용)
        - y (int, optional): 프론트 캔버스 y 좌표 (음수 허용)
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
    usage_count = WorkflowService.get_template_usage_count(db, template.id)
    result.usage_count = usage_count

    return result


@router.post("/templates/{template_id}/clone", response_model=WorkflowReadSchema)
def clone_from_template(
    *,
    db: Session = SessionDepends,
    template_id: str,  # UUID 문자열로 변경
    workflow_name: str = Query(..., description="새 워크플로우 이름"),
    service_id: Optional[str] = Query(None, description="연결할 서비스 ID (UUID)"),
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
    - **service_id** (str, optional): 연결할 서비스 ID (UUID)
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
    - **creator** (UserBriefSchema): 생성자 정보 (현재 사용자)
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
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
    - **workflow_definition** (WorkflowDefinition, optional): 새 템플릿 구조
        - components (List[ComponentCreateRequest]): 컴포넌트 목록
            - ref_id (str): 임시 참조 ID (연결과 매칭)
            - name (str): 컴포넌트 이름
            - type (ComponentType): 타입 (START/END/MODEL/KNOWLEDGE_BASE)
            - description (str, optional): 설명
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
            - knowledge_base_id (int, optional): KNOWLEDGE_BASE 타입인 경우 Knowledge Base ID
            - prompt_id (int, optional): MODEL 타입인 경우 프롬프트 ID
            - config (dict, optional): 타입별 세부 설정
            - x (int, optional): 프론트 캔버스 x 좌표 (음수 허용)
            - y (int, optional): 프론트 캔버스 y 좌표 (음수 허용)
        - connections (List[ConnectionCreateRequest]): 연결 목록
            - source_ref_id (str): 소스 컴포넌트 ref_id
            - target_ref_id (str): 타겟 컴포넌트 ref_id

    ## Response (WorkflowTemplateReadSchema)
    - **id** (str): 템플릿 UUID
    - **name** (str): 템플릿 이름
    - **description** (str): 템플릿 설명
    - **category** (str): 템플릿 카테고리
    - **status** (str): 템플릿 상태 (DRAFT)
    - **service_id** (str): 기본 서비스 ID (항상 null)
    - **creator_id** (int): 템플릿 생성자 ID
    - **creator** (UserBriefSchema): 생성자 정보
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
    try:
        # 템플릿인지 확인
        template = WorkflowService.get_workflow_by_id(db, template_id)
        if not template or not template.is_template:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} not found")

        if template_data.workflow_definition:
            _validate_workflow_definition_or_raise(db, template_data.workflow_definition)

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

        result = WorkflowTemplateReadSchema.model_validate(updated_template)

        # 사용 횟수 계산
        usage_count = WorkflowService.get_template_usage_count(db, template_id)
        result.usage_count = usage_count

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update template")


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workflow_template(
    *, db: Session = SessionDepends, template_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    워크플로우 템플릿 삭제

    템플릿은 워크플로 실행으로 생성된 클러스터 서빙 리소스가 없는 레코드이므로 DB에서 즉시 삭제됩니다.
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
    - **creator** (UserBriefSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
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
            - "KNOWLEDGE_BASE": 지식 베이스 검색 노드
        - model_id (int, optional): 모델 ID
            - MODEL 타입인 경우 필수, 다른 타입은 null
        - knowledge_base_id (int, optional): Knowledge Base ID
            - KNOWLEDGE_BASE 타입인 경우 필수, 다른 타입은 null
        - prompt_id (int, optional): 프롬프트 ID
            - MODEL 타입인 경우 선택, 다른 타입은 null
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
        - x (int, optional): 프론트 캔버스 x 좌표 (음수 허용)
        - y (int, optional): 프론트 캔버스 y 좌표 (음수 허용)
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
    - **public_url** (str|None): §2.6 — 요약용 첫 배포가 **KSERVE**이고 `KSERVE_GATEWAY_URL`이 설정된 경우에만
        `{게이트웨이}/v2/models/{model_name}/infer`, 그 외 null
    - **backend_api_url** (str|None): §2.6 — 요약용 첫 `model_workflow_deployments` 배포의 `internal_url` 기반(없으면 null;
        REMOTE 등은 스키마 계산상 null일 수 있음 — 상세는 `/workflows/{id}/models` 참고)
    - **created_at** (datetime): 워크플로우 생성 시각
    - **updated_at** (datetime): 워크플로우 수정 시각

    ## Notes
    - public_url·backend_api_url은 배포 레코드·설정에 따라 §2.6 정책으로 계산됨
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
    current_user: Optional[UserSchema] = Depends(get_current_user_or_internal),
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
    - **workflow_definition** (WorkflowDefinition, optional): 새 워크플로우 구조
        - components (List[ComponentCreateRequest]): 컴포넌트 목록
            - ref_id (str): 프론트 생성 임시 참조 ID
            - name (str): 컴포넌트 이름
            - type (ComponentType): 타입 (START/END/MODEL/KNOWLEDGE_BASE)
                - "START": 워크플로우 시작점
                - "END": 워크플로우 종료점
                - "MODEL": ML 모델 실행 노드
                - "KNOWLEDGE_BASE": 지식 베이스 검색 노드
            - description (str, optional): 설명
            - model_id (int, optional): MODEL 타입인 경우 모델 ID
                - MODEL 타입인 경우 필수, 다른 타입은 null
            - knowledge_base_id (int, optional): KNOWLEDGE_BASE 타입인 경우 Knowledge Base ID
                - KNOWLEDGE_BASE 타입인 경우 필수, 다른 타입은 null
            - prompt_id (int, optional): MODEL 타입인 경우 프롬프트 ID
                - MODEL 타입인 경우 선택, 다른 타입은 null
            - config (dict, optional): 타입별 세부 설정
            - x (int, optional): 프론트 캔버스 x 좌표 (음수 허용)
            - y (int, optional): 프론트 캔버스 y 좌표 (음수 허용)
        - connections (List[ConnectionCreateRequest]): 연결 목록
            - source_ref_id (str): 소스 컴포넌트 ref_id
            - target_ref_id (str): 타겟 컴포넌트 ref_id

    ## Response (WorkflowReadSchema)
    - **id** (str): 워크플로우 UUID
    - **name** (str): 워크플로우 이름
    - **description** (str): 워크플로우 설명
    - **category** (str): 워크플로우 카테고리
    - **status** (str): 워크플로우 상태 (DRAFT/ACTIVE/ERROR)
    - **service_id** (str): 연결된 서비스 ID
    - **service_name** (str): 연결된 서비스 이름
    - **creator_id** (int): 생성자 ID
    - **creator** (UserBriefSchema): 생성자 정보
    - **is_template** (bool): 템플릿 여부 (false)
    - **template_id** (str): 원본 템플릿 ID
    - **template_name** (str): 원본 템플릿 이름
    - **kubeflow_run_id** (str): Kubeflow 파이프라인 실행 ID
    - **components** (List[ComponentReadSchema]): 컴포넌트 목록
    - **component_connections** (List[ConnectionReadSchema]): 연결 정보
    - **public_url** (str|None): §2.6 — 첫 배포가 KSERVE이고 게이트웨이가 설정된 경우에만 공개 추론 URL
    - **backend_api_url** (str|None): §2.6 — 첫 배포 기준 요약 URL(없으면 null)
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
    - 409: 배포 중이거나 배포된 워크플로우의 구조 변경 시도
    - 500: 서버 내부 오류
    """
    try:
        if workflow_data.workflow_definition:
            if ModelWorkflowDeploymentService.has_active_deployment(db, workflow_id):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="배포 중이거나 배포된 워크플로우는 수정할 수 없습니다. 먼저 배포를 삭제한 후 수정해주세요.",
                )
            _validate_workflow_definition_or_raise(db, workflow_data.workflow_definition)

        workflow = WorkflowService.update_workflow(db=db, workflow_id=workflow_id, workflow_data=workflow_data)

        if not workflow:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

        # 관계를 다시 로드하여 스키마로 변환
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        return WorkflowReadSchema.model_validate(workflow)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update workflow")


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

    워크플로우 삭제를 시작합니다. Kubeflow 정리 파이프라인으로 **KServe InferenceService**,
    **Ollama Deployment/Service** 등 워크플로에 연결된 클러스터 서빙 리소스를 제거하고 `cleanup_run_id`를 반환합니다.
    실제 DB 삭제는 finalize-deletion API에서 클러스터 잔존 여부를 확인한 뒤 수행됩니다.

    ## Path Parameters
    - **workflow_id** (str): 삭제할 워크플로우 UUID

    ## Response (202 Accepted)
    - **message** (str): 상태 메시지 "Workflow deletion started"
    - **workflow_id** (str): 워크플로우 UUID
    - **cleanup_run_id** (str): 정리 파이프라인 실행 ID
    - **status** (str): 현재 상태 "cleanup_in_progress"
    - **next_step** (str): 다음 단계 API 안내
        - 형식: "Call /workflows/{workflow_id}/finalize-deletion to complete deletion"

    ## Deletion Process
    1. 현재 API 호출: 정리 파이프라인 시작
    2. Kubeflow Pipeline: 라벨 `workflow-id` 기준 InferenceService·Deployment·Service 등 삭제
    3. finalize-deletion API 호출: Kubernetes 잔존 리소스 확인 후 DB에서 워크플로 삭제

    ## Notes
    - 비동기 프로세스로 진행됨 (202 Accepted)
    - 클러스터 정리에 시간이 걸릴 수 있음
    - REMOTE 배포는 전용 클러스터 리소스가 없을 수 있음(파이프라인은 noop·콜백 위주)
    - 템플릿은 바로 DB에서 삭제됨 (배포 리소스 없음)

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 409: 배포가 진행 중인 워크플로우 삭제 시도
    - 500: 정리 파이프라인 시작 실패
    """
    try:
        # 워크플로우 존재 여부 확인
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

        if ModelWorkflowDeploymentService.has_deploying(db, workflow_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="배포가 진행 중인 워크플로우는 삭제할 수 없습니다. 배포가 완료된 후 다시 시도해주세요.",
            )

        # Kubeflow Pipeline을 통해 워크플로 서빙 리소스(InferenceService·Ollama 등) 삭제 시작
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
            "next_step": (f"Call /workflows/{workflow_id}/finalize-deletion to complete deletion"),
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
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 삭제 완료 처리

    Kubernetes 클러스터를 직접 조회하여 리소스가 실제로 삭제되었는지 확인하고,
    확인된 경우 DB에서 워크플로우를 삭제합니다.

    ## Path Parameters
    - **workflow_id** (str): 삭제할 워크플로우 UUID

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **status** (str): 삭제 상태
        - "completed": 삭제 완료
            - Kubernetes 리소스가 실제로 삭제되어 확인됨
            - DB에서 워크플로우가 삭제됨
        - "in_progress": 아직 진행중
            - Kubernetes에 리소스가 아직 존재함
            - 완료될 때까지 대기 후 재호출 필요
        - "failed": 삭제 실패
            - Kubernetes 리소스 확인 중 오류 발생
            - error_message에 상세 오류 정보 포함
    - **deleted_from_db** (bool): DB에서 삭제 여부
        - true: 완전히 삭제됨
        - false: 아직 삭제되지 않음
    - **message** (str): 상태 메시지

    ## Process
    1. 워크플로우 존재 여부 확인
    2. Kubernetes 클러스터에서 리소스 직접 조회
       - InferenceService 조회
       - Ollama Deployment/Service 조회
    3. 리소스가 모두 삭제된 경우: DB에서 워크플로우 삭제
    4. 리소스가 아직 존재하는 경우: 진행중 상태 반환 (재호출 필요)
    5. 확인 중 오류 발생: 실패 상태 반환

    ## Notes
    - REMOTE 전용 배포는 클러스터에 남는 리소스가 없을 수 있어, 위 k8s 검사만으로는 곧바로 DB 삭제로 이어질 수 있음
    - 이미 삭제된 워크플로우 호출 시 "already deleted" 반환
    - Kubernetes 리소스 확인에 실패해도 DB 조회 시도
    - 삭제는 되돌릴 수 없는 작업
    - 리소스가 아직 존재하면 재호출하여 완료 확인 필요

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
                "status": "completed",
                "deleted_from_db": True,
                "message": "Workflow already deleted",
            }

        # Kubernetes 클러스터에서 리소스 직접 조회하여 삭제 여부 확인
        try:
            from kubernetes import client
            from kubernetes import config as k8s_config

            # Kubernetes 설정
            try:
                k8s_config.load_incluster_config()
            except Exception:
                # 개발 환경에서는 kubeconfig 사용
                try:
                    k8s_config.load_kube_config()
                except Exception as e:
                    logger.error(f"Failed to load Kubernetes config: {e}")
                    raise Exception(f"Failed to load Kubernetes config: {str(e)}")

            namespace = settings.KUBEFLOW_NAMESPACE
            label_selector = f"workflow-id={workflow_id}"

            resources_exist = False

            # 1. InferenceService 확인
            try:
                api = client.CustomObjectsApi()

                result = api.list_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    label_selector=label_selector,
                )

                services = result.get("items", [])
                if len(services) > 0:
                    logger.info(f"Found {len(services)} InferenceServices still existing for workflow {workflow_id}")
                    resources_exist = True
            except Exception as e:
                logger.warning(f"Error checking InferenceServices: {e}")

            # 2. Ollama Deployment 확인
            if not resources_exist:
                try:
                    apps_v1 = client.AppsV1Api()
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    if len(deployments.items) > 0:
                        logger.info(
                            f"Found {len(deployments.items)} Deployments still existing for workflow {workflow_id}"
                        )
                        resources_exist = True
                except Exception as e:
                    logger.warning(f"Error checking Deployments: {e}")

            # 3. Ollama Service 확인
            if not resources_exist:
                try:
                    core_v1 = client.CoreV1Api()
                    services = core_v1.list_namespaced_service(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    if len(services.items) > 0:
                        logger.info(f"Found {len(services.items)} Services still existing for workflow {workflow_id}")
                        resources_exist = True
                except Exception as e:
                    logger.warning(f"Error checking Services: {e}")

            if resources_exist:
                # 리소스가 아직 존재함 - 진행중
                return {
                    "workflow_id": workflow_id,
                    "status": "in_progress",
                    "deleted_from_db": False,
                    "message": "Resources still exist in Kubernetes, waiting for cleanup",
                }

            # 리소스가 모두 삭제됨 - DB에서 워크플로우 삭제
            logger.info(f"All resources deleted for workflow {workflow_id}, deleting from DB")

            delete_success = WorkflowService.delete_workflow(db, workflow_id)

            if delete_success:
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "deleted_from_db": True,
                    "message": "Workflow deleted successfully",
                }
            else:
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "deleted_from_db": False,
                    "message": "Resources deleted but DB deletion failed",
                }

        except Exception as k8s_error:
            logger.error(f"Failed to check Kubernetes resources: {k8s_error}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "deleted_from_db": False,
                "message": f"Failed to check Kubernetes resources: {str(k8s_error)}",
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
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 실행 (서빙 배포 + Kubeflow 파이프라인)

    워크플로우를 실행하여 MODEL 컴포넌트를 배포합니다. 배포 유형(`deployment_type`)은 모델·설정에 따라
    **KServe InferenceService**, **Ollama Deployment/Service**, 또는 **REMOTE**(원격 LLM 스텁·콜백만)로 결정됩니다.
    Kubeflow 파이프라인은 위 유형에 맞는 태스크(또는 REMOTE 알림용 경량 컴포넌트)를 실행합니다.

    ## Path Parameters
    - **workflow_id** (str): 실행할 워크플로우 UUID

    ## Request Body
    - 없음. GPU/CPU·GPU 장 수·노드 지정 등은 백엔드가 사전정의 모델 메타 및 설정으로 결정한다.

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
    1. 지식베이스가 모델 앞에 있는지 검증
    2. MODEL별 서빙 계획·PVC(Ollama)·`model_workflow_deployments` 레코드 생성(REMOTE는 즉시 DEPLOYED 처리 가능)
    3. 워크플로우를 Kubeflow 파이프라인으로 변환·실행
    4. 파이프라인·콜백으로 배포 상태 및 워크플로 ACTIVE 반영

    ## Notes
    - 워크플로우 상태가 ERROR인 경우만 실행 불가
    - DRAFT 상태에서도 실행 가능 (파이프라인 완료 시 자동으로 ACTIVE로 변경됨)
    - 지식베이스 컴포넌트는 모델 컴포넌트보다 앞에 있어야 함
    - 배포된 모델은 /workflows/{workflow_id}/models로 확인
    - 실행 상태는 /workflows/{workflow_id}/status로 모니터링
    - 파이프라인 완료 시 워크플로우 상태가 자동으로 ACTIVE로 변경됨

    ## Errors
    - 400: MODEL 컴포넌트 없음, ERROR 상태, 지식베이스 순서 위반, 서빙 메타 검증 실패
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 409: 이미 배포 중이거나 배포 완료된 상태
    - 500: 실행 중 오류 발생
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    model_component_count = sum(1 for c in workflow.components if c.type == ComponentType.MODEL)
    if model_component_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="워크플로우에 MODEL 컴포넌트가 없어 배포할 수 없습니다.",
        )

    if ModelWorkflowDeploymentService.has_active_deployment(db, workflow_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 배포 중이거나 배포가 완료된 워크플로우입니다. 재배포가 필요하면 삭제·정리 후 다시 시도해 주세요.",
        )

    # ERROR 상태인 경우만 실행 불가
    if workflow.status == WorkflowStatus.ERROR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow has errors. Please fix the errors before execution.",
        )

    # 지식베이스가 모델 앞에 있는지 검증
    try:
        WorkflowService._validate_knowledge_base_before_model(workflow)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        from core.serving.serving_model_workflow_pvc import PvcReplicationConflict

        # WorkflowExecutor를 사용하여 워크플로우 실행
        executor = WorkflowExecutor(db)
        execution_result = executor.execute_workflow(workflow=workflow, parameters={})

        return WorkflowExecuteResponse(
            workflow_id=execution_result["workflow_id"],
            kubeflow_run_id=execution_result["kubeflow_run_id"],
            status=execution_result["status"],
            message=execution_result["message"],
        )

    except PvcReplicationConflict as e:
        logger.warning(f"Workflow execute PVC lock conflict: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message,
        )

    except ValueError as e:
        logger.warning(f"Workflow execution rejected (serving meta / validation): {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

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
    `model_workflow_deployments` 기반 배포 요약(KSERVE·OLLAMA·REMOTE)과 Kubeflow 파이프라인 실행 ID를 포함합니다.

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
        - **deployment_type** (str, optional): `KSERVE` | `OLLAMA` | `REMOTE`
        - **service_name** (str): Kubernetes 리소스 또는 식별용 이름 (DNS 1035 규칙 준수)
        - **service_hostname** (str): KServe 게이트웨이 라우팅용 호스트명 등 (Ollama·REMOTE는 빈 문자열일 수 있음)
        - **model_name** (str): 배포 레코드에 저장된 모델/컴포넌트 식별명 (REMOTE 시 맵 기준 원격 모델명 등)
        - **sanitized_model_name** (str): API 응답용 정제 이름 (레거시 필드, `model_name`과 동일할 수 있음)
        - **model_id** (int, optional): 컴포넌트에 연결된 모델 ID
        - **internal_url** (str, optional): 클러스터 내부 접근 URL (Ollama ClusterIP 등) 또는 REMOTE 베이스 URL
        - **gateway_url** (str|None): 설정된 KServe 게이트웨이 베이스 URL(없으면 null)
        - **public_url** (str|None): KSERVE+게이트웨이 설정 시에만 추론 URL
        - **backend_api_url** (str|None): §2.6 — `internal_url` 우선, REMOTE는 `remote_api_url` 폴백
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
    - 배포 상태는 `model_workflow_deployments` 테이블의 정보를 기반으로 함
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


@router.post(
    "/{workflow_id}/components/{component_id}/deployment-status", dependencies=[Depends(verify_internal_api_key)]
)
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
    컴포넌트의 워크플로 서빙 배포 상태를 업데이트합니다.

    **중요**: 이 API는 Kubeflow Pipeline(또는 파이프라인 내 경량 컴포넌트)에서만 호출되는 내부 API입니다.
    프론트엔드나 외부 클라이언트에서는 사용하지 않아야 합니다.

    MODEL 배포(KServe·Ollama·REMOTE 알림 등)가 끝나면 파이프라인이 이 API를 호출해
    `model_workflow_deployments` 상태를 갱신하고, 콜백 로직에 따라 워크플로를 ACTIVE로 전환할 수 있습니다.

    ## Path Parameters
    - **workflow_id** (str): 워크플로우 ID
    - **component_id** (str): 컴포넌트 UUID

    ## Request Body
    - **service_name** (str, required): 식별용 서비스/리소스 이름
    - **service_hostname** (str, required): KServe 라우팅용 호스트명(없으면 빈 문자열 가능)
    - **model_name** (str, required): 배포 레코드에 반영할 모델명(정제명 등)
    - **status** (str, required): `"deployed"` | `"failed"` 등 파이프라인에서 전달하는 상태 문자열
    - **internal_url** (str, optional): Ollama ClusterIP URL, REMOTE 베이스 URL 등
    - **error_message** (str, optional): 배포 실패 시 메시지

    ## Response
    - **message** (str): 업데이트 결과 메시지
    - **deployment_info** (dict): 배포 정보
        - service_name (str): 서비스 이름
        - service_hostname (str): 서비스 호스트명
        - model_name (str): 모델 이름
        - status (str): DB상 배포 상태 (`DEPLOYED`/`FAILED` 등 enum 값)
        - deployed_at (str, optional): 배포 완료 시각

    ## Notes
    - `X-Internal-API-Key` 헤더(설정 시)로만 호출 가능
    - KServe·Ollama·REMOTE noop 경로가 동일 엔드포인트를 사용할 수 있음

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
        deployment = ModelWorkflowDeploymentService.update_deployment_status(
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


def _get_workflow_execution_order(workflow: Workflow) -> List[WorkflowComponent]:
    """위상 정렬(Kahn's algorithm) 기반 실행 순서 결정"""
    component_map = {c.id: c for c in workflow.components}

    in_degree = {c.id: 0 for c in workflow.components}
    graph: Dict[str, List[str]] = {}
    for conn in workflow.component_connections:
        graph.setdefault(conn.source_component_id, []).append(conn.target_component_id)
        in_degree[conn.target_component_id] = in_degree.get(conn.target_component_id, 0) + 1

    queue = deque([cid for cid, deg in in_degree.items() if deg == 0])
    execution_order: List[WorkflowComponent] = []

    while queue:
        current_id = queue.popleft()
        component = component_map.get(current_id)
        if component and component.type in (ComponentType.KNOWLEDGE_BASE, ComponentType.MODEL):
            execution_order.append(component)

        for next_id in graph.get(current_id, []):
            in_degree[next_id] -= 1
            if in_degree[next_id] == 0:
                queue.append(next_id)

    return execution_order


# ============= Config Defaults & Effective Config =============

DEFAULT_LLM_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
}

DEFAULT_KB_CONFIG = {
    "top_k": 3,
}

CONFIG_TO_OLLAMA_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "num_predict",
}


def _get_effective_config(component: WorkflowComponent) -> dict:
    """실행 시 config 기본값 병합"""
    if component.type == ComponentType.MODEL:
        defaults = DEFAULT_LLM_CONFIG.copy()
    elif component.type == ComponentType.KNOWLEDGE_BASE:
        defaults = DEFAULT_KB_CONFIG.copy()
    else:
        return {}

    if component.config:
        defaults.update(component.config)
    return defaults


# ============= Workflow Validation Helper Functions =============


@dataclass
class ValidationCheckResult:
    rule: str
    passed: bool
    message: Optional[str] = None


def _has_cycle(definition: WorkflowDefinition) -> bool:
    """순환 참조 검증 (DFS)"""
    graph: Dict[str, List[str]] = {}
    for conn in definition.connections:
        graph.setdefault(conn.source_ref_id, []).append(conn.target_ref_id)

    visited: set = set()
    rec_stack: set = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.discard(node)
        return False

    for node in {c.ref_id for c in definition.components}:
        if node not in visited:
            if dfs(node):
                return True
    return False


def _validate_workflow_definition_checks(
    db: Session,
    definition: WorkflowDefinition,
) -> List[ValidationCheckResult]:
    """모든 검증 규칙을 순회하며 항목별 결과를 수집"""
    results: List[ValidationCheckResult] = []
    component_map = {c.ref_id: c for c in definition.components}

    # 0. START / END 각 1개 필수
    start_count = sum(1 for c in definition.components if c.type == ComponentType.START)
    end_count = sum(1 for c in definition.components if c.type == ComponentType.END)
    se_errors = []
    if start_count != 1:
        se_errors.append(f"START 컴포넌트는 정확히 1개여야 합니다. (현재 {start_count}개)")
    if end_count != 1:
        se_errors.append(f"END 컴포넌트는 정확히 1개여야 합니다. (현재 {end_count}개)")
    results.append(
        ValidationCheckResult(
            rule="single_start_end",
            passed=len(se_errors) == 0,
            message="; ".join(se_errors) if se_errors else None,
        )
    )

    # 1. ref_id 유효성
    ref_id_errors = []
    for conn in definition.connections:
        if conn.source_ref_id not in component_map:
            ref_id_errors.append(conn.source_ref_id)
        if conn.target_ref_id not in component_map:
            ref_id_errors.append(conn.target_ref_id)
    results.append(
        ValidationCheckResult(
            rule="ref_id_validity",
            passed=len(ref_id_errors) == 0,
            message=f"유효하지 않은 ref_id: {', '.join(ref_id_errors)}" if ref_id_errors else None,
        )
    )

    # 2. 자기 자신 연결
    self_conns = [c for c in definition.connections if c.source_ref_id == c.target_ref_id]
    results.append(
        ValidationCheckResult(
            rule="no_self_connection",
            passed=len(self_conns) == 0,
            message="컴포넌트가 자기 자신에게 연결될 수 없습니다." if self_conns else None,
        )
    )

    # 3. 순환 참조
    has_cycle = _has_cycle(definition)
    results.append(
        ValidationCheckResult(
            rule="no_cycle",
            passed=not has_cycle,
            message="순환 참조가 감지되었습니다." if has_cycle else None,
        )
    )

    # 4. OD/LLM/BFM 상호 배타 — 세 타입은 서로 혼합 불가
    #    (EMBEDDING 은 RAG 구성에서 LLM 과 공존하므로 배타 집합에서 제외)
    model_types = set()
    for comp in definition.components:
        if comp.type == ComponentType.MODEL and comp.model_id:
            model = ModelService.get(db, comp.model_id)
            if model and model.type_info:
                model_types.add(model.type_info.name)
    has_od = ModelTypeEnum.ODM.value in model_types
    has_llm = ModelTypeEnum.LLM.value in model_types
    has_bfm = ModelTypeEnum.BFM.value in model_types
    exclusive_present = [
        name
        for name, present in (
            (ModelTypeEnum.ODM.value, has_od),
            (ModelTypeEnum.LLM.value, has_llm),
            (ModelTypeEnum.BFM.value, has_bfm),
        )
        if present
    ]
    results.append(
        ValidationCheckResult(
            rule="no_incompatible_model_type_mix",
            passed=len(exclusive_present) <= 1,
            message=(
                f"하나의 워크플로우에 서로 혼합할 수 없는 모델 타입이 함께 있습니다: "
                f"{', '.join(exclusive_present)} (OD/LLM/BFM 은 상호 배타적입니다)."
                if len(exclusive_present) > 1
                else None
            ),
        )
    )

    # 5. OD+KB 공존
    has_kb = any(c.type == ComponentType.KNOWLEDGE_BASE for c in definition.components)
    results.append(
        ValidationCheckResult(
            rule="no_od_with_kb",
            passed=not (has_od and has_kb),
            message=(
                "ML 워크플로우(OD 모델)에는 KNOWLEDGE_BASE 컴포넌트를 포함할 수 없습니다."
                if (has_od and has_kb)
                else None
            ),
        )
    )

    # 5b. BFM + KB 공존 불가 (BFM 은 생체분자 서열 모델이라 RAG/KB 를 쓰지 않음; has_bfm 은 위 4번에서 계산)
    results.append(
        ValidationCheckResult(
            rule="no_bfm_with_kb",
            passed=not (has_bfm and has_kb),
            message=(
                "BFM 모델 워크플로우에는 KNOWLEDGE_BASE 컴포넌트를 포함할 수 없습니다."
                if (has_bfm and has_kb)
                else None
            ),
        )
    )

    # 5c. BFM + prompt 공존 불가 (BFM 은 프롬프트를 쓰지 않음; prompt_id 는 MODEL 컴포넌트의 필드)
    has_prompt = any(getattr(c, "prompt_id", None) is not None for c in definition.components)
    results.append(
        ValidationCheckResult(
            rule="no_bfm_with_prompt",
            passed=not (has_bfm and has_prompt),
            message=(
                "BFM 모델 워크플로우에는 프롬프트(prompt_id)를 사용할 수 없습니다."
                if (has_bfm and has_prompt)
                else None
            ),
        )
    )

    # 6. MODEL 수 제한
    model_count = sum(1 for c in definition.components if c.type == ComponentType.MODEL)
    results.append(
        ValidationCheckResult(
            rule="model_count_limit",
            passed=model_count <= 3,
            message=f"MODEL 컴포넌트는 최대 3개까지 허용됩니다. (현재 {model_count}개)" if model_count > 3 else None,
        )
    )

    # 7. 연결 타입 규칙 (KB→KB, END→*, *→START)
    conn_type_errors = []
    for conn in definition.connections:
        src = component_map.get(conn.source_ref_id)
        tgt = component_map.get(conn.target_ref_id)
        if not src or not tgt:
            continue
        if src.type == ComponentType.KNOWLEDGE_BASE and tgt.type == ComponentType.KNOWLEDGE_BASE:
            conn_type_errors.append("KNOWLEDGE_BASE → KNOWLEDGE_BASE 연결은 허용되지 않습니다.")
        if src.type == ComponentType.END:
            conn_type_errors.append("END 컴포넌트에서 나가는 연결은 허용되지 않습니다.")
        if tgt.type == ComponentType.START:
            conn_type_errors.append("START 컴포넌트로 들어오는 연결은 허용되지 않습니다.")
    results.append(
        ValidationCheckResult(
            rule="connection_type_rules",
            passed=len(conn_type_errors) == 0,
            message="; ".join(conn_type_errors) if conn_type_errors else None,
        )
    )

    # 8. 컴포넌트 필수 속성 검증
    required_field_errors = []
    for comp in definition.components:
        if comp.type == ComponentType.MODEL and not comp.model_id:
            required_field_errors.append(f"{comp.ref_id}: MODEL 컴포넌트에는 model_id가 필수입니다.")
        if comp.type == ComponentType.KNOWLEDGE_BASE and not comp.knowledge_base_id:
            required_field_errors.append(f"{comp.ref_id}: KNOWLEDGE_BASE 컴포넌트에는 knowledge_base_id가 필수입니다.")
    results.append(
        ValidationCheckResult(
            rule="required_fields",
            passed=len(required_field_errors) == 0,
            message="; ".join(required_field_errors) if required_field_errors else None,
        )
    )

    # 9. max_tokens vs 모델 context length 검증
    max_tokens_errors = []
    for comp in definition.components:
        if comp.type != ComponentType.MODEL or not comp.model_id:
            continue
        max_tokens_val = (comp.config or {}).get("max_tokens")
        if max_tokens_val is None:
            continue
        model = ModelService.get(db, comp.model_id)
        if not model or not model.repo_id:
            continue
        predefined = PREDEFINED_MODEL_CONFIGS.get(model.repo_id)
        if not predefined:
            continue
        ctx_len = predefined.get("max_context_length")
        if ctx_len and int(max_tokens_val) > ctx_len:
            max_tokens_errors.append(
                f"{comp.ref_id}: max_tokens({int(max_tokens_val)})가 "
                f"{model.repo_id}의 context length({ctx_len})를 초과합니다."
            )
    results.append(
        ValidationCheckResult(
            rule="max_tokens_within_context_length",
            passed=len(max_tokens_errors) == 0,
            message="; ".join(max_tokens_errors) if max_tokens_errors else None,
        )
    )

    # 10. config 키·범위 검증 (config_validation) — 요청 스키마(pydantic)가 아닌 규칙 엔진에서 평가한다.
    # 생성/수정 시엔 _validate_workflow_definition_or_raise 가 400 으로, 검증 API 에선 200 응답의
    # passed=false 항목으로 일관 보고한다. (MODEL: temperature/top_p 0.0~1.0, max_tokens 1~4096;
    # KNOWLEDGE_BASE: top_k 1~10; START/END: config 미사용)
    config_errors = []
    for comp in definition.components:
        cfg = comp.config
        if comp.type in (ComponentType.START, ComponentType.END):
            if cfg:
                config_errors.append(f"'{comp.name}': START/END 컴포넌트는 config 를 사용하지 않습니다")
            continue
        if not cfg:
            continue
        if comp.type == ComponentType.MODEL:
            for key in cfg:
                if key not in {"temperature", "top_p", "max_tokens"}:
                    config_errors.append(f"'{comp.name}': MODEL config 에 허용되지 않는 키: {key}")
            for k in ("temperature", "top_p"):
                if k in cfg:
                    try:
                        fv = float(cfg[k])
                    except (TypeError, ValueError):
                        config_errors.append(f"'{comp.name}': {k} 는 숫자여야 합니다")
                    else:
                        if not (0.0 <= fv <= 1.0):
                            config_errors.append(f"'{comp.name}': {k} 는 0.0~1.0 범위여야 합니다")
            if "max_tokens" in cfg:
                try:
                    iv = int(cfg["max_tokens"])
                except (TypeError, ValueError):
                    config_errors.append(f"'{comp.name}': max_tokens 는 정수여야 합니다")
                else:
                    if not (1 <= iv <= 4096):
                        config_errors.append(f"'{comp.name}': max_tokens 는 1~4096 범위여야 합니다")
        elif comp.type == ComponentType.KNOWLEDGE_BASE:
            for key in cfg:
                if key != "top_k":
                    config_errors.append(f"'{comp.name}': KNOWLEDGE_BASE config 에 허용되지 않는 키: {key}")
            if "top_k" in cfg:
                try:
                    iv = int(cfg["top_k"])
                except (TypeError, ValueError):
                    config_errors.append(f"'{comp.name}': top_k 는 정수여야 합니다")
                else:
                    if not (1 <= iv <= 10):
                        config_errors.append(f"'{comp.name}': top_k 는 1~10 범위여야 합니다")
    results.append(
        ValidationCheckResult(
            rule="config_validation",
            passed=len(config_errors) == 0,
            message="; ".join(config_errors) if config_errors else None,
        )
    )

    # 11. 경로당 컴포넌트 수 제한 (§4.2) — MODEL ≤ 2/경로, KB ≤ 1/경로
    path_limit_errors = []
    graph_fwd: Dict[str, List[str]] = {}
    for conn in definition.connections:
        graph_fwd.setdefault(conn.source_ref_id, []).append(conn.target_ref_id)

    start_ref_ids = [c.ref_id for c in definition.components if c.type == ComponentType.START]
    end_ref_ids = {c.ref_id for c in definition.components if c.type == ComponentType.END}

    all_paths: List[List[str]] = []
    for s in start_ref_ids:
        stack = [(s, [s])]
        while stack:
            node, path = stack.pop()
            if node in end_ref_ids:
                all_paths.append(path)
                continue
            for nxt in graph_fwd.get(node, []):
                if nxt not in set(path):
                    stack.append((nxt, path + [nxt]))

    for path in all_paths:
        model_in_path = sum(
            1 for rid in path if component_map.get(rid) and component_map[rid].type == ComponentType.MODEL
        )
        kb_in_path = sum(
            1 for rid in path if component_map.get(rid) and component_map[rid].type == ComponentType.KNOWLEDGE_BASE
        )
        if model_in_path > 2:
            names = [component_map[rid].name for rid in path if rid in component_map]
            path_limit_errors.append(
                f"경로({' → '.join(names)})에 MODEL 컴포넌트가 {model_in_path}개입니다. (최대 2개)"
            )
        if kb_in_path > 1:
            names = [component_map[rid].name for rid in path if rid in component_map]
            path_limit_errors.append(
                f"경로({' → '.join(names)})에 KNOWLEDGE_BASE 컴포넌트가 {kb_in_path}개입니다. (최대 1개)"
            )
    results.append(
        ValidationCheckResult(
            rule="path_component_limit",
            passed=len(path_limit_errors) == 0,
            message="; ".join(path_limit_errors) if path_limit_errors else None,
        )
    )

    # 12. KB로 들어오는 MODEL 연결은 최대 1개 (§4.3)
    kb_model_conn_errors = []
    kb_model_incoming: Dict[str, int] = {}
    for conn in definition.connections:
        src = component_map.get(conn.source_ref_id)
        tgt = component_map.get(conn.target_ref_id)
        if src and tgt and src.type == ComponentType.MODEL and tgt.type == ComponentType.KNOWLEDGE_BASE:
            kb_model_incoming[tgt.ref_id] = kb_model_incoming.get(tgt.ref_id, 0) + 1

    for kb_ref_id, cnt in kb_model_incoming.items():
        if cnt > 1:
            kb_comp = component_map.get(kb_ref_id)
            kb_name = kb_comp.name if kb_comp else kb_ref_id
            kb_model_conn_errors.append(f"KNOWLEDGE_BASE '{kb_name}'에 {cnt}개의 MODEL 연결이 들어옵니다. (최대 1개)")
    results.append(
        ValidationCheckResult(
            rule="kb_incoming_model_limit",
            passed=len(kb_model_conn_errors) == 0,
            message="; ".join(kb_model_conn_errors) if kb_model_conn_errors else None,
        )
    )

    # 13. START 데드 연결 검사 — START 직접 타겟이 동시에 MODEL/KB로부터 연결을 받으면 안 됨
    #     START는 출력을 생성하지 않으므로, 다른 소스와 공존하는 START 연결은 데이터 전달 효과가 없다.
    dead_start_errors = []
    start_targets = set()
    for conn in definition.connections:
        src = component_map.get(conn.source_ref_id)
        if src and src.type == ComponentType.START:
            start_targets.add(conn.target_ref_id)

    for target_ref in start_targets:
        other_source_names = []
        for conn in definition.connections:
            if conn.target_ref_id != target_ref:
                continue
            src = component_map.get(conn.source_ref_id)
            if src and src.type in (ComponentType.MODEL, ComponentType.KNOWLEDGE_BASE):
                other_source_names.append(f"'{src.name}'")

        if other_source_names:
            tgt = component_map.get(target_ref)
            tgt_name = tgt.name if tgt else target_ref
            dead_start_errors.append(
                f"'{tgt_name}'은(는) START에서 직접 연결되면서 동시에 "
                f"{', '.join(other_source_names)}에서도 연결을 받고 있습니다. "
                f"START는 출력을 생성하지 않으므로 이 연결은 데이터 전달 효과가 없습니다."
            )
    results.append(
        ValidationCheckResult(
            rule="no_dead_start_connection",
            passed=len(dead_start_errors) == 0,
            message="; ".join(dead_start_errors) if dead_start_errors else None,
        )
    )

    # 14. 모든 실행 가능 컴포넌트(MODEL, KB)는 START→END 경로에 포함되어야 함
    #     경로에 포함되지 않는 컴포넌트는 고아(orphan)이거나 데드엔드로, 불필요하게 실행될 수 있다.
    reachability_errors = []
    all_path_refs = set()
    for path in all_paths:
        all_path_refs.update(path)
    for comp in definition.components:
        if comp.type in (ComponentType.MODEL, ComponentType.KNOWLEDGE_BASE):
            if comp.ref_id not in all_path_refs:
                reachability_errors.append(
                    f"'{comp.name}'({comp.type.value})이(가) START→END 경로에 포함되지 않습니다. "
                    f"연결이 누락되었거나 불필요한 컴포넌트일 수 있습니다."
                )
    results.append(
        ValidationCheckResult(
            rule="component_reachability",
            passed=len(reachability_errors) == 0,
            message="; ".join(reachability_errors) if reachability_errors else None,
        )
    )

    # 15. 모든 START→END 경로에 최소 1개의 MODEL 포함
    #     MODEL 없는 경로(시작→종료, 시작→KB→종료 등)는 의미 있는 출력을 생성하지 않는다.
    no_model_path_errors = []
    for path in all_paths:
        model_count_in_path = sum(
            1 for rid in path if component_map.get(rid) and component_map[rid].type == ComponentType.MODEL
        )
        if model_count_in_path == 0:
            names = [component_map[rid].name for rid in path if rid in component_map]
            no_model_path_errors.append(
                f"경로({' → '.join(names)})에 MODEL 컴포넌트가 없습니다. " f"각 경로에는 최소 1개의 MODEL이 필요합니다."
            )
    results.append(
        ValidationCheckResult(
            rule="minimum_model_per_path",
            passed=len(no_model_path_errors) == 0,
            message="; ".join(no_model_path_errors) if no_model_path_errors else None,
        )
    )

    # 15b. §7.9 사전정의 서빙 메타(5키) 직접 또는 부모 파생으로 해소 가능한지
    serving_meta_errors: List[str] = []
    for comp in definition.components:
        if comp.type != ComponentType.MODEL or not comp.model_id:
            continue
        db_model = db.get(Model, comp.model_id)
        if not db_model:
            serving_meta_errors.append(f"{comp.ref_id}: model_id={comp.model_id}에 해당하는 모델이 없습니다.")
            continue
        err = serving_meta_validation_error(db, db_model, predefined_configs=PREDEFINED_MODEL_CONFIGS)
        if err:
            serving_meta_errors.append(f"{comp.ref_id}: {err}")
    results.append(
        ValidationCheckResult(
            rule="serving_predefined_meta_complete",
            passed=len(serving_meta_errors) == 0,
            message="; ".join(serving_meta_errors) if serving_meta_errors else None,
        )
    )

    # 16. 중복 연결 검사 — 동일한 source→target 쌍이 2회 이상 등장하면 안 됨
    seen_conns = set()
    duplicate_errors = []
    for conn in definition.connections:
        key = (conn.source_ref_id, conn.target_ref_id)
        if key in seen_conns:
            src = component_map.get(conn.source_ref_id)
            tgt = component_map.get(conn.target_ref_id)
            src_name = src.name if src else conn.source_ref_id
            tgt_name = tgt.name if tgt else conn.target_ref_id
            duplicate_errors.append(f"'{src_name}' → '{tgt_name}' 연결이 중복되었습니다.")
        seen_conns.add(key)
    results.append(
        ValidationCheckResult(
            rule="no_duplicate_connections",
            passed=len(duplicate_errors) == 0,
            message="; ".join(duplicate_errors) if duplicate_errors else None,
        )
    )

    return results


def _validate_workflow_definition_or_raise(db: Session, definition: WorkflowDefinition) -> None:
    """생성/수정 시 사용. 첫 번째 실패 시 즉시 400 응답."""
    if not definition or not definition.components:
        return
    checks = _validate_workflow_definition_checks(db, definition)
    for check in checks:
        if not check.passed:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=check.message)


# ============= Workflow Test Helper Functions =============


def _draw_predictions_on_image(image_bytes: bytes, predictions: List[dict], image_info: dict = None) -> bytes:
    """
    이미지에 예측 결과(bbox, label)를 그려서 반환

    Args:
        image_bytes: 원본 이미지 바이트 데이터
        predictions: 예측 결과 리스트 (각 항목은 score, label, box 포함)
        image_info: 이미지 메타데이터 (model_input_size 등)

    Returns:
        bbox와 label이 그려진 이미지의 바이트 데이터
    """
    try:
        # 이미지 로드
        image = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = image.size
        draw = ImageDraw.Draw(image)

        # 폰트 설정
        try:
            # 시스템 폰트 시도
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        except Exception:
            try:
                # Linux 폰트 시도
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except Exception:
                # 기본 폰트 사용
                font = ImageFont.load_default()

        # 크기 비율 계산 (bbox를 원본 이미지 크기로 스케일링)
        scale_x = 1.0
        scale_y = 1.0

        if image_info:
            model_input_size = image_info.get("model_input_size", {})
            model_height = model_input_size.get("height")
            model_width = model_input_size.get("width")

            if model_height and model_width:
                scale_x = original_width / model_width
                scale_y = original_height / model_height
                logger.info(
                    f"Scaling bbox: model_input={model_width}x{model_height}, "
                    f"original={original_width}x{original_height}, "
                    f"scale=({scale_x: .2f}, {scale_y: .2f})"
                )

        # 각 prediction에 대해 bbox와 label 그리기
        for pred in predictions:
            score = pred.get("score", 0)
            label = pred.get("label", "unknown")
            box = pred.get("box", [])

            # box는 [xmin, ymin, xmax, ymax] 형식
            if len(box) >= 4:
                # box가 리스트의 리스트인 경우 ([Array(4)]) 처리
                if isinstance(box[0], list):
                    box = box[0]

                # bbox를 원본 이미지 크기로 스케일링
                xmin = box[0] * scale_x
                ymin = box[1] * scale_y
                xmax = box[2] * scale_x
                ymax = box[3] * scale_y

                # 바운딩 박스 그리기 (빨간색, 두께 3)
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

                # 레이블 텍스트 (label + score)
                text = f"{label}: {score: .2f}"

                # 텍스트 배경 박스
                text_bbox = draw.textbbox((xmin, ymin - 25), text, font=font)
                draw.rectangle(text_bbox, fill="red")

                # 텍스트 그리기 (흰색)
                draw.text((xmin, ymin - 25), text, fill="white", font=font)

        # 이미지를 바이트로 변환
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="JPEG")
        return output_buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to draw predictions on image: {e}")
        # 에러 발생 시 원본 이미지 반환
        return image_bytes


def _validate_rag_workflow(db: Session, workflow: Workflow) -> None:
    """
    RAG 워크플로우 검증 (MODEL 컴포넌트나 KNOWLEDGE_BASE 컴포넌트 중 하나라도 있는지 확인)
    MODEL 컴포넌트가 있으면 모델 타입이 LLM인지 확인

    Args:
        db: 데이터베이스 세션
        workflow: 워크플로우

    Raises:
        HTTPException: RAG 워크플로우가 아닌 경우
    """
    has_knowledge_base = any(c.type == ComponentType.KNOWLEDGE_BASE for c in workflow.components)
    has_llm_model = False

    # MODEL 컴포넌트가 있으면 모델 타입이 LLM인지 확인
    for component in workflow.components:
        if component.type == ComponentType.MODEL and component.model_id:
            model = ModelService.get(db, component.model_id)
            if model and model.type_info and model.type_info.name == ModelTypeEnum.LLM.value:
                has_llm_model = True
                break

    if not has_knowledge_base and not has_llm_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RAG workflow requires at least one KNOWLEDGE_BASE component or LLM MODEL component",
        )


def _validate_ml_workflow(db: Session, workflow: Workflow) -> None:
    """
    ML 워크플로우 검증 (ODM 모델이 있는지 확인, KNOWLEDGE_BASE 컴포넌트가 없어야 함)

    Args:
        db: 데이터베이스 세션
        workflow: 워크플로우

    Raises:
        HTTPException: ML 워크플로우가 아닌 경우
    """
    # KNOWLEDGE_BASE 컴포넌트가 존재하면 안 됨
    knowledge_base_components = [c for c in workflow.components if c.type == ComponentType.KNOWLEDGE_BASE]
    if knowledge_base_components:
        component_names = ", ".join([c.name for c in knowledge_base_components])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ML workflow cannot have KNOWLEDGE_BASE components. Found: {component_names}",
        )

    # ODM 모델이 있는지 확인
    has_odm_model = False

    for component in workflow.components:
        if component.type == ComponentType.MODEL and component.model_id:
            model = ModelService.get(db, component.model_id)
            if model and model.type_info and model.type_info.name == ModelTypeEnum.ODM.value:
                has_odm_model = True
                break

    if not has_odm_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ML workflow requires at least one ODM (Object Detection Model) component",
        )


def _get_incoming_outputs(
    component: WorkflowComponent,
    workflow: Workflow,
    outputs: Dict[str, dict],
) -> dict:
    """컴포넌트에 연결된 이전 컴포넌트들의 출력을 타입별로 수집.

    다중 소스가 같은 타입으로 합류할 경우, 각 소스 컴포넌트 이름을 포함한
    구분자를 추가하여 후속 컴포넌트가 출처를 식별할 수 있도록 한다.
    """
    component_map = {c.id: c for c in workflow.components}

    kb_entries = []
    model_entries = []

    for conn in workflow.component_connections:
        if conn.target_component_id != component.id:
            continue
        source_output = outputs.get(conn.source_component_id)
        if not source_output:
            continue

        source_comp = component_map.get(conn.source_component_id)
        source_name = source_comp.name if source_comp else conn.source_component_id

        if source_output["type"] == "kb":
            kb_entries.append((source_name, source_output["text"]))
        elif source_output["type"] == "model":
            model_entries.append((source_name, source_output["text"]))

    incoming: Dict[str, str] = {}

    if kb_entries:
        if len(kb_entries) > 1:
            parts = [f"[{name} 검색결과]\n{text}" for name, text in kb_entries]
            incoming["kb_output"] = "\n\n".join(parts)
        else:
            incoming["kb_output"] = kb_entries[0][1]

    if model_entries:
        if len(model_entries) > 1:
            parts = [f"[{name} 응답결과]\n{text}" for name, text in model_entries]
            incoming["model_output"] = "\n\n".join(parts)
        else:
            incoming["model_output"] = model_entries[0][1]

    return incoming


async def _execute_workflow_graph(
    db: Session,
    workflow: Workflow,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    service_id: Optional[str] = None,
    current_user: Optional[UserSchema] = None,
) -> tuple[List[ComponentTestResult], List[str]]:
    """범용 그래프 실행 엔진 — 위상 정렬 순서로 컴포넌트를 실행하고 incoming outputs를 전달"""
    execution_order = _get_workflow_execution_order(workflow)

    if not execution_order:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No executable components found in workflow.",
        )

    outputs: Dict[str, dict] = {}
    results: List[ComponentTestResult] = []
    execution_order_ids: List[str] = []

    for component in execution_order:
        execution_order_ids.append(component.id)
        incoming = _get_incoming_outputs(component, workflow, outputs)
        config = _get_effective_config(component)

        if component.type == ComponentType.KNOWLEDGE_BASE:
            query = incoming.get("model_output") or text
            search_result_text, component_result = await _execute_knowledge_base_search(
                db, component, query, top_k=config.get("top_k", 3)
            )
            if search_result_text:
                outputs[component.id] = {"type": "kb", "text": search_result_text}
            results.append(component_result)

        elif component.type == ComponentType.MODEL:
            model = ModelService.get(db, component.model_id) if component.model_id else None
            is_od = model and model.type_info and model.type_info.name == ModelTypeEnum.ODM.value

            if is_od:
                component_result = await _execute_odm_inference(
                    db, workflow.id, component, image_base64, service_id, current_user
                )
                results.append(component_result)
            else:
                input_text = incoming.get("model_output") or text
                kb_context = incoming.get("kb_output")
                component_result = await _execute_llm_inference(
                    db, workflow.id, component, input_text, kb_context, service_id, current_user, config=config
                )
                if isinstance(component_result, ModelComponentTestResult):
                    if isinstance(component_result.result, ModelLLMTestResult):
                        outputs[component.id] = {
                            "type": "model",
                            "text": component_result.result.response,
                        }
                results.append(component_result)

    return results, execution_order_ids


async def _execute_knowledge_base_search(
    db: Session, component: WorkflowComponent, query: str, top_k: int = 3
) -> tuple[Optional[str], ComponentTestResult]:
    """
    지식베이스 검색 실행

    Args:
        db: 데이터베이스 세션
        component: 지식베이스 컴포넌트
        query: 검색 쿼리
        top_k: 검색 결과 상위 N건

    Returns:
        (search_result_text, component_result) 튜플
    """
    try:
        search_result = KnowledgeBaseService.search(
            db, knowledge_base_id=component.knowledge_base_id, query=query, top_k=top_k
        )

        # 검색 결과를 문자열로 변환
        search_result_strings = []
        for idx, result_item in enumerate(search_result.results):
            search_result_strings.append(f"[참고자료{idx + 1}]\n{result_item.text}")

        search_result_text = "\n\n".join(search_result_strings)

        component_result = KnowledgeBaseComponentTestResult(
            component_id=component.id,
            component_name=component.name,
            component_type="KNOWLEDGE_BASE",
            model_type="embedding",
            result=KnowledgeBaseTestResult(
                search_result=search_result_text,
                total=search_result.total,
                search_method=search_result.search_method,
            ),
        )

        return search_result_text, component_result

    except Exception as e:
        logger.error(f"Knowledge Base search failed for component {component.id}: {e}")
        error_result = ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="KNOWLEDGE_BASE",
            model_type="embedding",
            error=str(e),
        )
        return None, error_result


def _build_llm_chat_messages_for_component(
    db: Session,
    component: WorkflowComponent,
    text: str,
    search_text: Optional[str],
) -> List[Dict[str, Any]]:
    """Ollama·REMOTE 공통: 프롬프트·참고자료·user 메시지 구성."""
    messages: List[Dict[str, Any]] = []
    if component.prompt_id:
        prompt = prompt_repository.get_with_variables(db, component.prompt_id)
        if prompt:
            has_context_variable = False
            if prompt.prompt_variables:
                for var in prompt.prompt_variables:
                    if var.name == PromptVariableType.CONTEXT.value:
                        has_context_variable = True
                        break
            prompt_content = prompt.content
            if has_context_variable and search_text:
                prompt_content = prompt_content.replace("{context}", search_text)
                prompt_content = prompt_content.replace("{{context}}", search_text)
                messages.append({"role": "system", "content": prompt_content})
            else:
                messages.append({"role": "system", "content": prompt_content})
                if search_text:
                    messages.append({"role": "system", "content": f"[참고자료]\n{search_text}"})
    elif search_text:
        messages.append({"role": "system", "content": f"[참고자료]\n{search_text}"})
    messages.append({"role": "user", "content": text})
    return messages


async def _execute_llm_inference(
    db: Session,
    workflow_id: str,
    component: WorkflowComponent,
    text: str,
    search_text: Optional[str],
    service_id: Optional[str],
    current_user: UserSchema,
    config: Optional[dict] = None,
) -> ComponentTestResult:
    """
    LLM 모델 추론 실행

    Args:
        db: 데이터베이스 세션
        workflow_id: 워크플로우 ID
        component: 모델 컴포넌트
        text: 입력 텍스트
        search_text: 지식베이스 검색 결과 텍스트
        service_id: 서비스 ID
        current_user: 현재 사용자
        config: 실행 설정 (temperature, top_p, max_tokens)

    Returns:
        컴포넌트 테스트 결과
    """
    # 모델 정보 가져오기
    model_type_name = None
    is_ollama_model = False
    model_repo_id = None

    if component.model_id:
        model = ModelService.get(db, component.model_id)
        if model:
            model_type_name = model.type_info.name if model.type_info else None
            # Ollama 모델 확인
            if (
                hasattr(model, "provider_info")
                and model.provider_info
                and model.provider_info.name.lower() == ModelProviderEnum.OLLAMA.value.lower()
            ):
                if (
                    hasattr(model, "format_info")
                    and model.format_info
                    and model.format_info.name.lower() == ModelFormatEnum.GGUF.value.lower()
                ):
                    is_ollama_model = True
                    model_repo_id = model.repo_id

    # 배포 검증
    is_ready, error_msg, deployment = ModelWorkflowDeploymentService.validate_deployment_ready(
        db, workflow_id, component.id
    )

    if not is_ready:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=error_msg,
        )

    # 입력 검증
    if not text:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="LLM model requires 'text' parameter",
        )

    # 요청 시작 시간 기록
    start_time = time.time()

    try:
        if deployment.deployment_type == WorkflowServingDeploymentType.REMOTE:
            base_url = (deployment.remote_api_url or deployment.internal_url or "").strip().rstrip("/")
            if not base_url:
                return ComponentTestErrorResult(
                    component_id=component.id,
                    component_name=component.name,
                    component_type="MODEL",
                    model_type=model_type_name,
                    error="REMOTE 배포에 remote_api_url 또는 internal_url이 없습니다.",
                )
            messages = _build_llm_chat_messages_for_component(db, component, text, search_text)
            out_text, err = await remote_chat_completion_async(
                base_url, deployment.model_name, messages, timeout_sec=300.0
            )
            if err:
                return ComponentTestErrorResult(
                    component_id=component.id,
                    component_name=component.name,
                    component_type="MODEL",
                    model_type=model_type_name,
                    error=err,
                )
            response_time_ms = (time.time() - start_time) * 1000
            component_result = ModelComponentTestResult(
                component_id=component.id,
                component_name=component.name,
                component_type="MODEL",
                model_type=model_type_name or ModelTypeEnum.LLM.value,
                result=ModelLLMTestResult(
                    response=out_text or "",
                    full_response=None,
                ),
            )
            if service_id:
                try:
                    ServiceMonitoringService.record_inference_request(
                        db=db,
                        service_id=service_id,
                        workflow_id=workflow_id,
                        user_id=current_user.username,
                        response_time_ms=response_time_ms,
                        is_success=True,
                    )
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed to record monitoring data: {e}")
                    db.rollback()
            return component_result

        if not is_ollama_model:
            return ComponentTestErrorResult(
                component_id=component.id,
                component_name=component.name,
                component_type="MODEL",
                model_type=model_type_name,
                error="LLM model must be an Ollama model",
            )

        # Ollama 모델 처리
        if not deployment.internal_url:
            return ComponentTestErrorResult(
                component_id=component.id,
                component_name=component.name,
                component_type="MODEL",
                model_type=model_type_name,
                error="Internal URL not available for Ollama model deployment",
            )

        ollama_url = deployment.internal_url.rstrip("/")
        url = f"{ollama_url}/api/chat"

        messages = _build_llm_chat_messages_for_component(db, component, text, search_text)

        ollama_options = {}
        if config:
            for config_key, ollama_key in CONFIG_TO_OLLAMA_OPTIONS.items():
                if config_key in config:
                    ollama_options[ollama_key] = config[config_key]

        data = {
            "model": model_repo_id,
            "messages": messages,
            "stream": False,
            "keep_alive": "525600m",
        }
        if ollama_options:
            data["options"] = ollama_options
        headers = {"Content-Type": "application/json"}
        kf_manager = KubeflowManager()
        cookies = kf_manager.auth_session.session_cookie_dict if hasattr(kf_manager, "auth_session") else {}

        # 요청 실행 (비동기 httpx 사용)
        # LLM 모델의 경우 모델 로딩 시간과 추론 시간을 고려하여 타임아웃을 300초(5분)로 설정
        # 큰 모델(Qwen3-30B 등)의 경우 로딩에 30초 이상 소요될 수 있음
        async with httpx.AsyncClient(timeout=300.0, cookies=cookies) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result_data = response.json()

        response_time_ms = (time.time() - start_time) * 1000

        # 결과 처리
        ollama_message = result_data.get("message", {})
        ollama_content = ollama_message.get("content", "") if isinstance(ollama_message, dict) else ""

        component_result = ModelComponentTestResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name or ModelTypeEnum.LLM.value,
            result=ModelLLMTestResult(
                response=ollama_content,
                full_response=result_data,
            ),
        )

        # Ollama 응답의 토큰 사용량 = 입력(prompt_eval_count) + 출력(eval_count)
        token_usage = int(result_data.get("prompt_eval_count") or 0) + int(result_data.get("eval_count") or 0)

        # 모니터링 데이터 기록
        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.username,
                    response_time_ms=response_time_ms,
                    is_success=True,
                    token_usage=token_usage,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        return component_result

    except Exception as e:
        logger.error(f"LLM inference failed for component {component.id}: {e}")
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=str(e),
        )


async def _execute_odm_inference(
    db: Session,
    workflow_id: str,
    component: WorkflowComponent,
    image_base64: str,
    service_id: Optional[str],
    current_user: UserSchema,
) -> ComponentTestResult:
    """
    ODM 모델 추론 실행

    Args:
        db: 데이터베이스 세션
        workflow_id: 워크플로우 ID
        component: 모델 컴포넌트
        image_base64: base64로 인코딩된 이미지 문자열
        service_id: 서비스 ID
        current_user: 현재 사용자

    Returns:
        컴포넌트 테스트 결과
    """
    # 모델 정보 가져오기
    model_type_name = None

    if component.model_id:
        model = ModelService.get(db, component.model_id)
        if model:
            model_type_name = model.type_info.name if model.type_info else None

    # 배포 검증
    is_ready, error_msg, deployment = ModelWorkflowDeploymentService.validate_deployment_ready(
        db, workflow_id, component.id
    )

    if not is_ready:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=error_msg,
        )

    if deployment.deployment_type == WorkflowServingDeploymentType.REMOTE:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="REMOTE 배포 유형은 §6 Remote LLM API 연동 전까지 ODM 테스트를 지원하지 않습니다.",
        )

    # 입력 검증
    if not image_base64:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="ODM model requires 'image' parameter",
        )

    # 요청 시작 시간 기록
    start_time = time.time()

    try:
        # KServe 모델 처리
        # internal_url이 있으면 우선 사용, 없으면 게이트웨이 URL 사용
        service_hostname = deployment.service_hostname
        model_name = deployment.model_name

        if deployment.internal_url:
            # internal_url 사용 (클러스터 내부 접근)
            base_url = deployment.internal_url.rstrip("/")
            url = f"{base_url}/v2/models/{model_name}/infer"
            # internal_url 사용 시 Host 헤더는 필요 없음 (직접 접근)
            headers = {"Content-Type": "application/json"}
        else:
            # 게이트웨이 URL 사용 (Istio Gateway 경유)
            infer_svc_url = (settings.KSERVE_GATEWAY_URL or "").strip().rstrip("/")
            if not infer_svc_url:
                return ComponentTestErrorResult(
                    component_id=component.id,
                    component_name=component.name,
                    component_type="MODEL",
                    model_type=model_type_name,
                    error="KServe ODM 테스트에 필요한 internal_url 또는 KSERVE_GATEWAY_URL 설정이 없습니다.",
                )
            url = f"{infer_svc_url}/v2/models/{model_name}/infer"
            # 게이트웨이 사용 시 Host 헤더 필요 (Istio 라우팅용)
            headers = {"Content-Type": "application/json", "Host": service_hostname}

        payload = {"image": image_base64}
        data = {"inputs": [{"name": "INPUT_1", "shape": [1], "datatype": "BYTES", "data": [payload]}]}
        kf_manager = KubeflowManager()
        cookies = kf_manager.auth_session.session_cookie_dict if hasattr(kf_manager, "auth_session") else {}

        # 요청 실행 (비동기 httpx 사용)
        async with httpx.AsyncClient(timeout=30.0, cookies=cookies) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result_data = response.json()

        response_time_ms = (time.time() - start_time) * 1000

        # KServe 응답 처리
        outputs = result_data.get("outputs", [])
        if outputs and len(outputs) > 0:
            prediction_data = outputs[0].get("data", [])
            if prediction_data and len(prediction_data) > 0:
                response_data = prediction_data[0]

                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except Exception:
                        pass

                if isinstance(response_data, dict):
                    predictions = response_data.get("predictions", response_data)
                    image_info = response_data.get("image_info", {})

                    component_result = ModelComponentTestResult(
                        component_id=component.id,
                        component_name=component.name,
                        component_type="MODEL",
                        model_type=model_type_name or ModelTypeEnum.ODM.value,
                        result=ModelODMTestResult(
                            predictions=predictions if isinstance(predictions, list) else [predictions],
                            image_info=image_info if image_info else None,
                        ),
                    )

                    # 모니터링 데이터 기록
                    if service_id:
                        try:
                            ServiceMonitoringService.record_inference_request(
                                db=db,
                                service_id=service_id,
                                workflow_id=workflow_id,
                                user_id=current_user.username,
                                response_time_ms=response_time_ms,
                                is_success=True,
                            )
                            db.commit()
                        except Exception as e:
                            logger.error(f"Failed to record monitoring data: {e}")
                            db.rollback()

                    return component_result
                else:
                    component_result = ModelComponentTestResult(
                        component_id=component.id,
                        component_name=component.name,
                        component_type="MODEL",
                        model_type=model_type_name or ModelTypeEnum.ODM.value,
                        result=ModelODMTestResult(
                            predictions=[response_data] if not isinstance(response_data, list) else response_data,
                            image_info=None,
                        ),
                    )

                    # 모니터링 데이터 기록
                    if service_id:
                        try:
                            ServiceMonitoringService.record_inference_request(
                                db=db,
                                service_id=service_id,
                                workflow_id=workflow_id,
                                user_id=current_user.username,
                                response_time_ms=response_time_ms,
                                is_success=True,
                            )
                            db.commit()
                        except Exception as e:
                            logger.error(f"Failed to record monitoring data: {e}")
                            db.rollback()

                    return component_result
        else:
            # 예상치 못한 응답 형식
            component_result = ModelComponentTestResult(
                component_id=component.id,
                component_name=component.name,
                component_type="MODEL",
                model_type=model_type_name or ModelTypeEnum.ODM.value,
                result=ModelODMTestResult(
                    predictions=[result_data] if not isinstance(result_data, list) else result_data,
                    image_info=None,
                ),
            )
            return component_result

    except Exception as e:
        logger.error(f"ODM inference failed for component {component.id}: {e}")
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=str(e),
        )


async def _execute_plm_inference(
    db: Session,
    workflow_id: str,
    component: WorkflowComponent,
    epitope: str,
    cdr3b: str,
    service_id,
    current_user: UserSchema,
) -> ComponentTestResult:
    """protein-classification(ESM2) 모델 추론 실행. 단백질 서열 {epitope, cdr3b} 를 KServe predictor(transformers)로 보낸다."""
    model_type_name = None
    if component.model_id:
        model = ModelService.get(db, component.model_id)
        if model and model.type_info:
            model_type_name = model.type_info.name

    is_ready, error_msg, deployment = ModelWorkflowDeploymentService.validate_deployment_ready(
        db, workflow_id, component.id
    )
    if not is_ready:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=error_msg,
        )
    if deployment.deployment_type == WorkflowServingDeploymentType.REMOTE:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="REMOTE 배포 유형은 protein-classification 추론을 지원하지 않습니다.",
        )
    if not epitope or not cdr3b:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="protein-classification model requires non-empty 'epitope' and 'cdr3b'",
        )

    start_time = time.time()
    try:
        service_hostname = deployment.service_hostname
        model_name = deployment.model_name
        if deployment.internal_url:
            base_url = deployment.internal_url.rstrip("/")
            url = f"{base_url}/v2/models/{model_name}/infer"
            headers = {"Content-Type": "application/json"}
        else:
            infer_svc_url = (settings.KSERVE_GATEWAY_URL or "").strip().rstrip("/")
            if not infer_svc_url:
                return ComponentTestErrorResult(
                    component_id=component.id,
                    component_name=component.name,
                    component_type="MODEL",
                    model_type=model_type_name,
                    error="KServe protein-classification 테스트에 필요한 internal_url 또는 KSERVE_GATEWAY_URL 설정이 없습니다.",
                )
            url = f"{infer_svc_url}/v2/models/{model_name}/infer"
            headers = {"Content-Type": "application/json", "Host": service_hostname}

        # predictor 의 transformers 경로는 inputs[0].data[0] 의 dict 를 그대로 받는다.
        payload = {"epitope": epitope, "cdr3b": cdr3b}
        data = {"inputs": [{"name": "INPUT_1", "shape": [1], "datatype": "BYTES", "data": [payload]}]}
        kf_manager = KubeflowManager()
        cookies = kf_manager.auth_session.session_cookie_dict if hasattr(kf_manager, "auth_session") else {}

        async with httpx.AsyncClient(timeout=30.0, cookies=cookies) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result_data = response.json()

        response_time_ms = (time.time() - start_time) * 1000

        predictions = []
        input_info = None
        outputs = result_data.get("outputs", [])
        if outputs and outputs[0].get("data"):
            response_data = outputs[0]["data"][0]
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except Exception:
                    pass
            if isinstance(response_data, dict):
                preds = response_data.get("predictions", response_data)
                predictions = preds if isinstance(preds, list) else [preds]
                input_info = response_data.get("input_info")
            else:
                predictions = response_data if isinstance(response_data, list) else [response_data]

        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.username,
                    response_time_ms=response_time_ms,
                    is_success=True,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        return ModelComponentTestResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name or ModelTypeEnum.BFM.value,
            task=ModelTaskType.PROTEIN_CLASSIFICATION.value,
            result=ModelProteinClassificationTestResult(predictions=predictions, input_info=input_info),
        )

    except Exception as e:
        logger.error(f"protein-classification inference failed for component {component.id}: {e}")
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=str(e),
        )


async def _execute_fill_mask_inference(
    db: Session,
    workflow_id: str,
    component: WorkflowComponent,
    sequence: str,
    top_k: int,
    service_id,
    current_user: UserSchema,
) -> ComponentTestResult:
    """fill-mask(base BFM) 모델 추론 실행. 마스크 포함 서열 {sequence, top_k} 를 KServe predictor(transformers)로 보낸다."""
    model_type_name = None
    if component.model_id:
        model = ModelService.get(db, component.model_id)
        if model and model.type_info:
            model_type_name = model.type_info.name

    is_ready, error_msg, deployment = ModelWorkflowDeploymentService.validate_deployment_ready(
        db, workflow_id, component.id
    )
    if not is_ready:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=error_msg,
        )
    if deployment.deployment_type == WorkflowServingDeploymentType.REMOTE:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="REMOTE 배포 유형은 fill-mask 추론을 지원하지 않습니다.",
        )
    if not sequence:
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error="fill-mask model requires non-empty 'sequence'",
        )

    start_time = time.time()
    try:
        service_hostname = deployment.service_hostname
        model_name = deployment.model_name
        if deployment.internal_url:
            base_url = deployment.internal_url.rstrip("/")
            url = f"{base_url}/v2/models/{model_name}/infer"
            headers = {"Content-Type": "application/json"}
        else:
            infer_svc_url = (settings.KSERVE_GATEWAY_URL or "").strip().rstrip("/")
            if not infer_svc_url:
                return ComponentTestErrorResult(
                    component_id=component.id,
                    component_name=component.name,
                    component_type="MODEL",
                    model_type=model_type_name,
                    error="KServe fill-mask 테스트에 필요한 internal_url 또는 KSERVE_GATEWAY_URL 설정이 없습니다.",
                )
            url = f"{infer_svc_url}/v2/models/{model_name}/infer"
            headers = {"Content-Type": "application/json", "Host": service_hostname}

        # predictor 의 transformers 경로는 inputs[0].data[0] 의 dict 를 그대로 받는다.
        payload = {"sequence": sequence, "top_k": top_k}
        data = {"inputs": [{"name": "INPUT_1", "shape": [1], "datatype": "BYTES", "data": [payload]}]}
        kf_manager = KubeflowManager()
        cookies = kf_manager.auth_session.session_cookie_dict if hasattr(kf_manager, "auth_session") else {}

        async with httpx.AsyncClient(timeout=30.0, cookies=cookies) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result_data = response.json()

        response_time_ms = (time.time() - start_time) * 1000

        predictions = []
        input_info = None
        outputs = result_data.get("outputs", [])
        if outputs and outputs[0].get("data"):
            response_data = outputs[0]["data"][0]
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except Exception:
                    pass
            if isinstance(response_data, dict):
                preds = response_data.get("predictions", response_data)
                predictions = preds if isinstance(preds, list) else [preds]
                input_info = response_data.get("input_info")
            else:
                predictions = response_data if isinstance(response_data, list) else [response_data]

        if service_id:
            try:
                ServiceMonitoringService.record_inference_request(
                    db=db,
                    service_id=service_id,
                    workflow_id=workflow_id,
                    user_id=current_user.username,
                    response_time_ms=response_time_ms,
                    is_success=True,
                )
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record monitoring data: {e}")
                db.rollback()

        return ModelComponentTestResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name or ModelTypeEnum.BFM.value,
            task=ModelTaskType.FILL_MASK.value,
            result=ModelFillMaskTestResult(predictions=predictions, input_info=input_info),
        )

    except Exception as e:
        logger.error(f"fill-mask inference failed for component {component.id}: {e}")
        return ComponentTestErrorResult(
            component_id=component.id,
            component_name=component.name,
            component_type="MODEL",
            model_type=model_type_name,
            error=str(e),
        )


# ============= Workflow Test Endpoints =============


@router.post("/{workflow_id}/test/rag", response_model=WorkflowRAGTestResponse)
async def test_rag_workflow(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    text: str = Form(..., description="검색 쿼리 및 LLM 입력 텍스트"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    RAG 워크플로우 테스트

    Knowledge Base와 LLM(MODEL) 컴포넌트를 사용하는 RAG 워크플로우를 테스트합니다.
    지식베이스가 있으면 검색 후 결과를 LLM에 전달하고, 없으면 LLM만 실행합니다.
    LLM 배포 유형은 Ollama·REMOTE(원격 API 스텁) 등에 따라 `_execute_llm_inference`에서 분기됩니다.

    ## Path Parameters
    - **workflow_id** (str): 테스트할 워크플로우 UUID

    ## Request Body (Form Data)
    - **text** (str, required): 검색 쿼리 및 LLM 입력 텍스트

    ## Response (WorkflowTestResponse)
    - **workflow_id** (str): 워크플로우 UUID
    - **execution_order** (List[str]): 실행된 컴포넌트 ID 순서 (워크플로우 실행 순서)
    - **results** (List[ComponentTestResult]): 각 컴포넌트 실행 결과 목록
        - 각 항목은 다음 중 하나의 타입:

        **KnowledgeBaseComponentTestResult** (지식베이스 컴포넌트인 경우):
        - **component_id** (str): 컴포넌트 UUID
        - **component_name** (str): 컴포넌트 이름
        - **component_type** (str): "KNOWLEDGE_BASE"
        - **model_type** (str): "embedding"
        - **result** (KnowledgeBaseTestResult): 검색 결과
            - **search_result** (str): 검색 결과 문자열 (LLM 프롬프트에 사용 가능한 형식)
            - **total** (int): 검색 결과 총 개수
            - **search_method** (str): 검색 방법 (예: "similarity", "keyword")

        **ModelComponentTestResult** (LLM 모델 컴포넌트인 경우):
        - **component_id** (str): 컴포넌트 UUID
        - **component_name** (str): 컴포넌트 이름
        - **component_type** (str): "MODEL"
        - **model_type** (str): "LLM"
        - **result** (ModelLLMTestResult): LLM 추론 결과
            - **response** (str): LLM 응답 텍스트
            - **full_response** (dict, optional): 업스트림 LLM 전체 응답(Ollama 등·REMOTE 스텁은 null일 수 있음)

        **ComponentTestErrorResult** (오류 발생 시):
        - **component_id** (str): 컴포넌트 UUID
        - **component_name** (str): 컴포넌트 이름
        - **component_type** (str): "KNOWLEDGE_BASE" 또는 "MODEL"
        - **model_type** (str, optional): 모델 타입 (오류 발생 시 null 가능)
        - **error** (str): 오류 메시지

    - **final_result** (str, optional): 최종 결과 문자열
        - 우선순위: 마지막 LLM MODEL 컴포넌트의 응답 > 마지막 KNOWLEDGE_BASE 컴포넌트의 검색 결과
        - MODEL 컴포넌트가 있으면: LLM 응답 텍스트 (response)
        - MODEL 컴포넌트가 없고 KNOWLEDGE_BASE만 있으면: 검색 결과 문자열 (search_result)
        - 상세 정보는 results 배열의 각 컴포넌트 결과에서 확인 가능

    ## Notes
    - 워크플로우는 배포되어 있어야 함 (ACTIVE 상태)
    - 워크플로우에 최소 하나의 LLM MODEL 컴포넌트 또는 KNOWLEDGE_BASE 컴포넌트가 있어야 함
    - ODM(Object Detection) 전용 ML 그래프는 `POST .../test/ml`을 사용
    - Knowledge Base 컴포넌트는 선택 사항 (있으면 검색 후 결과를 LLM에 전달)
    - 지식베이스 검색 결과는 자동으로 LLM 모델의 context 파라미터로 전달됨
    - prompt_id가 설정된 경우:
        - prompt에 context 변수가 있으면: prompt의 {context} 또는 {{context}} 위치에 자동 치환
        - prompt에 context 변수가 없어도: [참고자료] 태그와 함께 별도의 system 메시지로 추가
    - prompt_id가 없는 경우: [참고자료] 태그와 함께 system 메시지로 추가
    - 각 컴포넌트의 실행 결과는 results 배열에 순서대로 포함됨
    - final_result는 최종 결과 문자열만 포함 (LLM 응답 또는 검색 결과)
    - 상세 정보 (total, search_method, full_response 등)는 results 배열의 각 컴포넌트 결과에서 확인 가능

    ## Errors
    - 400: 잘못된 요청 (RAG 워크플로우가 아님, 필수 파라미터 누락 등)
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 503: 모델 서비스가 준비되지 않음
    - 500: 서버 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # 워크플로우가 배포되어 있는지 확인
    if workflow.status != WorkflowStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow must be ACTIVE to test. Current status: {workflow.status.value}",
        )

    # RAG 워크플로우 검증
    _validate_rag_workflow(db, workflow)

    results, execution_order_ids = await _execute_workflow_graph(
        db,
        workflow,
        text=text,
        service_id=workflow.service_id,
        current_user=current_user,
    )

    # 최종 결과 문자열 추출 (마지막 MODEL > KNOWLEDGE_BASE)
    final_result = None
    for result in reversed(results):
        if isinstance(result, ModelComponentTestResult):
            if isinstance(result.result, ModelLLMTestResult):
                final_result = result.result.response
                break
        elif isinstance(result, KnowledgeBaseComponentTestResult):
            if final_result is None:
                final_result = result.result.search_result

    return WorkflowRAGTestResponse(
        workflow_id=workflow_id,
        execution_order=execution_order_ids,
        results=results,
        final_result=final_result,
    )


@router.post("/{workflow_id}/test/ml", response_model=WorkflowMLTestResponse)
async def test_ml_workflow(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    image: UploadFile = File(..., description="이미지 파일 (ODM 추론용)"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    ML 워크플로우 테스트

    Object Detection Model을 사용하는 ML 워크플로우를 테스트합니다.

    ## Path Parameters
    - **workflow_id** (str): 테스트할 워크플로우 UUID

    ## Request Body (Form Data)
    - **image** (file, required): 이미지 파일 (ODM 추론용)
        - 지원 형식: JPEG, PNG, GIF, WebP
        - Base64로 인코딩되어 서버로 전송

    ## Response (WorkflowTestResponse)
    - **workflow_id** (str): 워크플로우 UUID
    - **execution_order** (List[str]): 실행된 컴포넌트 ID 순서 (워크플로우 실행 순서)
    - **results** (List[ComponentTestResult]): 각 컴포넌트 실행 결과 목록
        - 각 항목은 다음 중 하나의 타입:

        **ModelComponentTestResult** (ODM 모델 컴포넌트인 경우):
        - **component_id** (str): 컴포넌트 UUID
        - **component_name** (str): 컴포넌트 이름
        - **component_type** (str): "MODEL"
        - **model_type** (str): "ODM"
        - **result** (ModelODMTestResult): ODM 추론 결과
            - **predictions** (List[dict]): 추론 결과 목록
                - 각 항목은 다음 필드를 포함:
                    - **score** (float): 객체 감지 신뢰도 점수 (0.0 ~ 1.0)
                    - **label** (str): 감지된 객체의 레이블 (예: "person", "laptop")
                    - **box** (List[float]): 바운딩 박스 좌표 [x1, y1, x2, y2]
            - **image_info** (dict, optional): 이미지 메타데이터
                - **original_size** (dict): 원본 이미지 크기
                - **model_input_size** (dict): 모델 입력 크기

        **ComponentTestErrorResult** (오류 발생 시):
        - **component_id** (str): 컴포넌트 UUID
        - **component_name** (str): 컴포넌트 이름
        - **component_type** (str): "MODEL"
        - **model_type** (str, optional): "ODM" (오류 발생 시 null 가능)
        - **error** (str): 오류 메시지

    - **final_result** (str, optional): 최종 결과 이미지 (base64 인코딩)
        - 입력 이미지에 마지막 ODM MODEL 컴포넌트의 predictions를 이용해 bbox와 label을 그린 이미지
        - base64로 인코딩된 JPEG 이미지 문자열
        - predictions가 없거나 에러 발생 시 원본 이미지를 base64로 인코딩하여 반환
        - 상세 정보 (predictions, image_info 등)는 results 배열의 각 컴포넌트 결과에서 확인 가능

    ## Notes
    - 워크플로우는 배포되어 있어야 함 (ACTIVE 상태)
    - 워크플로우에 최소 하나의 ODM MODEL 컴포넌트가 있어야 함
    - KNOWLEDGE_BASE 컴포넌트는 포함될 수 없음 (ML 워크플로우는 ODM만 지원)
    - 각 컴포넌트의 실행 결과는 results 배열에 순서대로 포함됨
    - final_result는 입력 이미지에 bbox와 label이 그려진 이미지를 base64로 인코딩한 문자열
    - bbox는 빨간색으로, label은 빨간 배경에 흰색 텍스트로 표시됨
    - 상세 정보 (predictions, image_info 등)는 results 배열의 각 컴포넌트 결과에서 확인 가능
    - 모든 추론 요청은 ServiceMonitoring 테이블에 자동 기록됨 (서비스와 연결된 경우)

    ## Errors
    - 400: 잘못된 요청 (ML 워크플로우가 아님, 필수 파라미터 누락 등)
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 503: 모델 서비스가 준비되지 않음
    - 500: 서버 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # 워크플로우가 배포되어 있는지 확인
    if workflow.status != WorkflowStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow must be ACTIVE to test. Current status: {workflow.status.value}",
        )

    # ML 워크플로우 검증
    _validate_ml_workflow(db, workflow)

    # 이미지 읽기 및 base64 인코딩 (한 번만 수행)
    image_base64 = None
    image_bytes = None
    if image:
        try:
            await image.seek(0)
            image_bytes = await image.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to read image file",
            )

    results, execution_order_ids = await _execute_workflow_graph(
        db,
        workflow,
        image_base64=image_base64,
        service_id=workflow.service_id,
        current_user=current_user,
    )

    # 최종 결과 이미지 생성 (마지막 MODEL 컴포넌트의 predictions를 이용해 이미지에 bbox와 label 그리기)
    final_result = None
    if image_bytes:
        for result in reversed(results):
            if isinstance(result, ModelComponentTestResult):
                # ODM 모델인 경우 predictions를 이용해 이미지에 bbox와 label 그리기
                if isinstance(result.result, ModelODMTestResult):
                    predictions = result.result.predictions
                    image_info = result.result.image_info

                    if predictions:
                        try:
                            # 이미지에 bbox와 label 그리기
                            annotated_image_bytes = _draw_predictions_on_image(
                                image_bytes,
                                predictions if isinstance(predictions, list) else [predictions],
                                image_info if image_info else None,
                            )
                            # base64로 인코딩
                            final_result = base64.b64encode(annotated_image_bytes).decode("utf-8")
                        except Exception as e:
                            logger.error(f"Failed to draw predictions on image: {e}")
                            # 에러 발생 시 원본 이미지를 base64로 인코딩
                            final_result = base64.b64encode(image_bytes).decode("utf-8")
                    else:
                        # predictions가 없으면 원본 이미지를 base64로 인코딩
                        final_result = base64.b64encode(image_bytes).decode("utf-8")
                    break

    return WorkflowMLTestResponse(
        workflow_id=workflow_id,
        execution_order=execution_order_ids,
        results=results,
        final_result=final_result,
    )


@router.post("/{workflow_id}/test/protein-classification", response_model=WorkflowProteinClassificationTestResponse)
async def test_protein_classification_workflow(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    epitope: str = Body(..., embed=True, description="단백질 epitope 서열"),
    cdr3b: str = Body(..., embed=True, description="TCR β-chain CDR3 서열"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    protein-classification(파인튜닝 ESM2/ESMC) 워크플로우 테스트 — 구 `/test/plm` 리네임.

    단백질 서열 분류(TCR-Epitope 결합) 모델을 배포한 워크플로우에 `{epitope, cdr3b}` 를 보내
    이진 분류 추론을 수행한다.

    ## Request Body (JSON)
    - **epitope** (str, required): epitope 서열
    - **cdr3b** (str, required): cdr3b 서열

    ## Response (WorkflowProteinClassificationTestResponse)
    - **results[].result** (ModelProteinClassificationTestResult): `predictions` + `input_info`

    ## Notes
    - 워크플로우는 ACTIVE(배포 완료) 상태여야 한다.
    - **task 가 protein-classification 이 아닌 모델 컴포넌트가 있으면 400** (해당 task 전용 엔드포인트).
    - **base(파인튜닝 안 된, parent_model_id IS NULL) 모델이면 400** — 어댑터가 없어 서빙 불가.

    ## Errors
    - 400: 해당 task 워크플로우가 아님 / base 모델 / ACTIVE 아님 / 필수 입력 누락
    - 401: 미인증 / 404: 워크플로우 없음 / 503: 모델 서비스 미준비 / 500: 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    if workflow.status != WorkflowStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow must be ACTIVE to test. Current status: {workflow.status.value}",
        )

    # MODEL 컴포넌트 수집 — task 가 protein-classification 이 아니면 거부, 하나도 없어도 거부.
    pc_task = ModelTaskType.PROTEIN_CLASSIFICATION.value
    pc_components = []
    for component in workflow.components:
        if component.type == ComponentType.MODEL and component.model_id:
            model = ModelService.get(db, component.model_id)
            task_name = getattr(model, "task", None) if model else None
            if task_name != pc_task:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"protein-classification 전용 추론 엔드포인트입니다. 비-protein-classification 모델 컴포넌트가 "
                    f"포함되어 있습니다: {component.name} (task={task_name})",
                )
            # base(파인튜닝 안 된) 모델은 LoRA 어댑터가 없어 서빙 불가 → 파인튜닝 자식을 서빙해야 함.
            if model.parent_model_id is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"base 모델({component.name})은 서빙할 수 없습니다. 파인튜닝된 자식 모델을 배포하세요.",
                )
            pc_components.append(component)

    if not pc_components:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="protein-classification 워크플로우가 아닙니다 (해당 task 모델 컴포넌트가 없습니다).",
        )

    results = []
    for component in pc_components:
        results.append(
            await _execute_plm_inference(db, workflow.id, component, epitope, cdr3b, workflow.service_id, current_user)
        )

    return WorkflowProteinClassificationTestResponse(
        workflow_id=workflow_id,
        execution_order=[c.id for c in pc_components],
        results=results,
    )


@router.post("/{workflow_id}/test/fill-mask", response_model=WorkflowFillMaskTestResponse)
async def test_fill_mask_workflow(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    sequence: str = Body(..., embed=True, description="마스크(<mask>) 토큰이 포함된 서열"),
    top_k: int = Body(5, embed=True, description="마스크 위치별 반환할 top-k 후보 수"),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    fill-mask(base BFM: ESM2/ESMC/RNA-FM/MoLFormer) 워크플로우 테스트.

    마스크 토큰이 포함된 서열을 배포된 base BFM 모델에 보내 각 마스크 위치의 top-k 토큰 예측을 받는다.
    protein-classification 과 달리 파인튜닝이 불필요하며 base 모델을 그대로 서빙한다.

    ## Request Body (JSON)
    - **sequence** (str, required): 마스크 토큰을 포함한 서열
    - **top_k** (int, optional, 기본 5): 마스크당 반환 후보 수

    ## Response (WorkflowFillMaskTestResponse)
    - **results[].result** (ModelFillMaskTestResult): 마스크 위치별 `predictions` + `input_info`

    ## Notes
    - 워크플로우는 ACTIVE(배포 완료) 상태여야 한다.
    - **task 가 fill-mask 가 아닌 모델 컴포넌트가 있으면 400** (해당 task 전용 엔드포인트).
    - base 모델 서빙이 정상이므로 protein-classification 과 달리 base 가드는 없다.

    ## Errors
    - 400: 해당 task 워크플로우가 아님 / ACTIVE 아님 / 필수 입력 누락
    - 401: 미인증 / 404: 워크플로우 없음 / 503: 모델 서비스 미준비 / 500: 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    if workflow.status != WorkflowStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow must be ACTIVE to test. Current status: {workflow.status.value}",
        )

    # MODEL 컴포넌트 수집 — task 가 fill-mask 가 아니면 거부, 하나도 없어도 거부.
    fm_task = ModelTaskType.FILL_MASK.value
    fm_components = []
    for component in workflow.components:
        if component.type == ComponentType.MODEL and component.model_id:
            model = ModelService.get(db, component.model_id)
            task_name = getattr(model, "task", None) if model else None
            if task_name != fm_task:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"fill-mask 전용 추론 엔드포인트입니다. 비-fill-mask 모델 컴포넌트가 "
                    f"포함되어 있습니다: {component.name} (task={task_name})",
                )
            fm_components.append(component)

    if not fm_components:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fill-mask 워크플로우가 아닙니다 (해당 task 모델 컴포넌트가 없습니다).",
        )

    results = []
    for component in fm_components:
        results.append(
            await _execute_fill_mask_inference(
                db, workflow.id, component, sequence, top_k, workflow.service_id, current_user
            )
        )

    return WorkflowFillMaskTestResponse(
        workflow_id=workflow_id,
        execution_order=[c.id for c in fm_components],
        results=results,
    )


@router.get("/{workflow_id}/models")
def get_deployed_models(
    *,
    db: Session = SessionDepends,
    workflow_id: str,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우에 배포된 모델 목록 조회

    워크플로우의 `model_workflow_deployments` 레코드를 조회합니다.
    KServe·Ollama·REMOTE 유형별 `internal_url`, `public_url`, `backend_api_url`(§2.6) 등을 포함합니다.

    ## Path Parameters
    - **workflow_id** (str): 조회할 워크플로우 UUID

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **backend_api_url** (str|None): 첫 번째 배포 기준 §2.6 요약 URL(없으면 null)
    - **deployed_models** (List[dict]): 배포된 모델 목록
        - workflow_id (str): 소속 워크플로우 ID
        - component_id (str): 컴포넌트 ID
        - component_name (str): 컴포넌트 이름
        - model_id (int): 모델 ID
        - model_name (str): 표시/배포용 모델 이름
        - sanitized_model_name (str): DNS 규칙에 맞게 변환된 모델 이름
        - service_name (str): Kubernetes 리소스 또는 식별용 이름
        - service_hostname (str): KServe 라우팅용 호스트명(없을 수 있음)
        - status (str): 배포 상태 (`DEPLOYING`/`DEPLOYED`/`FAILED`/`DELETED`)
        - internal_url (str|None): 클러스터 내부 URL 또는 REMOTE 베이스 URL
        - deployment_type (str): KSERVE | OLLAMA | REMOTE
        - public_url (str|None): KSERVE+게이트웨이 설정 시에만
        - backend_api_url (str|None): §2.6 — REMOTE는 internal·remote_api 폴백
        - gateway_url (str|None): KServe 게이트웨이 베이스 URL(없으면 null)
        - deployed_at (datetime): 배포 시각
        - deleted_at (datetime): 삭제 시각 (삭제된 경우)
        - error_message (str): 오류 메시지 (실패 시)
        - created_at (datetime): 레코드 생성 시각
        - updated_at (datetime): 레코드 업데이트 시각
    - **total** (int): 배포된 모델 총 개수

    ## Notes
    - 요약 필드 `backend_api_url`(루트)는 첫 배포 기준 §2.6과 동일
    - 컴포넌트 테스트·RAG/ML 테스트는 워크플로가 ACTIVE이고 해당 배포가 DEPLOYED일 때 가능

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    workflow = WorkflowService.get_workflow_by_id(db, workflow_id)

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

    # DB에서 배포된 모델 목록 조회
    deployed_models = ModelWorkflowDeploymentService.get_deployed_models(db, workflow_id, include_component_info=True)

    # backend_api_url — §2.6: 첫 배포의 internal_url 기준
    backend_api_url = None
    if deployed_models:
        first_deployment = deployed_models[0]
        backend_api_url = first_deployment.get("backend_api_url")

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

    배포된 **KServe InferenceService**·**Ollama Deployment/Service** 등 워크플로 서빙 리소스를 정리하는
    Kubeflow 파이프라인을 시작합니다. 워크플로우 레코드 자체는 유지됩니다.

    ## Path Parameters
    - **workflow_id** (str): 정리할 워크플로우 UUID

    ## Response (202 Accepted)
    - **message** (str): 상태 메시지 "Cleanup started"
    - **workflow_id** (str): 워크플로우 UUID
    - **cleanup_run_id** (str): 정리 파이프라인 실행 ID
    - **status** (str): 현재 상태 "cleanup_in_progress"
    - **next_step** (str): 다음 단계 API 안내
        - 형식: "Call /workflows/{workflow_id}/finalize-cleanup to check completion"

    ## Use Cases
    - 비용 절감을 위해 배포된 리소스 정리
    - 오류 발생 후 재배포 준비
    - 워크플로우 구조 변경 전 리소스 정리

    ## Process
    1. 정리용 Kubeflow 파이프라인 시작(라벨 `workflow-id` 기준 클러스터 리소스 삭제)
    2. cleanup_run_id 반환
    3. finalize-cleanup API로 완료 확인
    4. 완료 시 `model_workflow_deployments` 정리 및(조건부) 워크플로 DRAFT 전환

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
            "next_step": f"Call /workflows/{workflow_id}/finalize-cleanup to check completion",
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
    current_user: UserSchema = Depends(get_current_user),
):
    """
    워크플로우 정리 완료 처리

    Kubernetes 클러스터를 직접 조회하여 리소스가 실제로 삭제되었는지 확인하고,
    확인된 경우 워크플로우 상태를 업데이트합니다.
    워크플로우는 삭제되지 않고 리소스만 정리되며, 정리 후 재실행이 가능합니다.

    ## Path Parameters
    - **workflow_id** (str): 정리할 워크플로우 UUID
        - 워크플로우 목록 조회 API(/workflows)에서 확인 가능

    ## Response
    - **workflow_id** (str): 워크플로우 UUID
    - **status** (str): 정리 상태
        - "completed": 정리 완료
            - Kubernetes 리소스가 실제로 삭제되어 확인됨
            - 워크플로우 상태가 업데이트됨 (ERROR → DRAFT)
        - "in_progress": 아직 진행중
            - Kubernetes에 리소스가 아직 존재함
            - 완료될 때까지 대기 후 재호출 필요
        - "failed": 정리 실패
            - Kubernetes 리소스 확인 중 오류 발생
            - error_message에 상세 오류 정보 포함
    - **workflow_updated** (bool): 워크플로우 상태 업데이트 여부
        - true: 워크플로우 상태가 성공적으로 업데이트됨
            - ERROR 상태였던 경우 DRAFT로 변경됨
            - 재실행 가능한 상태로 변경됨
        - false: 워크플로우 상태가 업데이트되지 않음
            - 리소스가 아직 삭제되지 않았거나 실패한 경우
            - 또는 워크플로우가 이미 DRAFT 상태인 경우
    - **message** (str): 상태 메시지
        - 정리 완료: "Cleanup completed and workflow state updated"
        - 진행중: "Resources still exist in Kubernetes, waiting for cleanup"
        - 실패: "Failed to check Kubernetes resources: {error}"

    ## Process
    1. 워크플로우 존재 여부 확인
    2. Kubernetes 클러스터에서 리소스 직접 조회
       - InferenceService 조회
       - Ollama Deployment/Service 조회
    3. 리소스가 모두 삭제된 경우:
       - 워크플로우 상태가 ERROR인 경우 DRAFT로 변경
       - 워크플로 서빙 배포 데이터(`model_workflow_deployments`) 삭제
       - 재실행 가능한 상태로 업데이트
    4. 리소스가 아직 존재하는 경우: 진행중 상태 반환 (재호출 필요)
    5. 확인 중 오류 발생: 실패 상태 반환

    ## Notes
    - 워크플로우는 삭제되지 않고 리소스만 정리됨
    - 정리 완료 후 워크플로우를 재실행할 수 있음
    - Kubernetes 리소스 확인은 즉시 수행됨
    - 리소스가 아직 존재하면 재호출하여 완료 확인 필요
    - ERROR 상태의 워크플로우는 정리 완료 시 DRAFT로 변경됨
    - 이미 DRAFT 상태인 워크플로우는 상태 변경 없음
    - cleanup API 호출 후 이 API를 호출하여 완료 확인 필요

    ## Usage Example
    1. cleanup API 호출하여 정리 파이프라인 시작
    2. 이 API를 호출하여 완료 확인
    3. status가 "completed"이고 workflow_updated가 true면 정리 완료
    4. status가 "in_progress"면 잠시 후 재호출

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 워크플로우를 찾을 수 없음
        - workflow_id가 존재하지 않거나 삭제된 경우
    - 500: 정리 처리 중 오류 발생
        - Kubernetes 리소스 확인 실패 또는 워크플로우 상태 업데이트 실패
    """
    try:
        # 워크플로우 존재 여부 확인
        workflow = WorkflowService.get_workflow_by_id(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found")

        # Kubernetes 클러스터에서 리소스 직접 조회하여 삭제 여부 확인
        try:
            from kubernetes import client
            from kubernetes import config as k8s_config

            # Kubernetes 설정
            try:
                k8s_config.load_incluster_config()
            except Exception:
                # 개발 환경에서는 kubeconfig 사용
                try:
                    k8s_config.load_kube_config()
                except Exception as e:
                    logger.error(f"Failed to load Kubernetes config: {e}")
                    raise Exception(f"Failed to load Kubernetes config: {str(e)}")

            namespace = settings.KUBEFLOW_NAMESPACE
            label_selector = f"workflow-id={workflow_id}"

            resources_exist = False

            # 1. InferenceService 확인
            try:
                api = client.CustomObjectsApi()

                result = api.list_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    label_selector=label_selector,
                )

                services = result.get("items", [])
                if len(services) > 0:
                    logger.info(f"Found {len(services)} InferenceServices still existing for workflow {workflow_id}")
                    resources_exist = True
            except Exception as e:
                logger.warning(f"Error checking InferenceServices: {e}")

            # 2. Ollama Deployment 확인
            if not resources_exist:
                try:
                    apps_v1 = client.AppsV1Api()
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    if len(deployments.items) > 0:
                        logger.info(
                            f"Found {len(deployments.items)} Deployments still existing for workflow {workflow_id}"
                        )
                        resources_exist = True
                except Exception as e:
                    logger.warning(f"Error checking Deployments: {e}")

            # 3. Ollama Service 확인
            if not resources_exist:
                try:
                    core_v1 = client.CoreV1Api()
                    services = core_v1.list_namespaced_service(
                        namespace=namespace,
                        label_selector=label_selector,
                    )

                    if len(services.items) > 0:
                        logger.info(f"Found {len(services.items)} Services still existing for workflow {workflow_id}")
                        resources_exist = True
                except Exception as e:
                    logger.warning(f"Error checking Services: {e}")

            if resources_exist:
                # 리소스가 아직 존재함 - 진행중
                return {
                    "workflow_id": workflow_id,
                    "status": "in_progress",
                    "workflow_updated": False,
                    "message": "Resources still exist in Kubernetes, waiting for cleanup",
                }

            # 리소스가 모두 삭제됨 - 워크플로우 상태 업데이트 및 배포 데이터 삭제
            logger.info(
                f"All resources deleted for workflow {workflow_id}, "
                f"updating workflow state and cleaning up deployments"
            )

            # KServe 리소스가 모두 삭제됐으므로 워크플로우를 DRAFT 로 되돌린다(재배포 가능 상태).
            # ACTIVE(정상 배포본) / ERROR 어느 쪽이든 DRAFT 로 전환한다(설계서 §7-2).
            workflow_updated = False
            if workflow.status != WorkflowStatus.DRAFT:
                workflow.status = WorkflowStatus.DRAFT
                workflow_updated = True

            # 워크플로 서빙 배포 레코드(model_workflow_deployments) 삭제
            deleted_count = ModelWorkflowDeploymentService.delete_workflow_deployments(db, workflow_id)
            logger.info(f"Deleted {deleted_count} deployment records for workflow {workflow_id}")

            db.commit()

            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "workflow_updated": workflow_updated,
                "message": "Cleanup completed and workflow state updated",
            }

        except Exception as k8s_error:
            logger.error(f"Failed to check Kubernetes resources: {k8s_error}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "workflow_updated": False,
                "message": f"Failed to check Kubernetes resources: {str(k8s_error)}",
            }
    except Exception as e:
        logger.error(f"Failed to finalize cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to finalize cleanup: {str(e)}"
        )
