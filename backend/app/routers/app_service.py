"""Application Service API 라우터"""

import logging
from typing import List, Optional

from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, HTTPException, Query, status
from schemas.app_service import (
    ServiceBaseSchema,
    ServiceBriefSchema,
    ServiceCreateRequest,
    ServiceDeployRequest,
    ServiceDeployResponse,
    ServiceDetailSchema,
    ServiceListResponse,
    ServiceUpdateRequest,
)
from schemas.user import UserSchema
from services.app_service import AppServiceService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/services", tags=["Application Services"])


@router.post("", response_model=ServiceBriefSchema, status_code=status.HTTP_201_CREATED)
def create_service(
    *,
    db: Session = SessionDepends,
    service_data: ServiceCreateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    새로운 서비스 생성

    - **name**: 서비스 이름 (필수, 고유값)
    - **description**: 서비스 설명 (선택)
    - **tags**: 서비스 태그 리스트 (선택)
    """
    try:
        service = AppServiceService.create_service(db=db, service_data=service_data, creator_id=current_user.id)

        # Response 모델로 변환
        return ServiceBriefSchema(
            id=service.id,
            name=service.name,
            description=service.description,
            tags=service.tags or [],
            creator_id=service.creator_id,
            status=service.status,
            created_at=service.created_at,
            updated_at=service.updated_at,
            creator=current_user,
            workflow_count=0,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create service: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create service")


@router.get("", response_model=ServiceListResponse)
def list_services(
    *,
    db: Session = SessionDepends,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    creator_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    서비스 목록 조회 (대표정보)

    - **skip**: 건너뛸 항목 수
    - **limit**: 조회할 최대 항목 수
    - **creator_id**: 생성자 ID 필터
    - **status**: 상태 필터 (draft, active, inactive, deprecated)
    """
    services = AppServiceService.get_services(db=db, skip=skip, limit=limit, creator_id=creator_id, status=status)

    items = []
    for service in services:
        items.append(
            ServiceBriefSchema(
                id=service.id,
                name=service.name,
                description=service.description,
                tags=service.tags or [],
                creator_id=service.creator_id,
                status=service.status,
                created_at=service.created_at,
                updated_at=service.updated_at,
                creator=service.creator,
                workflow_count=len(service.workflows),
            )
        )

    return ServiceListResponse(total=len(items), items=items)


@router.get("/{service_id}", response_model=ServiceDetailSchema)
def get_service_detail(
    *, db: Session = SessionDepends, service_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    서비스 상세정보 조회

    연결된 워크플로우, 모델정보, 모니터링 정보를 모두 포함
    """
    service = AppServiceService.get_service_by_id(db, service_id)

    if not service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

    # 모니터링 데이터 조회
    monitoring_data = AppServiceService.get_service_monitoring_data(db, service_id)

    return ServiceDetailSchema(
        id=service.id,
        name=service.name,
        description=service.description,
        tags=service.tags or [],
        creator_id=service.creator_id,
        status=service.status,
        created_at=service.created_at,
        updated_at=service.updated_at,
        creator=service.creator,
        workflows=service.workflows,
        monitoring_data=monitoring_data,
        kserve_endpoint=service.kserve_endpoint,
        public_url=service.public_url,
        backend_api_url=service.backend_api_url,
    )


@router.put("/{service_id}", response_model=ServiceBriefSchema)
def update_service(
    *,
    db: Session = SessionDepends,
    service_id: str,
    service_data: ServiceUpdateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    서비스 정보 수정

    - **name**: 서비스 이름
    - **description**: 서비스 설명
    - **tags**: 서비스 태그
    - **status**: 서비스 상태
    """
    service = AppServiceService.update_service(db=db, service_id=service_id, service_data=service_data)

    if not service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

    return ServiceBriefSchema(
        id=service.id,
        name=service.name,
        description=service.description,
        tags=service.tags or [],
        creator_id=service.creator_id,
        status=service.status,
        created_at=service.created_at,
        updated_at=service.updated_at,
        creator=service.creator,
        workflow_count=len(service.workflows),
    )


@router.delete("/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_service(
    *, db: Session = SessionDepends, service_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    서비스 삭제

    연결된 워크플로우가 있는 경우 연결만 해제되고 워크플로우는 삭제되지 않음
    """
    success = AppServiceService.delete_service(db, service_id)

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

    return None


@router.post("/{service_id}/deploy", response_model=ServiceDeployResponse)
async def deploy_service(
    *,
    db: Session = SessionDepends,
    service_id: str,
    deploy_data: ServiceDeployRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    서비스를 KServe에 배포

    서비스와 연결된 모든 워크플로우를 Kubeflow 파이프라인으로 배포하고
    KServe 엔드포인트를 생성
    """
    # TODO: KServe 배포 로직 구현
    # 1. 서비스 상태를 ACTIVE로 변경
    # 2. 연결된 워크플로우들을 Kubeflow 파이프라인으로 배포
    # 3. KServe InferenceService 생성
    # 4. 엔드포인트 URL 저장

    service = AppServiceService.get_service_by_id(db, service_id)
    if not service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

    # 임시 응답 (실제 구현 필요)
    return ServiceDeployResponse(
        service_id=service_id,
        status="deploying",
        kserve_endpoint="https://kserve.example.com/v1/models/service-" + str(service_id),
        public_url="https://app.example.com/service/" + str(service_id),
        message="Service deployment initiated",
    )
