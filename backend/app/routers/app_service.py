"""Application Service API 라우터"""

import logging
import math
from typing import List, Optional

from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, HTTPException, Query, status
from schemas.app_service import (
    ServiceBaseSchema,
    ServiceBriefSchema,
    ServiceCreateRequest,
    ServiceDetailSchema,
    ServiceListResponse,
    ServiceResourceUsageResponse,
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

    서비스는 워크플로우를 그룹화하고 모니터링하는 최상위 단위입니다.
    하나의 서비스에 여러 워크플로우를 연결하여 통합 관리할 수 있습니다.

    ## Request Body (ServiceCreateRequest)
    - **name** (str, required): 서비스 이름 (1-255자, 고유값)
    - **description** (str, optional): 서비스에 대한 상세 설명
    - **tags** (List[str], optional): 서비스 분류/검색용 태그 리스트

    ## Response (ServiceBriefSchema)
    - **id** (str): 서비스 고유 ID (UUID)
    - **name** (str): 서비스 이름
    - **description** (str): 서비스 설명
    - **tags** (List[str]): 서비스 태그 목록
    - **creator_id** (int): 서비스 생성자 ID
    - **created_at** (datetime): 생성 시각
    - **updated_at** (datetime): 최종 수정 시각
    - **creator** (UserSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **workflow_count** (int): 연결된 워크플로우 수 (생성 직후는 0)

    ## Errors
    - 400: 이미 존재하는 서비스 이름이거나 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
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
    current_user: UserSchema = Depends(get_current_user),
):
    """
    서비스 목록 조회

    등록된 서비스들의 목록을 페이지네이션하여 조회합니다.
    각 서비스의 기본 정보와 연결된 워크플로우 수를 함께 제공합니다.

    ## Query Parameters
    - **page** (int, optional): 페이지 번호 (1부터 시작)
        - 생략 시: 전체 데이터 조회
        - 최소값: 1
    - **page_size** (int, optional): 페이지당 항목 수
        - 생략 시: 전체 데이터 조회
        - 범위: 1-1000
    - **creator_id** (int, optional): 특정 사용자가 생성한 서비스만 필터링

    ## Response (ServiceListResponse)
    - **total** (int): 조건에 맞는 전체 서비스 수
    - **items** (List[ServiceBriefSchema]): 서비스 목록
        - id (str): 서비스 고유 ID (UUID)
        - name (str): 서비스 이름
        - description (str): 서비스 설명
        - tags (List[str]): 서비스 태그 목록
        - creator_id (int): 생성자 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 최종 수정 시각
        - creator (UserSchema): 생성자 상세 정보
            - id (int): 사용자 ID
            - username (str): 사용자명
            - name (str): 사용자 이름
            - password (str): 비밀번호 (해시된 값)
            - created_at (datetime): 계정 생성 시각
            - updated_at (datetime): 계정 정보 수정 시각
            - created_by (str, optional): 계정 생성자
            - updated_by (str, optional): 계정 정보 수정자
        - workflow_count (int): 해당 서비스에 연결된 워크플로우 개수

    ## Notes
    - page와 page_size를 모두 생략하면 전체 데이터를 조회 (최대 10000개)
    - 페이지네이션 사용 시 total은 필터 조건에 맞는 전체 개수를 반환

    ## Errors
    - 401: 인증되지 않은 사용자
    - 500: 서버 내부 오류
    """
    # 페이지네이션 파라미터가 없는 경우 전체 데이터 조회
    if page is None or page_size is None:
        services = AppServiceService.get_services(db=db, skip=0, limit=10000, creator_id=creator_id)
        items = []
        for service in services:
            items.append(
                ServiceBriefSchema(
                    id=service.id,
                    name=service.name,
                    description=service.description,
                    tags=service.tags or [],
                    creator_id=service.creator_id,
                    created_at=service.created_at,
                    updated_at=service.updated_at,
                    creator=service.creator,
                    workflow_count=len(service.workflows),
                )
            )
        return ServiceListResponse(total=len(items), items=items)

    # 페이지네이션 적용
    total_count = AppServiceService.count_services(db=db, creator_id=creator_id)
    skip = page_size * (page - 1)

    services = AppServiceService.get_services(db=db, skip=skip, limit=page_size, creator_id=creator_id)

    items = []
    for service in services:
        items.append(
            ServiceBriefSchema(
                id=service.id,
                name=service.name,
                description=service.description,
                tags=service.tags or [],
                creator_id=service.creator_id,
                created_at=service.created_at,
                updated_at=service.updated_at,
                creator=service.creator,
                workflow_count=len(service.workflows),
            )
        )

    return ServiceListResponse(total=total_count, items=items)


@router.get("/{service_id}", response_model=ServiceDetailSchema)
def get_service_detail(
    *, db: Session = SessionDepends, service_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    서비스 상세정보 조회

    특정 서비스의 상세 정보를 조회합니다.
    연결된 모든 워크플로우 정보와 최근 1시간의 모니터링 메트릭을 포함합니다.

    ## Path Parameters
    - **service_id** (str): 조회할 서비스의 고유 ID (UUID)

    ## Response (ServiceDetailSchema)
    - **id** (str): 서비스 고유 ID (UUID)
    - **name** (str): 서비스 이름
    - **description** (str): 서비스 설명
    - **tags** (List[str]): 서비스 태그 목록
    - **creator_id** (int): 생성자 ID
    - **created_at** (datetime): 생성 시각
    - **updated_at** (datetime): 최종 수정 시각
    - **creator** (UserSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **workflows** (List[WorkflowBaseSchema]): 연결된 워크플로우 목록
        - id (str): 워크플로우 ID (UUID)
        - name (str): 워크플로우 이름
        - description (str): 워크플로우 설명
        - status (str): 워크플로우 상태 (DRAFT/ACTIVE/ERROR)
        - is_template (bool): 템플릿 여부
        - template_id (str): 원본 템플릿 ID (템플릿에서 생성된 경우)
        - category (str): 워크플로우 카테고리
        - tags (List[str]): 워크플로우 태그
        - workflow_definition (dict): 워크플로우 정의 (nodes, edges 포함)
        - service_id (str): 연결된 서비스 ID
        - creator_id (int): 생성자 ID
        - kubeflow_run_id (str): Kubeflow 실행 ID
        - created_at (datetime): 생성 시각
        - updated_at (datetime): 수정 시각
    - **monitoring_data** (ServiceMonitoringData): 모니터링 데이터
        - total_metrics (MonitoringMetrics): 전체 서비스 메트릭
            - message_count (int): 최근 1시간 총 메시지 수
            - active_users (int): 최근 1시간 활성 사용자 수
            - token_usage (int): 최근 1시간 토큰 사용량
            - avg_interaction_count (float): 최근 1시간 평균 사용자 상호작용 수
            - response_time_ms (float): 평균 응답 시간(ms)
            - error_count (int): 최근 1시간 오류 수
            - success_rate (float): 최근 1시간 성공률(%)
        - workflow_metrics (List[WorkflowMonitoring]): 워크플로우별 메트릭
            - workflow_id (str): 워크플로우 ID
            - workflow_name (str): 워크플로우 이름
            - metrics (MonitoringMetrics): 해당 워크플로우의 메트릭
            - last_updated (datetime): 마지막 업데이트 시각
        - period_start (datetime): 집계 시작 시간
        - period_end (datetime): 집계 종료 시간

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 서비스를 찾을 수 없음
    - 500: 서버 내부 오류
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
        created_at=service.created_at,
        updated_at=service.updated_at,
        creator=service.creator,
        workflows=service.workflows,
        monitoring_data=monitoring_data,
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

    기존 서비스의 정보를 부분적으로 또는 전체적으로 수정합니다.
    제공된 필드만 업데이트되며, 생략된 필드는 기존 값이 유지됩니다.

    ## Path Parameters
    - **service_id** (str): 수정할 서비스의 고유 ID (UUID)

    ## Request Body (ServiceUpdateRequest)
    - **name** (str, optional): 새로운 서비스 이름 (1-255자)
        - 다른 서비스와 중복 불가
    - **description** (str, optional): 새로운 서비스 설명
        - null 값으로 설명 제거 가능
    - **tags** (List[str], optional): 새로운 태그 목록
        - 기존 태그를 완전히 대체
        - 빈 리스트로 모든 태그 제거 가능

    ## Response (ServiceBriefSchema)
    - **id** (str): 서비스 고유 ID (UUID)
    - **name** (str): 수정된 서비스 이름
    - **description** (str): 수정된 서비스 설명
    - **tags** (List[str]): 수정된 태그 목록
    - **creator_id** (int): 생성자 ID (변경 불가)
    - **created_at** (datetime): 생성 시각 (변경 불가)
    - **updated_at** (datetime): 수정 시각 (현재 시각으로 자동 갱신)
    - **creator** (UserSchema): 생성자 정보
        - id (int): 사용자 ID
        - username (str): 사용자명
        - name (str): 사용자 이름
        - password (str): 비밀번호 (해시된 값)
        - created_at (datetime): 계정 생성 시각
        - updated_at (datetime): 계정 정보 수정 시각
        - created_by (str, optional): 계정 생성자
        - updated_by (str, optional): 계정 정보 수정자
    - **workflow_count** (int): 연결된 워크플로우 수

    ## Notes
    - 서비스 이름 변경 시 중복 검사 수행
    - 연결된 워크플로우는 영향받지 않음

    ## Errors
    - 400: 중복된 서비스 이름 또는 유효하지 않은 요청
    - 401: 인증되지 않은 사용자
    - 404: 서비스를 찾을 수 없음
    - 500: 서버 내부 오류
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

    서비스를 삭제합니다. 연결된 워크플로우가 있는 경우 연결만 해제되며,
    워크플로우 자체는 삭제되지 않고 독립적으로 유지됩니다.

    ## Path Parameters
    - **service_id** (str): 삭제할 서비스의 고유 ID (UUID)

    ## Response
    - **Status Code**: 204 No Content (성공 시 응답 본문 없음)

    ## Side Effects
    - 서비스와 연결된 모든 워크플로우의 service_id가 null로 설정됨
    - 서비스 관련 모니터링 데이터는 보존됨 (향후 분석용)
    - 서비스 정보는 데이터베이스에서 완전히 삭제됨

    ## Notes
    - 삭제는 되돌릴 수 없는 작업입니다
    - 워크플로우를 삭제하려면 별도로 워크플로우 삭제 API를 호출해야 함

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 서비스를 찾을 수 없음
    - 500: 서버 내부 오류
    """
    success = AppServiceService.delete_service(db, service_id)

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

    return None


@router.get("/{service_id}/resource-usages", response_model=ServiceResourceUsageResponse)
def get_service_resource_usages(
    *, db: Session = SessionDepends, service_id: str, current_user: UserSchema = Depends(get_current_user)
):
    """
    서비스 리소스 사용량 조회

    서비스에 속한 워크플로우의 배포된 모델들의 리소스 사용량을 조회합니다.
    k8s metrics API를 사용하여 CPU, Memory, GPU 사용량을 가져옵니다.

    ## Path Parameters
    - **service_id** (str): 조회할 서비스의 고유 ID (UUID)

    ## Response (ServiceResourceUsageResponse)
    - **service_id** (str): 서비스 고유 ID (UUID)
    - **service_name** (str): 서비스 이름
    - **deployments** (List[DeploymentResourceUsage]): 배포별 리소스 사용량 목록
        - **deployment_id** (str): KServe 배포 ID
        - **service_name** (str): 서비스 이름
        - **workflow_id** (str): 워크플로우 ID
        - **component_id** (str): 컴포넌트 ID
        - **model_name** (str): 모델 이름
        - **pods** (List[PodResourceUsage]): Pod별 리소스 사용량 목록
            - **pod_name** (str): Pod 이름
            - **namespace** (str): 네임스페이스
            - **deployment_type** (str): 배포 타입 (inferenceservice 또는 service)
            - **resource_usage** (ResourceUsage): 리소스 사용량
                - **cpu_usage_millicores** (float, optional): CPU 사용량 (밀리코어 단위)
                - **cpu_request_millicores** (float, optional): CPU 요청량 (밀리코어 단위)
                - **cpu_limit_millicores** (float, optional): CPU 제한량 (밀리코어 단위)
                - **memory_usage_bytes** (int, optional): 메모리 사용량 (바이트 단위)
                - **memory_request_bytes** (int, optional): 메모리 요청량 (바이트 단위)
                - **memory_limit_bytes** (int, optional): 메모리 제한량 (바이트 단위)
                - **gpu_usage_percent** (float, optional): GPU 사용률 (%)
                - **gpu_memory_usage_bytes** (int, optional): GPU 메모리 사용량 (바이트 단위)
            - **status** (str, optional): Pod 상태
    - **total_cpu_usage_millicores** (float, optional): 전체 CPU 사용량 (밀리코어 단위)
    - **total_memory_usage_bytes** (int, optional): 전체 메모리 사용량 (바이트 단위)
    - **total_gpu_usage_percent** (float, optional): 전체 GPU 사용률 (%)

    ## Notes
    - Metrics Server가 설치되어 있어야 실제 사용량을 조회할 수 있습니다.
    - Metrics Server가 없는 경우 리소스 요청/제한 정보만 반환됩니다.
    - GPU 사용량은 별도의 메트릭 수집기(dcgm-exporter 등)가 필요합니다.

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 서비스를 찾을 수 없음
    - 500: 서버 내부 오류 또는 Kubernetes API 접근 실패
    """
    service = AppServiceService.get_service_by_id(db, service_id)

    if not service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

    try:
        resource_usages = AppServiceService.get_service_resource_usages(db, service_id)
        if not resource_usages:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service {service_id} not found")

        return resource_usages

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resource usages for service {service_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource usages: {str(e)}",
        )
