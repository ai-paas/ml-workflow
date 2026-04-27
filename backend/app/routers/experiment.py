from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, HTTPException, Query, status
from schemas.experiment import (
    ExperimentBaseSchema,
    ExperimentCreateRequest,
    ExperimentDetailResponse,
    ExperimentInternalUpdateRequest,
    ExperimentListResponse,
    ExperimentMetricsSchema,
    ExperimentReadSchema,
    ExperimentUpdateRequest,
    HyperparameterReadSchema,
)
from schemas.user import UserSchema
from services.experiment import ExperimentService
from sqlalchemy.orm import Session
from utils.authentication import get_current_user, verify_internal_api_key

router = APIRouter(prefix="/experiments", tags=["Experiment"])


@router.get("", response_model=list[ExperimentListResponse])
async def list_experiments(
    db: Session = SessionDepends,
    page: int | None = Query(None, ge=1),
    page_size: int | None = Query(None, ge=1),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    실험 목록 조회

    모든 학습 실험의 목록을 반환합니다.
    모델 등록 상태, 경과 시간, 종료 시간 등 요약 정보를 포함합니다.

    ## 응답 원소(`ExperimentListResponse`) — status·등록 관련
    - **status** (str, 필수): 학습 실험 진행 상태. 예: `CREATED`, `RUNNING`, `COMPLETED`, `FAILED`
      (리모델링 설계서에는 완료를 `FINISHED`로 적는 경우가 있으나, DB/동기화에서는 `COMPLETED`가 쓰임)
    - **registration_status** (str, 필수): 모델 등록 파이프라인/`registered_model` 기준
      `NOT_REQUESTED` | `PIPELINE_SUBMITTED` | `SUCCESS` | `FAILED`
    - **registered_model_id** (int, 선택): `registration_status` 가 `SUCCESS`일 때만 의미 있음(등록된 모델 ID)
    - **elapsed_time** (int, 선택): 경과 시간(초)
    - **end_time** (datetime, 선택): 학습 종료 시각(메트릭 기반)
    """
    try:
        skip = ((page - 1) * page_size) if (page and page_size) else 0
        limit = page_size if page_size else 100

        experiments = ExperimentService.get_multi(db, skip=skip, limit=limit)

        results = []
        for exp in experiments:
            metrics = getattr(exp, "metrics", None)
            ref_model = exp.reference_model
            ds = exp.dataset

            results.append(
                ExperimentListResponse(
                    id=exp.id,
                    name=exp.name,
                    description=exp.description,
                    reference_model_id=exp.reference_model_id,
                    dataset_id=exp.dataset_id,
                    status=exp.status,
                    registration_status=getattr(exp, "registration_status", "NOT_REQUESTED") or "NOT_REQUESTED",
                    registered_model_id=getattr(exp, "registered_model_id", None),
                    elapsed_time=metrics.elapsed_time if metrics else None,
                    end_time=metrics.end_time if metrics else None,
                    reference_model={"id": ref_model.id, "name": ref_model.name} if ref_model else None,
                    dataset={"id": ds.id, "name": ds.name} if ds else None,
                    created_at=exp.created_at,
                    updated_at=exp.updated_at,
                )
            )
        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/{experiment_id}", response_model=ExperimentReadSchema)
async def update_experiment(
    db: Session = SessionDepends,
    *,
    experiment_id: int,
    experiment_update_request: ExperimentUpdateRequest,
    current_user: UserSchema = Depends(get_current_user),
):
    """
    실험 정보 수정

    학습이 진행 중이거나 완료된 실험의 이름과 설명을 수정합니다.
    학습 결과의 무결성을 위해 모델, 데이터셋, 하이퍼파라미터 등은 수정할 수 없습니다.

    ## Path Parameters
    - **experiment_id** (int): 수정할 실험 ID

    ## Request Body (ExperimentUpdateRequest)
    - **name** (str, optional): 새로운 실험 이름
        - 실험을 식별하기 위한 이름
        - 생략 시 기존 값 유지
    - **description** (str, optional): 새로운 실험 설명
        - 실험에 대한 상세 설명
        - null 값으로 설명 제거 가능
        - 생략 시 기존 값 유지

    ## Response (ExperimentReadSchema)
    - **id** (int): 실험 ID
    - **name** (str): 실험 이름
    - **description** (str): 실험 설명
    - **reference_model_id** (int): 참조 모델 ID
        - 학습에 사용된 원본 모델 ID
    - **dataset_id** (int): 데이터셋 ID
        - 학습에 사용된 데이터셋 ID
    - **kubeflow_run_id** (str, optional): Kubeflow 파이프라인 실행 ID
        - Kubeflow에서 학습 파이프라인 실행 시 생성된 ID
    - **mlflow_run_id** (str, optional): MLflow 실행 ID
        - MLflow에서 학습 실행 시 생성된 ID
    - **status** (str): 실험 상태
        - 예: `CREATED`, `RUNNING`, `COMPLETED`, `FAILED` (리모델링 설계서의 `FINISHED`는 구현에서 `COMPLETED`로 저장될 수 있음)
    - **reference_model** (ModelReadSchema): 참조 모델 상세 정보
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
        - parent_model (ModelReadParentSchema, optional): 부모 모델 정보
        - child_models (List[ModelReadChildSchema], optional): 자식 모델 목록
        - created_at (datetime): 모델 생성 시각
        - updated_at (datetime): 모델 수정 시각
    - **dataset** (DatasetReadSchema): 데이터셋 상세 정보
        - id (int): 데이터셋 ID
        - name (str): 데이터셋 이름
        - dataset_registry (DatasetRegistryReadSchema): 데이터셋 레지스트리 정보
            - id (int): 레지스트리 ID
            - artifact_path (str): 아티팩트 경로
            - uri (str): 데이터셋 URI
            - dataset_id (int): 데이터셋 ID
            - created_at (datetime): 생성 시각
            - updated_at (datetime): 수정 시각
        - created_at (datetime): 데이터셋 생성 시각
        - updated_at (datetime): 데이터셋 수정 시각
    - **hyperparameters** (List[HyperparameterReadSchema]): 하이퍼파라미터 목록
        - id (int): 하이퍼파라미터 ID
        - value (str): 하이퍼파라미터 값
        - experiment_id (int): 소속 실험 ID
        - hyperparameter_type_id (int): 하이퍼파라미터 타입 ID
        - hyperparameter_type (HyperparameterTypeReadSchema): 하이퍼파라미터 타입 정보
            - id (int): 타입 ID
            - param_name (str): 파라미터 이름
            - param_type (str): 파라미터 타입
            - default_value (str): 기본값
    - **created_at** (datetime): 실험 생성 시각
    - **updated_at** (datetime): 실험 수정 시각

    ## Notes
    - 학습이 진행 중이거나 완료된 실험에서는 name과 description만 수정 가능
    - reference_model_id, dataset_id, hyperparameters 등은 학습 결과의 무결성을 위해 수정 불가
    - 제공된 필드만 업데이트되며, 생략된 필드는 기존 값이 유지됨

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 실험을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        return ExperimentService().update(db, experiment_id=experiment_id, obj_in=experiment_update_request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    db: Session = SessionDepends, *, experiment_id: int, current_user: UserSchema = Depends(get_current_user)
):
    """
    실험 상세정보 조회

    특정 실험의 상세 정보를 조회합니다.
    목록 필드에 더해 학습 메트릭, 등록 상태, 메시지 정보를 통합 제공합니다.

    ## 응답 — status·등록·메시지(문서 §7.2과 정합)
    - **status** (str): `CREATED`, `RUNNING`, `COMPLETED`, `FAILED` 등(완료는 주로 `COMPLETED`)
    - **registration_status** (str): `NOT_REQUESTED` | `PIPELINE_SUBMITTED` | `SUCCESS` | `FAILED`
    - **registered_model_id** (int, 선택): 등록 성공 시에만 값
    - **train_msg** (str, 선택): 학습 단계 안내(예: 실패/완료 메시지; 메트릭 폴링이 갱신)
    - **model_register_msg** (str, 선택): 모델 등록 파이프라인 결과 안내
    - **mlflow_run_id** (str, 선택): MLflow run(상세·디버깅용; 설계에 따라 내부용으로 둘 수 있음)
    - 메트릭: **max_epoch**, **current_epoch**, **loss**, **loss_history**, **average_precision**, **accuracy**,
      **precision**, **recall** 등(일부는 UI·기획용으로 null일 수 있음)
    """
    try:
        experiment = ExperimentService().get(db, pk=experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail=f"실험 ID '{experiment_id}'을 찾을 수 없습니다.")

        metrics = getattr(experiment, "metrics", None)
        if metrics:
            metrics_data = ExperimentMetricsSchema.from_orm_model(metrics)
        else:
            metrics_data = ExperimentMetricsSchema()

        ref_model = experiment.reference_model
        ds = experiment.dataset

        hp_list = []
        for hp in experiment.hyperparameters:
            hp_list.append(HyperparameterReadSchema.model_validate(hp).model_dump())

        return ExperimentDetailResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            reference_model_id=experiment.reference_model_id,
            dataset_id=experiment.dataset_id,
            status=experiment.status,
            mlflow_run_id=experiment.mlflow_run_id,
            registration_status=getattr(experiment, "registration_status", "NOT_REQUESTED") or "NOT_REQUESTED",
            registered_model_id=getattr(experiment, "registered_model_id", None),
            model_register_msg=getattr(experiment, "model_register_msg", None),
            train_msg=getattr(experiment, "train_msg", None),
            reference_model={"id": ref_model.id, "name": ref_model.name} if ref_model else None,
            dataset={"id": ds.id, "name": ds.name} if ds else None,
            hyperparameters=hp_list,
            created_at=experiment.created_at,
            updated_at=experiment.updated_at,
            **metrics_data.model_dump(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{experiment_id}")
async def delete_experiment_internal(
    db: Session = SessionDepends, *, experiment_id: int, current_user: UserSchema = Depends(get_current_user)
):
    """
    실험을 삭제합니다.
    MLflow artifacts와 S3 object도 함께 삭제됩니다.

    ## Path Parameters
    - **experiment_id** (int): 삭제할 실험 ID

    ## Response
    - **message** (str): 삭제 성공 메시지

    ## Errors
    - 404: 실험을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        ExperimentService().delete(db, experiment_id=experiment_id)
        return {"message": f"실험 {experiment_id}가 성공적으로 삭제되었습니다."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch(
    "/{experiment_id}/internal-access",
    response_model=ExperimentReadSchema,
    dependencies=[Depends(verify_internal_api_key)],
)
async def update_experiment_internal(
    db: Session = SessionDepends,
    *,
    experiment_id: int,
    experiment_update_request: ExperimentInternalUpdateRequest,
):
    """
    내부 통신 전용 실험 정보 수정 API

    시스템 내부 통신에서 사용하는 API로, status, mlflow_run_id, kubeflow_run_id를 수정할 수 있습니다.
    인증이 필요합니다.

    ## Path Parameters
    - **experiment_id** (int): 수정할 실험 ID

    ## Request Body (ExperimentInternalUpdateRequest)
    - **status** (str, optional): `experiment` 행의 학습 상태. 예: `CREATED`, `RUNNING`, `COMPLETED`, `FAILED`
    - **mlflow_run_id** (str, optional): MLflow run ID(학습 컨테이너가 run 생성 후 갱신)
    - **kubeflow_run_id** (str, optional): Kubeflow 학습 파이프라인 run ID(해당되면)
    - **registration_kubeflow_run_id** (str, optional): 모델 **등록**용 Kubeflow run ID(등록 API가 설정)

    ## Response (ExperimentReadSchema)
    - 실험의 전체 정보(참조 모델·데이터셋·하이퍼파라미터·status·run id 등)를 반환합니다.

    ## Errors
    - 401: 인증되지 않은 사용자
    - 404: 실험을 찾을 수 없음
    - 500: 서버 내부 오류
    """
    try:
        return ExperimentService().update_internal(db, experiment_id=experiment_id, obj_in=experiment_update_request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
