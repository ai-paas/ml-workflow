from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum

from config.db.enums import ModelFormatEnum, ModelProviderEnum
from config.db.session import SessionLocal
from config.optimization_sources import optimization_task_allowlist_for_repo
from config.settings import get_settings
from core.optimization.optimization_client import OptimizationClient, get_optimization_client
from db.models.model import ModelRegistry
from db.models.model_improvement_task import ModelImprovementTask
from fastapi import HTTPException, status
from repos.model import model_format_repository, model_registry_repository, model_repository
from schemas.model import ModelBaseSchema, ModelRegistryBaseSchema
from schemas.model_improvement import (
    CreateImprovementRequest,
    CreateImprovementResponse,
    ImprovementStatusResponse,
    TaskTypeResponse,
)
from services.model import ModelProviderService, ModelService
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"PENDING", "RUNNING"})


class RemoteTaskStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def _map_optimizer_type(optimizer_type: str | None) -> str:
    m = {"optimize": "optimization", "lightweight": "lightweight"}
    return m.get(optimizer_type or "", optimizer_type or "unknown")


def _determine_remote_task_status(task: dict) -> RemoteTaskStatus:
    progress_status = task.get("progress_status", False)
    model_path_output = task.get("model_path_output")
    if not progress_status:
        return RemoteTaskStatus.RUNNING
    if model_path_output and str(model_path_output).strip():
        return RemoteTaskStatus.SUCCESS
    return RemoteTaskStatus.FAILED


def _to_api_status(remote: RemoteTaskStatus) -> str:
    if remote == RemoteTaskStatus.RUNNING:
        return "RUNNING"
    if remote == RemoteTaskStatus.SUCCESS:
        return "SUCCEEDED"
    return "FAILED"


_STATUS_MESSAGES = {
    "RUNNING": "최적화 작업이 진행 중입니다.",
    "PENDING": "최적화 작업이 큐에 등록되었습니다.",
    "SUCCEEDED": "최적화가 완료되었습니다.",
    "FAILED": "최적화 작업이 실패하였습니다.",
}

_TASK_TYPE_TO_FORMAT_NAME: dict[str, str] = {
    "pruning": ModelFormatEnum.ONNX.value,
    "tensorrt": ModelFormatEnum.TENSORRT.value,
    "openvino": ModelFormatEnum.OPENVINO.value,
    "sklearn-onnx": ModelFormatEnum.ONNX.value,
}


def registry_uri_suffix_after_artifacts(artifact_path: str) -> str:
    """MLflow artifact URI에서 run의 artifacts 디렉터리 이하 상대 경로만 반환.

    예: ``s3://.../4640.../artifacts/hustvl-yolos-tiny`` → ``hustvl-yolos-tiny``
    ``mlflow-artifacts:/3/.../artifacts/hustvl-yolos-tiny`` → ``hustvl-yolos-tiny``

    ``/artifacts/``가 없으면 빈 문자열.
    """
    s = (artifact_path or "").strip().replace("\\", "/")
    if not s:
        return ""
    lower = s.lower()
    needle = "/artifacts/"
    i = lower.find(needle)
    if i == -1:
        return ""
    return s[i + len(needle) :].lstrip("/")


def _isoformat_utc(dt: datetime | None) -> str:
    if dt is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelImprovementService:
    """최적화/경량화 서버 연동 및 task·자동 모델 등록."""

    def __init__(self, client: OptimizationClient):
        self._client = client

    async def get_task_types(
        self,
        db: Session,
        category: str | None = None,
        *,
        source_model_id: int | None = None,
    ) -> list[TaskTypeResponse]:
        data = await self._client.get_optimizer_list(page_size=100)
        items = data.get("items") or []
        allow: frozenset[str] | None = None
        if source_model_id is not None:
            src = model_repository.get(db, source_model_id)
            if src is not None:
                allow = optimization_task_allowlist_for_repo(src.repo_id, model_name=src.name)
        allow_lower = {n.lower() for n in allow} if allow is not None else None

        results: list[TaskTypeResponse] = []
        for item in items:
            mapped_cat = _map_optimizer_type(item.get("optimizer_type"))
            if category is not None and mapped_cat != category:
                continue
            name = item.get("optimizer_name") or ""
            if not name:
                continue
            if allow_lower is not None and name.strip().lower() not in allow_lower:
                continue
            results.append(
                TaskTypeResponse(
                    name=name,
                    category=mapped_cat,
                    description=item.get("accelerator"),
                )
            )
        return results

    async def _reconcile_open_tasks(self, db: Session, source_model_id: int) -> None:
        rows = (
            db.query(ModelImprovementTask)
            .filter(
                ModelImprovementTask.source_model_id == source_model_id,
                ModelImprovementTask.last_known_status.in_(_ACTIVE_STATUSES),
            )
            .all()
        )
        for row in rows:
            try:
                remote_raw = await self._client.get_task_detail(row.task_id)
            except HTTPException as e:
                if e.status_code == status.HTTP_404_NOT_FOUND:
                    row.last_known_status = "FAILED"
                    row.updated_at = datetime.now(timezone.utc)
                    continue
                raise
            remote = _determine_remote_task_status(remote_raw)
            row.last_known_status = _to_api_status(remote)
            row.updated_at = datetime.now(timezone.utc)
            if remote == RemoteTaskStatus.SUCCESS:
                self._maybe_register_on_success(db, row, remote_raw)
        db.flush()

    def _maybe_register_on_success(
        self,
        db: Session,
        row: ModelImprovementTask,
        raw: dict,
    ) -> None:
        mlflow_run_id = raw.get("mlflow_run_id")
        path = raw.get("model_path_output")
        if not mlflow_run_id or not path or not str(path).strip():
            return
        existing = db.query(ModelRegistry).filter(ModelRegistry.run_id == str(mlflow_run_id)).first()
        if existing:
            row.result_model_id = existing.reference_model_id
            return
        if row.result_model_id is not None:
            return
        raw_tt = raw.get("task_type")
        task_type = raw_tt or row.task_type
        logger.info(
            "최적화등록 task=%s used=%r remote=%r row=%r",
            row.task_id,
            task_type,
            raw_tt,
            row.task_type,
        )
        new_id = self._register_optimized_model(
            db=db,
            task_data=raw,
            source_model_id=row.source_model_id,
            task_type=task_type,
        )
        row.result_model_id = new_id

    def _resolve_format_id(self, db: Session, task_type: str, source_model: object) -> int:
        key = (task_type or "").strip().lower()
        fmt_name = _TASK_TYPE_TO_FORMAT_NAME.get(key)

        if fmt_name:
            fmt = model_format_repository.get_by_name(db, fmt_name)
            if fmt:
                logger.info("최적화포맷 %s→%s (id=%s)", key or "-", fmt_name, fmt.id)
                return fmt.id
            logger.warning(
                "최적화포맷 %s→%s 인데 DB에 없음, 원본 format_id=%s 유지",
                key or "-",
                fmt_name,
                source_model.format_id,
            )
            return source_model.format_id

        logger.warning(
            "최적화포맷 '%s' 매핑 없음(%s), 원본 format_id=%s 유지",
            key or "-",
            ",".join(sorted(_TASK_TYPE_TO_FORMAT_NAME.keys())),
            source_model.format_id,
        )
        return source_model.format_id

    def _register_optimized_model(
        self,
        db: Session,
        task_data: dict,
        source_model_id: int,
        task_type: str,
    ) -> int:
        source_model = model_repository.get(db, source_model_id)
        if not source_model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="원본 모델을 찾을 수 없습니다.")

        custom_provider = ModelProviderService.get_by_name(db, ModelProviderEnum.CUSTOM.value)
        if not custom_provider:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="custom model provider가 구성되어 있지 않습니다.",
            )

        model_name = task_data.get("model_name") or source_model.name
        out_path = str(task_data.get("model_path_output") or "").strip()
        mlflow_run_id = str(task_data.get("mlflow_run_id") or "").strip()
        name_suffix = uuid.uuid4().hex[:6]

        lineage_root_id = ModelService.resolve_lineage_root_model_id(db, source_model_id)

        model_schema = ModelBaseSchema(
            name=f"{model_name}_{task_type}_{name_suffix}",
            description=f"{task_type} 최적화 적용 모델 (원본: {model_name})",
            repo_id=None,
            provider_id=custom_provider.id,
            type_id=source_model.type_id,
            format_id=self._resolve_format_id(db, task_type, source_model),
            parent_model_id=lineage_root_id,
            learning_enable_yn=False,
            opt_enable_yn=False,
            version=1,
            subversion=1,
            task=source_model.task,
        )

        model_obj = model_repository.create(db, obj_in=model_schema)

        reg_uri = registry_uri_suffix_after_artifacts(out_path)
        if not reg_uri:
            reg_uri = out_path.rstrip("/").rsplit("/", maxsplit=1)[-1] if out_path else ""

        model_registry_repository.create(
            db,
            obj_in=ModelRegistryBaseSchema(
                artifact_path=out_path,
                uri=reg_uri,
                run_id=mlflow_run_id or None,
                reference_model_id=model_obj.id,
            ),
        )
        db.flush()
        logger.info("최적화완료 model_id=%s format_id=%s", model_obj.id, model_obj.format_id)
        return model_obj.id

    async def create_task(
        self,
        db: Session,
        request: CreateImprovementRequest,
        *,
        created_by_username: str,
    ) -> CreateImprovementResponse:
        source = model_repository.get(db, request.source_model_id)
        if not source:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="모델을 찾을 수 없습니다.")

        if not source.opt_enable_yn:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="OPTIMIZATION_SOURCE_NOT_ELIGIBLE",
            )

        registry = source.registry
        if registry is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="OPTIMIZATION_SOURCE_NOT_ELIGIBLE",
            )
        saved_path = (registry.uri or "").strip()
        if not registry.run_id or not saved_path:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="OPTIMIZATION_SOURCE_NOT_ELIGIBLE",
            )

        allow = optimization_task_allowlist_for_repo(source.repo_id, model_name=source.name)
        if allow is not None:
            tt = request.task_type.strip().lower()
            if tt not in {n.lower() for n in allow}:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="OPTIMIZATION_TASK_TYPE_NOT_ALLOWED_FOR_MODEL",
                )

        await self._reconcile_open_tasks(db, source.id)

        open_row = (
            db.query(ModelImprovementTask)
            .filter(
                ModelImprovementTask.source_model_id == source.id,
                ModelImprovementTask.last_known_status.in_(_ACTIVE_STATUSES),
            )
            .first()
        )
        if open_row:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="OPTIMIZATION_TASK_ALREADY_RUNNING",
            )

        data = await self._client.get_optimizer_list(page_size=100)
        items = data.get("items") or []
        optimizer_id: int | None = None
        tt_lower = request.task_type.strip().lower()
        for item in items:
            if (item.get("optimizer_name") or "").strip().lower() == tt_lower:
                oid = item.get("id")
                if oid is not None:
                    optimizer_id = int(oid)
                break
        if optimizer_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"유효하지 않은 task_type입니다: {request.task_type}",
            )

        opt_model_name = (source.repo_id or "").strip() or source.name

        resp = await self._client.create_optimize_task(
            optimizer_id=optimizer_id,
            saved_model_run_id=str(registry.run_id),
            saved_model_path=saved_path,
            model_name=opt_model_name,
            args={},
        )
        task_uuid = resp.get("task_uuid")
        if not task_uuid:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="최적화 서버 응답에 task_uuid가 없습니다.",
            )

        task_row = ModelImprovementTask(
            task_id=str(task_uuid),
            source_model_id=source.id,
            task_type=request.task_type,
            created_by_username=created_by_username,
            last_known_status="PENDING",
        )
        db.add(task_row)
        db.commit()
        db.refresh(task_row)

        return CreateImprovementResponse(
            task_id=str(task_uuid),
            status="PENDING",
            source_model_id=source.id,
            created_at=_isoformat_utc(task_row.created_at),
        )

    def get_task_status(
        self,
        db: Session,
        task_id: str,
        *,
        current_username: str,
    ) -> ImprovementStatusResponse:
        """DB 에 기록된 상태만 돌려준다.

        최적화 서버 조회와 결과 모델 등록은 백그라운드 폴링이 전담한다. 조회 때마다 원격을 부르면
        아무도 조회하지 않는 동안 작업이 끝나도 결과가 반영되지 않는다.
        """
        row = db.query(ModelImprovementTask).filter(ModelImprovementTask.task_id == task_id).first()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"task_id '{task_id}'를 찾을 수 없습니다.",
            )
        if row.created_by_username != current_username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="이 작업을 조회할 권한이 없습니다.",
            )

        return ImprovementStatusResponse(
            task_id=row.task_id,
            status=row.last_known_status,
            source_model_id=row.source_model_id,
            created_at=_isoformat_utc(row.created_at),
            updated_at=_isoformat_utc(row.updated_at),
            message=_STATUS_MESSAGES.get(row.last_known_status),
            result_model_id=row.result_model_id,
            error=row.error_message,
        )


def _poll_deadline(created_at: datetime) -> datetime:
    """작업 생성 시각 기준 추적 상한. 재시작해도 늘어나지 않도록 폴링 시작 시각이 아닌 created_at 을 쓴다."""
    base = created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=timezone.utc)
    return base + timedelta(seconds=int(get_settings().MODEL_IMPROVEMENT_POLL_TIMEOUT_SEC or 7200))


def _finish_task(db: Session, row: ModelImprovementTask, api_status: str, error: str | None) -> None:
    row.last_known_status = api_status
    row.error_message = error
    row.updated_at = datetime.now(timezone.utc)
    db.commit()


async def poll_improvement_task(task_id: str) -> None:
    """최적화 작업이 끝날 때까지 서버에 물어보며 DB 를 갱신한다.

    성공 시 결과 모델 등록까지 이 루프가 처리하므로, 사용자가 화면을 보고 있지 않아도 반영된다.
    어떤 경로로 끝나든 상태를 SUCCEEDED 또는 FAILED 로 확정한다. 활성 상태로 남겨두면 조회 API 가
    DB 만 읽는 구조에서 영원히 진행 중으로 보이고, 같은 모델에 대한 재시도까지 막힌다.
    """
    settings = get_settings()
    interval = max(1, int(settings.MODEL_IMPROVEMENT_POLL_INTERVAL_SEC or 5))
    service = ModelImprovementService(get_optimization_client())

    while True:
        db = SessionLocal()
        try:
            row = db.query(ModelImprovementTask).filter(ModelImprovementTask.task_id == task_id).first()
            if row is None:
                logger.info("최적화 폴링 종료: task %s 행이 없음", task_id)
                return
            if row.last_known_status not in _ACTIVE_STATUSES:
                return

            expired = datetime.now(timezone.utc) >= _poll_deadline(row.created_at)

            try:
                raw = await service._client.get_task_detail(task_id)
            except HTTPException as e:
                if e.status_code == status.HTTP_404_NOT_FOUND:
                    _finish_task(db, row, "FAILED", "최적화 서버에서 작업을 찾을 수 없습니다.")
                    logger.warning("최적화 폴링 종료: task %s 원격에 없음", task_id)
                    return
                # 일시적인 연결·게이트웨이 오류는 다음 주기에 다시 시도한다.
                logger.warning("최적화 폴링 조회 실패(task=%s): %s", task_id, e.detail)
                if expired:
                    _finish_task(db, row, "FAILED", "최적화 서버 상태를 확인하지 못한 채 추적 시간이 초과되었습니다.")
                    return
                raw = None

            if raw is not None:
                remote = _determine_remote_task_status(raw)
                if remote == RemoteTaskStatus.SUCCESS:
                    row.last_known_status = _to_api_status(remote)
                    row.updated_at = datetime.now(timezone.utc)
                    service._maybe_register_on_success(db, row, raw)
                    row.error_message = None
                    db.commit()
                    logger.info("최적화 완료: task=%s result_model_id=%s", task_id, row.result_model_id)
                    return
                if remote == RemoteTaskStatus.FAILED:
                    err = raw.get("error") if isinstance(raw.get("error"), str) else None
                    _finish_task(db, row, "FAILED", err or "최적화 작업이 실패하였습니다.")
                    logger.info("최적화 실패: task=%s", task_id)
                    return

                api_status = _to_api_status(remote)
                if row.last_known_status != api_status:
                    row.last_known_status = api_status
                row.updated_at = datetime.now(timezone.utc)
                db.commit()

                if expired:
                    # 상한을 넘겼지만 원격은 아직 진행 중이다. 여기서 종결짓지 않으면 활성 상태로 방치된다.
                    _finish_task(
                        db,
                        row,
                        "FAILED",
                        "추적 시간이 초과되었습니다. 최적화 서버에서는 계속 진행 중일 수 있습니다.",
                    )
                    logger.warning("최적화 추적 시간 초과: task=%s", task_id)
                    return
        except Exception as e:
            logger.error("최적화 폴링 오류(task=%s): %s", task_id, e)
        finally:
            db.close()

        await asyncio.sleep(interval)


def resume_open_improvement_polls() -> int:
    """활성 상태로 남은 최적화 작업의 폴링을 다시 건다. 시작한 개수를 돌려준다.

    백그라운드 폴링은 프로세스와 함께 사라지므로, 재배포·재시작을 건너뛴 작업은 여기서 이어받는다.
    상한을 이미 넘긴 작업도 그대로 태운다. 폴링이 첫 주기에서 종결 처리한다.
    """
    db = SessionLocal()
    try:
        rows = db.query(ModelImprovementTask).filter(ModelImprovementTask.last_known_status.in_(_ACTIVE_STATUSES)).all()
        task_ids = [r.task_id for r in rows]
    except Exception as e:
        logger.error("최적화 폴링 재개 대상 조회 실패: %s", e)
        return 0
    finally:
        db.close()

    for tid in task_ids:
        asyncio.create_task(poll_improvement_task(tid))
    if task_ids:
        logger.info("최적화 폴링 재개: %d건 %s", len(task_ids), task_ids)
    return len(task_ids)
