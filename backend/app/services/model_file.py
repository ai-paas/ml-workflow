"""모델 저장 파일 조회·다운로드 URL 발급.

저장 위치가 셋(MLflow S3 / Ollama 볼륨 / 없음)이라 응답 형태는 하나로 두고 내용만 달리한다.
파일 목록과 다운로드를 실제로 제공하는 것은 MLflow 아티팩트를 쓰는 모델뿐이다.
"""

from __future__ import annotations

import logging
import posixpath
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from config.db.enums import ModelFormatEnum, ModelProviderEnum
from core.kubeflow.s3.mlflow_s3_manager import MLFlowS3Manager
from db.models.model import Model
from fastapi import HTTPException, status
from schemas.model_file import ModelFileDownloadUrlResponse, ModelFileEntry, ModelFileListResponse, ModelStorageType
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

#: 한 쪽에 담는 최대 파일 수.
PAGE_SIZE = 1000

#: 서명 URL 유효기간(초). 발급 직후 쓰이는 값이라 짧게 둔다.
DOWNLOAD_URL_EXPIRES_SEC = 300

#: 목록에서 빼는 디렉터리. 모델을 내려받는 과정에서 생긴 캐시 부산물이라 사용자가 받을 일이 없고,
#: 수십 개가 섞이면 정작 필요한 가중치 파일이 묻힌다.
_EXCLUDED_DIRS = (".cache",)

_MSG_OLLAMA = "Ollama 가 자체 형식으로 관리하는 볼륨이라 파일 목록을 제공하지 않습니다."
_MSG_REMOTE = "원격 서빙 전용 모델이라 플랫폼이 보관하는 파일이 없습니다."
_MSG_NO_ARTIFACT = "이 모델에 연결된 아티팩트 경로가 없습니다. 모델 등록이 정상적으로 끝나지 않았을 수 있습니다."
_MSG_EMPTY = "아티팩트 경로에 파일이 없습니다."


def _is_remote_only(model: Model) -> bool:
    """원격 서빙 전용 모델인지. 서빙 경로 판정과 같은 출처를 본다."""
    from core.serving.serving_mode import ServingMode
    from core.serving.serving_workflow_deployment_policy import resolve_serving_mode_by_repo_id

    rid = (model.repo_id or "").strip()
    if not rid:
        return False
    return resolve_serving_mode_by_repo_id(rid) is ServingMode.REMOTE_ONLY


def _is_ollama(model: Model) -> bool:
    provider = (model.provider_info.name if model.provider_info else "").lower()
    fmt = (model.format_info.name if model.format_info else "").lower()
    return provider == ModelProviderEnum.OLLAMA.value.lower() and fmt == ModelFormatEnum.GGUF.value.lower()


def resolve_storage_type(model: Model) -> ModelStorageType:
    """모델의 파일이 어디에 있는지. 위에서 걸리면 아래는 보지 않는다."""
    if _is_remote_only(model):
        return ModelStorageType.NONE
    if _is_ollama(model):
        return ModelStorageType.OLLAMA
    return ModelStorageType.MLFLOW


def _artifact_prefix(model: Model) -> Optional[str]:
    """모델 아티팩트의 S3 오브젝트 prefix. 없으면 None."""
    registry = model.registry
    if not registry:
        return None
    path = MLFlowS3Manager.s3_path_from_artifact_uri(registry.artifact_path)
    if not path:
        return None
    return path.rstrip("/")


def _is_excluded(rel_name: str) -> bool:
    """제외 대상 경로인지. 경로 조각이 정확히 일치할 때만 뺀다(`foo.cache` 는 대상이 아니다)."""
    return any(seg in _EXCLUDED_DIRS for seg in rel_name.split("/"))


def _iso_utc(dt: datetime) -> str:
    """boto3 가 주는 tz-aware 시각을 ISO8601 'Z' 표기로."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _download_url_path(model_id: int, name: str) -> str:
    from urllib.parse import quote

    return f"/api/v1/models/{model_id}/files/download-url?name={quote(name, safe='')}"


def _collect_page(prefix: str, cursor: Optional[str]) -> Tuple[List[dict], Optional[str]]:
    """제외 규칙을 적용한 한 쪽을 만든다.

    제외 때문에 쪽이 통째로 비면 다음 쪽을 이어 읽는다. 빈 목록과 next_cursor 를 함께 내보내면
    클라이언트가 "파일 없음" 으로 오해한다. 필터는 서버 사정이지 클라이언트가 알 일이 아니다.

    쪽 크기(1000)가 아티팩트 하나의 파일 수보다 훨씬 커서 평소에는 반복하지 않는다.
    파일이 1000건을 넘고 그 앞부분이 전부 제외 대상인 모델이 들어왔을 때를 위한 것이다.
    """
    s3 = MLFlowS3Manager.get_instance()
    collected: List[dict] = []
    next_cursor = cursor
    while True:
        items, next_cursor = s3.list_objects(prefix + "/", limit=PAGE_SIZE, cursor=next_cursor)
        for obj in items:
            rel = obj["key"][len(prefix) + 1 :]
            if not rel or _is_excluded(rel):
                continue
            collected.append({"name": rel, "size": obj["size"], "last_modified": obj["last_modified"]})
        if collected or not next_cursor:
            break
    return collected, next_cursor


class ModelFileService:
    """모델 파일 목록·다운로드 URL."""

    @staticmethod
    def list_files(db: Session, model_id: int, cursor: Optional[str] = None) -> ModelFileListResponse:
        model = db.get(Model, model_id)
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다: {model_id}")

        storage = resolve_storage_type(model)

        if storage is ModelStorageType.NONE:
            return ModelFileListResponse(
                model_id=model.id, model_name=model.name, storage_type=storage, message=_MSG_REMOTE
            )

        if storage is ModelStorageType.OLLAMA:
            pvc = (model.registry.pvc if model.registry else None) or None
            return ModelFileListResponse(
                model_id=model.id,
                model_name=model.name,
                storage_type=storage,
                location=pvc,
                message=_MSG_OLLAMA,
            )

        prefix = _artifact_prefix(model)
        if not prefix:
            return ModelFileListResponse(
                model_id=model.id, model_name=model.name, storage_type=storage, message=_MSG_NO_ARTIFACT
            )

        try:
            items, next_cursor = _collect_page(prefix, cursor)
        except Exception as e:
            logger.error("모델 %s 아티팩트 조회 실패(prefix=%s): %s", model_id, prefix, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="모델 파일 저장소를 조회하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            )

        files = [
            ModelFileEntry(
                name=it["name"],
                size_bytes=it["size"],
                last_modified=_iso_utc(it["last_modified"]),
                download_url=_download_url_path(model.id, it["name"]),
            )
            for it in sorted(items, key=lambda x: x["name"])
        ]
        return ModelFileListResponse(
            model_id=model.id,
            model_name=model.name,
            storage_type=storage,
            location=prefix,
            files=files,
            next_cursor=next_cursor,
            message=None if files else _MSG_EMPTY,
        )

    @staticmethod
    def issue_download_url(db: Session, model_id: int, name: str) -> ModelFileDownloadUrlResponse:
        model = db.get(Model, model_id)
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다: {model_id}")

        storage = resolve_storage_type(model)
        if storage is not ModelStorageType.MLFLOW:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="이 모델은 파일 다운로드를 제공하지 않습니다.",
            )

        prefix = _artifact_prefix(model)
        if not prefix:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_MSG_NO_ARTIFACT)

        rel = (name or "").strip()
        if not rel:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="요청한 파일을 찾을 수 없습니다.")

        # `name` 은 사용자 입력이다. 경로를 정규화한 뒤 이 모델의 prefix 안에 남아 있는지 확인한다.
        # 정규화 전에 확인하면 '../' 로 다른 모델의 아티팩트를 가리키는 값을 통과시킨다.
        key = posixpath.normpath(f"{prefix}/{rel}")
        if key != prefix and not key.startswith(prefix + "/"):
            logger.warning("모델 %s 다운로드 요청이 아티팩트 경로를 벗어남: %r", model_id, name)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="요청한 파일을 찾을 수 없습니다.")
        if _is_excluded(key[len(prefix) + 1 :]):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="요청한 파일을 찾을 수 없습니다.")

        s3 = MLFlowS3Manager.get_instance()
        try:
            meta = s3.object_exists(key)
        except Exception as e:
            logger.error("모델 %s 파일 확인 실패: %s", model_id, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="모델 파일 저장소를 조회하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            )
        if not meta:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="요청한 파일을 찾을 수 없습니다.")

        try:
            url = s3.presigned_get_url(key, expires_in=DOWNLOAD_URL_EXPIRES_SEC)
        except Exception as e:
            logger.error("모델 %s 다운로드 URL 발급 실패: %s", model_id, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="다운로드 링크를 만들지 못했습니다. 잠시 후 다시 시도해 주세요.",
            )

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=DOWNLOAD_URL_EXPIRES_SEC)
        return ModelFileDownloadUrlResponse(
            model_id=model.id,
            name=key[len(prefix) + 1 :],
            size_bytes=meta["size"],
            download_url=url,
            expires_at=_iso_utc(expires_at),
        )
