import logging
from functools import lru_cache
from typing import Any

import httpx
from config.settings import get_settings
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class OptimizationClient:
    """최적화/경량화 서버 HTTP 클라이언트 (AsyncClient)."""

    def __init__(self, base_url: str, timeout: int = 30):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base_url}{path}" if path.startswith("/") else f"{self._base_url}/{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                if response.content:
                    return response.json()
                return {}
        except httpx.ConnectError as e:
            logger.warning("optimization server connect error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="최적화/경량화 서버에 연결할 수 없습니다.",
            ) from e
        except httpx.TimeoutException as e:
            logger.warning("optimization server timeout: %s", e)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="최적화/경량화 서버 응답 시간이 초과되었습니다.",
            ) from e
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            body = (e.response.text or "")[:2000]
            if sc == status.HTTP_404_NOT_FOUND:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="해당 작업을 찾을 수 없습니다.",
                ) from e
            if sc >= 500:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="최적화/경량화 서버 내부 오류가 발생했습니다.",
                ) from e
            raise HTTPException(status_code=sc, detail=body or e.response.reason_phrase) from e

    async def get_optimizer_list(
        self,
        page_num: int = 1,
        page_size: int = 100,
        name: str | None = None,
        model_id: int | None = None,
    ) -> dict:
        params: dict[str, str | int] = {"page_num": page_num, "page_size": page_size}
        if name is not None:
            params["name"] = name
        if model_id is not None:
            params["model_id"] = model_id
        return await self._request("GET", "/api/v1/checked/optimizer", params=params)

    async def create_optimize_task(
        self,
        optimizer_id: int,
        saved_model_run_id: str,
        saved_model_path: str,
        model_name: str,
        args: dict | None = None,
    ) -> dict:
        payload = {
            "saved_model_run_id": saved_model_run_id,
            "saved_model_path": saved_model_path,
            "model_name": model_name,
            "args": args if args is not None else {},
        }
        return await self._request(
            "POST",
            f"/api/v1/optimize/optimize/{optimizer_id}",
            json=payload,
        )

    async def get_task_detail(self, task_id: str) -> dict:
        return await self._request("GET", f"/api/v1/tasks/{task_id}")


@lru_cache
def get_optimization_client() -> OptimizationClient:
    settings = get_settings()
    return OptimizationClient(
        base_url=settings.OPTIMIZATION_SERVER_URL,
        timeout=settings.OPTIMIZATION_SERVER_TIMEOUT,
    )
