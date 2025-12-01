from fastapi import APIRouter

from .app_service import router as app_service_router
from .authentication import router as auth_router
from .dataset import router as dataset_router
from .experiment import router as experiment_router
from .model import router as model_router

# TODO: 현재 사용하지 않음 - 추후 필요시 활성화
# from .monitoring import router as monitoring_router
from .pipeline import router as pipeline_router
from .workflow import router as workflow_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(pipeline_router)
api_router.include_router(model_router)
api_router.include_router(dataset_router)
api_router.include_router(auth_router)
# TODO: 현재 사용하지 않음 - 추후 필요시 활성화
# api_router.include_router(monitoring_router)
api_router.include_router(experiment_router)
api_router.include_router(app_service_router)
api_router.include_router(workflow_router)
