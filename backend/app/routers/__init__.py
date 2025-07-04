from fastapi import APIRouter

from .authentication import router as auth_router
from .dataset import router as dataset_router
from .inference import router as inference_router
from .model import router as model_router
from .monitoring import router as monitoring_router
from .pipeline import router as pipeline_router
from .v2.dataset import router as v2_dataset_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(pipeline_router)
api_router.include_router(model_router)
api_router.include_router(dataset_router)
api_router.include_router(inference_router)
api_router.include_router(auth_router)


api_v2_router = APIRouter(prefix="/api/v2")
api_v2_router.include_router(v2_dataset_router)

api_router.include_router(monitoring_router)
