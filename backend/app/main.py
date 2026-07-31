from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api_router
from services.polling_recovery import resume_background_polls


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 최적화·학습 진행 상태는 백그라운드 폴링이 갱신하는데, 그 폴링은 프로세스와 함께 사라진다.
    # 재배포를 건너뛴 작업이 진행 중인 채로 남지 않도록 기동할 때 이어받는다.
    resume_background_polls()
    yield


app = FastAPI(lifespan=lifespan)

# CORS 설정
origins = [
    "*",  # 모든 출처 허용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
