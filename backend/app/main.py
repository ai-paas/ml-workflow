from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api_router

app = FastAPI()

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
