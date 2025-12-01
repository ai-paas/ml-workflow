import uuid
from datetime import timedelta

from config.db.connect import SessionDepends
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from schemas.user import UserSchema
from services.user import UserService
from sqlalchemy.orm import Session
from utils.authentication import (
    ACCESS_TOKEN_EXPIRE_SECONDS,
    Token,
    authenticate_user,
    create_access_token,
    get_current_user,
)
from utils.crypto import get_sha256_hash

user_service = UserService()

router = APIRouter(prefix="/authentications", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(*, db: Session = SessionDepends, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    로그인 기능 및 토큰 발급 기능

    * Request Body
        username: 사용자ID
        password: 비밀번호

    * Response Body
        access_token: str - 액세스 토큰 값
        token_type: str - 토큰 타입. "bearer" 고정
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=UserSchema)
async def read_users_me(current_user: UserSchema = Depends(get_current_user)):
    """Login한 사용자 자신의 정보를 return하는 API

    * Response Body
        * id: int - pk
        * username: str - 사용자ID
        * name: str - 사용자명
        * password: str - 비밀번호 hash값 (decode 불가능)
    """
    return current_user
