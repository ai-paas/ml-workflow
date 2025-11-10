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
    로그인 및 액세스 토큰 발급

    사용자 인증을 수행하고 JWT 액세스 토큰을 발급합니다.
    발급된 토큰은 이후 모든 보호된 API 요청에 사용됩니다.
    OAuth2 Password Flow 표준을 따르며, form-data 형식으로 요청을 받습니다.

    ## Request Body (Form Data)
    - **username** (str, required): 사용자 ID
        - 데이터베이스에 등록된 사용자 이름
    - **password** (str, required): 비밀번호
        - 평문 비밀번호 (서버에서 SHA-256 해시로 검증)

    ## Response (Token)
    - **access_token** (str): JWT 액세스 토큰
        - 이후 API 요청 시 Authorization 헤더에 사용
        - 형식: `Bearer {access_token}`
        - 토큰 유효기간: 24시간 (86400초)
        - HS256 알고리즘으로 서명됨
    - **token_type** (str): 토큰 타입
        - 항상 `"bearer"`로 고정
        - OAuth2 표준에 따른 값

    ## Notes
    - Content-Type은 `application/x-www-form-urlencoded` 형식 사용
    - 토큰 만료 시 재로그인 필요 (자동 갱신 기능 없음)
    - 발급된 토큰은 클라이언트에 안전하게 저장 권장
    - 모든 보호된 API 요청 시 `Authorization: Bearer {access_token}` 헤더 필수
    - 잘못된 자격증명 시 401 Unauthorized 응답 반환

    ## Errors
    - 401: 사용자 이름 또는 비밀번호가 잘못됨
        - 응답 헤더에 `WWW-Authenticate: Bearer` 포함
        - 응답 본문: `{"detail": "Incorrect username or password"}`
    - 500: 서버 내부 오류
        - 인증 처리 중 예상치 못한 오류 발생 시
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
    """
    현재 로그인한 사용자 정보 조회

    인증된 사용자 자신의 정보를 조회합니다.
    요청 헤더의 Bearer 토큰에서 사용자 정보를 추출하여 반환합니다.

    ## Request Headers
    - **Authorization** (str, required): Bearer 토큰
        - 형식: `Bearer {access_token}`
        - `/authentications/token` API에서 발급받은 토큰 사용

    ## Response (UserSchema)
    - **id** (int): 사용자 고유 ID (PK)
    - **username** (str): 사용자 ID
        - 로그인 시 사용하는 사용자 이름
    - **name** (str): 사용자 이름
        - 사용자의 표시명
    - **password** (str): 비밀번호 해시값
        - SHA-256 알고리즘으로 해시된 값
        - 복호화 불가능 (보안상 이유)
    - **created_at** (datetime): 계정 생성 시각
    - **updated_at** (datetime): 계정 정보 수정 시각
    - **created_by** (str, optional): 계정 생성자
    - **updated_by** (str, optional): 계정 정보 수정자

    ## Notes
    - 인증이 필요한 API (Bearer 토큰 필수)
    - 토큰에서 사용자 정보를 자동으로 추출하므로 별도 파라미터 불필요
    - 토큰이 만료되었거나 유효하지 않으면 401 에러 반환
    - `password` 필드는 보안상 해시값만 반환 (평문 비밀번호는 반환하지 않음)
    - 현재 로그인한 사용자의 정보만 조회 가능

    ## Errors
    - 401: 토큰이 없거나 유효하지 않음
        - 토큰이 없는 경우
        - 토큰 형식이 잘못된 경우
        - 토큰이 만료된 경우
        - 토큰에 해당하는 사용자가 존재하지 않는 경우
        - 응답 헤더에 `WWW-Authenticate: Bearer` 포함
        - 응답 본문: `{"detail": "Could not validate credentials"}`
    - 500: 서버 내부 오류
        - 사용자 정보 조회 중 예상치 못한 오류 발생 시
    """
    return current_user
