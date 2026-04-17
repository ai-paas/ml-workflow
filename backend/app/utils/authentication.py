import hashlib
from datetime import datetime, timedelta
from typing import Optional

from config.db.connect import SessionDepends
from config.settings import get_settings
from db.models.user import UserModel
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from schemas.user import UserSchema
from services.user import UserService
from sqlalchemy.orm import Session
from utils.crypto import get_sha256_hash

settings = get_settings()

# 설정
SECRET_KEY = settings.LOGIN_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 86400  # 24시간

_INTERNAL_API_KEY_HASH: Optional[str] = None
if settings.INTERNAL_API_KEY:
    _INTERNAL_API_KEY_HASH = get_sha256_hash(settings.INTERNAL_API_KEY)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/authentications/token")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/authentications/token", auto_error=False)
interview_scheme = HTTPBearer()
user_service = UserService()


# Pydantic 모델
class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None


# 유틸리티 함수
def authenticate_user(db: Session, username: str, password: str) -> Optional[UserModel]:
    user_model = user_service.get_by_username(db, username)
    if user_model:
        if get_sha256_hash(password) == user_model.password:
            return user_model
        else:
            return None
    else:
        return None


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(db: Session = SessionDepends, token: str = Depends(oauth2_scheme)) -> UserSchema:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_model = user_service.get_by_username(db, username)
        if username is None:
            raise credentials_exception
        if not user_model:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    result = UserSchema.model_validate(user_model)
    return result


def _validate_internal_api_key(key: str) -> bool:
    if not _INTERNAL_API_KEY_HASH:
        return False
    return get_sha256_hash(key) == _INTERNAL_API_KEY_HASH


async def verify_internal_api_key(
    x_internal_api_key: Optional[str] = Header(None),
) -> None:
    """내부 전용 API 인증. X-Internal-API-Key 헤더의 키를 SHA-256 해시로 검증한다."""
    if not x_internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Internal-API-Key header is required",
        )
    if not _validate_internal_api_key(x_internal_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API key",
        )


async def get_current_user_or_internal(
    db: Session = SessionDepends,
    token: Optional[str] = Depends(oauth2_scheme_optional),
    x_internal_api_key: Optional[str] = Header(None),
) -> Optional[UserSchema]:
    """Bearer 토큰 또는 X-Internal-API-Key 중 하나로 인증. 외부 겸용 API용."""
    if x_internal_api_key and _validate_internal_api_key(x_internal_api_key):
        return None

    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username:
                user_model = user_service.get_by_username(db, username)
                if user_model:
                    return UserSchema.model_validate(user_model)
        except JWTError:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
