"""세션 관리 - 로그인 상태 유지"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """세션 관리 클래스"""

    def __init__(self, session_file: str = ".session.json", token_expiry_hours: int = 24):
        """
        Args:
            session_file: 세션 정보를 저장할 파일 경로
            token_expiry_hours: 토큰 만료 시간 (시간)
        """
        self.session_file = Path(__file__).parent / session_file
        self.token_expiry_hours = token_expiry_hours

    def save_session(self, username: str, token: str) -> None:
        """세션 정보 저장"""
        try:
            session_data = {
                "username": username,
                "token": token,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=self.token_expiry_hours)).isoformat(),
            }

            with open(self.session_file, "w") as f:
                json.dump(session_data, f)

            logger.info(f"Session saved for user: {username}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load_session(self) -> Optional[dict]:
        """저장된 세션 정보 로드"""
        try:
            if not self.session_file.exists():
                logger.info("No session file found")
                return None

            with open(self.session_file, "r") as f:
                session_data = json.load(f)

            # 만료 시간 확인
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                logger.info("Session expired")
                self.clear_session()
                return None

            logger.info(f"Session loaded for user: {session_data['username']}")
            return session_data

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def clear_session(self) -> None:
        """세션 정보 삭제"""
        try:
            if self.session_file.exists():
                os.remove(self.session_file)
                logger.info("Session cleared")
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")

    def is_session_valid(self) -> bool:
        """세션이 유효한지 확인"""
        session = self.load_session()
        return session is not None
