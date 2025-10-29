"""세션 관리 - 브라우저 localStorage 기반"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """세션 관리 클래스 - 브라우저 localStorage와 연동"""

    def __init__(self, token_expiry_hours: int = 24):
        """
        Args:
            token_expiry_hours: 토큰 만료 시간 (시간)
        """
        self.token_expiry_hours = token_expiry_hours

    def create_session_data(self, username: str, token: str, current_tab: int = 1) -> str:
        """세션 데이터 생성 (JSON 문자열 반환)

        Args:
            username: 사용자명
            token: 인증 토큰
            current_tab: 현재 선택된 탭 ID

        Returns:
            JSON 문자열
        """
        session_data = {
            "username": username,
            "token": token,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=self.token_expiry_hours)).isoformat(),
            "current_tab": current_tab,
        }
        return json.dumps(session_data)

    def parse_session_data(self, session_json: str) -> Optional[dict]:
        """세션 데이터 파싱 및 검증

        Args:
            session_json: localStorage에서 가져온 JSON 문자열

        Returns:
            유효한 세션 데이터 또는 None
        """
        if not session_json or session_json.strip() == "":
            return None

        try:
            session_data = json.loads(session_json)

            # 만료 시간 확인
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                logger.info("Session expired")
                return None

            logger.info(f"Session validated for user: {session_data['username']}")
            return session_data

        except Exception as e:
            logger.error(f"Failed to parse session data: {e}")
            return None
