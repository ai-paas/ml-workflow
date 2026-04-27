import pytest
import requests
from config import API_PREFIX, BASE_URL, PASSWORD, USERNAME


@pytest.fixture(scope="session")
def api_url() -> str:
    return f"{BASE_URL}{API_PREFIX}"


@pytest.fixture(scope="session")
def auth_token(api_url: str) -> str:
    """로그인하여 Bearer 토큰을 획득한다."""
    resp = requests.post(
        f"{api_url}/authentications/token",
        data={"username": USERNAME, "password": PASSWORD},
    )
    assert resp.status_code == 200, f"로그인 실패: {resp.status_code} {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def auth_headers(auth_token: str) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}
