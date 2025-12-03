from pydantic import BaseModel, Field


class AuthenticationSessionSchema(BaseModel):
    endpoint_url: str  # KF endpoint URL
    redirect_url: str  # KF redirect URL, if applicable
    dex_login_url: str  # Dex login URL (for POST of credentials)
    is_secured: bool  # True if KF endpoint is secured
    session_cookie: str  # Resulting session cookies in the form "key1=value1; key2=value2"
    session_cookie_dict: dict
    # session_cookie_route: str
    # session_cookie_csrf: str

    class Config:
        from_orm = True
