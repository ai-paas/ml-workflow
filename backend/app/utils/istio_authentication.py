import re
from urllib.parse import urlencode, urlsplit

import requests
from schemas.authentication import AuthenticationSessionSchema


# TODO: LDAP 타입도 받을수 있도록 추가 작업 필요.
def get_istio_auth_session(url: str, username: str, password: str) -> AuthenticationSessionSchema:
    """
    Authenticate against a Dex-secured Istio endpoint and obtain a session cookie.

    This function performs the following steps:
    1. Checks if the provided URL is secured.
    2. If secured, navigates through the Dex authentication process.
    3. Attempts to log in using the provided credentials.
    4. Handles any necessary approval steps.
    5. Retrieves and returns the session cookie upon successful authentication.

    Args:
        url (str): The Istio endpoint URL, including the protocol (e.g., "https://example.com").
        username (str): The username for Dex authentication.
        password (str): The password corresponding to the provided username.

    Returns:
        AuthenticationSessionSchema: An object containing authentication session information:
            - endpoint_url (str): The original Istio endpoint URL.
            - redirect_url (str | None): The URL after any redirects, if applicable.
            - dex_login_url (str | None): The Dex login URL used for credential submission.
            - is_secured (bool | None): Indicates whether the endpoint is secured.
            - session_cookie (str | None): The resulting session cookie string if authentication is successful.

    Raises:
        RuntimeError: If there are issues with HTTP responses, redirects, or the login process.

    Note:
        - This function supports 'staticPasswords' authentication methods.
        - The function uses a requests.Session to maintain cookies throughout the authentication process.
        - If the endpoint is not secured, the function will return early with is_secured set to False.
    """
    # define the default return object
    auth_session = {
        "endpoint_url": url,  # KF endpoint URL
        "redirect_url": "",  # KF redirect URL, if applicable
        "dex_login_url": "",  # Dex login URL (for POST of credentials)
        "is_secured": False,  # True if KF endpoint is secured
        "session_cookie": "",  # Resulting session cookies in the form "key1=value1; key2=value2"
    }

    # use a persistent session (for cookies)
    with requests.Session() as s:
        ################
        # Determine if Endpoint is Secured
        ################
        resp = s.get(url, allow_redirects=True)
        if resp.status_code == 200:
            pass
        elif resp.status_code == 403:
            # if we get 403, we might be at the oauth2-proxy sign-in page
            # the default path to start the sign-in flow is `/oauth2/start?rd=<url>`
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(path="/oauth2/start", query=urlencode({"rd": url_obj.path}))
            resp = s.get(url_obj.geturl(), allow_redirects=True)
        else:
            raise RuntimeError(f"HTTP status code '{resp.status_code}' for GET against: {url}")

        auth_session["redirect_url"] = resp.url

        # if we were NOT redirected, then the endpoint is UNSECURED
        if len(resp.history) == 0:
            auth_session["is_secured"] = False
            return auth_session
        else:
            auth_session["is_secured"] = True

        ################
        # Get Dex Login URL
        ################
        redirect_url_obj = urlsplit(auth_session["redirect_url"])

        # if we are at `/auth?=xxxx` path, we need to select an auth type
        if re.search(r"/auth$", redirect_url_obj.path):
            #######
            # TIP: choose the default auth type by including ONE of the following
            #######

            # OPTION 1: set "staticPasswords" as default auth type
            redirect_url_obj = redirect_url_obj._replace(path=re.sub(r"/auth$", "/auth/local", redirect_url_obj.path))
            # OPTION 2: set "ldap" as default auth type
            # redirect_url_obj = redirect_url_obj._replace(
            #     path=re.sub(r"/auth$", "/auth/ldap", redirect_url_obj.path)
            # )

        # if we are at `/auth/xxxx/login` path, then no further action is needed (we can use it for login POST)
        if re.search(r"/auth/.*/login$", redirect_url_obj.path):
            auth_session["dex_login_url"] = redirect_url_obj.geturl()

        # else, we need to be redirected to the actual login page
        else:
            # this GET should redirect us to the `/auth/xxxx/login` path
            resp = s.get(redirect_url_obj.geturl(), allow_redirects=True)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {redirect_url_obj.geturl()}"
                )

            # set the login url
            auth_session["dex_login_url"] = resp.url

        ################
        # Attempt Dex Login
        ################
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        resp = s.post(
            auth_session["dex_login_url"],
            data={"login": username, "password": password},
            headers=headers,
            allow_redirects=True,
        )
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials were probably invalid - "
                f"No redirect after POST to: {auth_session['dex_login_url']}"
            )

        # Approval Grant Access
        url_obj = urlsplit(resp.url)
        if re.search(r"/approval$", url_obj.path):
            dex_approval_url = url_obj.geturl()

            # approve the login
            resp = s.post(
                dex_approval_url,
                data={"approval": "approve"},
                allow_redirects=True,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP status code '{resp.status_code}' for POST against: {url_obj.geturl()}")

        # store the session cookies in a "key1=value1; key2=value2" string.
        auth_session["session_cookie"] = "; ".join([f"{c.name}={c.value}" for c in s.cookies])
        auth_session["session_cookie_dict"] = s.cookies.get_dict()

    return AuthenticationSessionSchema(**auth_session)
