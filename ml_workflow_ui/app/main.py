import json

import requests
import streamlit as st
from config.settings import get_settings

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

settings = get_settings()


def is_authenticated():
    return "token" in st.session_state


def authentication(username: str, password: str):
    response = requests.post(
        f"{settings.REST_API_URL}/api/v1/authentications/token", data={"username": username, "password": password}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    return None


def login():
    st.caption("🚀 AI-PaaS MLOps 서비스 입니다.")
    st.title("Login")
    username = st.text_input("Username", "surromind", disabled=True)
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        token = authentication(username, password)
        if token:
            st.session_state["token"] = token
            st.success("로그인 성공!")
            st.rerun()
        else:
            st.error("로그인 실패. 사용자 이름과 비밀번호를 확인해주세요.")

    # if st.button("Log in"):
    #     st.session_state.logged_in = True
    #     st.rerun()


def logout():
    if st.button("Log out"):
        if "token" in st.session_state:
            del st.session_state["token"]
        # st.session_state.logged_in = False
        st.rerun()


login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")


inferences = st.Page("workspaces/inference.py", title="inference", icon=":material/terminal:")
pg = st.navigation(
    {
        "Inference": [inferences],
        "Account": [logout_page],
    }
)

if not is_authenticated():
    login()
else:
    pg.run()

st.markdown(
    """
        <style>
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
        }
        .rounded-box {
            background-color: white;
            color: darkgrey;
            padding: 20px;
            margin: 10px;
            border-radius: 15px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }

        .rounded-box:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        </style>
        """,
    unsafe_allow_html=True,
)
