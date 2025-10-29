"""Gradio 기반 ML Workflow UI"""

import logging
from typing import Optional

import gradio as gr
from api_client import APIClient
from session_manager import SessionManager
from ui.deployment_management import create_deployment_management_ui
from ui.playground import create_playground_ui
from ui.quick_deployment import create_quick_deployment_ui

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 세션 관리자 초기화
session_manager = SessionManager(token_expiry_hours=24)


# 앱 상태 클래스
class AppState:
    """앱 상태"""

    def __init__(self):
        self.token: Optional[str] = None
        self.api_client: Optional[APIClient] = None
        self.username: Optional[str] = None


# 전역 앱 상태 (단일 사용자 환경)
app_state = AppState()


def authenticate(username: str, password: str):
    """사용자 인증"""
    token = APIClient.authenticate(username, password)
    if token:
        app_state.token = token
        app_state.api_client = APIClient(token)
        app_state.username = username

        # 브라우저 localStorage에 저장할 세션 데이터 생성
        session_json = session_manager.create_session_data(username, token, current_tab=1)
        logger.info(f"User {username} logged in")

        return True, f"안녕하세요, {username}님! (세션이 24시간 동안 유지됩니다)", session_json
    return False, "로그인 실패. 사용자 이름과 비밀번호를 확인해주세요.", ""


def logout():
    """로그아웃"""
    app_state.token = None
    app_state.api_client = None
    app_state.username = None

    logger.info("User logged out")

    return "로그아웃되었습니다."


def restore_session(session_json: str):
    """브라우저 localStorage에서 세션 복원

    Args:
        session_json: localStorage에서 가져온 JSON 문자열
    """
    session_data = session_manager.parse_session_data(session_json)
    if session_data:
        try:
            app_state.token = session_data["token"]
            app_state.api_client = APIClient(session_data["token"])
            app_state.username = session_data["username"]
            current_tab = session_data.get("current_tab", 1)

            logger.info(f"Session restored for user: {app_state.username}, tab: {current_tab}")
            return True, f"✅ 로그인됨: {app_state.username}", current_tab
        except Exception as e:
            logger.error(f"Failed to restore session: {e}")
            return False, "❌ 로그인 필요", 0

    return False, "❌ 로그인 필요", 0


def create_app():
    """Gradio 앱 생성"""
    with gr.Blocks(
        title="AI-PaaS MLOps 서비스",
        theme=gr.themes.Soft(),
        head="""
        <script>
            // localStorage에서 세션 로드
            function loadSession() {
                const session = localStorage.getItem('ml_workflow_session');
                return session || '';
            }

            // localStorage에 세션 저장
            function saveSession(sessionData) {
                if (sessionData) {
                    localStorage.setItem('ml_workflow_session', sessionData);
                }
            }

            // localStorage에서 세션 삭제
            function clearSession() {
                localStorage.removeItem('ml_workflow_session');
            }
        </script>
    """,
    ) as app:
        gr.Markdown("# 🚀 AI-PaaS MLOps 서비스")

        # localStorage에서 세션 데이터 로드 (숨겨진 텍스트박스)
        session_storage = gr.Textbox(visible=False, elem_id="session_storage")

        # 로그인 상태 표시
        login_status = gr.Markdown("❌ 로그인 필요")

        # 기본적으로 로그인 탭으로 시작
        with gr.Tabs(selected=0) as tabs:
            with gr.Tab("계정", id=0):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## 🔐 계정 관리")

                        # 로그인 폼
                        with gr.Group() as login_form:
                            username_input = gr.Textbox(label="사용자 이름", value="surromind")
                            password_input = gr.Textbox(label="비밀번호", type="password")
                            login_btn = gr.Button("로그인", variant="primary", size="lg")

                        # 로그인 상태 정보
                        with gr.Group(visible=False) as logout_form:
                            logout_btn = gr.Button("로그아웃", variant="stop", size="lg")

                        # 상태 메시지
                        status_msg = gr.Textbox(label="상태", interactive=False, visible=False)

                        # 세션 데이터를 저장할 숨겨진 필드
                        save_session_trigger = gr.Textbox(visible=False, elem_id="save_session_trigger")

                def handle_login(username, password):
                    """로그인 처리"""
                    success, message, session_json = authenticate(username, password)
                    if success:
                        # 로그인 성공 시
                        return (
                            gr.update(visible=False),  # login_form 숨기기
                            gr.update(visible=True),  # logout_form 보이기
                            gr.update(value=message, visible=True),  # status_msg
                            f"✅ 로그인됨: {app_state.username}",  # login_status
                            gr.Tabs(selected=1),  # Quick Service Deployment 탭으로 이동
                            session_json,  # localStorage에 저장할 세션 데이터
                        )
                    else:
                        # 로그인 실패 시
                        return (
                            gr.update(visible=True),  # login_form 유지
                            gr.update(visible=False),  # logout_form 숨김
                            gr.update(value=message, visible=True),  # status_msg
                            "❌ 로그인 필요",  # login_status
                            gr.Tabs(),  # 탭 변경 없음
                            "",  # 빈 세션 데이터
                        )

                def handle_logout():
                    """로그아웃 처리"""
                    msg = logout()
                    return (
                        gr.update(visible=True),  # login_form 보이기
                        gr.update(visible=False),  # logout_form 숨기기
                        gr.update(value=msg, visible=True),  # status_msg
                        "❌ 로그인 필요",  # login_status
                        gr.Tabs(selected=0),  # 계정 탭으로 이동
                    )

                login_btn.click(
                    fn=handle_login,
                    inputs=[username_input, password_input],
                    outputs=[login_form, logout_form, status_msg, login_status, tabs, save_session_trigger],
                ).then(
                    fn=None,
                    inputs=[save_session_trigger],
                    outputs=[],
                    js="""
                    (sessionData) => {
                        if (sessionData) {
                            localStorage.setItem('ml_workflow_session', sessionData);
                            console.log('Session saved to localStorage');
                        }
                    }
                    """,
                )

                logout_btn.click(
                    fn=handle_logout,
                    inputs=[],
                    outputs=[login_form, logout_form, status_msg, login_status, tabs],
                ).then(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="""
                    () => {
                        localStorage.removeItem('ml_workflow_session');
                        console.log('Session cleared from localStorage');
                    }
                    """,
                )

            with gr.Tab("Quick Service Deployment", id=1):
                gr.Markdown(
                    """
                ## Quick Service Deployment
                템플릿을 사용하여 빠르게 모델을 배포할 수 있습니다.
                """
                )
                quick_deployment_info = create_quick_deployment_ui(app_state, tabs)

            with gr.Tab("배포 관리", id=2):
                gr.Markdown(
                    """
                ## 📊 배포 관리
                배포된 워크플로우의 상태를 확인하고 관리합니다.
                """
                )
                deployment_info = create_deployment_management_ui(app_state)
                delete_confirm_box = deployment_info["delete_confirm_box"]
                delete_confirm_btn = deployment_info["delete_confirm_btn"]

            with gr.Tab("Playground", id=3):
                gr.Markdown(
                    """
                ## Playground
                배포된 워크플로우를 사용하여 추론을 수행합니다.
                """
                )
                playground_info = create_playground_ui(app_state)

        # 탭 변경 시 삭제 확인 메시지 숨기기
        def on_tab_change(evt: gr.SelectData):
            """탭 변경 시 처리"""
            logger.info(f"Tab changed to: {evt.index}")
            # 삭제 확인 메시지 숨기기
            return gr.update(visible=False), gr.update(visible=False)

        tabs.select(
            fn=on_tab_change,
            inputs=[],
            outputs=[delete_confirm_box, delete_confirm_btn],
        )

        # 앱 로드 시 localStorage에서 세션 복원
        def restore_session_on_load(session_json):
            """페이지 로드 시 localStorage에서 세션 복원"""
            session_restored, status_text, current_tab = restore_session(session_json)

            if session_restored:
                return (
                    gr.update(visible=False),  # login_form 숨기기
                    gr.update(visible=True),  # logout_form 보이기
                    status_text,  # login_status
                    gr.Tabs(selected=current_tab),  # 저장된 탭으로 이동
                )
            else:
                return (
                    gr.update(visible=True),  # login_form 보이기
                    gr.update(visible=False),  # logout_form 숨기기
                    status_text,  # login_status
                    gr.Tabs(selected=0),  # 로그인 탭으로
                )

        app.load(
            fn=restore_session_on_load,
            inputs=[session_storage],
            outputs=[login_form, logout_form, login_status, tabs],
            js="""
            () => {
                const sessionData = localStorage.getItem('ml_workflow_session') || '';
                console.log('Loading session from localStorage:', sessionData ? 'found' : 'not found');
                return [sessionData];
            }
            """,
        )

        # 앱 로드 시 Quick Service Deployment, 배포 관리, Playground의 목록 자동 로드
        app.load(
            fn=quick_deployment_info["load_fn"],
            inputs=[],
            outputs=quick_deployment_info["load_outputs"],
        )

        app.load(
            fn=deployment_info["load_fn"],
            inputs=[],
            outputs=deployment_info["load_outputs"],
        )

        app.load(
            fn=playground_info["load_fn"],
            inputs=[],
            outputs=playground_info["load_outputs"],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
