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


# 전역 상태
class AppState:
    """앱 전역 상태"""

    def __init__(self):
        self.token: Optional[str] = None
        self.api_client: Optional[APIClient] = None
        self.username: Optional[str] = None


app_state = AppState()


def authenticate(username: str, password: str):
    """사용자 인증"""
    token = APIClient.authenticate(username, password)
    if token:
        app_state.token = token
        app_state.api_client = APIClient(token)
        app_state.username = username

        # 세션 저장
        session_manager.save_session(username, token)
        logger.info(f"User {username} logged in and session saved")

        return True, f"안녕하세요, {username}님! (세션이 24시간 동안 유지됩니다)"
    return False, "로그인 실패. 사용자 이름과 비밀번호를 확인해주세요."


def logout():
    """로그아웃"""
    app_state.token = None
    app_state.api_client = None
    app_state.username = None

    # 세션 삭제
    session_manager.clear_session()
    logger.info("User logged out and session cleared")

    return "로그아웃되었습니다."


def restore_session():
    """저장된 세션 복원"""
    session_data = session_manager.load_session()
    if session_data:
        try:
            app_state.token = session_data["token"]
            app_state.api_client = APIClient(session_data["token"])
            app_state.username = session_data["username"]

            logger.info(f"Session restored for user: {app_state.username}")
            return True, f"✅ 세션 복원됨: {app_state.username}"
        except Exception as e:
            logger.error(f"Failed to restore session: {e}")
            session_manager.clear_session()
            return False, "❌ 로그인 필요"

    return False, "❌ 로그인 필요"


def create_app():
    """Gradio 앱 생성"""
    with gr.Blocks(title="AI-PaaS MLOps 서비스", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 AI-PaaS MLOps 서비스")

        # 저장된 세션 복원 시도
        session_restored, initial_status = restore_session()

        # 로그인 상태 표시
        login_status = gr.Markdown(initial_status)

        with gr.Tab("로그인"):
            with gr.Row():
                with gr.Column():
                    username_input = gr.Textbox(label="사용자 이름", value="surromind")
                    password_input = gr.Textbox(label="비밀번호", type="password")
                    login_btn = gr.Button("로그인", variant="primary")
                    login_message = gr.Textbox(label="로그인 상태", interactive=False)

            def handle_login(username, password):
                success, message = authenticate(username, password)
                if success:
                    return message, f"✅ 로그인됨: {app_state.username}"
                return message, "❌ 로그인 필요"

            login_btn.click(
                fn=handle_login,
                inputs=[username_input, password_input],
                outputs=[login_message, login_status],
            )

        with gr.Tab("Quick Service Deployment"):
            gr.Markdown(
                """
            ## Quick Service Deployment
            템플릿을 사용하여 빠르게 모델을 배포할 수 있습니다.
            """
            )
            create_quick_deployment_ui(app_state)

        with gr.Tab("Playground"):
            gr.Markdown(
                """
            ## Playground
            배포된 워크플로우를 사용하여 추론을 수행합니다.
            """
            )
            create_playground_ui(app_state)

        with gr.Tab("배포 관리"):
            gr.Markdown(
                """
            ## 📊 배포 관리
            배포된 워크플로우의 상태를 확인하고 관리합니다.
            """
            )
            create_deployment_management_ui(app_state)

        with gr.Tab("로그아웃"):
            logout_btn = gr.Button("로그아웃", variant="stop")
            logout_message = gr.Textbox(label="상태", interactive=False)

            def handle_logout():
                msg = logout()
                return msg, "❌ 로그인 필요"

            logout_btn.click(fn=handle_logout, inputs=[], outputs=[logout_message, login_status])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
