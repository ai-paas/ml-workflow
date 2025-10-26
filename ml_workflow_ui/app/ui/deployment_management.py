"""Deployment Management UI - 배포된 워크플로우 관리"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr

logger = logging.getLogger(__name__)


def create_deployment_management_ui(app_state):
    """Deployment Management UI 생성"""

    # 상태 저장
    workflows_list_state = gr.State([])
    selected_workflow_id_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
            ### 📊 배포된 워크플로우 관리

            배포된 모든 워크플로우의 상태를 확인하고 관리할 수 있습니다.
            """
            )

            refresh_btn = gr.Button("🔄 배포된 워크플로우 새로고침", variant="primary", size="lg")

            workflows_dropdown = gr.Dropdown(
                label="📦 배포된 워크플로우 선택",
                choices=[],
                interactive=True,
                info="관리할 워크플로우를 선택하세요",
            )

            workflow_info_display = gr.Markdown(value="워크플로우를 선택하면 상세 정보가 표시됩니다.", label="📋 워크플로우 정보")

            status_message = gr.Textbox(label="⚠️ 상태 메시지", interactive=False, visible=False, lines=2)

        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ 관리 기능")

            check_status_btn = gr.Button("🔍 배포 상태 확인 (DB 기반)", variant="primary", size="lg")

            gr.Markdown(
                """
            ---
            #### 고급 기능
            """
            )

            cleanup_btn = gr.Button("🗑️ 리소스 정리 (KServe 서비스 삭제)", variant="stop", size="lg")

            action_output = gr.Textbox(label="📊 작업 결과", interactive=False, lines=8, placeholder="작업 결과가 여기에 표시됩니다...")

            gr.Markdown("---")

            deployed_models_display = gr.Dataframe(
                headers=["Component ID", "서비스명", "모델명", "상태", "배포시간", "Hostname", "Gateway URL"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                label="🚀 배포된 모델 목록",
                interactive=False,
                wrap=True,
            )

    def load_deployed_workflows():
        """배포된 워크플로우 목록 로드"""
        if not app_state.api_client:
            error_msg = "❌ 로그인이 필요합니다."
            logger.warning(error_msg)
            return (
                gr.update(choices=[], value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                [],
                [],
            )

        try:
            logger.info("Loading deployed workflows...")
            result = app_state.api_client.get_all_deployed_workflows()

            workflows = result.get("items", [])
            logger.info(f"Found {len(workflows)} deployed workflows")

            if not workflows:
                msg = "⚠️ 배포된 워크플로우가 없습니다."
                return (
                    gr.update(choices=[], value=None),
                    "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                    gr.update(visible=True, value=msg),
                    [],
                    [],
                )

            # 드롭다운 선택지 생성 (이름과 상태 표시)
            choices = [(f"{w.get('name')} ({w.get('status')})", w.get("id")) for w in workflows]

            success_msg = f"✅ {len(workflows)}개의 배포된 워크플로우를 찾았습니다."
            return (
                gr.update(choices=choices, value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=success_msg),
                workflows,
                [],
            )

        except Exception as e:
            error_msg = f"❌ 워크플로우 로드 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return (
                gr.update(choices=[], value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                [],
                [],
            )

    def on_workflow_selected(workflow_id: Optional[str], workflows_list: List[Dict]):
        """워크플로우 선택 시 상세 정보 표시"""
        if not workflow_id or not workflows_list:
            return "워크플로우를 선택하면 상세 정보가 표시됩니다.", workflow_id or "", []

        # 선택된 워크플로우 찾기
        selected = None
        for w in workflows_list:
            if w.get("id") == workflow_id:
                selected = w
                break

        if not selected:
            return "워크플로우 정보를 찾을 수 없습니다.", "", []

        # 상세 정보 포맷팅
        info = f"""
### 📋 {selected.get('name', 'N/A')}

**🆔 Workflow ID:** `{selected.get('id', 'N/A')}`

**📊 상태:** {selected.get('status', 'N/A')}

**📁 카테고리:** {selected.get('category', 'N/A')}

**📝 설명:** {selected.get('description', '설명이 없습니다.')}

**🔗 Kubeflow Run ID:** {selected.get('kubeflow_run_id', 'N/A')}

**📅 생성일:** {selected.get('created_at', 'N/A')}

**🔄 업데이트:** {selected.get('updated_at', 'N/A')}

---

하단의 버튼을 사용하여 배포 상태를 확인하거나 관리할 수 있습니다.
        """

        # 배포된 모델 정보 조회
        deployed_models_data = []
        try:
            if app_state.api_client:
                models_result = app_state.api_client.get_deployed_models(workflow_id)
                deployed_models = models_result.get("deployed_models", [])

                for model in deployed_models:
                    deployed_models_data.append(
                        [
                            model.get("component_id", "N/A"),
                            model.get("service_name", "N/A"),
                            model.get("model_name", "N/A"),
                            model.get("status", "N/A"),
                            model.get("deployed_at", "N/A"),
                            model.get("service_hostname", "N/A"),
                            model.get("gateway_url", "N/A"),
                        ]
                    )
        except Exception as e:
            logger.error(f"Failed to load deployed models: {e}")

        return info, workflow_id, deployed_models_data

    def check_workflow_status(workflow_id: str):
        """워크플로우 상태 확인 (DB 기반)"""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다."

        if not workflow_id:
            return "❌ 워크플로우를 먼저 선택해주세요."

        try:
            # 워크플로우 상태 조회 (이제 DB 기반)
            result = app_state.api_client.get_workflow_status(workflow_id=workflow_id)

            deployed_models = result.get("deployed_models", [])

            output = []
            output.append("=" * 70)
            output.append("📊 워크플로우 상태 정보 (DB 기반)")
            output.append("=" * 70)
            output.append("")
            output.append(f"🆔 Workflow ID: {result.get('workflow_id')}")
            output.append(f"📌 워크플로우 상태: {result.get('status')}")
            output.append(f"🔗 Kubeflow Run ID: {result.get('kubeflow_run_id', 'N/A')}")

            if result.get("kubeflow_status"):
                output.append(f"🚀 Kubeflow Pipeline 상태: {result.get('kubeflow_status')}")

            # ✅ DB 기반 배포 정보 (kserve_deployments 테이블)
            if deployed_models:
                output.append("")
                output.append("=" * 70)
                output.append("🚀 배포된 모델 정보 (kserve_deployments 테이블)")
                output.append("=" * 70)
                for idx, model in enumerate(deployed_models, 1):
                    status_icon = "✅" if model.get("status") == "deployed" else "⏳"
                    output.append("")
                    output.append(f"{status_icon} [모델 #{idx}]")
                    output.append(f"  Component ID    : {model.get('component_id')}")
                    output.append(f"  서비스명         : {model.get('service_name')}")
                    output.append(f"  모델명          : {model.get('model_name')}")
                    output.append(f"  상태            : {model.get('status').upper()}")
                    output.append(f"  배포 시간        : {model.get('deployed_at', 'N/A')}")
                    output.append(f"  Service Hostname: {model.get('service_hostname')}")
                    output.append(f"  Gateway URL     : {model.get('gateway_url')}")
                    output.append(f"  Internal URL    : {model.get('internal_url', 'N/A')}")
                    if model.get("error_message"):
                        output.append(f"  ⚠️ 에러          : {model.get('error_message')}")
            else:
                output.append("")
                output.append("⚠️ 배포된 모델 정보가 없습니다.")
                output.append("   'Quick Service 배포' 또는 '워크플로우 실행'으로 모델을 배포하세요.")

            output.append("")
            output.append("=" * 70)
            output.append("💡 이 정보는 데이터베이스(kserve_deployments 테이블)를 기반으로 합니다.")
            output.append("=" * 70)

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Failed to check status: {e}")
            return f"❌ 상태 확인 실패: {str(e)}"

    def cleanup_workflow_resources(workflow_id: str):
        """워크플로우 리소스 정리"""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다."

        if not workflow_id:
            return "❌ 워크플로우를 먼저 선택해주세요."

        try:
            result = app_state.api_client.cleanup_workflow(workflow_id=workflow_id)

            output = []
            output.append("=" * 50)
            output.append("🗑️ 리소스 정리 완료")
            output.append("=" * 50)
            output.append("")
            output.append(f"삭제된 서비스 수: {result.get('deleted_services')}")
            output.append(f"메시지: {result.get('message')}")
            output.append("")
            output.append("✅ 워크플로우 리소스가 정리되었습니다.")
            output.append("⚠️  워크플로우 목록을 새로고침하세요.")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")
            return f"❌ 리소스 정리 실패: {str(e)}"

    # 이벤트 핸들러 연결
    refresh_btn.click(
        fn=load_deployed_workflows,
        inputs=[],
        outputs=[
            workflows_dropdown,
            workflow_info_display,
            status_message,
            workflows_list_state,
            deployed_models_display,
        ],
    )

    workflows_dropdown.change(
        fn=on_workflow_selected,
        inputs=[workflows_dropdown, workflows_list_state],
        outputs=[workflow_info_display, selected_workflow_id_state, deployed_models_display],
    )

    check_status_btn.click(
        fn=check_workflow_status,
        inputs=[selected_workflow_id_state],
        outputs=[action_output],
    )

    cleanup_btn.click(
        fn=cleanup_workflow_resources,
        inputs=[selected_workflow_id_state],
        outputs=[action_output],
    )
