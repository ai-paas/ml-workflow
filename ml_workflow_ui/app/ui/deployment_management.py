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

            workflow_count_message = gr.Markdown(value="", visible=False)

            workflows_dropdown = gr.Dropdown(
                label="📦 배포된 워크플로우 선택",
                choices=[],
                interactive=True,
                info="관리할 워크플로우를 선택하세요",
            )

            workflow_info_display = gr.Markdown(
                value="워크플로우를 선택하면 \
상세 정보가 표시됩니다.",
                label="📋 워크플로우 정보",
            )

            status_message = gr.Textbox(label="⚠️ 상태 메시지", interactive=False, visible=False, lines=2)

            gr.Markdown(
                """
            ---
            ### ⚠️ 위험 영역
            """
            )

            delete_workflow_btn = gr.Button("🗑️ Workflow 삭제", variant="stop", size="lg")

            delete_confirm_box = gr.Markdown(
                value="",
                visible=False,
            )

            delete_confirm_btn = gr.Button(
                "✅ 예, 삭제합니다",
                variant="stop",
                size="lg",
                visible=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 배포 상태")

            refresh_status_btn = gr.Button("🔄 배포 상태 새로고침", variant="secondary", size="sm")

            deployment_status_display = gr.Markdown(
                value="워크플로우를 선택하면 \
배포 상태가 표시됩니다.",
                elem_classes="deployment-status",
            )

    def load_deployed_workflows():
        """배포된 워크플로우 목록 로드"""
        if not app_state.api_client:
            error_msg = "❌ 로그인이 필요합니다."
            logger.warning(error_msg)
            return (
                gr.update(value=error_msg, visible=True),
                gr.update(choices=[], value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                [],
                "워크플로우를 선택하면 배포 상태가 표시됩니다.",
                gr.update(visible=False),  # delete_confirm_box
                gr.update(visible=False),  # delete_confirm_btn
            )

        try:
            logger.info("Loading deployed workflows...")
            result = app_state.api_client.get_all_deployed_workflows()

            workflows = result.get("items", [])
            logger.info(f"Found {len(workflows)} deployed workflows")

            if not workflows:
                msg = "⚠️ 배포된 워크플로우가 없습니다."
                return (
                    gr.update(value=msg, visible=True),
                    gr.update(choices=[], value=None),
                    "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                    gr.update(visible=True, value=msg),
                    [],
                    "워크플로우를 선택하면 배포 상태가 표시됩니다.",
                    gr.update(visible=False),  # delete_confirm_box
                    gr.update(visible=False),  # delete_confirm_btn
                )

            # 드롭다운 선택지 생성 (워크플로우명 + 템플릿명)
            choices = []
            for w in workflows:
                workflow_name = w.get("name", "N/A")
                template_id = w.get("template_id")

                # 템플릿 정보 조회
                template_name = None
                if template_id and app_state.api_client:
                    try:
                        template_info = app_state.api_client.get_workflow_template(template_id)
                        template_name = template_info.get("name")
                    except Exception as e:
                        logger.warning(f"Failed to fetch template info for {template_id}: {e}")

                # 표시 텍스트 구성
                if template_name:
                    display_text = f"{workflow_name} (템플릿: {template_name})"
                else:
                    display_text = workflow_name

                choices.append((display_text, w.get("id")))

            success_msg = f"✅ {len(workflows)}개의 배포된 워크플로우를 찾았습니다."
            return (
                gr.update(value=success_msg, visible=True),
                gr.update(choices=choices, value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=False),
                workflows,
                "워크플로우를 선택하면 배포 상태가 표시됩니다.",
                gr.update(visible=False),  # delete_confirm_box
                gr.update(visible=False),  # delete_confirm_btn
            )

        except Exception as e:
            error_msg = f"❌ 워크플로우 로드 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return (
                gr.update(value=error_msg, visible=True),
                gr.update(choices=[], value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                [],
                "워크플로우를 선택하면 배포 상태가 표시됩니다.",
                gr.update(visible=False),  # delete_confirm_box
                gr.update(visible=False),  # delete_confirm_btn
            )

    def on_workflow_selected(workflow_id: Optional[str], workflows_list: List[Dict]):
        """워크플로우 선택 시 상세 정보 표시"""
        if not workflow_id or not workflows_list:
            return (
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                workflow_id or "",
                "워크플로우를 선택하면 배포 상태가 표시됩니다.",
                gr.update(visible=False),  # delete_confirm_box
                gr.update(visible=False),  # delete_confirm_btn
            )

        # 선택된 워크플로우 찾기
        selected = None
        for w in workflows_list:
            if w.get("id") == workflow_id:
                selected = w
                break

        if not selected:
            return (
                "워크플로우 정보를 찾을 수 없습니다.",
                "",
                "배포 상태를 확인할 수 없습니다.",
                gr.update(visible=False),  # delete_confirm_box
                gr.update(visible=False),  # delete_confirm_btn
            )

        # 템플릿 정보 조회
        template_name = "N/A"
        template_id = selected.get("template_id")

        if template_id and app_state.api_client:
            try:
                template_info = app_state.api_client.get_workflow_template(template_id)
                template_name = template_info.get("name", "N/A")
            except Exception as e:
                logger.warning(f"Failed to fetch template info for {template_id}: {e}")
                template_name = template_id  # fallback to template_id

        # 상세 정보 포맷팅 (상태, 카테고리, 템플릿명만 표시)
        info = f"""
### 📋 {selected.get('name', 'N/A')}

**📊 상태:** {selected.get('status', 'N/A')}

**📁 카테고리:** {selected.get('category', 'N/A')}

**📑 템플릿:** {template_name}

        """

        # 배포 상태 조회 (kserve_deployments 테이블에서)
        deployment_status_html = ""
        try:
            if app_state.api_client:
                models_result = app_state.api_client.get_deployed_models(workflow_id)
                deployed_models = models_result.get("deployed_models", [])

                if deployed_models:
                    for idx, model in enumerate(deployed_models):
                        status = model.get("status", "UNKNOWN")
                        component_id = model.get("component_id", "N/A")
                        model_name = model.get("model_name", "N/A")
                        service_name = model.get("service_name", "N/A")
                        service_hostname = model.get("service_hostname", "N/A")
                        gateway_url = model.get("gateway_url", "N/A")
                        internal_url = model.get("internal_url", "N/A")
                        deployed_at = model.get("deployed_at", "N/A")

                        # 상태에 따라 아이콘과 색상 변경
                        if status == "DEPLOYED":
                            status_icon = "✅"
                            status_text = "**DEPLOYED**"
                        elif status == "DEPLOYING":
                            status_icon = "🔄"
                            status_text = "**DEPLOYING**"
                        elif status == "FAILED":
                            status_icon = "❌"
                            status_text = "**FAILED**"
                        else:
                            status_icon = "⚠️"
                            status_text = f"**{status}**"

                        deployment_status_html += f"""
## {status_icon} {component_id}

**상태:** {status_text}

**모델명:** `{model_name}`

**서비스명:** `{service_name}`

**Service Hostname:** `{service_hostname}`

**Gateway URL:** `{gateway_url}`

**Internal URL:** `{internal_url}`

**배포시간:** {deployed_at}

"""
                        # 구분선 추가 (마지막 항목 제외)
                        if idx < len(deployed_models) - 1:
                            deployment_status_html += "\n---\n\n"
                else:
                    deployment_status_html = "⚠️ 배포된 모델이 없습니다."
        except Exception as e:
            logger.error(f"Failed to load deployment status: {e}")
            deployment_status_html = f"❌ 배포 상태 조회 실패: {str(e)}"

        return (
            info,
            workflow_id,
            deployment_status_html,
            gr.update(visible=False),  # delete_confirm_box
            gr.update(visible=False),  # delete_confirm_btn
        )

    def refresh_deployment_status(workflow_id: str):
        """배포 상태 새로고침"""
        if not workflow_id:
            return "워크플로우를 선택하면 배포 상태가 표시됩니다."

        if not app_state.api_client:
            return "❌ 로그인이 필요합니다."

        # 배포 상태 조회 (kserve_deployments 테이블에서)
        deployment_status_html = ""
        try:
            models_result = app_state.api_client.get_deployed_models(workflow_id)
            deployed_models = models_result.get("deployed_models", [])

            if deployed_models:
                for idx, model in enumerate(deployed_models):
                    status = model.get("status", "UNKNOWN")
                    component_id = model.get("component_id", "N/A")
                    model_name = model.get("model_name", "N/A")
                    service_name = model.get("service_name", "N/A")
                    service_hostname = model.get("service_hostname", "N/A")
                    gateway_url = model.get("gateway_url", "N/A")
                    internal_url = model.get("internal_url", "N/A")
                    deployed_at = model.get("deployed_at", "N/A")

                    # 상태에 따라 아이콘과 색상 변경
                    if status == "DEPLOYED":
                        status_icon = "✅"
                        status_text = "**DEPLOYED**"
                    elif status == "DEPLOYING":
                        status_icon = "🔄"
                        status_text = "**DEPLOYING**"
                    elif status == "FAILED":
                        status_icon = "❌"
                        status_text = "**FAILED**"
                    else:
                        status_icon = "⚠️"
                        status_text = f"**{status}**"

                    deployment_status_html += f"""
## {status_icon} {component_id}

**상태:** {status_text}

**모델명:** `{model_name}`

**서비스명:** `{service_name}`

**Service Hostname:** `{service_hostname}`

**Gateway URL:** `{gateway_url}`

**Internal URL:** `{internal_url}`

**배포시간:** {deployed_at}

"""
                    # 구분선 추가 (마지막 항목 제외)
                    if idx < len(deployed_models) - 1:
                        deployment_status_html += "\n---\n\n"
            else:
                deployment_status_html = "⚠️ 배포된 모델이 없습니다."
        except Exception as e:
            logger.error(f"Failed to load deployment status: {e}")
            deployment_status_html = f"❌ 배포 상태 조회 실패: {str(e)}"

        return deployment_status_html

    def show_delete_confirmation(workflow_id: str, workflows_list: List[Dict]):
        """삭제 확인 메시지 표시"""
        if not workflow_id:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # 선택된 워크플로우 정보 찾기
        workflow_name = "알 수 없음"
        for w in workflows_list:
            if w.get("id") == workflow_id:
                workflow_name = w.get("name", "알 수 없음")
                break

        confirm_message = f"""
### ⚠️ 정말로 삭제하시겠습니까?

**워크플로우:** `{workflow_name}`

**ID:** `{workflow_id}`

**경고:**
- 이 작업은 되돌릴 수 없습니다.
- 배포된 KServe InferenceService도 함께 삭제됩니다.
- 삭제 프로세스는 최대 5분 정도 소요될 수 있습니다.

아래 "예, 삭제합니다" 버튼을 클릭하면 삭제가 진행됩니다.
        """

        return (
            gr.update(value=confirm_message, visible=True),
            gr.update(visible=True),
        )

    def delete_workflow(workflow_id: str):
        """워크플로우 삭제 실행 (2단계 프로세스)"""
        if not app_state.api_client:
            logger.warning("로그인이 필요합니다.")
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

        if not workflow_id:
            logger.warning("워크플로우를 먼저 선택해주세요.")
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

        try:
            # Step 1: 삭제 시작 - Kubeflow Pipeline cleanup 시작
            logger.info(f"Starting workflow deletion: {workflow_id}")
            result = app_state.api_client.delete_workflow(workflow_id=workflow_id)

            cleanup_run_id = result.get("cleanup_run_id")
            if not cleanup_run_id:
                logger.error("Cleanup pipeline 시작 실패: run_id를 받지 못했습니다.")
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            logger.info(f"Cleanup pipeline started with run_id: {cleanup_run_id}")

            # Step 2: Polling으로 완료 확인
            import time

            max_attempts = 60  # 최대 5분 (5초 간격)
            attempt = 0

            while attempt < max_attempts:
                try:
                    # 완료 확인 API 호출
                    finalize_result = app_state.api_client.finalize_workflow_deletion(
                        workflow_id=workflow_id, run_id=cleanup_run_id
                    )

                    status = finalize_result.get("status")

                    if status == "completed":
                        deleted_from_db = finalize_result.get("deleted_from_db", False)
                        if deleted_from_db:
                            logger.info(f"Workflow {workflow_id} 삭제 완료")
                            return (
                                gr.update(visible=False),
                                gr.update(visible=False),
                            )
                        else:
                            logger.warning("Pipeline은 완료되었으나 DB 삭제에 실패했습니다.")
                            return (
                                gr.update(visible=False),
                                gr.update(visible=False),
                            )

                    elif status == "failed":
                        message = finalize_result.get("message", "Unknown error")
                        logger.error(f"Pipeline 실패: {message}")
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    # 아직 진행중
                    if attempt % 6 == 0:  # 30초마다 로그
                        logger.info(f"Cleanup still in progress for {workflow_id}, attempt {attempt}/{max_attempts}")

                    time.sleep(5)  # 5초 대기
                    attempt += 1

                except Exception as e:
                    logger.error(f"Error checking cleanup status: {e}")
                    time.sleep(5)
                    attempt += 1

            # Timeout
            logger.warning("정리 작업 확인 시간 초과 (5분). 백그라운드에서 계속 진행 중일 수 있습니다.")
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

    # 이벤트 핸들러 연결
    refresh_btn.click(
        fn=load_deployed_workflows,
        inputs=[],
        outputs=[
            workflow_count_message,
            workflows_dropdown,
            workflow_info_display,
            status_message,
            workflows_list_state,
            deployment_status_display,
            delete_confirm_box,
            delete_confirm_btn,
        ],
    )

    workflows_dropdown.change(
        fn=on_workflow_selected,
        inputs=[workflows_dropdown, workflows_list_state],
        outputs=[
            workflow_info_display,
            selected_workflow_id_state,
            deployment_status_display,
            delete_confirm_box,
            delete_confirm_btn,
        ],
    )

    refresh_status_btn.click(
        fn=refresh_deployment_status,
        inputs=[selected_workflow_id_state],
        outputs=[deployment_status_display],
    )

    delete_workflow_btn.click(
        fn=show_delete_confirmation,
        inputs=[selected_workflow_id_state, workflows_list_state],
        outputs=[delete_confirm_box, delete_confirm_btn],
    )

    delete_confirm_btn.click(
        fn=delete_workflow,
        inputs=[selected_workflow_id_state],
        outputs=[delete_confirm_box, delete_confirm_btn],
    )

    # 탭 전환 시 삭제 확인 메시지를 숨기기 위해 컴포넌트 반환
    # 페이지 로드를 위한 컴포넌트들도 함께 반환
    return {
        "delete_confirm_box": delete_confirm_box,
        "delete_confirm_btn": delete_confirm_btn,
        "load_fn": load_deployed_workflows,
        "load_outputs": [
            workflow_count_message,
            workflows_dropdown,
            workflow_info_display,
            status_message,
            workflows_list_state,
            deployment_status_display,
            delete_confirm_box,
            delete_confirm_btn,
        ],
    }
