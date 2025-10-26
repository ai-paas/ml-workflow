"""Quick Service Deployment UI"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr

logger = logging.getLogger(__name__)


def create_quick_deployment_ui(app_state):
    """Quick Service Deployment UI 생성"""

    # 상태 저장
    template_list_state = gr.State([])
    deployed_workflow_id_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
            ### 🚀 Quick Service Deployment

            간편하게 템플릿을 선택하고 배포하세요!
            """
            )

            category_filter = gr.Dropdown(
                label="📁 카테고리 필터",
                choices=["전체", "Object Detection", "Image Classification", "Text Analysis"],
                value="전체",
            )

            refresh_templates_btn = gr.Button("🔄 템플릿 목록 새로고침", size="sm")

            # 템플릿 선택 드롭다운
            template_dropdown = gr.Dropdown(label="✨ 템플릿 선택", choices=[], interactive=True, info="배포할 템플릿을 선택하세요")

            # 템플릿 상세 정보 표시
            template_info_display = gr.Markdown(value="템플릿을 선택하면 상세 정보가 표시됩니다.", label="📋 템플릿 상세 정보")

            template_error_msg = gr.Textbox(label="⚠️ 상태 메시지", interactive=False, visible=False, lines=2)

            # 선택된 템플릿 ID (숨김)
            selected_template_id = gr.Textbox(label="선택된 템플릿 ID", visible=False)

        with gr.Column(scale=1):
            gr.Markdown("### 📦 배포 진행")

            # 한 번에 배포하는 버튼
            deploy_btn = gr.Button("🚀 Quick Service 배포 시작", variant="primary", size="lg")

            deployment_output = gr.Textbox(
                label="📊 배포 진행 상황", interactive=False, lines=10, placeholder="템플릿을 선택하고 '배포 시작' 버튼을 클릭하세요..."
            )

            workflow_id_display = gr.Textbox(label="🆔 배포된 워크플로우 ID", interactive=False, visible=False)

            gr.Markdown("---")

            check_status_btn = gr.Button("🔍 배포 상태 확인 (DB 기반)", variant="secondary", size="lg")

            status_output = gr.Textbox(label="📈 상세 상태", interactive=False, lines=5)

    def load_templates(category: str):
        """템플릿 목록 로드"""
        if not app_state.api_client:
            error_msg = "❌ API client가 초기화되지 않았습니다. 먼저 로그인해주세요."
            logger.warning(error_msg)
            return (
                gr.update(choices=[], value=None),
                "템플릿을 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                "",
                [],
            )

        try:
            logger.info(f"Loading templates with category: {category}")
            category_param = None if category == "전체" else category

            # API 호출 시 자세한 로깅
            logger.info(f"Calling API: {app_state.api_client.base_url}/api/v1/workflows/templates")
            templates = app_state.api_client.get_workflow_templates(category=category_param)

            logger.info(f"Received {len(templates)} templates from API")

            if not templates:
                msg = "⚠️ 템플릿이 없습니다. 먼저 템플릿을 생성해주세요."
                logger.warning(msg)
                return (
                    gr.update(choices=[], value=None),
                    "템플릿을 선택하면 상세 정보가 표시됩니다.",
                    gr.update(visible=True, value=msg),
                    "",
                    [],
                )

            # 드롭다운 선택지 생성
            choices = [(f"{t.get('name')} ({t.get('category', 'N/A')})", t.get("id")) for t in templates]

            success_msg = f"✅ {len(templates)}개의 템플릿을 불러왔습니다."
            return (
                gr.update(choices=choices, value=None),
                "템플릿을 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=success_msg),
                "",
                templates,
            )

        except Exception as e:
            error_msg = f"❌ 템플릿 로드 실패: {str(e)}\n\n자세한 내용은 콘솔 로그를 확인해주세요."
            logger.error(f"Failed to load templates: {e}", exc_info=True)
            return (
                gr.update(choices=[], value=None),
                "템플릿을 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                "",
                [],
            )

    def on_template_selected(template_id: Optional[str], template_list: list):
        """템플릿 선택 시 상세 정보 표시"""
        if not template_id or not template_list:
            return "템플릿을 선택하면 상세 정보가 표시됩니다.", template_id or ""

        # 선택된 템플릿 찾기
        selected = None
        for t in template_list:
            if t.get("id") == template_id:
                selected = t
                break

        if not selected:
            return "템플릿 정보를 찾을 수 없습니다.", ""

        # 상세 정보 포맷팅
        info = f"""
### 📋 {selected.get('name', 'N/A')}

**🆔 템플릿 ID:** `{selected.get('id', 'N/A')}`

**📁 카테고리:** {selected.get('category', 'N/A')}

**📝 설명:** {selected.get('description', '설명이 없습니다.')}

**📊 상태:** {selected.get('status', 'N/A')}

**👤 생성자:** {selected.get('creator_id', 'N/A')}

**📅 생성일:** {selected.get('created_at', 'N/A')}

---

이 템플릿을 사용하여 Quick Service를 배포하려면 아래 '배포 시작' 버튼을 클릭하세요.
        """

        return info, template_id

    def deploy_quick_service(template_id: Optional[str], template_list: list):
        """Quick Service 원클릭 배포 - 워크플로우 생성 + 실행"""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다.", "", gr.update(visible=False)

        if not template_id or not template_id.strip():
            return "❌ 템플릿을 선택해주세요.", "", gr.update(visible=False)

        template_id = template_id.strip()

        output_messages = []

        try:
            # 1단계: 워크플로우 자동 생성 (타임스탬프로 고유한 이름 생성)
            output_messages.append("=" * 60)
            output_messages.append("🚀 Quick Service 배포 시작")
            output_messages.append("=" * 60)
            output_messages.append("")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workflow_name = f"quick-service-{timestamp}"

            output_messages.append("📝 1단계: 워크플로우 생성 중...")
            output_messages.append(f"   - 템플릿 ID: {template_id}")
            output_messages.append(f"   - 워크플로우 이름: {workflow_name}")

            result = app_state.api_client.clone_from_template(
                template_id=template_id,  # 문자열 그대로 전달
                workflow_name=workflow_name,
            )

            workflow_id = result.get("id")
            output_messages.append("   ✅ 워크플로우 생성 완료!")
            output_messages.append(f"   - Workflow ID: {workflow_id}")
            output_messages.append(f"   - 상태: {result.get('status')}")
            output_messages.append("")

            # 2단계: 워크플로우 자동 실행
            output_messages.append("🚀 2단계: 워크플로우 실행 중...")

            exec_result = app_state.api_client.execute_workflow(workflow_id=workflow_id)

            output_messages.append("   ✅ 워크플로우 실행 시작!")
            output_messages.append(f"   - Kubeflow Run ID: {exec_result.get('kubeflow_run_id')}")
            output_messages.append(f"   - 실행 상태: {exec_result.get('status')}")
            output_messages.append(f"   - 메시지: {exec_result.get('message')}")
            output_messages.append("")

            output_messages.append("=" * 60)
            output_messages.append("✅ Quick Service 배포가 시작되었습니다!")
            output_messages.append("=" * 60)
            output_messages.append("")
            output_messages.append("💡 다음 단계:")
            output_messages.append("   1. '배포 상태 확인' 버튼으로 진행 상황 모니터링")
            output_messages.append("   2. 배포 완료 후 '배포 정보 동기화' 버튼 클릭")
            output_messages.append("   3. Playground 탭에서 추론 테스트")
            output_messages.append("")
            output_messages.append("⏱️  예상 소요 시간: 2-3분")

            return ("\n".join(output_messages), workflow_id, gr.update(visible=True, value=workflow_id))

        except Exception as e:
            logger.error(f"Failed to deploy quick service: {e}")
            output_messages.append("")
            output_messages.append("=" * 60)
            output_messages.append("❌ 배포 실패")
            output_messages.append("=" * 60)
            output_messages.append(f"오류: {str(e)}")

            return ("\n".join(output_messages), "", gr.update(visible=False))

    def check_deployment_status(workflow_id: str):
        """배포 상태 확인 (상세)"""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다."

        if not workflow_id:
            return "❌ 먼저 Quick Service를 배포해주세요."

        try:
            result = app_state.api_client.get_workflow_status(workflow_id=workflow_id)

            status_lines = []
            status_lines.append("=" * 50)
            status_lines.append("📊 배포 상태 상세 정보")
            status_lines.append("=" * 50)
            status_lines.append("")
            status_lines.append(f"🆔 Workflow ID: {result.get('workflow_id')}")
            status_lines.append(f"📌 상태: {result.get('status')}")
            status_lines.append(f"🔗 Kubeflow Run ID: {result.get('kubeflow_run_id')}")
            status_lines.append("")

            # 배포된 모델 정보
            if result.get("deployed_models"):
                status_lines.append("🚀 배포된 모델 정보:")
                status_lines.append("-" * 50)
                for idx, model in enumerate(result.get("deployed_models", []), 1):
                    status_lines.append("")
                    status_lines.append(f"[모델 #{idx}]")
                    status_lines.append(f"  📦 Component ID: {model.get('component_id')}")
                    status_lines.append(f"  ✅ 상태: {model.get('status')}")
                    status_lines.append(f"  🏷️  서비스명: {model.get('service_name')}")
                    if model.get("service_hostname"):
                        status_lines.append(f"  🌐 Hostname: {model.get('service_hostname')}")
            else:
                status_lines.append("⏳ 모델 배포 진행 중...")
                status_lines.append("   잠시 후 다시 확인해주세요.")

            status_lines.append("")
            status_lines.append("=" * 50)

            return "\n".join(status_lines)

        except Exception as e:
            logger.error(f"Failed to check status: {e}")
            return f"❌ 상태 확인 실패: {str(e)}"

    # 이벤트 핸들러 연결
    refresh_templates_btn.click(
        fn=load_templates,
        inputs=[category_filter],
        outputs=[
            template_dropdown,
            template_info_display,
            template_error_msg,
            selected_template_id,
            template_list_state,
        ],
    )

    category_filter.change(
        fn=load_templates,
        inputs=[category_filter],
        outputs=[
            template_dropdown,
            template_info_display,
            template_error_msg,
            selected_template_id,
            template_list_state,
        ],
    )

    # 템플릿 선택 시 상세 정보 표시
    template_dropdown.change(
        fn=on_template_selected,
        inputs=[template_dropdown, template_list_state],
        outputs=[template_info_display, selected_template_id],
    )

    # Quick Service 원클릭 배포
    deploy_btn.click(
        fn=deploy_quick_service,
        inputs=[selected_template_id, template_list_state],
        outputs=[deployment_output, deployed_workflow_id_state, workflow_id_display],
    )

    # 배포 상태 확인
    check_status_btn.click(
        fn=check_deployment_status,
        inputs=[deployed_workflow_id_state],
        outputs=[status_output],
    )
