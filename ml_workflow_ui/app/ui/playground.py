"""Playground UI - 배포된 워크플로우로 추론 수행"""

import base64
import io
import logging
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_playground_ui(app_state):
    """Playground UI 생성"""

    # 상태 저장
    workflows_state = gr.State([])
    selected_workflow_id_state = gr.State("")
    deployed_models_state = gr.State([])
    selected_component_id_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
            ### 1️⃣ 워크플로우 선택

            배포된 워크플로우를 선택하세요.
            """
            )

            refresh_workflows_btn = gr.Button("🔄 워크플로우 목록 새로고침", variant="secondary", size="lg")

            workflow_count_message = gr.Markdown(value="", visible=False)

            workflow_dropdown = gr.Dropdown(
                label="📦 워크플로우 선택", choices=[], interactive=True, info="추론을 실행할 워크플로우를 선택하세요"
            )

            workflow_info_display = gr.Markdown(
                value="워크플로우를 선택하면 \
상세 정보가 표시됩니다.",
                label="📋 워크플로우 정보",
            )

            status_message = gr.Textbox(label="⚠️ 상태 메시지", interactive=False, visible=False, lines=2)

            gr.Markdown("---")

            gr.Markdown(
                """
            ### 2️⃣ 모델 선택

            배포된 모델을 선택하세요.
            """
            )

            model_dropdown = gr.Dropdown(
                label="🤖 모델 선택", choices=[], interactive=True, info="추론을 실행할 모델을 선택하세요"
            )

            model_info_display = gr.Markdown(value="모델을 선택하면 상세 정보가 표시됩니다.", label="📋 모델 정보")

        with gr.Column(scale=1):
            gr.Markdown(
                """### 3️⃣ 워크플로 테스트 실행
            백엔드 `POST .../test/ml`(이미지·ODM) 또는 `POST .../test/rag`(텍스트·RAG/LLM)로 **전체 그래프**를 실행합니다."""
            )

            image_input = gr.Image(
                label="이미지 업로드 (ODM / test/ml)",
                type="filepath",
                height=300,
                visible=True,
            )

            text_input = gr.Textbox(
                label="텍스트 입력 (RAG·LLM / test/rag)",
                placeholder="텍스트를 입력하세요...",
                lines=5,
                visible=False,
            )

            run_inference_btn = gr.Button("추론 실행 🚀", variant="primary", size="lg")

            gr.Markdown("### 📊 추론 결과")

            inference_status = gr.Textbox(label="실행 상태", interactive=False)

            inference_output_image = gr.Image(
                label="결과 이미지 (KServe 모델용)",
                type="numpy",
                height=400,
                visible=True,
            )

            inference_output_text = gr.Textbox(
                label="LLM 응답 (Ollama 모델용)",
                lines=10,
                visible=False,
            )

            inference_output_json = gr.JSON(label="상세 결과 (JSON)")

    def load_workflows():
        """워크플로우 목록 로드"""
        if not app_state.api_client:
            error_msg = "❌ 로그인이 필요합니다."
            return (
                gr.update(value=error_msg, visible=True),
                gr.update(choices=[], value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                gr.update(choices=[], value=None),
                "모델을 선택하면 상세 정보가 표시됩니다.",
                [],
                "",
                [],
                "",
            )

        try:
            result = app_state.api_client.get_workflows()
            workflows = result.get("items", [])

            # ACTIVE 상태이고 배포된 모델이 있는 워크플로우만 필터링
            active_workflows = []
            for w in workflows:
                if w.get("status") == "ACTIVE":
                    # 배포된 모델이 있는지 확인
                    try:
                        models_result = app_state.api_client.get_deployed_models(w.get("id"))
                        if models_result.get("deployed_models"):
                            active_workflows.append(w)
                    except Exception:
                        pass

            if not active_workflows:
                msg = "⚠️ 배포된 워크플로우가 없습니다."
                return (
                    gr.update(value=msg, visible=True),
                    gr.update(choices=[], value=None),
                    "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                    gr.update(visible=True, value=msg),
                    gr.update(choices=[], value=None),
                    "모델을 선택하면 상세 정보가 표시됩니다.",
                    [],
                    "",
                    [],
                    "",
                )

            # 드롭다운 선택지 생성 (워크플로우명 + 템플릿명)
            choices = []
            for w in active_workflows:
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

            success_msg = f"✅ {len(active_workflows)}개의 워크플로우를 찾았습니다."
            return (
                gr.update(value=success_msg, visible=True),
                gr.update(choices=choices, value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                "모델을 선택하면 상세 정보가 표시됩니다.",
                active_workflows,
                "",
                [],
                "",
            )

        except Exception as e:
            logger.error(f"Failed to load workflows: {e}")
            error_msg = f"❌ 워크플로우 로드 실패: {str(e)}"
            return (
                gr.update(value=error_msg, visible=True),
                gr.update(choices=[], value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=error_msg),
                gr.update(choices=[], value=None),
                "모델을 선택하면 상세 정보가 표시됩니다.",
                [],
                "",
                [],
                "",
            )

    def on_workflow_selected(workflow_id: Optional[str], workflows_list: list):
        """워크플로우 선택 시 상세 정보 및 모델 로드"""
        logger.info(f"on_workflow_selected called with workflow_id: {workflow_id}")

        if not workflow_id or not workflows_list:
            logger.warning("No workflow_id or workflows_list")
            return (
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(choices=[], value=None),
                "모델을 선택하면 상세 정보가 표시됩니다.",
                [],
                workflow_id or "",
                "",
            )

        # 선택된 워크플로우 찾기
        selected = None
        for w in workflows_list:
            if w.get("id") == workflow_id:
                selected = w
                break

        if not selected:
            logger.error(f"Workflow {workflow_id} not found in workflows_list")
            return (
                "워크플로우 정보를 찾을 수 없습니다.",
                gr.update(choices=[], value=None),
                "모델을 선택하면 상세 정보가 표시됩니다.",
                [],
                "",
                "",
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

        # 워크플로우 상세 정보 (상태, 카테고리, 템플릿명만 표시)
        workflow_info = f"""
### 📋 {selected.get('name', 'N/A')}

**📊 상태:** {selected.get('status', 'N/A')}

**📁 카테고리:** {selected.get('category', 'N/A')}

**📑 템플릿:** {template_name}
        """

        # 배포된 모델 로드
        try:
            if not app_state.api_client:
                logger.error("No api_client")
                return (
                    workflow_info,
                    gr.update(choices=[], value=None),
                    "모델을 선택하면 상세 정보가 표시됩니다.",
                    [],
                    workflow_id,
                    "",
                )

            logger.info(f"Fetching deployed models for workflow {workflow_id}")
            result = app_state.api_client.get_deployed_models(workflow_id)
            models = result.get("deployed_models", [])

            logger.info(f"Found {len(models)} models")
            for m in models:
                logger.info(
                    f"  - Model: {m.get('model_name')}, Component: {m.get('component_id')}, Status: {m.get('status')}"
                )

            if not models:
                logger.warning("No models found")
                return (
                    workflow_info,
                    gr.update(choices=[], value=None),
                    "⚠️ 배포된 모델이 없습니다.",
                    [],
                    workflow_id,
                    "",
                )

            # 모델 드롭다운 선택지 생성 (모든 모델 표시, deployed 필터링 제거)
            model_choices = [
                (
                    f"{m.get('model_name', 'N/A')} ({m.get('component_id')}) - [{m.get('status', 'unknown')}]",
                    m.get("component_id"),
                )
                for m in models
            ]

            logger.info(f"Created {len(model_choices)} model choices")

            if not model_choices:
                logger.warning("No model choices created")
                return (
                    workflow_info,
                    gr.update(choices=[], value=None),
                    "⚠️ 모델 선택지를 생성할 수 없습니다.",
                    [],
                    workflow_id,
                    "",
                )

            logger.info(f"Returning model choices: {model_choices}")
            return (
                workflow_info,
                gr.update(choices=model_choices, value=None),
                f"✅ {len(model_choices)}개의 모델을 찾았습니다.",
                models,
                workflow_id,
                "",
            )

        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            return (
                workflow_info,
                gr.update(choices=[], value=None),
                f"❌ 모델 로드 실패: {str(e)}",
                [],
                workflow_id,
                "",
            )

    def on_model_selected(component_id: Optional[str], models_list: list, workflow_id: str):
        """모델 선택 시 상세 정보 표시 및 입력 필드 업데이트"""
        if not component_id or not models_list:
            return (
                "모델을 선택하면 상세 정보가 표시됩니다.",
                component_id or "",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        # 선택된 모델 찾기
        selected = None
        for m in models_list:
            if m.get("component_id") == component_id:
                selected = m
                break

        if not selected:
            return (
                "모델 정보를 찾을 수 없습니다.",
                "",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        # 워크플로우에서 모델 타입 확인
        is_ollama_model = False
        if workflow_id and app_state.api_client:
            try:
                workflow_info = app_state.api_client.get_workflow(workflow_id)
                components = workflow_info.get("components", [])
                for comp in components:
                    if comp.get("id") == component_id:
                        model_info = comp.get("model")
                        if model_info:
                            model_format = model_info.get("format_info", {}).get("name", "").lower()
                            provider_name = model_info.get("provider_info", {}).get("name", "").lower()
                            if provider_name == "ollama" and model_format == "gguf":
                                is_ollama_model = True
                        break
            except Exception as e:
                logger.warning(f"Failed to check model type: {e}")

        # 모델 상세 정보
        model_info = f"""
### 🤖 {selected.get('model_name', 'N/A')}

**🆔 Component ID:** `{selected.get('component_id', 'N/A')}`

**📊 상태:** {selected.get('status', 'N/A').upper()}

**🚀 서비스명:** `{selected.get('service_name', 'N/A')}`

**🌐 Hostname:** `{selected.get('service_hostname', 'N/A')}`

**🔗 Gateway URL:** `{selected.get('gateway_url', 'N/A')}`

**📅 배포 시간:** {selected.get('deployed_at', 'N/A')}
        """

        # 모델 타입에 따라 입력 필드 표시/숨김
        if is_ollama_model:
            return (
                model_info,
                component_id,
                gr.update(visible=False),  # image_input 숨김
                gr.update(visible=True),  # text_input 표시
                gr.update(visible=False),  # inference_output_image 숨김
                gr.update(visible=True),  # inference_output_text 표시
            )
        else:
            return (
                model_info,
                component_id,
                gr.update(visible=True),  # image_input 표시
                gr.update(visible=False),  # text_input 숨김
                gr.update(visible=True),  # inference_output_image 표시
                gr.update(visible=False),  # inference_output_text 숨김
            )

    def run_inference(
        workflow_id: str,
        component_id: str,
        image_path: Optional[str],
        text: Optional[str],
    ):
        """워크플로 전체 그래프 테스트 (RAG/LLM: POST .../test/rag, ODM: POST .../test/ml)."""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다.", None, None, None

        if not workflow_id:
            return "❌ 워크플로우 ID를 입력해주세요.", None, None, None

        hint = ""
        if not component_id:
            hint = "\n(Component ID는 표시용입니다. API는 워크플로 전체 그래프를 실행합니다.)"

        try:
            if image_path:
                result = app_state.api_client.test_ml_workflow(workflow_id, image_path)
                status_msg = f"✅ ML(ODM) 워크플로 테스트 완료!{hint}\n- workflow_id: {result.get('workflow_id')}"
                result_image = None
                fr = result.get("final_result")
                if fr:
                    try:
                        raw = base64.b64decode(fr)
                        img = Image.open(io.BytesIO(raw))
                        result_image = np.array(img)
                    except Exception as e:
                        logger.error(f"Failed to decode final_result image: {e}")
                return status_msg, result_image, None, result

            if text:
                result = app_state.api_client.test_rag_workflow(workflow_id, text)
                result_text = result.get("final_result") or ""
                status_msg = (
                    f"✅ RAG/LLM 워크플로 테스트 완료!{hint}\n"
                    f"- workflow_id: {result.get('workflow_id')}\n"
                    f"- execution_order: {result.get('execution_order', [])}"
                )
                return status_msg, None, result_text, result

            return (
                "❌ RAG/LLM 워크플로는 텍스트, ODM 워크플로는 이미지를 입력하세요.",
                None,
                None,
                None,
            )
        except Exception as e:
            logger.error(f"Workflow test failed: {e}")
            return f"❌ 테스트 실패: {str(e)}", None, None, None

    # 이벤트 핸들러 연결
    refresh_workflows_btn.click(
        fn=load_workflows,
        inputs=[],
        outputs=[
            workflow_count_message,
            workflow_dropdown,
            workflow_info_display,
            status_message,
            model_dropdown,
            model_info_display,
            workflows_state,
            selected_workflow_id_state,
            deployed_models_state,
            selected_component_id_state,
        ],
    )

    workflow_dropdown.change(
        fn=on_workflow_selected,
        inputs=[workflow_dropdown, workflows_state],
        outputs=[
            workflow_info_display,
            model_dropdown,
            model_info_display,
            deployed_models_state,
            selected_workflow_id_state,
            selected_component_id_state,
        ],
    )

    model_dropdown.change(
        fn=on_model_selected,
        inputs=[model_dropdown, deployed_models_state, selected_workflow_id_state],
        outputs=[
            model_info_display,
            selected_component_id_state,
            image_input,
            text_input,
            inference_output_image,
            inference_output_text,
        ],
    )

    run_inference_btn.click(
        fn=run_inference,
        inputs=[selected_workflow_id_state, selected_component_id_state, image_input, text_input],
        outputs=[inference_status, inference_output_image, inference_output_text, inference_output_json],
    )

    # 페이지 로드를 위한 함수와 출력 컴포넌트 반환
    return {
        "load_fn": load_workflows,
        "load_outputs": [
            workflow_count_message,
            workflow_dropdown,
            workflow_info_display,
            status_message,
            model_dropdown,
            model_info_display,
            workflows_state,
            selected_workflow_id_state,
            deployed_models_state,
            selected_component_id_state,
        ],
    }
