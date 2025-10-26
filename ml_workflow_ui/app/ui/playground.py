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

            workflow_dropdown = gr.Dropdown(
                label="📦 워크플로우 선택", choices=[], interactive=True, info="추론을 실행할 워크플로우를 선택하세요"
            )

            workflow_info_display = gr.Markdown(value="워크플로우를 선택하면 상세 정보가 표시됩니다.", label="📋 워크플로우 정보")

            status_message = gr.Textbox(label="⚠️ 상태 메시지", interactive=False, visible=False, lines=2)

            gr.Markdown("---")

            gr.Markdown(
                """
            ### 2️⃣ 모델 선택

            배포된 모델을 선택하세요.
            """
            )

            model_dropdown = gr.Dropdown(label="🤖 모델 선택", choices=[], interactive=True, info="추론을 실행할 모델을 선택하세요")

            model_info_display = gr.Markdown(value="모델을 선택하면 상세 정보가 표시됩니다.", label="📋 모델 정보")

        with gr.Column(scale=1):
            gr.Markdown("### 3️⃣ 추론 실행")

            image_input = gr.Image(
                label="이미지 업로드",
                type="filepath",
                height=300,
            )

            gr.Markdown("#### 텍스트 입력 (레이블)")

            # 동적 텍스트 입력 필드
            with gr.Column():
                text_inputs = []
                for i in range(5):
                    text_input = gr.Textbox(
                        label=f"텍스트 {i+1}",
                        placeholder="예: a cat, a remote control",
                        visible=(i == 0),  # 첫 번째만 기본으로 표시
                    )
                    text_inputs.append(text_input)

                add_text_btn = gr.Button("텍스트 입력 추가", size="sm")
                visible_count_state = gr.State(1)

            run_inference_btn = gr.Button("추론 실행 🚀", variant="primary", size="lg")

            gr.Markdown("### 📊 추론 결과")

            inference_status = gr.Textbox(label="실행 상태", interactive=False)

            inference_output_image = gr.Image(
                label="결과 이미지",
                type="numpy",
                height=400,
            )

            inference_output_json = gr.JSON(label="상세 결과 (JSON)")

    def load_workflows():
        """워크플로우 목록 로드"""
        if not app_state.api_client:
            error_msg = "❌ 로그인이 필요합니다."
            return (
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
            result = app_state.api_client.get_workflows(is_template=False)
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

            # 드롭다운 선택지 생성 (이름과 ID 표시)
            choices = [(f"{w.get('name')} (ID: {w.get('id')[:8]}...)", w.get("id")) for w in active_workflows]

            success_msg = f"✅ {len(active_workflows)}개의 워크플로우를 찾았습니다."
            return (
                gr.update(choices=choices, value=None),
                "워크플로우를 선택하면 상세 정보가 표시됩니다.",
                gr.update(visible=True, value=success_msg),
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
            return ("워크플로우 정보를 찾을 수 없습니다.", gr.update(choices=[], value=None), "모델을 선택하면 상세 정보가 표시됩니다.", [], "", "")

        # 워크플로우 상세 정보
        workflow_info = f"""
### 📋 {selected.get('name', 'N/A')}

**🆔 Workflow ID:** `{selected.get('id', 'N/A')}`

**📊 상태:** {selected.get('status', 'N/A')}

**📁 카테고리:** {selected.get('category', 'N/A')}

**📝 설명:** {selected.get('description', '설명이 없습니다.')}

**📅 생성일:** {selected.get('created_at', 'N/A')}

---

배포된 모델을 선택하세요.
        """

        # 배포된 모델 로드
        try:
            if not app_state.api_client:
                logger.error("No api_client")
                return (workflow_info, gr.update(choices=[], value=None), "모델을 선택하면 상세 정보가 표시됩니다.", [], workflow_id, "")

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
                return (workflow_info, gr.update(choices=[], value=None), "⚠️ 배포된 모델이 없습니다.", [], workflow_id, "")

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
                return (workflow_info, gr.update(choices=[], value=None), "⚠️ 모델 선택지를 생성할 수 없습니다.", [], workflow_id, "")

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
            return (workflow_info, gr.update(choices=[], value=None), f"❌ 모델 로드 실패: {str(e)}", [], workflow_id, "")

    def on_model_selected(component_id: Optional[str], models_list: list):
        """모델 선택 시 상세 정보 표시"""
        if not component_id or not models_list:
            return "모델을 선택하면 상세 정보가 표시됩니다.", component_id or ""

        # 선택된 모델 찾기
        selected = None
        for m in models_list:
            if m.get("component_id") == component_id:
                selected = m
                break

        if not selected:
            return "모델 정보를 찾을 수 없습니다.", ""

        # 모델 상세 정보
        model_info = f"""
### 🤖 {selected.get('model_name', 'N/A')}

**🆔 Component ID:** `{selected.get('component_id', 'N/A')}`

**📊 상태:** {selected.get('status', 'N/A').upper()}

**🚀 서비스명:** `{selected.get('service_name', 'N/A')}`

**🌐 Hostname:** `{selected.get('service_hostname', 'N/A')}`

**🔗 Gateway URL:** `{selected.get('gateway_url', 'N/A')}`

**📅 배포 시간:** {selected.get('deployed_at', 'N/A')}

---

이미지를 업로드하고 텍스트 레이블을 입력한 후 추론을 실행하세요.
        """

        return model_info, component_id

    def add_text_input_field(visible_count: int):
        """텍스트 입력 필드 추가 (가시성 토글)"""
        if visible_count >= 5:
            return visible_count, *[gr.update() for _ in range(5)]

        new_count = visible_count + 1
        updates = []
        for i in range(5):
            if i < new_count:
                updates.append(gr.update(visible=True))
            else:
                updates.append(gr.update(visible=False))

        return new_count, *updates

    def run_inference(
        workflow_id: str,
        component_id: str,
        image_path: Optional[str],
        *text_values,
    ):
        """추론 실행"""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다.", None, None

        if not workflow_id or not component_id:
            return "❌ 워크플로우 ID와 Component ID를 입력해주세요.", None, None

        if not image_path:
            return "❌ 이미지를 업로드해주세요.", None, None

        # 빈 텍스트 입력 제외
        labels = [text for text in text_values if text and text.strip()]

        if not labels:
            return "❌ 최소 1개의 텍스트 레이블을 입력해주세요.", None, None

        try:
            # 추론 요청
            result = app_state.api_client.inference(
                workflow_id=workflow_id,
                component_id=component_id,
                image_path=image_path,
                labels=labels,
            )

            status_msg = (
                f"✅ 추론 완료!\n"
                f"- Workflow: {result.get('workflow_id')}\n"
                f"- Component: {result.get('component_id')}\n"
                f"- Labels: {', '.join(result.get('labels', []))}"
            )

            # 결과 이미지 처리
            predictions = result.get("predictions")
            result_image = None

            if predictions:
                if isinstance(predictions, str):
                    # Base64 인코딩된 이미지 문자열인 경우
                    try:
                        image_bytes = base64.b64decode(predictions)
                        image = Image.open(io.BytesIO(image_bytes))
                        result_image = np.array(image)
                        logger.info("Successfully decoded base64 image")
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        # 디코딩 실패 시 predictions 내용 확인
                        pred_len = len(predictions) if isinstance(predictions, str) else "N/A"
                        logger.debug(f"predictions type: {type(predictions)}, length: {pred_len}")

                elif isinstance(predictions, dict):
                    # JSON 객체인 경우 (바운딩 박스, 점수 등)
                    logger.info(f"Predictions is a dict: {predictions.keys()}")
                    # 이미지 데이터가 dict 안에 있을 수 있음
                    if "image" in predictions and isinstance(predictions["image"], str):
                        try:
                            image_bytes = base64.b64decode(predictions["image"])
                            image = Image.open(io.BytesIO(image_bytes))
                            result_image = np.array(image)
                            logger.info("Successfully decoded image from predictions dict")
                        except Exception as e:
                            logger.error(f"Failed to decode image from dict: {e}")

            return status_msg, result_image, result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"❌ 추론 실패: {str(e)}", None, None

    # 이벤트 핸들러 연결
    refresh_workflows_btn.click(
        fn=load_workflows,
        inputs=[],
        outputs=[
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
        inputs=[model_dropdown, deployed_models_state],
        outputs=[model_info_display, selected_component_id_state],
    )

    add_text_btn.click(
        fn=add_text_input_field,
        inputs=[visible_count_state],
        outputs=[visible_count_state] + text_inputs,
    )

    run_inference_btn.click(
        fn=run_inference,
        inputs=[selected_workflow_id_state, selected_component_id_state, image_input] + text_inputs,
        outputs=[inference_status, inference_output_image, inference_output_json],
    )
