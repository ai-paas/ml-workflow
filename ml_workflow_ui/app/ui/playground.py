"""Playground UI - 배포된 워크플로우로 추론 수행"""

import base64
import io
import logging
import os
import tempfile
from typing import List, Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def draw_predictions_on_image(image_path: str, predictions: List[dict], image_info: dict = None) -> np.ndarray:
    """이미지에 예측 결과(bbox, label)를 그려서 반환"""
    try:
        # 이미지 로드
        image = Image.open(image_path)
        original_width, original_height = image.size
        draw = ImageDraw.Draw(image)

        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # 크기 비율 계산 (bbox를 원본 이미지 크기로 스케일링)
        scale_x = 1.0
        scale_y = 1.0

        if image_info:
            model_input_size = image_info.get("model_input_size", {})
            model_height = model_input_size.get("height")
            model_width = model_input_size.get("width")

            if model_height and model_width:
                scale_x = original_width / model_width
                scale_y = original_height / model_height
                logger.info(
                    f"Scaling bbox: model_input={model_width}x{model_height}, "
                    f"original={original_width}x{original_height}, "
                    f"scale=({scale_x:.2f}, {scale_y:.2f})"
                )

        # 각 prediction에 대해 bbox와 label 그리기
        for pred in predictions:
            score = pred.get("score", 0)
            label = pred.get("label", "unknown")
            box = pred.get("box", [])

            # box는 [xmin, ymin, xmax, ymax] 형식
            if len(box) >= 4:
                # box가 리스트의 리스트인 경우 ([Array(4)]) 처리
                if isinstance(box[0], list):
                    box = box[0]

                # bbox를 원본 이미지 크기로 스케일링
                xmin = box[0] * scale_x
                ymin = box[1] * scale_y
                xmax = box[2] * scale_x
                ymax = box[3] * scale_y

                # 바운딩 박스 그리기 (빨간색, 두께 3)
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

                # 레이블 텍스트 (label + score)
                text = f"{label}: {score:.2f}"

                # 텍스트 배경 박스
                text_bbox = draw.textbbox((xmin, ymin - 25), text, font=font)
                draw.rectangle(text_bbox, fill="red")

                # 텍스트 그리기 (흰색)
                draw.text((xmin, ymin - 25), text, fill="white", font=font)

        # numpy array로 변환하여 반환
        return np.array(image)

    except Exception as e:
        logger.error(f"Failed to draw predictions on image: {e}")
        # 에러 발생 시 원본 이미지 반환
        try:
            image = Image.open(image_path)
            return np.array(image)
        except Exception:
            return None


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
                """### 3️⃣ 추론 실행
            모델 타입에 따라 이미지 또는 텍스트를 입력하세요."""
            )

            image_input = gr.Image(
                label="이미지 업로드 (KServe 모델용)",
                type="filepath",
                height=300,
                visible=True,
            )

            text_input = gr.Textbox(
                label="텍스트 입력 (Ollama 모델용)",
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
        """추론 실행"""
        if not app_state.api_client:
            return "❌ 로그인이 필요합니다.", None, None, None

        if not workflow_id or not component_id:
            return "❌ 워크플로우 ID와 Component ID를 입력해주세요.", None, None, None

        # 모델 타입 확인
        is_ollama_model = False
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

        # 입력 검증
        if is_ollama_model:
            if not text:
                return "❌ Ollama 모델은 텍스트 입력이 필요합니다.", None, None, None
        else:
            if not image_path:
                return "❌ KServe 모델은 이미지 업로드가 필요합니다.", None, None, None

        try:
            # 이미지가 있는 경우 먼저 메모리로 로드 (파일이 닫히는 것을 방지)
            loaded_image = None
            if image_path:
                try:
                    loaded_image = Image.open(image_path)
                    # 이미지를 메모리에 복사 (파일 핸들 닫힘 방지)
                    loaded_image.load()
                except Exception as e:
                    logger.error(f"Failed to load image: {e}")
                    return f"❌ 이미지 로드 실패: {str(e)}", None, None, None

            # 추론 요청
            result = app_state.api_client.inference(
                workflow_id=workflow_id,
                component_id=component_id,
                image_path=image_path,
                text=text,
            )

            status_msg = (
                f"✅ 추론 완료!\n"
                f"- Workflow: {result.get('workflow_id')}\n"
                f"- Component: {result.get('component_id')}"
            )

            # 통일된 응답 형식 처리
            result_data = result.get("result", {})
            model_type = result_data.get("model_type", "KServe")

            result_image = None
            result_text = None

            if model_type == "Ollama":
                # Ollama 모델 응답 처리
                result_text = result_data.get("response", "")
                status_msg += "\n- 모델 타입: LLM (Ollama)"
            else:
                # KServe 모델 응답 처리
                predictions = result_data.get("predictions")
                image_info = result_data.get("image_info", {})

                if predictions:
                    if isinstance(predictions, list):
                        # 리스트 형태의 predictions (bbox, label, score 포함)
                        logger.info(f"Predictions is a list with {len(predictions)} items")
                        logger.info(f"Image info: {image_info}")
                        try:
                            # 로드된 이미지가 있으면 사용, 없으면 경로로 다시 로드
                            if loaded_image:
                                # 이미지를 임시 파일로 저장하여 draw_predictions_on_image 함수에 전달
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                    loaded_image.save(tmp_file.name, format="JPEG")
                                    tmp_path = tmp_file.name

                                try:
                                    # 원본 이미지에 predictions 그리기 (image_info 전달)
                                    result_image = draw_predictions_on_image(tmp_path, predictions, image_info)
                                    logger.info(f"Successfully drew {len(predictions)} predictions on image")
                                finally:
                                    # 임시 파일 삭제
                                    try:
                                        os.unlink(tmp_path)
                                    except Exception:
                                        pass
                            else:
                                # 이미지가 없는 경우 경로로 다시 시도
                                result_image = draw_predictions_on_image(image_path, predictions, image_info)
                                logger.info(f"Successfully drew {len(predictions)} predictions on image")

                            # 상태 메시지에 감지된 객체 수 추가
                            status_msg += f"\n- 감지된 객체: {len(predictions)}개"
                        except Exception as e:
                            logger.error(f"Failed to draw predictions: {e}", exc_info=True)
                            # 에러 발생 시 원본 이미지 표시
                            try:
                                if loaded_image:
                                    result_image = np.array(loaded_image)
                                else:
                                    image = Image.open(image_path)
                                    result_image = np.array(image)
                            except Exception as img_err:
                                logger.error(f"Failed to load image for fallback: {img_err}")
                                pass

                    elif isinstance(predictions, str):
                        # Base64 인코딩된 이미지 문자열인 경우
                        try:
                            image_bytes = base64.b64decode(predictions)
                            image = Image.open(io.BytesIO(image_bytes))
                            result_image = np.array(image)
                            logger.info("Successfully decoded base64 image")
                        except Exception as e:
                            logger.error(f"Failed to decode base64 image: {e}")

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

                status_msg += "\n- 모델 타입: Object Detection (KServe)"

            return status_msg, result_image, result_text, result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"❌ 추론 실패: {str(e)}", None, None, None

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
