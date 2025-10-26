"""워크플로우 템플릿 생성 유틸리티

facebook/detr-resnet-50 모델을 사용하는 Object Detection 워크플로우 템플릿 생성
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_client import APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_detr_template(api_client: APIClient, model_id: int) -> dict:
    """DETR Object Detection 워크플로우 템플릿 생성

    Args:
        api_client: API 클라이언트
        model_id: facebook/detr-resnet-50 모델의 ID

    Returns:
        생성된 템플릿 정보
    """
    template_data = {
        "name": "DETR Object Detection Template",
        "description": "facebook/detr-resnet-50 모델을 사용한 Object Detection 워크플로우 템플릿",
        "category": "Object Detection",
        "is_template": True,
        "workflow_definition": {
            "components": [
                {
                    "component_id": "start",
                    "name": "Start",
                    "type": "START",
                    "config": {},
                },
                {
                    "component_id": "detr_model",
                    "name": "DETR Object Detection",
                    "type": "MODEL",
                    "model_id": model_id,
                    "config": {
                        "model_name": "facebook/detr-resnet-50",
                        "task": "object-detection",
                        "framework": "transformers",
                    },
                },
                {
                    "component_id": "end",
                    "name": "End",
                    "type": "END",
                    "config": {},
                },
            ],
            "connections": [
                {
                    "source_component_id": "start",
                    "target_component_id": "detr_model",
                    "connection_type": "DATA",
                    "config": {},
                },
                {
                    "source_component_id": "detr_model",
                    "target_component_id": "end",
                    "connection_type": "DATA",
                    "config": {},
                },
            ],
        },
    }

    # 템플릿 생성 API 호출
    result = api_client._post("/api/v1/workflows/templates", data=template_data)
    return result


def main():
    parser = argparse.ArgumentParser(description="워크플로우 템플릿 생성")
    parser.add_argument("--username", default="surromind", help="사용자 이름")
    parser.add_argument("--password", required=True, help="비밀번호")
    parser.add_argument("--model-id", type=int, required=True, help="모델 ID")

    args = parser.parse_args()

    # 로그인
    logger.info("로그인 중...")
    token = APIClient.authenticate(args.username, args.password)

    if not token:
        logger.error("로그인 실패")
        return

    logger.info("로그인 성공")
    api_client = APIClient(token)

    # 템플릿 생성
    logger.info(f"DETR Object Detection 템플릿 생성 중... (Model ID: {args.model_id})")
    template = create_detr_template(api_client, args.model_id)

    logger.info("템플릿 생성 완료!")
    logger.info(f"템플릿 ID: {template.get('id')}")
    logger.info(f"템플릿 이름: {template.get('name')}")
    logger.info(f"카테고리: {template.get('category')}")

    print("\n" + "=" * 60)
    print("템플릿 생성 완료!")
    print("=" * 60)
    print(json.dumps(template, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
