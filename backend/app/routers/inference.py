import base64
import json
import logging
from enum import Enum
from typing import Annotated

import requests
from config.settings import get_settings
from core.kubeflow.kubeflow_manager import KubeflowManager
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from schemas.user import UserSchema
from utils.authentication import get_current_user

router = APIRouter(prefix="/inference", tags=["Inference Service"])

# TODO: 추후 전역 레벨 관리로 수정 필요
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@router.post("")
async def external_inference(
    *,
    infer_svc_url: Annotated[str, Form()],
    service_hostname: Annotated[str, Form()],
    model_name: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    labels: Annotated[list[str], Form()],
    current_user: UserSchema = Depends(get_current_user),
):
    # TODO: mocking data. 이후 외부에서 인자로 받아야함.
    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error(f"error occured when get image bytes : {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="receive unreadable image. check image file"
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    # 데이터 준비
    payload = {"image": image_base64, "text": [labels]}  # [['a cat', 'remote control']]
    # infer_svc_url = "0.0.0.0:8000"
    # 요청 URL

    # TODO: 현재 Custom Model Server 사용하므로 v2로 띄워지도록 유지.
    # 이후 Custom Model Server가 아니라 다른 모델 서버로 띄워야 할 경우 내부적으로 관리할수있도록 개선 필요.
    url = f"{infer_svc_url}/v2/models/{model_name}/infer"

    kf = KubeflowManager()
    # 헤더 설정
    headers = {"Content-Type": "application/json"}
    if service_hostname:
        headers["Host"] = service_hostname

    # headers["kubeflow-userid"] = "user@example.com"

    # TODO : mocking data

    # KServe v2 protocol request 형식
    data = {"inputs": [{"name": "INPUT_1", "shape": [1], "datatype": "BYTES", "data": [payload]}]}

    try:
        # 요청 보내기 (verbose 모드와 유사하게 정보 출력)
        logger.info("\n=== Request Details ===")
        logger.info(f"URL: {url}")
        logger.info("Headers:", headers)
        # print("Request Body:", data)
        logger.info("\n=== Making Request ===")

        response = requests.post(url, cookies=kf.auth_session.session_cookie_dict, headers=headers, json=data)

        # Response 정보 출력
        logger.info("\n=== Response Details ===")
        logger.info(f"Status Code: {response.status_code}")
        logger.info("Response Headers:")
        for key, value in response.headers.items():
            print(f"{key}: {value}")

        logger.info(f"response ={response.content}")
        result = response.json()
        # HTTP 상태 코드 확인
        response.raise_for_status()
        # TODO: 다중 input에 대한 처리 필요.
        return result["outputs"][0]["data"][0]
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"\nHTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"\nError connecting to the server: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"\nTimeout error: {timeout_err}")
    except requests.exceptions.RequestException as err:
        logger.error(f"\nAn error occurred: {err}")


@router.post("/status")
def check_status(
    *,
    infer_svc_url: str = Body(...),
    service_hostname: str = Body(None),
    model_name: str = Body(...),
    current_user: UserSchema = Depends(get_current_user),
):
    # 요청 URL
    url = f"{infer_svc_url}/v2/models/{model_name}/ready"

    kf = KubeflowManager()
    # 헤더 설정
    headers = {
        # "Content-Type": "application/json"
    }
    if service_hostname:
        headers["Host"] = service_hostname

    # TODO : mocking data

    try:
        # 요청 보내기 (verbose 모드와 유사하게 정보 출력)
        logger.info("\n=== Request Details ===")
        logger.info(f"URL: {url}")
        logger.info("Headers:", headers)
        # print("Request Body:", data)
        logger.info("\n=== Making Request ===")

        response = requests.get(url, cookies=kf.auth_session.session_cookie_dict, headers=headers)

        # Response 정보 출력
        logger.info("\n=== Response Details ===")
        logger.info(f"Status Code: {response.status_code}")
        logger.info("Response Headers:")
        for key, value in response.headers.items():
            logger.info(f"{key}: {value}")

        logger.info("\nResponse Body:")
        result = response.text
        logger.info(json.loads(result))
        # HTTP 상태 코드 확인
        response.raise_for_status()
        return json.loads(result)["ready"]
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"\nHTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"\nError connecting to the server: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"\nTimeout error: {timeout_err}")
    except requests.exceptions.RequestException as err:
        logger.error(f"\nAn error occurred: {err}")
