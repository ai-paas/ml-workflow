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

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# TODO: 추후 전역 레벨 관리로 수정 필요
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/metrics")
def get_metric_result(
    *,
    infer_svc_url: str = Body(...),
    service_hostname: str = Body(None),
    current_user: UserSchema = Depends(get_current_user),
):
    # 요청 URL
    url = f"{infer_svc_url}/metrics"

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
        logger.info(f"Headers:{headers}")
        # print("Request Body:", data)
        logger.info("\n=== Making Request ===")

        response = requests.get(
            url,
            cookies=kf.auth_session.session_cookie_dict,
            headers=headers,
        )

        # Response 정보 출력
        logger.info("\n=== Response Details ===")
        logger.info(f"Status Code: {response.status_code}")
        logger.info("Response Headers:")
        for key, value in response.headers.items():
            logger.info(f"{key}: {value}")

        logger.info("\nResponse Body:")
        # logger.info(response.json())
        result = response.text
        result_list = result.split("\n")
        logger.info(result_list)
        # logger.info(json.loads(result))
        # HTzcTP 상태 코드 확인
        response.raise_for_status()
        return result_list
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"\nHTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"\nError connecting to the server: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"\nTimeout error: {timeout_err}")
    except requests.exceptions.RequestException as err:
        logger.error(f"\nAn error occurred: {err}")
