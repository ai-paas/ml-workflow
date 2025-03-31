import requests
import streamlit as st
from config.settings import get_settings

settings = get_settings()


def fetch_data(url):
    headers = {"Authorization": f"Bearer {st.session_state['token']}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    return response.json()


def request_inference(inference_service_url: str, service_hostname: str, model_name: str, image, text: list[str]):
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    url = f"{settings.REST_API_URL}/api/v1/inference"

    files = {"image": ("image.jpg", image, "image/jpeg")}

    data = {
        "infer_svc_url": inference_service_url,
        "service_hostname": service_hostname,
        "model_name": model_name,
        "image": image,
        "labels": text,
    }

    # print(f"request inference data = {data}")

    response = requests.post(url, data=data, files=files, headers=headers)
    print(f"raw response type = {type(response)}")
    return response.json()
