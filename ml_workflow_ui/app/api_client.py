"""API 클라이언트 - Backend API와 통신"""

import base64
import logging
from typing import Any, Dict, List, Optional

import requests
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class APIClient:
    """Backend API 클라이언트"""

    def __init__(self, token: str):
        self.token = token
        self.base_url = settings.REST_API_URL
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """GET 요청"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            # 더 자세한 에러 메시지
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except Exception:
                    error_detail = e.response.text or str(e)
                raise Exception(f"API 오류 ({e.response.status_code}): {error_detail}")
            raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def _post(self, endpoint: str, data: Optional[Dict] = None, files=None, params: Optional[Dict] = None) -> Any:
        """POST 요청"""
        url = f"{self.base_url}{endpoint}"
        headers = self.headers.copy()

        try:
            if files:
                # multipart/form-data인 경우 Content-Type 제거
                headers.pop("Content-Type", None)
                response = requests.post(url, headers=headers, data=data, files=files, params=params)
            elif data is not None:
                # data가 있을 때만 json으로 전송
                response = requests.post(url, headers=headers, json=data, params=params)
            else:
                # data가 None이면 Content-Type 제거하고 Query 파라미터만 사용
                headers.pop("Content-Type", None)
                response = requests.post(url, headers=headers, params=params)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            # 더 자세한 에러 메시지
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except Exception:
                    error_detail = e.response.text or str(e)
                raise Exception(f"API 오류 ({e.response.status_code}): {error_detail}")
            raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def _delete(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """DELETE 요청"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.delete(url, headers=self.headers, params=params)
            response.raise_for_status()

            # DELETE는 204 No Content를 반환할 수 있으므로 빈 응답 처리
            if response.status_code == 204:
                return {"message": "Successfully deleted"}

            # 202 Accepted - 비동기 처리 시작 (run_id 등 포함)
            if response.status_code == 202:
                try:
                    return response.json()
                except Exception:
                    return {"message": "Deletion accepted and processing"}

            # 기타 응답 본문이 있으면 JSON 파싱
            try:
                return response.json()
            except Exception:
                return {"message": "Successfully deleted"}

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            # 더 자세한 에러 메시지
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except Exception:
                    error_detail = e.response.text or str(e)
                raise Exception(f"API 오류 ({e.response.status_code}): {error_detail}")
            raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    # ============= Authentication =============
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[str]:
        """로그인 - 토큰 반환"""
        try:
            response = requests.post(
                f"{settings.REST_API_URL}/api/v1/authentications/token",
                data={"username": username, "password": password},
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    # ============= Component Types =============
    def get_component_types(self) -> List[Dict]:
        """사용 가능한 컴포넌트 타입 조회"""
        return self._get("/api/v1/workflows/component-types")

    # ============= Workflow Templates =============
    def get_workflow_templates(
        self,
        category: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Dict:
        """워크플로우 템플릿 목록 조회

        Returns:
            {
                "total": int,
                "items": List[Dict]  # 템플릿 목록
            }
        """
        params = {}
        if category:
            params["category"] = category
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        return self._get("/api/v1/workflows/templates", params=params)

    def get_workflow_template(self, template_id: str) -> Dict:
        """워크플로우 템플릿 상세 조회"""
        return self._get(f"/api/v1/workflows/templates/{template_id}")

    def create_workflow_template(
        self,
        name: str,
        workflow_definition: Optional[Dict] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict:
        """워크플로우 템플릿 생성

        Args:
            name: 템플릿 이름 (필수)
            workflow_definition: 워크플로우 정의 (선택)
                - components: 컴포넌트 목록 (START, END, MODEL)
                - connections: 컴포넌트 간 연결 정보
            description: 템플릿 설명 (선택)
            category: 템플릿 카테고리 (선택)

        Returns:
            생성된 템플릿 정보
        """
        data = {"name": name}
        if description:
            data["description"] = description
        if category:
            data["category"] = category
        if workflow_definition:
            data["workflow_definition"] = workflow_definition
        return self._post("/api/v1/workflows/templates", data=data)

    def clone_from_template(self, template_id: str, workflow_name: str, service_id: Optional[int] = None) -> Dict:
        """템플릿으로부터 워크플로우 생성"""
        params = {"workflow_name": workflow_name}
        if service_id:
            params["service_id"] = service_id
        # Query 파라미터로 전송 (라우터가 Query()로 정의됨)
        return self._post(f"/api/v1/workflows/templates/{template_id}/clone", data=None, params=params)

    # ============= Workflow =============
    def get_workflows(
        self,
        status: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Dict:
        """워크플로우 목록 조회 (템플릿 제외)"""
        params = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        if status:
            params["status"] = status
        return self._get("/api/v1/workflows", params=params)

    def get_workflow(self, workflow_id: str) -> Dict:
        """워크플로우 상세 조회"""
        return self._get(f"/api/v1/workflows/{workflow_id}")

    def execute_workflow(self, workflow_id: str, parameters: Optional[Dict] = None) -> Dict:
        """워크플로우 실행"""
        data = {"parameters": parameters or {}}
        return self._post(f"/api/v1/workflows/{workflow_id}/execute", data=data)

    def get_workflow_status(self, workflow_id: str) -> Dict:
        """워크플로우 실행 상태 조회"""
        return self._get(f"/api/v1/workflows/{workflow_id}/status")

    # ============= Deployed Models =============
    def get_deployed_models(self, workflow_id: str) -> Dict:
        """워크플로우에 배포된 모델 목록 조회"""
        return self._get(f"/api/v1/workflows/{workflow_id}/models")

    def get_all_deployed_workflows(self) -> Dict:
        """모든 배포된 워크플로우 목록 조회 (템플릿 제외)"""
        return self.get_workflows()

    def cleanup_workflow(self, workflow_id: str) -> Dict:
        """워크플로우 리소스 정리 (배포된 서비스 삭제)"""
        return self._post(f"/api/v1/workflows/{workflow_id}/cleanup")

    def delete_workflow(self, workflow_id: str) -> Dict:
        """
        워크플로우 삭제 시작

        KServe InferenceService 삭제를 위한 Kubeflow Pipeline을 시작하고 cleanup_run_id를 반환합니다.

        Returns:
            {
                "workflow_id": str,
                "cleanup_run_id": str,
                "status": "cleanup_in_progress",
                ...
            }
        """
        return self._delete(f"/api/v1/workflows/{workflow_id}")

    def finalize_workflow_deletion(self, workflow_id: str, run_id: str) -> Dict:
        """
        워크플로우 삭제 완료 확인 및 DB 삭제

        Args:
            workflow_id: 워크플로우 ID
            run_id: Cleanup pipeline run ID

        Returns:
            {
                "status": "completed" | "in_progress" | "failed",
                "deleted_from_db": bool,
                ...
            }
        """
        params = {"run_id": run_id}
        return self._post(f"/api/v1/workflows/{workflow_id}/finalize-deletion", params=params)

    # ============= Inference =============
    def inference(
        self,
        workflow_id: str,
        component_id: str,
        image_path: str,
    ) -> Dict:
        """배포된 모델에 추론 요청

        Args:
            workflow_id: 워크플로우 ID (path parameter)
            component_id: 컴포넌트 ID (path parameter)
            image_path: 이미지 파일 경로

        Returns:
            {
                "workflow_id": str,
                "component_id": str,
                "predictions": str | dict,  # base64 문자열 또는 JSON 객체
                "model_info": dict
            }
        """
        with open(image_path, "rb") as f:
            files = {"image": ("image.jpg", f, "image/jpeg")}

            # multipart/form-data 요청
            # component_id는 경로 파라미터로 전달
            url = f"{self.base_url}/api/v1/workflows/{workflow_id}/models/{component_id}/inference"
            headers = {"Authorization": f"Bearer {self.token}"}

            response = requests.post(
                url,
                headers=headers,
                files=files,
            )
            response.raise_for_status()
            return response.json()
