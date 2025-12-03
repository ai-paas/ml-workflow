import mlflow


class BaseModelManager:
    """
    모델 관리자 기본 클래스
    프레임워크별 모델 관리자가 상속받아 사용

    1. 프레임워크별 모델 load 후 클래스 저장
    2. 프레임워크별 모델 추론 로직 구현
    3. 필요시 전처리, 후처리 로직 구현하여 추론 함수에서 일괄 실행
    """

    def __init__(self):
        """
        필요시 초기화 로직 구현
        """
        pass

    def load_model(self, model_name: str, run_id: str):
        """
        모델 로드 로직 구현
        """
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, data, device):
        """
        모델 추론 로직 구현
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _load_artifacts(self, run_id, model_path):
        """
        mlflow 아티팩트 다운로드 기능(공통기능)
        :param run_id: mlflow run ID
        :param model_path: 저장소 내 아티팩트 저장 경로
        :return: 아티팩트 다운로드 경로
        """
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_path)
