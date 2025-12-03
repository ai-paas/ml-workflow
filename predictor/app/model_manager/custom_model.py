"""
Sample Custom Model Manager
"""

from functools import lru_cache

from app.logging_inference import logger
from app.model_manager.base import BaseModelManager

FILE_INPUT_TYPE = "이미지/테이블"
ACCELERATOR = "cpu, gpu"


class SampleModelManager(BaseModelManager):
    """
    Sample Custom Model Manager
    """

    def __init__(self):
        """
        ex)
        ```
        super().__init__()
        self.model_name = model_name
        self.run_id = run_id
        self.model_path = model_path
        ```
        """
        super().__init__()
        logger.error("모델관리자가 적용되지 않음")

    def load_model(self, model_name: str, run_id: str):
        """
        AutoModel 등을 활용한 load_model 로직 구현
        ex)
        ```
        if not self.model:
            local_path = self._load_artifacts(self.run_id, self.model_path)
            self.model = AutoModel.from_pretrained(local_path)
        return self.model
        ```
        """
        raise NotImplementedError("프레임워크별 모델 호출 로직이 적용되지 않았습니다")

    def predict(self, data, device):
        """
        AutoModel 등을 활용한 predict 로직 구현
        ex)
        ```
        return self.model.eval(data)
        ```
        """
        raise NotImplementedError("프레임워크별 추론 로직이 적용되지 않았습니다")


# 모델 관리자 주입을 위한 wrapping
@lru_cache
def get_model_manager():
    model_manager = SampleModelManager()
    return model_manager
