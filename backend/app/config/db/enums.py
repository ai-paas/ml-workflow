"""모델 관련 Enum 정의"""

from enum import Enum


class ModelFormatEnum(str, Enum):
    """모델 포맷 Enum"""

    TRANSFORMERS = "transformers"
    PYTORCH = "pytorch"
    KERAS = "keras"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    YOLOX = "yolox"
    GGUF = "gguf"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"

    def __str__(self) -> str:
        return self.value


class ModelProviderEnum(str, Enum):
    """모델 제공자 Enum"""

    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    OLLAMA = "ollama"
    KAGGLE = "kaggle"

    def __str__(self) -> str:
        return self.value


class ModelTypeEnum(str, Enum):
    """모델 타입 Enum"""

    ODM = "ODM"
    LLM = "LLM"
    EMBEDDING = "Embedding"
    PLM = "pLM"

    def __str__(self) -> str:
        return self.value


class ModelVisibility(str, Enum):
    """모델 가시성 구분 Enum"""

    CATALOG = "CATALOG"
    CUSTOM = "CUSTOM"

    def __str__(self) -> str:
        return self.value


class DatasetKindEnum(str, Enum):
    """데이터셋 분류(학습 태스크 분류명) Enum

    - object-detection: YOLOX 계열 객체 감지 데이터셋
    - protein-classification: ESM2 단백질 서열 분류(TCR-Epitope) 데이터셋
    """

    OBJECT_DETECTION = "object-detection"
    PROTEIN_CLASSIFICATION = "protein-classification"

    def __str__(self) -> str:
        return self.value
