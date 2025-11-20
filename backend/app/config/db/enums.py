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

    def __str__(self) -> str:
        return self.value


class ModelProviderEnum(str, Enum):
    """모델 제공자 Enum"""

    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    OLLAMA = "ollama"

    def __str__(self) -> str:
        return self.value


class ModelTypeEnum(str, Enum):
    """모델 타입 Enum"""

    ODM = "ODM"
    LLM = "LLM"
    EMBEDDING = "Embedding"

    def __str__(self) -> str:
        return self.value
