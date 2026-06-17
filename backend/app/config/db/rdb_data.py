"""DB 시드 데이터(순수 모듈).

이 파일은 settings·datetime·해시 등 **런타임 의존을 import 시점에 평가하지 않는다.**
- 비밀번호는 평문/해시를 박지 않고 `_password_env`(settings 속성명)로 두어, 적용 시점에
  `data_initializer`의 transform 이 sha-256 해시로 변환한다.
- `created_at` 등 타임스탬프는 모델의 `default=func.now()`가 채우므로 시드에 두지 않는다.
밑줄(`_`)로 시작하는 키는 적용 시점 전용 메타데이터이며 DB 컬럼으로 들어가지 않는다.
"""

from config.db.enums import ModelFormatEnum, ModelProviderEnum, ModelTypeEnum

USER_DATA = [
    {
        "username": "surromind",
        "name": "surromind",
        # 적용 시점에 settings.DEMO_PASSWORD 를 sha-256 해시로 변환 (data_initializer._user_transform)
        "_password_env": "DEMO_PASSWORD",
    }
]

MODEL_FORMAT_DATA = [
    {"name": ModelFormatEnum.TRANSFORMERS.value, "description": "Huggingface transformers"},
    {"name": ModelFormatEnum.PYTORCH.value, "description": "Pytorch"},
    {"name": ModelFormatEnum.KERAS.value, "description": "Keras"},
    {"name": ModelFormatEnum.ONNX.value, "description": "ONNX"},
    {"name": ModelFormatEnum.TENSORFLOW.value, "description": "TensorFlow"},
    {"name": ModelFormatEnum.YOLOX.value, "description": "YOLOX Object Detection"},
    {"name": ModelFormatEnum.GGUF.value, "description": "GGUF format for Ollama"},
    {"name": ModelFormatEnum.TENSORRT.value, "description": "NVIDIA TensorRT optimized model"},
    {"name": ModelFormatEnum.OPENVINO.value, "description": "Intel OpenVINO IR optimized model"},
]

MODEL_PROVIDER_DATA = [
    {"name": ModelProviderEnum.HUGGINGFACE.value, "description": "huggingface"},
    {"name": ModelProviderEnum.CUSTOM.value, "description": "user uploaded"},
    {"name": ModelProviderEnum.OLLAMA.value, "description": "Ollama"},
    {"name": ModelProviderEnum.KAGGLE.value, "description": "Kaggle"},
]

MODEL_TYPE_DATA = [
    {"name": ModelTypeEnum.ODM.value, "description": "Object Detection Model"},
    {"name": ModelTypeEnum.LLM.value, "description": "Large Language Model"},
    {"name": ModelTypeEnum.EMBEDDING.value, "description": "Embedding Model"},
    {"name": ModelTypeEnum.PLM.value, "description": "Protein Language Model"},
]

HYPERPARAMETER_TYPE_DATA = [
    {
        "param_name": "epochs",
        "param_type": "int",
        "default_value": "10",
    },
    {
        "param_name": "batch_size",
        "param_type": "int",
        "default_value": "16",
    },
    {
        "param_name": "weight_decay",
        "param_type": "float",
        "default_value": "0.0001",
    },
    {
        "param_name": "save_period",
        "param_type": "int",
        "default_value": "1",
    },
    {
        "param_name": "lr0",
        "param_type": "float",
        "default_value": "0.01",
    },
    {
        "param_name": "lrf",
        "param_type": "float",
        "default_value": "0.05",
    },
    {
        "param_name": "gpus",
        "param_type": "int",
        "default_value": "1",
    },
]

CHUNK_TYPE_DATA = [
    {
        "name": "RecursiveCharacterSplitter",
        "description": None,
    },
]

LANGUAGE_DATA = [
    {
        "name": "KO",
        "description": "한국어",
    },
    {
        "name": "EN",
        "description": "영어",
    },
]

SEARCH_METHOD_DATA = [
    {
        "name": "vector",
        "description": "vector search",
    },
]
