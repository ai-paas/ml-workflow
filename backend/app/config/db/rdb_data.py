import datetime

from config.settings import get_settings
from utils.crypto import get_sha256_hash

settings = get_settings()

USER_DATA = [
    {
        "username": "surromind",
        "name": "surromind",
        "password": get_sha256_hash(settings.DEMO_PASSWORD),
        "created_at": datetime.datetime.now(),
    }
]

MODEL_FORMAT_DATA = [
    {
        "name": "transformers",
        "description": "Huggingface transformers",
    }
]

MODEL_FORMAT_DATA_2 = [
    {
        "name": "pytorch",
        "description": "Pytorch",
    },
    {
        "name": "keras",
        "description": "Keras",
    },
    {
        "name": "onnx",
        "description": "ONNX",
    },
]

MODEL_FORMAT_DATA_3 = [
    {
        "name": "tensorflow",
        "description": "TensorFlow",
    },
    {
        "name": "yolox",
        "description": "YOLOX Object Detection",
    },
]

MODEL_FORMAT_DATA_4 = [
    {
        "name": "gguf",
        "description": "GGUF format for Ollama",
    },
]

MODEL_PROVIDER_DATA = [
    {
        "name": "huggingface",
        "description": "huggingface",
    },
    {
        "name": "custom",
        "description": "user uploaded",
    },
]

MODEL_PROVIDER_DATA_2 = [
    {
        "name": "ollama",
        "description": "Ollama",
    },
]

MODEL_TYPE_DATA = [
    {
        "name": "ODM",
        "description": "Object Detection Model",
    }
]

MODEL_TYPE_DATA_2 = [
    {
        "name": "LLM",
        "description": "Large Language Model",
    },
]

MODEL_TYPE_DATA_3 = [
    {
        "name": "Embedding",
        "description": "Embedding Model",
    },
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
