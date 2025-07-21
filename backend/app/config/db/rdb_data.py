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
    }
]

MODEL_PROVIDER_DATA = [
    {
        "name": "Huggingface",
        "description": "huggingface",
    },
    {
        "name": "custom",
        "description": "user uploaded",
    },
]

MODEL_TYPE_DATA = [
    {
        "name": "ODM",
        "description": "Object Detection Model",
    },
    {
        "name": "Fine-Tuned",
        "description": "Fine-Tuned Model",
    },
]
