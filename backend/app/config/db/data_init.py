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
