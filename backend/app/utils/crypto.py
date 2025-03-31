import hashlib

from config.settings import get_settings

settings = get_settings()


def get_sha256_hash(input_string: str):
    # 입력 문자열을 바이트로 인코딩. SHA-256 해시 객체 생성 및 해시 계산
    sha256_hash = hashlib.sha256(input_string.encode("utf-8")).hexdigest()

    return sha256_hash
