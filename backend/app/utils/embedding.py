"""임베딩 유틸리티 모듈

Ollama Embedding Service를 사용하여 텍스트를 벡터로 변환하고,
Milvus에 저장할 수 있는 형식으로 변환합니다.
"""

import logging
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class OllamaEmbedding:
    """Ollama Embedding Service를 사용한 임베딩 클래스

    BGEM3Embedding과 유사한 인터페이스를 제공하지만,
    실제로는 Ollama Embedding Service를 사용합니다.
    """

    def __init__(self, internal_url: str, model_name: str, texts: List[str]):
        """
        Args:
            internal_url: Ollama 서비스 내부 URL
            model_name: 모델 이름 (model_base_deployment의 model_name)
            texts: 임베딩할 텍스트 리스트
        """
        self._internal_url = internal_url
        self._model_name = model_name
        self._texts = texts
        self._embeddings = self.get_embeddings(texts)

    def get_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        Ollama Embedding Service를 사용하여 텍스트를 벡터로 변환합니다.

        참고: Ollama Embedding API
        - 엔드포인트: /api/embed
        - 요청 형식: {"model": "model_name", "input": "text"}
        - 응답 형식: {"embedding": [float, ...]}

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            dict: dense_vector와 sparse_vector를 포함한 딕셔너리
                - dense_vecs: 밀집 벡터 리스트 (numpy array 또는 list)
                - lexical_weights: 희소 벡터 리스트 (Ollama는 제공하지 않으므로 빈 딕셔너리 리스트)
        """
        # Ollama 서비스 URL 구성
        ollama_url = self._internal_url.rstrip("/")
        url = f"{ollama_url}/api/embed"

        # 각 텍스트에 대해 임베딩 요청
        dense_vectors = []
        sparse_vectors = []  # Ollama는 sparse vector를 제공하지 않으므로 빈 딕셔너리

        for text in texts:
            try:
                # Ollama Embedding API 요청
                data = {
                    "model": self._model_name,
                    "input": text,
                }

                headers = {"Content-Type": "application/json"}

                logger.debug(f"Sending embedding request to {url} with model {self._model_name}")

                # 타임아웃 설정: 텍스트 길이에 따라 동적 조정 (최소 30초, 최대 300초)
                timeout = min(30 + (len(text) // 1000) * 10, 300)
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
                response.raise_for_status()

                result = response.json()

                # Ollama 응답에서 embedding 추출
                # 응답 형식: {"embedding": [...]} 또는 {"embeddings": [[...]]}
                embedding = result.get("embedding")
                if embedding is None:
                    # embeddings (복수형)로 시도
                    embeddings_list = result.get("embeddings", [])
                    if embeddings_list and len(embeddings_list) > 0:
                        # embeddings가 2차원 배열인 경우 첫 번째 요소 사용
                        embedding = embeddings_list[0] if isinstance(embeddings_list[0], list) else embeddings_list
                    else:
                        embedding = []

                if not embedding:
                    raise ValueError(f"No embedding in response: {result}")

                dense_vectors.append(embedding)

                # Ollama는 sparse vector를 제공하지 않으므로 빈 딕셔너리 추가
                # Milvus는 빈 딕셔너리를 허용하지 않으므로 최소한의 유효한 형식 사용
                # {0: 0.0} 형식은 유효하지만 의미 없는 sparse vector입니다
                sparse_vectors.append({0: 0.0})

            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error occurred during embedding: {http_err}")
                logger.error(f"Response content: {http_err.response.text if hasattr(http_err, 'response') else 'N/A'}")
                raise ValueError(f"Failed to get embedding from Ollama service: {str(http_err)}")
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection error: {conn_err}")
                raise ValueError(f"Unable to connect to Ollama service at {url}")
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Request timeout: {timeout_err}")
                raise ValueError(f"Embedding request timed out for model {self._model_name}")
            except Exception as e:
                logger.error(f"Unexpected error during embedding: {e}")
                raise ValueError(f"Failed to get embedding: {str(e)}")

        return {
            "dense_vecs": dense_vectors,
            "lexical_weights": sparse_vectors,
        }

    @property
    def dense_vector(self) -> List[List[float]]:
        """밀집 벡터 반환"""
        return self._embeddings["dense_vecs"]

    @property
    def sparse_vector(self) -> List[Dict]:
        """희소 벡터 반환 (Ollama는 제공하지 않으므로 빈 딕셔너리 리스트)"""
        return self._embeddings["lexical_weights"]


def get_embeddings_for_milvus(internal_url: str, model_name: str, texts: List[str]) -> Dict[str, Any]:
    """
    Milvus에 저장할 수 있는 형식으로 임베딩을 생성합니다.

    Args:
        internal_url: Ollama 서비스 내부 URL
        model_name: 모델 이름 (model_base_deployment의 model_name)
        texts: 임베딩할 텍스트 리스트

    Returns:
        dict: Milvus에 저장할 수 있는 형식의 임베딩 데이터
            - dense_vectors: 밀집 벡터 리스트
            - sparse_vectors: 희소 벡터 리스트 (Ollama는 제공하지 않으므로 빈 딕셔너리)
            - texts: 원본 텍스트 리스트
    """
    embedding = OllamaEmbedding(internal_url=internal_url, model_name=model_name, texts=texts)

    return {
        "dense_vectors": embedding.dense_vector,
        "sparse_vectors": embedding.sparse_vector,
        "texts": texts,
    }


def create_milvus_entities(internal_url: str, model_name: str, texts: List[str]) -> List[Dict[str, Any]]:
    """
    Milvus에 삽입할 수 있는 엔티티 리스트를 생성합니다.

    Args:
        internal_url: Ollama 서비스 내부 URL
        model_name: 모델 이름 (model_base_deployment의 model_name)
        texts: 임베딩할 텍스트 리스트

    Returns:
        list: Milvus 엔티티 리스트
            각 엔티티는 다음 필드를 포함:
            - dense_vector: 밀집 벡터 (List[float])
            - sparse_vector: 희소 벡터 (Dict, Ollama는 빈 딕셔너리)
            - text: 원본 텍스트 (str)
    """
    embeddings_data = get_embeddings_for_milvus(internal_url=internal_url, model_name=model_name, texts=texts)

    entities = []
    for i, text in enumerate(embeddings_data["texts"]):
        sparse_vec = embeddings_data["sparse_vectors"][i]
        # Milvus는 빈 딕셔너리를 허용하지 않으므로, 빈 딕셔너리인 경우 유효한 형식으로 변환
        if not sparse_vec or sparse_vec == {}:
            sparse_vec = {0: 0.0}

        entity = {
            "dense_vector": embeddings_data["dense_vectors"][i],
            "sparse_vector": sparse_vec,
            "text": text,
        }
        entities.append(entity)

    return entities
