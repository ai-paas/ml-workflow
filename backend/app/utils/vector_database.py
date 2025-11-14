from typing import Any

import numpy as np
from config.settings import get_settings
from pymilvus import AnnSearchRequest, Collection, DataType, MilvusClient, WeightedRanker, connections

settings = get_settings()


class Client:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Client, cls).__new__(cls)
            cls._instance.client = MilvusClient(
                uri=f"http://{settings.MILVUS_DB_HOST}:{settings.MILVUS_DB_PORT}",
                token=f"{settings.MILVUS_DB_USERNAME}:{settings.MILVUS_DB_PASSWORD}",
                db_name=settings.MILVUS_DB_NAME,
            )
        return cls._instance

    def get(self) -> MilvusClient:
        return self.client


class MilvusManager:
    _client = Client().get()

    @classmethod
    def create_collection(cls, name: str, *, dimension: int = 1024) -> str:
        """
        Milvus에 컬렉션이 존재하지 않을 경우 컬렉션을 생성합니다.

        매개변수:
            name (str): 생성할 컬렉션의 이름.
            dimension (int, 선택적): 벡터 필드의 차원 수. 기본값은 1024.

        반환:
            str: 생성된 컬렉션의 이름.
        """
        if not cls._client.has_collection(name):
            schemas = cls._create_collection_schemas(dimension=dimension)
            cls._client.create_collection(
                collection_name=name,
                schema=schemas,
            )

            dense_vector_index_params = cls._client.prepare_index_params()
            dense_vector_index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
            sparse_vector_index_params = cls._client.prepare_index_params()
            sparse_vector_index_params.add_index(
                field_name="sparse_vector", index_type="SPARSE_WAND", metric_type="IP", params={"drop_ratio_build": 0.5}
            )
            cls._client.create_index(collection_name=name, index_params=dense_vector_index_params)
            cls._client.create_index(collection_name=name, index_params=sparse_vector_index_params)
            cls._client.load_collection(collection_name=name)
        return name

    # TODO: 필요시 별도 API로 Index 생성하도록 수정 <- 전체 collection, schema 목록 확인 및 선택 후 추가 가능하도록 선행되어야 함
    @classmethod
    def create_index(
        cls, collection_name: str, *, field_name: str = "vector", index_type: str = "HNSW", metric_type: str = "L2"
    ):
        """
        Milvus 컬렉션의 지정된 필드에 인덱스를 생성합니다.

        매개변수:
            collection_name (str): 컬렉션의 이름.
            field_name (str, 선택적): 인덱스를 생성할 필드의 이름. 기본값은 "vector".
            index_type (str, 선택적): 인덱스의 유형. 기본값은 "HNSW".
            metric_type (str, 선택적): 인덱스의 메트릭 유형. 기본값은 "L2".
        """
        index_params = cls._client.prepare_index_params()
        # TODO: 관리자 페이지에서 직접 설정하도록
        index_params.add_index(
            field_name=field_name, index_type=index_type, metric_type=metric_type, params={"M": 8, "efConstruction": 64}
        )
        cls._client.create_index(collection_name=collection_name, index_params=index_params)

    # TODO: 필요시 schema 동적으로 추가할 수 있도록 <- 단순 vector store 역할만 수행하면되기에 필요여부 확인
    @classmethod
    def _create_collection_schemas(cls, *, dimension: int = 1024, max_length: int = 8192):
        """
        컬렉션의 스키마를 생성합니다.

        매개변수:
            dimension (int, 선택적): 벡터 필드의 차원 수. 기본값은 1024.
            max_length (int, 선택적): 벡터 필드의 차원 수. 기본값은 8192.
        반환:
            schema: 컬렉션의 스키마 객체.
        """
        schema = cls._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=max_length)
        return schema

    @classmethod
    def create_partition(cls, collection_name: str, partition_name: str) -> str:
        """
        Milvus 컬렉션에 새로운 파티션을 생성합니다.

        매개변수:
            collection_name (str): 컬렉션의 이름.
            partition_name (str): 생성할 파티션의 이름.

        반환:
            str: 생성된 파티션의 이름.
        """
        if not cls._client.has_partition(collection_name, partition_name):
            cls._client.create_partition(collection_name=collection_name, partition_name=partition_name)
        return partition_name

    @classmethod
    def drop_partition(cls, collection_name: str, partition_name: str) -> bool:
        """
        Milvus 컬렉션에서 파티션을 삭제합니다.

        매개변수:
            collection_name (str): 컬렉션의 이름.
            partition_name (str): 삭제할 파티션의 이름.

        반환:
            bool: 파티션이 삭제되었는지 여부 (존재하지 않으면 False).
        """
        try:
            if cls._client.has_partition(collection_name, partition_name):
                cls._client.drop_partition(collection_name=collection_name, partition_name=partition_name)
                return True
            else:
                return False
        except Exception as e:
            raise Exception(f"파티션 '{partition_name}' 삭제 중 오류 발생: {e}")

    @classmethod
    def embed_documents(cls, collection_name: str, entities: list[dict[str, Any]], partition_name: str = None):
        """
        지정된 컬렉션에 문서(엔티티)를 삽입합니다.

        매개변수:
            collection_name (str): 컬렉션의 이름.
            entities (list[dict[str, int | float]]): 컬렉션에 삽입할 엔티티 목록.
        """
        if not cls._client.has_partition(collection_name, partition_name):
            partition_name = cls.create_partition(collection_name, partition_name)
        cls._client.insert(collection_name=collection_name, data=entities, partition_name=partition_name)
        cls._client.load_collection(collection_name=collection_name)

    @classmethod
    def drop_collection(cls, collection_name: str) -> bool:
        """
        Safely drops a Milvus collection if it exists.

        Args:
            collection_name (str): The name of the collection to drop.

        Returns:
            bool: True if the collection was dropped, False if it didn't exist.
        """
        try:
            if cls._client.has_collection(collection_name):
                cls._client.drop_collection(collection_name)
                return True
            else:
                return False
        except Exception as e:
            raise Exception(f"An error occurred while droping the collection '{collection_name}': {e}")

    @classmethod
    def load_collection(cls, collection_name: str) -> bool:
        """
        Loads a collection into memory if it exists.

        Args:
            collection_name (str): The name of the collection to load.

        Returns:
            bool: True if the collection exists and is loaded successfully, False otherwise.

        Raises:
            Exception: If there is an error during the loading process.
        """
        try:
            if cls._client.has_collection(collection_name):
                cls._client.load_collection(collection_name)
                return True
            else:
                return False
        except Exception as e:
            raise Exception(f"An error occurred while loading the collection '{collection_name}': {e}")

    @classmethod
    def release_collection(cls, collection_name: str) -> bool:
        """
        Releases a collection from memory if it exists.

        Args:
            collection_name (str): The name of the collection to release.

        Returns:
            bool: True if the collection exists and is released successfully, False otherwise.

        Raises:
            Exception: If there is an error during the releasing process.
        """
        try:
            if cls._client.has_collection(collection_name):
                cls._client.release_collection(collection_name)
                return True
            else:
                return False
        except Exception as e:
            raise Exception(f"An error occurred while releasing the collection '{collection_name}': {e}")

    @classmethod
    def check_connection(cls) -> bool:
        try:
            cls._client.list_users
        except Exception:
            return False
        return True


class MilvusSearchManager:
    def __init__(self, collection_name: str, top_k: int):
        """
        MilvusSearchManager 클래스의 생성자. 주어진 컬렉션 이름과 상위 k개의 결과 제한을 설정하고,
        Milvus에 연결하여 컬렉션을 초기화합니다.

        Args:
            collection_name (str): 사용할 Milvus 컬렉션의 이름.
            top_k (int): 검색 시 반환할 상위 k개의 결과 수.
        """
        self.connect_to_milvus()
        self._collection = Collection(collection_name)
        self._top_k = top_k
        self._dense_search_param = {"metric_type": "COSINE", "params": {}}
        self._sparse_search_param = {"metric_type": "IP", "params": {}}

    @staticmethod
    def connect_to_milvus():
        """
        Milvus 서버에 연결합니다. 연결 정보는 환경 변수에서 설정된 값을 사용합니다.
        """
        connections.connect(
            uri=f"http://{settings.MILVUS_DB_HOST}:{settings.MILVUS_DB_PORT}",
            token=f"{settings.MILVUS_DB_USERNAME}:{settings.MILVUS_DB_PASSWORD}",
            db_name=settings.MILVUS_DB_NAME,
        )

    def dense_search(self, embeded_query: np.ndarray):
        """
        주어진 밀집(dense) 임베딩을 기반으로 Milvus에서 검색을 수행합니다.

        Args:
            embeded_query (np.ndarray): 검색에 사용할 밀집 임베딩 벡터.

        Returns:
            list: 검색된 결과를 반환합니다.
        """
        search_results = self._collection.search(
            embeded_query,
            anns_field="dense_vector",
            limit=self._top_k,
            output_fields=["text"],
            param=self._dense_search_param,
        )
        return search_results

    def sparse_search(self, embeded_query: np.ndarray):
        """
        주어진 희소(sparse) 임베딩을 기반으로 Milvus에서 검색을 수행합니다.

        Args:
            embeded_query (np.ndarray): 검색에 사용할 희소 임베딩 벡터.

        Returns:
            list: 검색된 결과를 반환합니다.
        """
        search_results = self._collection.search(
            embeded_query,
            anns_field="sparse_vector",
            limit=self._top_k,
            output_fields=["text"],
            param=self._sparse_search_param,
        )
        return search_results

    def hybrid_search(
        self,
        dense_embeded_query: np.ndarray,
        sparse_embeded_query: np.ndarray,
        dense_weight=0.6,
        sparse_weight=0.4,
    ):
        """
        밀집(dense) 및 희소(sparse) 임베딩을 결합하여 하이브리드 검색을 수행합니다. 각 검색 결과에
        가중치를 부여하여 순위를 재조정합니다.

        Args:
            dense_embeded_query (np.ndarray): 밀집 임베딩 벡터.
            sparse_embeded_query (np.ndarray): 희소 임베딩 벡터.
            sparse_weight (float, optional): 희소 임베딩 결과에 부여할 가중치. 기본값은 0.6.
            dense_weight (float, optional): 밀집 임베딩 결과에 부여할 가중치. 기본값은 0.4.

        Returns:
            list: 하이브리드 검색을 통해 얻은 결과를 반환합니다.
        """
        dense_req = AnnSearchRequest(dense_embeded_query, "dense_vector", self._dense_search_param, limit=self._top_k)
        sparse_req = AnnSearchRequest(
            sparse_embeded_query, "sparse_vector", self._sparse_search_param, limit=self._top_k
        )
        rerank = WeightedRanker(dense_weight, sparse_weight)
        search_results = self._collection.hybrid_search(
            [dense_req, sparse_req], rerank=rerank, limit=self._top_k, output_fields=["text"]
        )
        return search_results
