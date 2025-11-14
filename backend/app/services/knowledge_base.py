import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from db.models.knowledge_base import KnowledgeBase, KnowledgeBaseFile
from db.models.model_base_deployment import BaseDeploymentStatus
from fastapi import UploadFile
from langchain_core.documents import Document
from repos.knowledge_base import (
    chunk_type_repository,
    knowledge_base_file_repository,
    knowledge_base_repository,
    knowledge_base_search_record_repository,
    language_repository,
    search_method_repository,
)
from repos.model import model_repository
from repos.model_base_deployment import model_base_deployment_repository
from schemas.knowledge_base import (
    ChunkTypeReadSchema,
    KnowledgeBaseBaseSchema,
    KnowledgeBaseCreateSchema,
    KnowledgeBaseFileCreateSchema,
    KnowledgeBaseReadSchema,
    KnowledgeBaseSearchRecordCreateSchema,
    KnowledgeBaseSearchRequestSchema,
    KnowledgeBaseSearchResponseSchema,
    KnowledgeBaseUpdateSchema,
    LanguageReadSchema,
    SearchMethodReadSchema,
    SearchResultItemSchema,
)
from sqlalchemy.orm import Session
from utils.chunk import file_load_and_split
from utils.embedding import create_milvus_entities, get_embeddings_for_milvus
from utils.vector_database import MilvusManager, MilvusSearchManager

logger = logging.getLogger(__name__)


class ChunkTypeService:
    @staticmethod
    def get_all(db: Session) -> list[ChunkTypeReadSchema]:
        chunk_types = chunk_type_repository.get_all(db)
        return [ChunkTypeReadSchema.model_validate(chunk_type) for chunk_type in chunk_types]


class LanguageService:
    @staticmethod
    def get_all(db: Session) -> list[LanguageReadSchema]:
        languages = language_repository.get_all(db)
        return [LanguageReadSchema.model_validate(language) for language in languages]


class SearchMethodService:
    @staticmethod
    def get_all(db: Session) -> list[SearchMethodReadSchema]:
        search_methods = search_method_repository.get_all(db)
        return [SearchMethodReadSchema.model_validate(search_method) for search_method in search_methods]


class KnowledgeBaseService:
    @staticmethod
    def get(db: Session, pk: int) -> Optional[KnowledgeBaseReadSchema]:
        kb = knowledge_base_repository.get(db, pk)
        if not kb:
            return None
        return KnowledgeBaseReadSchema.model_validate(kb)

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> list[KnowledgeBaseReadSchema]:
        kb_list = knowledge_base_repository.get_multi(db, skip=skip, limit=limit)
        return [KnowledgeBaseReadSchema.model_validate(kb) for kb in kb_list]

    @staticmethod
    def create(
        db: Session,
        *,
        obj_in: KnowledgeBaseCreateSchema,
        file: UploadFile,
    ) -> KnowledgeBaseReadSchema:
        """Knowledge Base 생성

        Args:
            db: 데이터베이스 세션
            obj_in: Knowledge Base 생성 스키마
            file: 업로드된 파일

        Returns:
            생성된 Knowledge Base 읽기 스키마

        Raises:
            ValueError: 검증 오류 발생 시
            Exception: Knowledge Base 생성 중 오류 발생 시
        """
        try:
            # 1. 모델 조회 및 검증
            model = model_repository.get(db, obj_in.embedding_model_id)
            if not model:
                raise ValueError(f"Embedding model not found: {obj_in.embedding_model_id}")

            # 2. 배포 정보 조회 및 검증
            deployment = model_base_deployment_repository.get_by_model_id(db, obj_in.embedding_model_id)
            if not deployment:
                raise ValueError(f"Deployment not found for model_id: {obj_in.embedding_model_id}")

            if deployment.status != BaseDeploymentStatus.DEPLOYED:
                raise ValueError(f"Deployment is not ready. Status: {deployment.status}")

            if not deployment.internal_url:
                raise ValueError(f"Internal URL not available for model_id: {obj_in.embedding_model_id}")

            if not deployment.model_name:
                raise ValueError(f"Model name not available for model_id: {obj_in.embedding_model_id}")

            # 3. Collection 이름 생성 (고유해야 함)
            collection_name = f"kb_{uuid.uuid4().hex[:16]}"

            # 4. 파일을 청크로 분할
            file.file.seek(0)
            file_data = file.file.read()
            documents: list[Document] = file_load_and_split(
                file_data=file_data,
                filename=file.filename,
                chunk_size=obj_in.chunk_size,
                overlap=obj_in.chunk_overlap,
            )

            # 5. 청크 텍스트 추출
            chunk_texts = [doc.page_content for doc in documents]
            if not chunk_texts:
                raise ValueError("파일에서 청크를 생성할 수 없습니다.")

            # 6. 임베딩 생성 및 Milvus 엔티티 생성
            entities = create_milvus_entities(
                internal_url=deployment.internal_url,
                model_name=deployment.model_name,
                texts=chunk_texts,
            )

            # 7. Milvus Collection 생성 (임베딩 차원 확인 필요)
            # 모델의 임베딩 차원을 확인해야 하지만, 일단 기본값 사용
            # TODO: 모델에서 임베딩 차원 정보를 가져오도록 수정 필요
            embedding_dimension = 1024  # 기본값, 실제로는 모델에서 가져와야 함
            MilvusManager.create_collection(name=collection_name, dimension=embedding_dimension)

            # 8. Partition 이름 생성 (파일별로 고유)
            partition_name = f"file_{uuid.uuid4().hex[:16]}"

            # 9. Milvus에 문서 삽입
            MilvusManager.embed_documents(
                collection_name=collection_name,
                entities=entities,
                partition_name=partition_name,
            )

            # 10. Knowledge Base 생성

            kb_obj = knowledge_base_repository.create(
                db,
                obj_in=KnowledgeBaseBaseSchema(
                    name=obj_in.name,
                    description=obj_in.description,
                    embedding_model_id=obj_in.embedding_model_id,
                    language_id=obj_in.language_id,
                    collection_name=collection_name,
                    chunk_size=obj_in.chunk_size,
                    chunk_overlap=obj_in.chunk_overlap,
                    chunk_type_id=obj_in.chunk_type_id,
                    search_method_id=obj_in.search_method_id,
                    top_k=obj_in.top_k,
                    threshold=obj_in.threshold,
                ),
            )
            kb_id = kb_obj.id

            # 11. Knowledge Base File 생성
            knowledge_base_file_repository.create(
                db,
                obj_in=KnowledgeBaseFileCreateSchema(
                    name=file.filename,
                    object_storage_uri=None,
                    knowledge_base_id=kb_id,
                    chunk_number=len(chunk_texts),
                    partition_name=partition_name,
                ),
            )

            db.commit()
            logger.info(f"Knowledge Base 생성 성공: {obj_in.name} (ID: {kb_id})")
            return KnowledgeBaseService.get(db, kb_id)

        except Exception as e:
            db.rollback()
            logger.error(f"Knowledge Base 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def add_file(
        db: Session,
        *,
        knowledge_base_id: int,
        file: UploadFile,
    ) -> KnowledgeBaseReadSchema:
        """Knowledge Base에 파일 추가

        Args:
            db: 데이터베이스 세션
            knowledge_base_id: Knowledge Base ID
            file: 업로드된 파일

        Returns:
            업데이트된 Knowledge Base 읽기 스키마

        Raises:
            ValueError: Knowledge Base를 찾을 수 없거나 검증 오류 발생 시
            Exception: 파일 추가 중 오류 발생 시
        """
        try:
            # 1. Knowledge Base 조회
            kb = knowledge_base_repository.get(db, knowledge_base_id)
            if not kb:
                raise ValueError(f"Knowledge Base not found: {knowledge_base_id}")

            # 2. 배포 정보 조회 및 검증
            deployment = model_base_deployment_repository.get_by_model_id(db, kb.embedding_model_id)
            if not deployment:
                raise ValueError(f"Deployment not found for model_id: {kb.embedding_model_id}")

            if deployment.status != BaseDeploymentStatus.DEPLOYED:
                raise ValueError(f"Deployment is not ready. Status: {deployment.status}")

            if not deployment.internal_url:
                raise ValueError(f"Internal URL not available for model_id: {kb.embedding_model_id}")

            if not deployment.model_name:
                raise ValueError(f"Model name not available for model_id: {kb.embedding_model_id}")

            # 3. 파일을 청크로 분할
            file.file.seek(0)
            file_data = file.file.read()
            documents: list[Document] = file_load_and_split(
                file_data=file_data,
                filename=file.filename,
                chunk_size=kb.chunk_size,
                overlap=kb.chunk_overlap,
            )

            # 4. 청크 텍스트 추출
            chunk_texts = [doc.page_content for doc in documents]
            if not chunk_texts:
                raise ValueError("파일에서 청크를 생성할 수 없습니다.")

            # 5. 임베딩 생성 및 Milvus 엔티티 생성
            entities = create_milvus_entities(
                internal_url=deployment.internal_url,
                model_name=deployment.model_name,
                texts=chunk_texts,
            )

            # 6. Partition 이름 생성 (파일별로 고유)
            partition_name = f"file_{uuid.uuid4().hex[:16]}"

            # 7. Milvus에 문서 삽입 (기존 Collection에 Partition으로 추가)
            MilvusManager.embed_documents(
                collection_name=kb.collection_name,
                entities=entities,
                partition_name=partition_name,
            )

            # 8. Knowledge Base File 생성
            knowledge_base_file_repository.create(
                db,
                obj_in=KnowledgeBaseFileCreateSchema(
                    name=file.filename,
                    object_storage_uri=None,
                    knowledge_base_id=knowledge_base_id,
                    chunk_number=len(chunk_texts),
                    partition_name=partition_name,
                ),
            )

            db.commit()
            logger.info(f"Knowledge Base에 파일 추가 성공: {file.filename} (KB ID: {knowledge_base_id})")
            return KnowledgeBaseService.get(db, knowledge_base_id)

        except Exception as e:
            db.rollback()
            logger.error(f"Knowledge Base 파일 추가 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def delete_file(
        db: Session,
        *,
        knowledge_base_id: int,
        file_id: int,
    ) -> KnowledgeBaseReadSchema:
        """Knowledge Base에서 파일 삭제

        Args:
            db: 데이터베이스 세션
            knowledge_base_id: Knowledge Base ID
            file_id: 삭제할 파일 ID

        Returns:
            업데이트된 Knowledge Base 읽기 스키마

        Raises:
            ValueError: Knowledge Base 또는 파일을 찾을 수 없을 때
            Exception: 파일 삭제 중 오류 발생 시
        """
        try:
            # 1. Knowledge Base 조회
            kb = knowledge_base_repository.get(db, knowledge_base_id)
            if not kb:
                raise ValueError(f"Knowledge Base not found: {knowledge_base_id}")

            # 2. 파일 조회
            kb_file = knowledge_base_file_repository.get(db, file_id)
            if not kb_file or kb_file.knowledge_base_id != knowledge_base_id:
                raise ValueError(f"Knowledge Base File not found: {file_id}")

            # 3. Milvus에서 Partition 삭제
            MilvusManager.drop_partition(kb.collection_name, kb_file.partition_name)

            # 4. DB에서 파일 삭제
            knowledge_base_file_repository.delete(db, pk=file_id)

            db.commit()
            logger.info(f"Knowledge Base 파일 삭제 성공: {file_id} (KB ID: {knowledge_base_id})")
            return KnowledgeBaseService.get(db, knowledge_base_id)

        except Exception as e:
            db.rollback()
            logger.error(f"Knowledge Base 파일 삭제 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def delete(db: Session, knowledge_base_id: int) -> bool:
        """Knowledge Base 삭제

        Args:
            db: 데이터베이스 세션
            knowledge_base_id: 삭제할 Knowledge Base ID

        Returns:
            삭제 성공 여부

        Raises:
            ValueError: Knowledge Base를 찾을 수 없을 때
            Exception: 삭제 중 오류 발생 시
        """
        try:
            # 1. Knowledge Base 조회
            kb = knowledge_base_repository.get(db, knowledge_base_id)
            if not kb:
                raise ValueError(f"Knowledge Base not found: {knowledge_base_id}")

            # 2. Milvus Collection 삭제
            MilvusManager.drop_collection(kb.collection_name)

            # 3. DB에서 Knowledge Base 삭제 (CASCADE로 파일도 자동 삭제됨)
            knowledge_base_repository.delete(db, pk=knowledge_base_id)

            db.commit()
            logger.info(f"Knowledge Base 삭제 성공: {knowledge_base_id}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Knowledge Base 삭제 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def update(
        db: Session,
        *,
        knowledge_base_id: int,
        obj_in: KnowledgeBaseUpdateSchema,
    ) -> KnowledgeBaseReadSchema:
        """Knowledge Base 수정 (name, description만 수정 가능)

        Args:
            db: 데이터베이스 세션
            knowledge_base_id: 수정할 Knowledge Base ID
            obj_in: 수정 스키마

        Returns:
            수정된 Knowledge Base 읽기 스키마

        Raises:
            ValueError: Knowledge Base를 찾을 수 없을 때
            Exception: 수정 중 오류 발생 시
        """
        try:
            kb = knowledge_base_repository.get(db, knowledge_base_id)
            if not kb:
                raise ValueError(f"Knowledge Base not found: {knowledge_base_id}")

            update_data = obj_in.model_dump(exclude_unset=True)
            if not update_data:
                raise ValueError("수정할 데이터가 없습니다.")

            # name과 description만 수정 가능
            allowed_fields = {"name", "description"}
            update_data = {k: v for k, v in update_data.items() if k in allowed_fields}

            if not update_data:
                raise ValueError("name 또는 description만 수정할 수 있습니다.")

            knowledge_base_repository.update(db, db_obj=kb, obj_in=KnowledgeBaseUpdateSchema(**update_data))
            db.commit()
            logger.info(f"Knowledge Base 수정 성공: {knowledge_base_id}")
            return KnowledgeBaseService.get(db, knowledge_base_id)

        except Exception as e:
            db.rollback()
            logger.error(f"Knowledge Base 수정 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def search(
        db: Session,
        *,
        knowledge_base_id: int,
        query: str,
    ) -> KnowledgeBaseSearchResponseSchema:
        """Knowledge Base 검색 테스트

        Args:
            db: 데이터베이스 세션
            knowledge_base_id: Knowledge Base ID
            query: 검색할 쿼리 텍스트

        Returns:
            검색 결과 스키마

        Raises:
            ValueError: Knowledge Base를 찾을 수 없거나 검증 오류 발생 시
            Exception: 검색 중 오류 발생 시
        """
        try:
            # 1. Knowledge Base 조회
            kb = knowledge_base_repository.get(db, knowledge_base_id)
            if not kb:
                raise ValueError(f"Knowledge Base not found: {knowledge_base_id}")

            # 2. 배포 정보 조회 및 검증
            deployment = model_base_deployment_repository.get_by_model_id(db, kb.embedding_model_id)
            if not deployment:
                raise ValueError(f"Deployment not found for model_id: {kb.embedding_model_id}")

            if deployment.status != BaseDeploymentStatus.DEPLOYED:
                raise ValueError(f"Deployment is not ready. Status: {deployment.status}")

            if not deployment.internal_url:
                raise ValueError(f"Internal URL not available for model_id: {kb.embedding_model_id}")

            if not deployment.model_name:
                raise ValueError(f"Model name not available for model_id: {kb.embedding_model_id}")

            # 3. Search Method 조회
            search_method = search_method_repository.get(db, kb.search_method_id)
            if not search_method:
                raise ValueError(f"Search method not found: {kb.search_method_id}")

            search_method_name = search_method.name.lower()

            # 4. 검색 방법 검증 (현재는 vector만 지원)
            if search_method_name not in ["vector", "dense"]:
                raise ValueError(
                    f"현재 지원하지 않는 검색 방법입니다: {search_method_name}. 현재는 'vector' 또는 'dense'만 지원합니다."
                )

            # 5. 쿼리 텍스트 임베딩 생성
            embeddings_data = get_embeddings_for_milvus(
                internal_url=deployment.internal_url,
                model_name=deployment.model_name,
                texts=[query],
            )

            dense_query_vector = np.array(embeddings_data["dense_vectors"][0])

            # 6. MilvusSearchManager를 사용하여 검색
            # 현재는 vector/dense 검색만 지원 (dense_search 사용)
            # keyword, hybrid는 향후 지원 예정
            search_manager = MilvusSearchManager(collection_name=kb.collection_name, top_k=kb.top_k)
            search_results = search_manager.dense_search(dense_query_vector.reshape(1, -1))

            # 7. 검색 결과 파싱
            # 참고 코드 형식: search_result[0]에서 data.distance > threshold_score로 필터링
            # 검색 테스트에서는 top_k만큼의 결과를 전부 보여줘야 함
            results = []
            chunk_ids = []  # 파티션 조회를 위한 chunk_id 수집

            if search_results and len(search_results) > 0:
                # search_results는 리스트의 리스트 형식, 첫 번째 결과 리스트 사용
                search_result = search_results[0]

                for data in search_result:
                    score = getattr(data, "score", None)

                    # text 접근: Milvus Hit 객체의 경우 entity 속성 사용
                    text = ""
                    if hasattr(data, "entity"):
                        entity_data = data.entity
                        if isinstance(entity_data, dict):
                            text = entity_data.get("text", "")
                        elif hasattr(entity_data, "text"):
                            text = getattr(entity_data, "text", "")
                        elif hasattr(entity_data, "get") and callable(getattr(entity_data, "get")):
                            try:
                                text = entity_data.get("text", "")
                            except TypeError:
                                text = ""
                    elif isinstance(data, dict):
                        text = data.get("text", "")
                    elif hasattr(data, "text"):
                        text = getattr(data, "text", "")

                    # Milvus의 id가 chunk_id
                    chunk_id = -1
                    if hasattr(data, "id"):
                        chunk_id = data.id
                    elif hasattr(data, "entity"):
                        entity_data = data.entity
                        if isinstance(entity_data, dict):
                            chunk_id = entity_data.get("id", -1)
                        elif hasattr(entity_data, "id"):
                            chunk_id = getattr(entity_data, "id", -1)

                    # 검색 테스트에서는 top_k만큼의 결과를 전부 보여줌
                    results.append(
                        {
                            "text": text,
                            "score": float(score) if score is not None else 0.0,
                            "chunk_id": chunk_id,
                        }
                    )

                    if chunk_id != -1:
                        chunk_ids.append(chunk_id)

            # 8. chunk_id로 파티션 정보 조회 (Milvus의 query API 사용)
            partition_names = set()
            chunk_id_to_partition = {}

            if chunk_ids:
                try:
                    partitions = search_manager._collection.partitions

                    for partition in partitions:
                        partition_name = partition.name
                        try:
                            query_results = partition.query(expr=f"id in {chunk_ids}", output_fields=["id"])
                            for result in query_results:
                                result_chunk_id = result.get("id")
                                if result_chunk_id in chunk_ids:
                                    chunk_id_to_partition[result_chunk_id] = partition_name
                                    partition_names.add(partition_name)
                        except Exception as e:
                            logger.warning(f"파티션 {partition_name}에서 조회 실패: {str(e)}")
                            continue
                except Exception as e:
                    logger.warning(f"파티션 정보 조회 실패: {str(e)}")

            # 9. partition_name으로 파일명 조회
            partition_to_filename = {}
            if partition_names:
                kb_files = knowledge_base_file_repository.get_by_knowledge_base_id(db, knowledge_base_id)
                for kb_file in kb_files:
                    if kb_file.partition_name in partition_names:
                        partition_to_filename[kb_file.partition_name] = kb_file.name

            # 10. 최종 결과 생성 (파티션명, 파일명 포함)
            final_results = []
            for result in results:
                chunk_id = result["chunk_id"]
                partition_name = chunk_id_to_partition.get(chunk_id, "")
                file_name = partition_to_filename.get(partition_name, "")

                result_item = SearchResultItemSchema(
                    text=result["text"],
                    score=result["score"],
                    chunk_id=str(chunk_id),  # 큰 정수 정밀도 보존을 위해 문자열로 변환
                    partition_name=partition_name,
                    file_name=file_name,
                )
                final_results.append(result_item)

            # 11. 검색 기록 저장
            knowledge_base_search_record_repository.create(
                db,
                obj_in=KnowledgeBaseSearchRecordCreateSchema(
                    knowledge_base_id=knowledge_base_id,
                    source=kb.collection_name,
                    text=query,
                ),
            )
            db.commit()

            logger.info(f"Knowledge Base 검색 성공: {knowledge_base_id}, 결과 수: {len(final_results)}")
            return KnowledgeBaseSearchResponseSchema(
                results=final_results,
                total=len(final_results),
                search_method=search_method_name,
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Knowledge Base 검색 중 오류 발생: {str(e)}", exc_info=True)
            raise
