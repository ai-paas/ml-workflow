# 지식베이스(Knowledge Base) 관련 API 사용 가이드

UI에서 지식베이스 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 지식베이스 워크플로우

### 1. Knowledge Base 생성 전 필요한 정보 조회

#### 1-1. 청크 타입 목록 조회
**API**: `GET /api/v1/knowledge-bases/chunk-types`

- 사용 가능한 모든 청크 타입 목록을 조회합니다
- 응답: 청크 타입 목록 (List[ChunkTypeReadSchema])
  - `id`: 청크 타입 ID
  - `name`: 청크 타입 이름 (예: "RecursiveTextSplitter", "RecursiveCharacterSplitter")
  - `description`: 청크 타입 설명
- Knowledge Base 생성 시 `chunk_type_id`로 사용

#### 1-2. 언어 목록 조회
**API**: `GET /api/v1/knowledge-bases/languages`

- 사용 가능한 모든 언어 목록을 조회합니다
- 응답: 언어 목록 (List[LanguageReadSchema])
  - `id`: 언어 ID
  - `name`: 언어 코드 (예: "KO", "EN")
  - `description`: 언어 설명 (예: "한국어", "영어")
- Knowledge Base 생성 시 `language_id`로 사용

#### 1-3. 검색 방법 목록 조회
**API**: `GET /api/v1/knowledge-bases/search-methods`

- 사용 가능한 모든 검색 방법 목록을 조회합니다
- 응답: 검색 방법 목록 (List[SearchMethodReadSchema])
  - `id`: 검색 방법 ID
  - `name`: 검색 방법 이름 (예: "vector")
  - `description`: 검색 방법 설명
- Knowledge Base 생성 시 `search_method_id`로 사용

#### 1-4. 임베딩 모델 목록 조회
**API**: `GET /api/v1/models?model_type_id={embedding_type_id}`

- Knowledge Base 생성에 사용할 임베딩 모델을 조회합니다
- 쿼리 파라미터: `model_type_id` (Embedding 타입의 ID)
  - Embedding 타입 ID는 `GET /api/v1/models/types?type_name=Embedding`로 조회 가능
- 응답: 임베딩 모델 목록 (List[ModelBriefReadSchema])
- Knowledge Base 생성 시 `embedding_model_id`로 사용

### 2. Knowledge Base 생성
**API**: `POST /api/v1/knowledge-bases`

- **Content-Type**: `multipart/form-data`
- 필수 파라미터:
  - `name`: Knowledge Base 이름
  - `language_id`: 언어 ID (1-2 단계에서 조회)
  - `embedding_model_id`: 임베딩 모델 ID (1-4 단계에서 조회)
  - `chunk_size`: 청크 크기
  - `chunk_overlap`: 청크 오버랩 크기
  - `chunk_type_id`: 청크 타입 ID (1-1 단계에서 조회)
  - `search_method_id`: 검색 방법 ID (1-3 단계에서 조회)
  - `top_k`: 검색 시 반환할 상위 k개 결과 수
  - `threshold`: 검색 임계값 (0.0 ~ 1.0)
  - `file`: 업로드할 문서 파일
- 선택 파라미터:
  - `description`: Knowledge Base 설명
- 응답: 생성된 Knowledge Base 정보 (KnowledgeBaseReadSchema)
  - `id`: Knowledge Base ID
  - `name`: Knowledge Base 이름
  - `description`: Knowledge Base 설명
  - `collection_name`: Milvus Collection 이름
  - `files`: 파일 목록
  - 기타 설정 정보
- **주의**: 파일은 청크로 분할되고 임베딩되어 Milvus에 저장됩니다

### 3. Knowledge Base에 파일 추가
**API**: `POST /api/v1/knowledge-bases/{knowledge_base_id}/files`

- **Content-Type**: `multipart/form-data`
- 경로 파라미터: `knowledge_base_id` (Knowledge Base ID)
- 필수 파라미터:
  - `file`: 추가할 문서 파일
- 응답: 업데이트된 Knowledge Base 정보 (KnowledgeBaseReadSchema)
- **주의**: 파일은 청크로 분할되고 임베딩되어 Milvus의 동일한 Collection에 Partition으로 추가됩니다

### 4. Knowledge Base에서 파일 삭제
**API**: `DELETE /api/v1/knowledge-bases/{knowledge_base_id}/files/{file_id}`

- 경로 파라미터:
  - `knowledge_base_id`: Knowledge Base ID
  - `file_id`: 삭제할 파일 ID
- 응답: 업데이트된 Knowledge Base 정보 (KnowledgeBaseReadSchema)
- **주의**: DB에서 파일 정보를 삭제하고, Milvus에서 해당 Partition을 삭제합니다

### 5. Knowledge Base 목록 조회
**API**: `GET /api/v1/knowledge-bases`

- 쿼리 파라미터: `page`, `page_size` (선택사항)
  - 생략 시: 전체 데이터 조회 (최대 10000개)
  - 제공 시: 페이지네이션 적용
- 응답: Knowledge Base 목록 (List[KnowledgeBaseBriefReadSchema])
  - 각 항목은 간단한 정보만 포함 (파일 목록 제외)

### 6. Knowledge Base 상세 조회
**API**: `GET /api/v1/knowledge-bases/{knowledge_base_id}`

- 경로 파라미터: `knowledge_base_id` (Knowledge Base ID)
- 응답: Knowledge Base 상세 정보 (KnowledgeBaseReadSchema)
  - 모든 설정 정보 및 파일 목록 포함

### 7. Knowledge Base 수정
**API**: `PUT /api/v1/knowledge-bases/{knowledge_base_id}`

- 경로 파라미터: `knowledge_base_id` (Knowledge Base ID)
- 요청 바디: `KnowledgeBaseUpdateSchema`
  - `name` (선택): 수정할 이름
  - `description` (선택): 수정할 설명
- 응답: 수정된 Knowledge Base 정보 (KnowledgeBaseReadSchema)
- **주의**: 이름과 설명만 수정 가능합니다 (청크 설정, 검색 설정 등은 수정 불가)

### 8. Knowledge Base 삭제
**API**: `DELETE /api/v1/knowledge-bases/{knowledge_base_id}`

- 경로 파라미터: `knowledge_base_id` (Knowledge Base ID)
- 응답: `{ "success": true, "message": "Knowledge Base가 성공적으로 삭제되었습니다." }`
- **주의**: DB에서 Knowledge Base 정보를 삭제하고, Milvus에서 Collection을 삭제합니다

### 9. Knowledge Base 검색 테스트
**API**: `POST /api/v1/knowledge-bases/{knowledge_base_id}/search`

- 경로 파라미터: `knowledge_base_id` (Knowledge Base ID)
- 요청 바디: `KnowledgeBaseSearchRequestSchema`
  - `text` (필수): 검색할 쿼리 텍스트
- 응답: 검색 결과 (KnowledgeBaseSearchResponseSchema)
  - `results`: 검색 결과 목록 (List[SearchResultItemSchema])
    - `text`: 검색된 문서 텍스트
    - `score`: 검색 점수 (유사도)
    - `chunk_id`: 청크 ID
    - `partition_name`: 파티션 이름
    - `file_name`: 파일명
  - `total`: 검색 결과 총 개수
  - `search_method`: 사용된 검색 방법 (dense/sparse/hybrid)
- **주의**: Knowledge Base의 설정된 검색 방법(search_method), top_k, threshold를 사용하여 검색을 수행합니다

### 10. Knowledge Base 검색 기록 조회
**API**: `GET /api/v1/knowledge-bases/{knowledge_base_id}/search-records`

- 경로 파라미터: `knowledge_base_id` (Knowledge Base ID)
- 응답: 검색 기록 목록 (List[KnowledgeBaseSearchRecordReadSchema])
  - `id`: 검색 기록 ID
  - `knowledge_base_id`: Knowledge Base ID
  - `source`: Collection 이름
  - `text`: 검색 쿼리 텍스트
  - `created_at`: 검색 기록 생성 시간

## Knowledge Base API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 청크 타입 목록 조회 | `/api/v1/knowledge-bases/chunk-types` | GET | - |
| 언어 목록 조회 | `/api/v1/knowledge-bases/languages` | GET | - |
| 검색 방법 목록 조회 | `/api/v1/knowledge-bases/search-methods` | GET | - |
| 임베딩 모델 목록 조회 | `/api/v1/models` | GET | `model_type_id` (쿼리) |
| Knowledge Base 생성 | `/api/v1/knowledge-bases` | POST | `name`, `language_id`, `embedding_model_id`, `chunk_size`, `chunk_overlap`, `chunk_type_id`, `search_method_id`, `top_k`, `threshold`, `file` (multipart/form-data) |
| 파일 추가 | `/api/v1/knowledge-bases/{knowledge_base_id}/files` | POST | `knowledge_base_id` (경로), `file` (multipart/form-data) |
| 파일 삭제 | `/api/v1/knowledge-bases/{knowledge_base_id}/files/{file_id}` | DELETE | `knowledge_base_id`, `file_id` (경로) |
| Knowledge Base 목록 조회 | `/api/v1/knowledge-bases` | GET | `page`, `page_size` (선택) |
| Knowledge Base 상세 조회 | `/api/v1/knowledge-bases/{knowledge_base_id}` | GET | `knowledge_base_id` (경로) |
| Knowledge Base 수정 | `/api/v1/knowledge-bases/{knowledge_base_id}` | PUT | `knowledge_base_id` (경로), `name`, `description` (요청 바디) |
| Knowledge Base 삭제 | `/api/v1/knowledge-bases/{knowledge_base_id}` | DELETE | `knowledge_base_id` (경로) |
| 검색 테스트 | `/api/v1/knowledge-bases/{knowledge_base_id}/search` | POST | `knowledge_base_id` (경로), `text` (요청 바디) |
| 검색 기록 조회 | `/api/v1/knowledge-bases/{knowledge_base_id}/search-records` | GET | `knowledge_base_id` (경로) |

## Knowledge Base 참고사항

- 모든 API는 Bearer Token 인증 필요
- Knowledge Base 생성 시 필요한 ID 값들은 각각 해당하는 조회 API를 먼저 호출하여 확인해야 합니다:
  - `language_id`: `GET /api/v1/knowledge-bases/languages`
  - `embedding_model_id`: `GET /api/v1/models?model_type_id={embedding_type_id}`
  - `chunk_type_id`: `GET /api/v1/knowledge-bases/chunk-types`
  - `search_method_id`: `GET /api/v1/knowledge-bases/search-methods`
- 파일 업로드 시 파일은 자동으로 청크로 분할되고 임베딩되어 Milvus에 저장됩니다
- 추가된 파일은 Milvus의 동일한 Collection에 Partition으로 저장됩니다
- Knowledge Base 수정은 이름과 설명만 가능하며, 청크 설정이나 검색 설정은 수정할 수 없습니다
- 검색은 Knowledge Base에 설정된 `search_method`, `top_k`, `threshold` 값을 사용합니다
- `threshold`는 0.0 ~ 1.0 사이의 값이어야 합니다
- 상세한 요청/응답 형식은 각 API의 docstring 참조
