# 프롬프트 관련 API 사용 가이드

UI에서 프롬프트 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 프롬프트 워크플로우

### 1. 프롬프트 생성

**API**: `POST /api/v1/prompts`

- 요청 바디: `PromptCreateSchema`
  - **prompt** (PromptBaseSchema, 필수): 프롬프트 기본 정보
    - `name` (str, 필수): 프롬프트 이름
    - `description` (str, 선택): 프롬프트 설명
    - `content` (str, 필수): 프롬프트 내용
  - **prompt_variable** (List[str], 선택): 프롬프트 변수 이름 목록
- 응답: 생성된 프롬프트 정보 (PromptReadSchema)
  - `id`: 프롬프트 ID
  - `name`: 프롬프트 이름
  - `description`: 프롬프트 설명
  - `content`: 프롬프트 내용
  - `prompt_variable`: 프롬프트 변수 목록 (각 변수의 id, name, prompt_id 포함)

**예시 요청**:
```json
{
  "prompt": {
    "name": "고객 서비스 프롬프트",
    "description": "고객 문의에 대한 응답 프롬프트",
    "content": "안녕하세요, {customer_name}님. {question}에 대해 답변드리겠습니다."
  },
  "prompt_variable": ["customer_name", "question"]
}
```

### 2. 프롬프트 목록 조회

**API**: `GET /api/v1/prompts`

- 쿼리 파라미터: `page`, `page_size` (선택사항)
  - `page`: 페이지 번호 (1부터 시작, 최소값: 1)
  - `page_size`: 페이지당 항목 수 (범위: 1-1000)
- 생략 시 전체 데이터 조회 (최대 10000개)
- `page`와 `page_size` 중 하나라도 생략하면 전체 데이터 조회
- 응답: 프롬프트 목록 (List[PromptReadSchema])
  - 각 프롬프트의 기본 정보와 변수 목록 포함

### 3. 프롬프트 상세 조회

**API**: `GET /api/v1/prompts/{prompt_id}`

- 경로 파라미터: `prompt_id` (int)
- 응답: 프롬프트 상세 정보 (PromptReadSchema)
  - `id`: 프롬프트 ID
  - `name`: 프롬프트 이름
  - `description`: 프롬프트 설명
  - `content`: 프롬프트 내용
  - `prompt_variable`: 프롬프트 변수 목록
    - 각 변수: `id`, `name`, `prompt_id`

### 4. 프롬프트 수정

**API**: `PUT /api/v1/prompts/{prompt_id}`

- 경로 파라미터: `prompt_id` (int)
- 요청 바디: `PromptUpdateSchema`
  - `name` (str, 선택): 프롬프트 이름
  - `description` (str, 선택): 프롬프트 설명
  - `content` (str, 선택): 프롬프트 내용
  - `prompt_variable` (List[str], 선택): 프롬프트 변수 이름 목록
- 응답: 수정된 프롬프트 정보 (PromptReadSchema)
- **주의**:
  - `prompt_variable`을 제공하면 기존 변수는 모두 삭제되고 새로운 변수로 대체됨
  - 수정하지 않을 필드는 생략 가능

**예시 요청**:
```json
{
  "name": "수정된 프롬프트 이름",
  "content": "수정된 프롬프트 내용 {new_variable}",
  "prompt_variable": ["new_variable"]
}
```

### 5. 프롬프트 삭제

**API**: `DELETE /api/v1/prompts/{prompt_id}`

- 경로 파라미터: `prompt_id` (int)
- 응답: 204 No Content
- **주의**:
  - 프롬프트 삭제 시 관련된 모든 변수(prompt_variable)도 자동으로 삭제됨 (CASCADE DELETE)
  - 삭제는 되돌릴 수 없음

## 프롬프트 API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 프롬프트 생성 | `/api/v1/prompts` | POST | `prompt` (name, description, content), `prompt_variable` (선택) |
| 프롬프트 목록 조회 | `/api/v1/prompts` | GET | `page`, `page_size` (선택) |
| 프롬프트 상세 조회 | `/api/v1/prompts/{prompt_id}` | GET | `prompt_id` (경로) |
| 프롬프트 수정 | `/api/v1/prompts/{prompt_id}` | PUT | `prompt_id` (경로), `name`, `description`, `content`, `prompt_variable` (선택) |
| 프롬프트 삭제 | `/api/v1/prompts/{prompt_id}` | DELETE | `prompt_id` (경로) |

## 프롬프트 참고사항

- 모든 API는 Bearer Token 인증 필요
- 프롬프트 변수는 프롬프트 내용에서 `{variable_name}` 형식으로 사용할 수 있는 변수
- 프롬프트 수정 시 `prompt_variable`을 제공하면 기존 변수는 모두 삭제되고 새로 생성됨
- 프롬프트 삭제 시 관련된 모든 변수도 자동 삭제됨 (CASCADE DELETE)
- 프롬프트 내용에 변수를 사용할 때는 `{변수명}` 형식으로 작성
- 상세한 요청/응답 형식은 각 API의 docstring 참조
