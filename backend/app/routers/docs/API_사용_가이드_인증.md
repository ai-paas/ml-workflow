# 인증(Authentication) 관련 API 사용 가이드

UI에서 인증 관련 기능을 구현할 때 사용할 API 호출 순서를 정리한 가이드입니다.
상세한 요청/응답 형식은 각 API의 docstring을 참조하세요.

## 인증 워크플로우

### 1. 로그인 및 토큰 발급
**API**: `POST /api/v1/authentications/token`

- **Content-Type**: `application/x-www-form-urlencoded` (OAuth2 표준)
- 필수 파라미터: `username`, `password`
- 응답: `{ "access_token": "...", "token_type": "bearer" }`
- 토큰 유효기간: 24시간
- 발급된 토큰은 이후 모든 API 요청에 `Authorization: Bearer {access_token}` 헤더로 사용

### 2. 현재 사용자 정보 조회
**API**: `GET /api/v1/authentications/users/me`

- **Authorization 헤더**: `Bearer {access_token}` (필수)
- 응답: 현재 로그인한 사용자 정보 (UserSchema)
  - `id`, `username`, `name`, `password` (해시값), `created_at`, `updated_at` 등
- 토큰에서 사용자 정보를 자동으로 추출하여 반환

## API 요약

| 작업 | API | 메서드 | 주요 파라미터 |
|------|-----|--------|--------------|
| 로그인 및 토큰 발급 | `/api/v1/authentications/token` | POST | `username`, `password` (form-data) |
| 현재 사용자 정보 조회 | `/api/v1/authentications/users/me` | GET | Authorization 헤더 (Bearer Token) |

## 참고사항

- 모든 보호된 API는 Bearer Token 인증 필요
- 토큰 만료 시 재로그인 필요
- OAuth2 Password Flow 표준 사용
- JWT 기반 인증 (HS256 알고리즘)
- 상세한 요청/응답 형식은 각 API의 docstring 참조
