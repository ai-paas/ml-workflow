#!/bin/bash

# Repository Root에서 시작
cd .git/hooks

# prepare-commit-msg 파일 생성
touch prepare-commit-msg

# 실행 권한 부여
chmod +x prepare-commit-msg

# .git/hooks/prepare-commit-msg 내용 작성
cat << 'EOF' > prepare-commit-msg
#!/bin/sh
echo "🔄 커밋 메시지 검사 중..."
# 머지 커밋인지 확인
if grep -q "^Merge " "$1"; then
  echo "🔄 머지 커밋 감지: 컨벤션 체크 스킵"
  exit 0
fi
COMMIT_MESSAGE_FILE=$1
COMMIT_MESSAGE=$(cat "$COMMIT_MESSAGE_FILE")
# 커밋 타입 정의
VALID_TYPES="feat|refactor|fix|chore|style|test|docs|perf|ci|build|revert"
# 커밋 메시지가 '타입: 메시지 (티켓넘버)' 형식인지 검사
# 티켓 넘버는 (SDOCS-123) 또는 (SDOCS-123, SDOCS-124) 형식을 허용합니다.
if ! echo "$COMMIT_MESSAGE" | grep -qE "^($VALID_TYPES): .+ \([A-Za-z]+-[0-9]+(, [A-Za-z]+-[0-9]+)*\)$"; then
  echo "❌ 커밋 메시지는 반드시 '타입: 메시지 (티켓넘버)' 형식이어야 합니다."
  echo "   예: feat: 테스트 커밋 (SDOCS-123)"
  echo "   예: fix: 버그 수정 (SDOCS-123, SDOCS-124)"
  echo ""
  echo "   [허용 타입]"
  echo "   - feat: 새로운 기능 추가"
  echo "   - refactor: 코드 리팩토링"
  echo "   - fix: 버그 수정"
  echo "   - chore: 빌드 프로세스 또는 보조 도구 변경"
  echo "   - style: 코드 스타일 변경 (포맷팅, 세미콜론 등)"
  echo "   - test: 테스트 코드 추가 또는 수정"
  echo "   - docs: 문서 변경"
  echo "   - perf: 성능 개선"
  echo "   - ci: CI/CD 설정 변경"
  echo "   - build: 빌드 설정/의존성 변경"
  echo "   - revert: 이전 커밋으로 되돌리기"
  exit 1
fi
echo "🔆 브랜치 명이 포함된 커밋 메시지가 완성되었습니다! 🔆"
EOF
