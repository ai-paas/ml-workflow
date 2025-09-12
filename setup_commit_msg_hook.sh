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

echo "🔄 커밋 메시지 검사 및 포매팅 중..."

# 머지 커밋인지 확인
if grep -q "^Merge " "$1"; then
  echo "🔄 머지 커밋 감지: 포매팅 스킵"
  exit 0
fi

COMMIT_MESSAGE_FILE=$1
COMMIT_MESSAGE=$(cat "$COMMIT_MESSAGE_FILE")

# 현재 브랜치 명 가져오기
CURRENT_BRANCH=$(git branch --show-current)

# 브랜치 명이 이미 포함되어 있는지 확인
if echo "$COMMIT_MESSAGE" | grep -q "^$CURRENT_BRANCH: "; then
  echo "✅ 브랜치 명이 이미 포함되어 있습니다."
  exit 0
fi

# 커밋 타입 정의
VALID_TYPES="feat|refactor|fix|chore|style|test"

# 기존 커밋 메시지에서 타입 추출 또는 자동 생성
FORMATTED_MESSAGE=""

if echo "$COMMIT_MESSAGE" | grep -qE "^($VALID_TYPES): "; then
  # 이미 올바른 형식이면 브랜치 명만 앞에 추가
  FORMATTED_MESSAGE="$CURRENT_BRANCH $COMMIT_MESSAGE"
else
  # 형식이 맞지 않으면 자동 포매팅
  echo "⚠️  커밋 메시지 형식을 자동으로 수정합니다..."

  # 첫 번째 줄과 나머지 줄 분리
  FIRST_LINE=$(echo "$COMMIT_MESSAGE" | head -n1)
  REST_LINES=$(echo "$COMMIT_MESSAGE" | tail -n +2)

  # 키워드 기반 자동 타입 감지
  AUTO_TYPE="feat"  # 기본값

  if echo "$FIRST_LINE" | grep -qiE "(fix|bug|error|issue)"; then
    AUTO_TYPE="fix"
  elif echo "$FIRST_LINE" | grep -qiE "(refactor|restructure|reorganize)"; then
    AUTO_TYPE="refactor"
  elif echo "$FIRST_LINE" | grep -qiE "(test|spec)"; then
    AUTO_TYPE="test"
  elif echo "$FIRST_LINE" | grep -qiE "(style|format|indent|prettier|eslint)"; then
    AUTO_TYPE="style"
  elif echo "$FIRST_LINE" | grep -qiE "(chore|config|build|deps|dependency)"; then
    AUTO_TYPE="chore"
  fi

  # 브랜치 명 + 타입 + 메시지 조합
  FORMATTED_MESSAGE="$CURRENT_BRANCH $AUTO_TYPE: $FIRST_LINE"

  echo "✅ 커밋 메시지가 '$CURRENT_BRANCH $AUTO_TYPE:' 형식으로 자동 수정되었습니다."
fi

# 새로운 커밋 메시지 작성
echo "$FORMATTED_MESSAGE" > "$COMMIT_MESSAGE_FILE"
if [ -n "$REST_LINES" ]; then
  echo "$REST_LINES" >> "$COMMIT_MESSAGE_FILE"
fi

echo "🔆 브랜치 명이 포함된 커밋 메시지가 완성되었습니다! 🔆"
EOF
