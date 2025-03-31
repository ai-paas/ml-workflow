#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_step() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 에러 핸들링 함수
handle_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 스크립트 시작
echo -e "${BLUE}=== Python Development Environment Setup ===${NC}"

# python 가상환경 세팅 및 package 설치
log_step "Upgrading pip..."
pip install --upgrade pip || handle_error "pip 업그레이드 실패"
log_success "Pip 업그레이드 완료"

# pipenv 설치
log_step "Installing pipenv..."
pip install pipenv || handle_error "pipenv 설치 실패"
log_success "Pipenv 설치 완료"

# 의존성 설치
log_step "Installing dependencies..."
pipenv install --dev || handle_error "의존성 설치 실패"
log_success "의존성 설치 완료"

# pre-commit 설치
log_step "Installing pre-commit hooks..."
pipenv run pre-commit install || handle_error "pre-commit 설치 실패"
log_success "Pre-commit hooks 설치 완료"

log_step "Configuring prepare-commit-msg hook..."
# Repository Root에서 시작
cd .git/hooks

# prepare-commit-msg 파일 생성
touch prepare-commit-msg

# 실행 권한 부여
chmod +x prepare-commit-msg

# .git/hooks/prepare-commit-msg 내용 작성
cat << 'EOF' > prepare-commit-msg
#!/bin/bash
FIRST_LINE=$(head -n 1 $1)

COMMITFORMAT="(feat|fix|docs|style|refactor|test|chore|build|ci|perf|security|hotfix): (.*) \\(PAAS-[0-9]{1,4}\\)$"

if ! [[ "$FIRST_LINE" =~ $COMMITFORMAT ]]; then
  echo ""
  echo " Commit Message 포맷을 아래 예시와 같이 지켜주세요."
  echo " Prefix : 사용가능한 commit의 Prefix는 아래와 같습니다."
  echo " Suffix : 반드시 commit에 해당하는 JIRA Ticket 번호를 괄호(DOCS-000)로 묶어서 마지막에 기입해주세요."
  echo ""
  echo "======================= 반드시 콜론(:) 을 붙여야 합니다. ========================="
  echo ""
  echo "1. feat: 새로운 기능 추가"
  echo "  - 새로운 기능이나 기능적 변경 사항을 추가할 때 사용합니다."
  echo "    예: feat(user): 로그인 기능 추가"
  echo ""
  echo "2. fix: 버그 수정"
  echo "  - 기존 코드에서 발견된 버그를 수정할 때 사용합니다."
  echo "    예: fix(auth): 로그인 페이지에서 발생하는 오류 수정"
  echo ""
  echo "3. docs: 문서 변경"
  echo "  - 문서(README, 도움말, 코드 주석 등)에 대한 수정이나 추가에 사용합니다."
  echo "    예: docs: API 문서 업데이트"
  echo ""
  echo "4. style: 코드 스타일 수정"
  echo "  - 코드의 동작에 영향을 미치지 않는 스타일 수정을 나타냅니다."
  echo "    예: style: 들여쓰기 문제 수정"
  echo ""
  echo "5. refactor: 리팩토링"
  echo "  - 기능 변경 없이 코드 구조나 효율성을 개선하는 변경에 사용합니다."
  echo "    예: refactor(auth): 로그인 기능 리팩토링"
  echo ""
  echo "6. test: 테스트 코드 추가/수정"
  echo "  - 새로운 테스트 추가나 기존 테스트 수정에 사용합니다."
  echo "    예: test: 로그인 기능 유닛 테스트 추가"
  echo ""
  echo "7. chore: 잡다한 작업"
  echo "  - 코드, 문서 등 주요 기능과 관련이 없는 기타 작업에 사용합니다."
  echo "    예: chore: 의존성 패키지 업데이트"
  echo ""
  echo "8. build: 빌드 관련 변경"
  echo "  - 빌드 시스템(webpack, Gulp, Gradle 등)이나 의존성 관련 변경에 사용합니다."
  echo "    예: build: 프로젝트 빌드 설정 업데이트"
  echo ""
  echo "9. ci: CI/CD 설정 변경"
  echo "  - CI/CD 파이프라인 설정 변경에 사용됩니다."
  echo "    예: ci: GitHub Actions 설정 추가"
  echo ""
  echo "10. perf: 성능 개선"
  echo "  - 성능을 개선하기 위한 변경 사항에 사용됩니다."
  echo "    예: perf: 이미지 로딩 속도 개선"
  echo ""
  echo "11. security: 보안 관련 변경"
  echo "  - 보안을 강화하기 위한 변경 사항에 사용됩니다."
  echo "    예: security: XSS 취약점 수정"
  echo ""
  echo "12. hotfix: 긴급 수정"
  echo "  - 배포 후 긴급하게 수정해야 할 버그를 수정하는 경우에 사용됩니다."
  echo "    예: hotfix: 프로덕션 환경에서 발생한 로그인 버그 수정"
  echo ""
  echo "=================================================================================="
  echo ""
  echo -e " 아래 EXAMPLE과 같이 첫째 줄에 Prefix와 함께 요약을 남기고 한 줄 개행 후 상세 내용을 작성해주세요. \n Merge Request 시 Overview에 자동으로 Title, Description 작성이 완료됩니다."
  echo ""
  echo "================================== E X A M P L E ================================="
  echo ""
  echo -e " git commit -m \"feat: 기능 A 추가 (DOCS-123)\n\n  1. 000파일 추가 \n  2. 2222파일추가\n  3. 00 관련 비즈니스 로직 추가\""
  echo ""
  echo "=================================================================================="
  echo ""
  exit 1
fi
EOF

log_success "Prepare-commit-msg hook 설정 완료"

# 최종 완료 메시지
echo -e "${GREEN}=== Development Environment Setup Complete ===${NC}"
echo -e "  - ${BLUE}Pipfile 명시 패키지 설치 완료${NC}"
echo -e "  - ${BLUE}pre-commit-hook 설정 완료${NC}"
echo -e "${YELLOW}사용 방법:${NC}"
echo -e "  - 가상환경 활성화(개발시 필수): ${BLUE}pipenv shell${NC}"
echo -e "  - 문의: ${BLUE}조한슬(Teams, email(hsjo@surromind.ai)${NC}"
