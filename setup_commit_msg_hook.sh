#!/bin/bash

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

COMMITFORMAT="(feat|fix|docs|style|refactor|design|comment|rename|remove|chore|!HOTFIX|!BREAKING CHANGE|revert): (.*) \\((CSG|LS)-[0-9]{2,4}\\)$"

if ! [[ "$FIRST_LINE" =~ $COMMITFORMAT ]]; then
  echo ""
  echo " Commit Message 포맷을 아래 예시와 같이 지켜주세요."
  echo " Prefix : 사용가능한 commit의 Prefix는 아래와 같습니다."
  echo " Suffix : 반드시 commit에 해당하는 JIRA Ticket 번호를 괄호(CSG-000 또는 LS-000)로 묶어서 마지막에 기입해주세요."
  echo ""
  echo "======================= 반드시 콜론(:) 을 붙여야 합니다. ========================="
  echo ""
  echo "  feat:             새로운 기능을 추가"
  echo "  fix:              버그 수정"
  echo "  design:           CSS 등 사용자 UI 디자인 변경"
  echo "  !BREAKING CHANGE: 커다란 API 변경의 경우"
  echo "  !HOTFIX:          급하게 치명적인 버그를 고쳐야하는 경우"
  echo "  style:            코드 포맷 변경, 세미 콜론 누락, 코드 수정이 없는 경우"
  echo "  refactor:         코드 리팩토링"
  echo "  comment:          필요한 주석 추가 및 변경"
  echo "  docs:             문서 수정"
  echo "  chore:            빌드 업무 수정, 패키지 매니저 수정, 패키지 관리자 구성 등 업데이트, Production Code 변경 없음"
  echo "  rename:           파일 혹은 폴더명을 수정하거나 옮기는 작업만인 경우"
  echo "  remove:           파일을 삭제하는 작업만 수행한 경우"
  echo "  test:             테스트 코드 작성 등 테스트 관련 작업"
  echo "  revert:           커밋 되돌리기 작업"
  echo ""
  echo "=================================================================================="
  echo ""
  echo -e " 아래 EXAMPLE과 같이 첫째 줄에 Prefix와 함께 요약을 남기고 한 줄 개행 후 상세 내용을 작성해주세요. \n Merge Request 시 Overview에 자동으로 Title, Description 작성이 완료됩니다."
  echo ""
  echo "================================== E X A M P L E ================================="
  echo ""
  echo -e " git commit -m \"feat: 기능 A 추가 (CSG-123)\n\n  1. 000파일 추가 \n  2. 2222파일추가\n  3. 00 관련 비즈니스 로직 추가\""
  echo -e " git commit -m \"fix: 버그 수정 (LS-456)\n\n  1. 000버그 수정 \n  2. 2222오류 해결\""
  echo ""
  echo "=================================================================================="
  echo ""
  exit 1
fi
EOF
