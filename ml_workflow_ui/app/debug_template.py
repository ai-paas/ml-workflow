"""템플릿 생성 및 조회 디버깅 스크립트"""

import sys
from pathlib import Path

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from api_client import APIClient


def main():
    print("=" * 60)
    print("🔍 템플릿 디버깅 스크립트")
    print("=" * 60)
    print()

    # 1. 로그인
    print("1️⃣ 로그인 시도...")
    username = "surromind"
    password = "mindsurro"  # 실제 비밀번호로 변경

    token = APIClient.authenticate(username, password)
    if not token:
        print("❌ 로그인 실패")
        return

    print("✅ 로그인 성공!")
    print()

    # 2. API 클라이언트 생성
    client = APIClient(token)

    # 3. 템플릿 목록 조회
    print("2️⃣ 템플릿 목록 조회...")
    try:
        templates = client.get_workflow_templates()
        print(f"✅ API 응답 받음: {len(templates)}개의 템플릿")
        print()

        if not templates:
            print("⚠️  템플릿이 없습니다!")
            print()
            print("📝 템플릿 생성 방법:")
            print("   python utils/create_template.py --password YOUR_PASSWORD --model-id 19")
            print()
        else:
            print(f"✨ 템플릿 목록 ({len(templates)}개):")
            print("-" * 60)
            for t in templates:
                print(f"\n📦 템플릿 #{t.get('id')}")
                print(f"   이름: {t.get('name')}")
                print(f"   카테고리: {t.get('category', 'N/A')}")
                print(f"   상태: {t.get('status')}")
                print(f"   템플릿 여부: {t.get('is_template')}")
                print(f"   설명: {t.get('description', '')[:80]}")
                if t.get("creator"):
                    print(f"   생성자: {t.get('creator', {}).get('username', 'N/A')}")
            print()

        # 4. 워크플로우 조회 (템플릿 제외)
        print("3️⃣ 워크플로우 조회 (템플릿 제외)...")
        all_workflows = client.get_workflows()
        print(f"✅ 일반 워크플로우: {all_workflows.get('total', 0)}개")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 60)
    print("✅ 디버깅 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
