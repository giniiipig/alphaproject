# -*- coding: utf-8 -*-
"""
외부 실행용 런처 코드
recommand1.py를 독립적으로 실행시킵니다.
"""

import runpy
import os
import sys

if __name__ == "__main__":
    # recommand1.py 파일 경로 (현재 폴더 기준)
    target = os.path.join(os.path.dirname(__file__), "src/recommendation/recommand1.py")

    if not os.path.exists(target):
        print(f"❌ 실행 대상 파일을 찾을 수 없습니다: {target}")
        sys.exit(1)

    print("🚀 recommand1 실행 시작...")
    runpy.run_path(target, run_name="__main__")
    print("✅ recommand1 실행 종료")
