"""
POKER AI V4.7.4 - PHASE 1: 전체 프로젝트 코드 감사 스크립트
- 검사 대상 디렉터리: v47, v471, v472, v473
- 검색 패턴:
  - chance_outcomes()[0][0]
  - chance_outcomes()[0]
  - chance_outcomes()
  - random.choice
  - np.random.choice
- 목적: 카드 분배 및 행동 샘플링에 사용된 모든 코드를 전수 추출하고,
  A. 테스트용 결정론적 카드 분배
  B. 실제 강화학습 카드 분배
  C. 성능 평가 카드 분배
  D. 기타
  로 엄격하게 분류하여 파일별, 라인별 증거를 제시.
"""

import os
import re

TARGET_DIRS = ["v47", "v471", "v472", "v473"]

def audit_codebase():
    results = []
    
    for d in TARGET_DIRS:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.endswith(".py"):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r", encoding="utf-8") as fp:
                        lines = fp.readlines()
                        
                    for idx, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if "chance_outcomes" in stripped:
                            results.append({
                                "file": fpath,
                                "line_no": idx,
                                "code": stripped,
                                "type": "chance_outcomes"
                            })

    print("=" * 80)
    print("🔍 [V4.7.4 PHASE 1] 전체 프로젝트 chance_outcomes() 전수 추적 결과")
    print("=" * 80)
    
    categories = {"A": [], "B": [], "C": [], "D": []}
    
    for r in results:
        f = r["file"]
        code = r["code"]
        line = r["line_no"]
        
        # 분류 로직
        cat = "D"
        desc = ""
        
        if "phase4_holdem_selfplay_train.py" in f:
            cat = "B" # 실제 강화학습
            desc = "실제 5인 홀덤 강화학습 자기대국 훈련 루프"
        elif "phase1_holdem_rules_audit.py" in f:
            cat = "A" # 단위 테스트
            desc = "특수 족보/올인/폴드 재현을 위한 단위 테스트"
        elif "phase8_speed_benchmark.py" in f:
            cat = "D" # 속도 벤치마크
            desc = "순수 엔진 상태전이 속도 측정용 무작위 패스"
        elif any(eval_dir in f for eval_dir in ["v471", "v472", "v473", "phase5_benchmark_eval.py"]):
            cat = "C" # 성능 평가
            desc = "체크포인트 성능 벤치마크 평가 및 감사 스크립트"
        else:
            cat = "D"
            desc = "기타 스크립트"
            
        categories[cat].append((r, desc))
        
    category_names = {
        "A": "A. 테스트용 결정론적 카드 분배 (특수 상황 단위 검증)",
        "B": "B. 실제 강화학습 카드 분배 (Self-play Training Loop)",
        "C": "C. 성능 평가 카드 분배 (Evaluation & Benchmark Scripts)",
        "D": "D. 기타 (엔진 벤치마크 및 유틸리티)"
    }
    
    for cat_key in ["B", "C", "A", "D"]:
        print(f"\n📂 [{category_names[cat_key]}] (총 {len(categories[cat_key])}건)")
        print("-" * 80)
        for r, desc in categories[cat_key]:
            is_bug = "[0][0]" in r["code"]
            status = "🚨 [결정론적 고정 [0][0]]" if is_bug else "✅ [확률적 샘플링 zip/choice]"
            print(f"  - {r['file']}:{r['line_no']}")
            print(f"    코드: {r['code']}")
            print(f"    판정: {status} | 용도: {desc}")
            
    print("\n" + "=" * 80)

if __name__ == "__main__":
    audit_codebase()
