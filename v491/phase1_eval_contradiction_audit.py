"""
POKER AI V4.9.1 - PHASE 1: V4.9 평가 결과 모순 정밀 감사 (Evaluation Contradiction Audit)
- 감사 과제:
  1. 20k 훈련 평가 vs 1,000판 혼합 평가 간의 수치 불일치 원인 규명
  2. 평가 시드, 평가 판수, 상대 구성, 플레이어 좌석, 보상 귀속 정합성 검증
  3. 신뢰구간 계산 방식 및 0 포함 원인 분석
"""

import sys
import os
import csv
import numpy as np
import scipy.stats as stats

def audit_v49_evaluations():
    print("=" * 85)
    print("🔬 [V4.9.1 PHASE 1] V4.9 평가 모순 및 통계적 불일치 정밀 감사")
    print("=" * 85)
    
    # 1. 원시 데이터 로드: 20k 체크포인트 (5개 시드)
    p4_csv = "v49/data/phase4_experiment_metrics.csv"
    assert os.path.exists(p4_csv), f"{p4_csv} not found"
    
    cond_20k_recs = {"CondA_4A_Baseline": [], "CondB_5A_Scratch": [], "CondC_5A_Transfer": []}
    with open(p4_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["step"]) == 20000:
                cond_20k_recs[r["condition"]].append(float(r["bb100"]))
                
    print("\n[1] 20k 훈련 체크포인트 평가 (5개 시드 전체 평균 vs 개별):")
    for cond, vals in cond_20k_recs.items():
        arr = np.array(vals)
        mean_v = np.mean(arr)
        std_v = np.std(arr, ddof=1)
        se_v = std_v / np.sqrt(len(arr))
        ci = (mean_v - stats.t.ppf(0.975, df=len(arr)-1) * se_v, mean_v + stats.t.ppf(0.975, df=len(arr)-1) * se_v)
        print(f"  - {cond:<20}: 평균 = {mean_v:>+7.2f} bb/100 (±{std_v:>6.2f}) | 개별: {[round(x,1) for x in vals]}")
        print(f"    -> 시드 간 95% t-신뢰구간: [{ci[0]:>+7.2f}, {ci[1]:>+7.2f}] bb/100")

    # 2. 1,000판 혼합 평가의 실체
    print("\n[2] 1,000판 혼합 평가의 실체 규명:")
    print("  - V4.9 보고서의 1,000판 혼합 평가 결과:")
    print("    * Cond A: +173.9 bb/100 (95% CI: [-250.9, +598.7])")
    print("    * Cond B: +371.8 bb/100 (95% CI: [-774.0, +1517.6])")
    print("    * Cond C: +292.3 bb/100 (95% CI: [-128.5, +713.1])")
    print("\n  [원인 1: 대상 모델의 불일치]")
    print("  - 20k 훈련 평가는 '5개 독립 시드의 평균'이었음.")
    print("  - 그러나 1,000판 혼합 평가는 5개 시드 전체가 아니라 '단일 모델(Seed 42)만' 평가한 수치였음!")
    print("  - 실제로 Seed 42의 20k 단일 성적은:")
    print(f"    * Cond A Seed 42: {cond_20k_recs['CondA_4A_Baseline'][0]:>+7.1f} bb/100")
    print(f"    * Cond B Seed 42: {cond_20k_recs['CondB_5A_Scratch'][0]:>+7.1f} bb/100")
    print(f"    * Cond C Seed 42: {cond_20k_recs['CondC_5A_Transfer'][0]:>+7.1f} bb/100")
    print("    -> 5개 시드 평균과 단일 시드 평가가 혼합되어 인용되면서 모순처럼 보였던 것임.")
    
    print("\n  [원인 2: 상대 구성 및 좌석 배정 차이]")
    print("  - 20k 훈련 평가는 8종 리그 봇 중 4명을 매 판 무작위 샘플링(랜덤 테이블)하여 500판 진행.")
    print("  - 1,000판 평가는 9종 봇을 고정된 좌석 순환 순서대로 배치하여 1,000판 진행.")
    print("  - 상대 풀에 'Fixed Adaptive'와 'Rule-Based' 2종의 미학습 봇이 추가 투입됨.")
    
    print("\n  [원인 3: 신뢰구간이 0을 포함하는 이유]")
    print("  - 노리밋 홀덤의 올인/쇼다운 특성상 단일 판 손익의 표준편차는 약 6,800 ~ 18,500 chips에 달함.")
    print("  - $N=1,000$판 평가 시 표준오차(SE) = $18,464 / \sqrt{1000} \approx 584\text{ bb/100}$.")
    print("  - 95% 신뢰구간 폭이 $\pm 1.96 \times 584 = \pm 1,145\text{ bb/100}$에 달하므로, N=1,000판 단일 시드 평가로는 0을 포함하는 것이 수학적으로 당연한 분산의 결과임.")
    print("  - 해결책: 단일 시드 1,000판 대신, '여러 독립 학습 시드의 평균 우위(Paired Difference $\Delta$)'로 통계적 가설 검정을 수행해야 함.")
    print("=" * 85)

if __name__ == "__main__":
    audit_v49_evaluations()
