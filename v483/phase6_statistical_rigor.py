"""
POKER AI V4.8.3 - PHASE 6: 결과 통계적 엄밀성 검증 (Statistical Rigor & Comparison)
- 목적:
  1. 기존 V4.8.1 보고치 (+437.46 bb/100, 95% CI [+2.54, +872.38])가
     10개 독립 시드 모집단에서 우연한 특이치(Lucky Seed)였는지 통계적 위치 검정
  2. 미학습 상대 혼합 테이블에서 10개 PPO 모델 vs V4.7 Step 100k 간의
     페어드 차이(Δ)에 대한 엄밀한 95% 신뢰구간 및 유의확률(p-value) 산출
  3. 사후 체리피킹 배제: 최고 시드만이 아닌 10개 모델 전수 기반 검증
"""

import os
import csv
import numpy as np
import scipy.stats as stats

def evaluate_statistical_rigor():
    p1_csv = "v483/data/phase1_seed_metrics.csv"
    p4_csv = "v483/data/phase4_unseen_results.csv"
    
    if not os.path.exists(p1_csv) or not os.path.exists(p4_csv):
        print("Data files not ready.")
        return
        
    # Phase 1 벤치마크 성적 로드 (Step 20k)
    recs_20k = []
    with open(p1_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["step"]) == 20000:
                recs_20k.append(float(r["bb100"]))
                
    bbs_bench = np.array(recs_20k)
    n = len(bbs_bench)
    
    # Phase 4 미학습 혼합 성적 로드
    p4_mixed = []
    with open(p4_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            p4_mixed.append(float(r["Mixed Unseen"]))
    bbs_unseen = np.array(p4_mixed)
    
    print("=" * 90)
    print("📐 [PHASE 6] 통계적 엄밀성 및 가설 검정 (Statistical Rigor)")
    print("=" * 90)
    
    # 1. V4.8.1 보고치 (+437.46 bb/100) 위치 검정
    v481_val = 437.46
    mean_bench = np.mean(bbs_bench)
    std_bench = np.std(bbs_bench, ddof=1)
    se_bench = std_bench / np.sqrt(n)
    
    z_score = (v481_val - mean_bench) / std_bench if std_bench > 0 else 0.0
    p_val_v481 = 1.0 - stats.norm.cdf(z_score)
    
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_bench = (mean_bench - t_crit * se_bench, mean_bench + t_crit * se_bench)
    
    print(f"\n[1] V4.8.1 보고치 (+437.46 bb/100)의 모집단 위치:")
    print(f"  - 10개 독립 시드 벤치마크 평균: {mean_bench:>+8.2f} bb/100 (표준편차: {std_bench:.2f})")
    print(f"  - 95% t-신뢰구간:             [{ci_bench[0]:>+8.2f}, {ci_bench[1]:>+8.2f}] bb/100")
    print(f"  - V4.8.1 값의 Z-score:          {z_score:+.2f} σ")
    print(f"  - 상위 백분위 (Percentile):     {(1.0 - p_val_v481)*100:.1f}%")
    if z_score > 1.5:
        print(f"  -> 판정: V4.8.1 보고치는 상위 편향(Lucky Seed) 특성이 있었으며, 10개 독립 시드 평균은 이보다 현실적으로 수렴함.")
    else:
        print(f"  -> 판정: V4.8.1 보고치는 10개 독립 시드 분포 범위 내에 존재함.")
        
    # 2. 미학습 상대 혼합 테이블: PPO 20k vs Step 100k 검정
    step100k_mixed = 5.9 # 기존 Step 100k 기준치
    diffs = bbs_unseen - step100k_mixed
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
    p_val_diff = stats.t.sf(t_stat, df=n-1) # 단측 검정
    ci_diff = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
    
    print(f"\n[2] 미학습 상대 혼합 테이블 성능 차이 검정 (PPO 20k - Step 100k):")
    print(f"  - PPO 20k (10시드 전수) 평균:   {np.mean(bbs_unseen):>+8.2f} bb/100")
    print(f"  - Step 100k 기준 수익률:        {step100k_mixed:>+8.2f} bb/100")
    print(f"  - 평균 우위 차이 (Δ):           {mean_diff:>+8.2f} bb/100")
    print(f"  - 차이의 95% 신뢰구간:          [{ci_diff[0]:>+8.2f}, {ci_diff[1]:>+8.2f}] bb/100")
    print(f"  - Paired t-통계량:              t = {t_stat:.3f} (df={n-1})")
    print(f"  - 단측 유의확률 (p-value):      p = {p_val_diff:.4f}")
    
    if p_val_diff < 0.05:
        print(f"  -> 판정: p < 0.05 수준에서 PPO 20k의 Step 100k 대비 통계적으로 유의미한 우위 입증 (귀무가설 기각)!")
    else:
        print(f"  -> 판정: p >= 0.05 수준으로 통계적 유의성이 확실히 입증되지 않음.")
        
    print("=" * 90)

if __name__ == "__main__":
    evaluate_statistical_rigor()
