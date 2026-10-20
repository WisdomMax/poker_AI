"""
POKER AI V4.8.3 - PHASE 2: 학습 재현성 분석 (Reproducibility & Policy Bifurcation Analysis)
- 목적: 10개 독립 시드 모델의 최종 20k 성능 및 학습 진행 경로를 통계적으로 정밀 분석.
- 산출 지표:
  1. 20k 모델 수익률 분포 (평균, 중앙값, 표본 표준편차, 95% 신뢰구간 t-분포 및 부트스트랩)
  2. 수익 모델 비율(Win Rate) 및 손실 모델 비율
  3. 카드 인식 분화 성공률 (TV Distance >= 0.50 및 >= 0.80)
  4. 단계별(0k, 5k, 10k, 15k, 20k) 평균 및 시드별 궤적 추적
"""

import os
import csv
import numpy as np
import scipy.stats as stats

def analyze_reproducibility():
    metrics_csv = "v483/data/phase1_seed_metrics.csv"
    if not os.path.exists(metrics_csv):
        print(f"Error: {metrics_csv} not found.")
        return
        
    records = []
    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "seed": int(row["seed"]),
                "step": int(row["step"]),
                "bb100": float(row["bb100"]),
                "win_rate": float(row["win_rate"]),
                "allin_pct": float(row["allin_pct"]),
                "fold_pct": float(row["fold_pct"]),
                "tv_aa_72o": float(row["tv_aa_72o"]),
                "ent_aa": float(row["ent_aa"]),
                "ent_72o": float(row["ent_72o"]),
                "aa_allin": float(row["aa_allin"]),
                "72o_fold": float(row["72o_fold"]),
                "72o_allin": float(row["72o_allin"]),
            })
            
    steps = [0, 5000, 10000, 15000, 20000]
    seeds = sorted(list(set(r["seed"] for r in records)))
    
    print("=" * 90)
    print("📊 [V4.8.3 PHASE 2] 10개 독립 학습 시드 재현성 및 분포 정밀 분석")
    print("=" * 90)
    
    # 1. Step별 진행 추이
    print(f"\n[1] 단계별 평균 지표 추이 (10개 시드 평균):")
    print(f"{'Step':<8} | {'평균 TV Dist':<14} | {'평균 수익률 (bb/100)':<20} | {'평균 AA 올인율':<15} | {'평균 72o 폴드율':<15} | {'평균 엔트로피(AA)'}")
    print("-" * 95)
    
    for s in steps:
        step_recs = [r for r in records if r["step"] == s]
        tvs = [r["tv_aa_72o"] for r in step_recs]
        bbs = [r["bb100"] for r in step_recs]
        aa_allins = [r["aa_allin"] for r in step_recs]
        t72_folds = [r["72o_fold"] for r in step_recs]
        ent_aas = [r["ent_aa"] for r in step_recs]
        
        print(f"{s:<8d} | {np.mean(tvs):>10.4f}     | {np.mean(bbs):>+14.2f} bb/100     | {np.mean(aa_allins)*100:>11.1f}%    | {np.mean(t72_folds)*100:>11.1f}%    | {np.mean(ent_aas):>10.4f}")

    # 2. 최종 20k 모델 성능 상세
    recs_20k = [r for r in records if r["step"] == 20000]
    bbs_20k = np.array([r["bb100"] for r in recs_20k])
    tvs_20k = np.array([r["tv_aa_72o"] for r in recs_20k])
    
    n = len(bbs_20k)
    mean_bb = np.mean(bbs_20k)
    median_bb = np.median(bbs_20k)
    std_bb = np.std(bbs_20k, ddof=1)
    se_bb = std_bb / np.sqrt(n)
    
    # 95% t-신뢰구간
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_t = (mean_bb - t_crit * se_bb, mean_bb + t_crit * se_bb)
    
    # 부트스트랩 95% 신뢰구간
    rng_bs = np.random.RandomState(42)
    bs_means = [np.mean(rng_bs.choice(bbs_20k, size=n, replace=True)) for _ in range(10000)]
    ci_bs = (np.percentile(bs_means, 2.5), np.percentile(bs_means, 97.5))
    
    win_models = [r for r in recs_20k if r["bb100"] > 0]
    loss_models = [r for r in recs_20k if r["bb100"] <= 0]
    bifurcated_50 = [r for r in recs_20k if r["tv_aa_72o"] >= 0.50]
    bifurcated_80 = [r for r in recs_20k if r["tv_aa_72o"] >= 0.80]
    
    print(f"\n[2] 최종 20k 모델 10종 개별 성적:")
    print(f"{'시드':<8} | {'수익률 (bb/100)':<18} | {'TV Distance':<12} | {'AA 올인율':<11} | {'72o 폴드율':<11} | {'72o 올인율':<11} | {'판정'}")
    print("-" * 95)
    for r in recs_20k:
        status = "🟢 수익+분화" if r["bb100"] > 0 and r["tv_aa_72o"] >= 0.5 else ("🟡 수익(미분화)" if r["bb100"] > 0 else "🔴 손실")
        print(f"Seed {r['seed']:<4d} | {r['bb100']:>+12.2f} bb/100 | {r['tv_aa_72o']:>10.4f} | {r['aa_allin']*100:>8.1f}%  | {r['72o_fold']*100:>8.1f}%  | {r['72o_allin']*100:>8.1f}%  | {status}")
        
    print(f"\n[3] 20k 모델 통계 요약 (N={n}):")
    print(f"  - 평균 수익률 (Mean):        {mean_bb:>+10.2f} bb/100")
    print(f"  - 중앙값 수익률 (Median):    {median_bb:>+10.2f} bb/100")
    print(f"  - 표본 표준편차 (Std Dev):   {std_bb:>10.2f} bb/100")
    print(f"  - 95% t-신뢰구간:            [{ci_t[0]:>+8.2f}, {ci_t[1]:>+8.2f}] bb/100")
    print(f"  - 부트스트랩 95% 신뢰구간:   [{ci_bs[0]:>+8.2f}, {ci_bs[1]:>+8.2f}] bb/100")
    print(f"  - 수익 모델 비율:            {len(win_models)}/{n} ({len(win_models)/n*100:.1f}%)")
    print(f"  - 손실 모델 비율:            {len(loss_models)}/{n} ({len(loss_models)/n*100:.1f}%)")
    print(f"  - 카드 분화 성공률 (TV >= 0.50): {len(bifurcated_50)}/{n} ({len(bifurcated_50)/n*100:.1f}%)")
    print(f"  - 완전 카드 분화율 (TV >= 0.80): {len(bifurcated_80)}/{n} ({len(bifurcated_80)/n*100:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    analyze_reproducibility()
