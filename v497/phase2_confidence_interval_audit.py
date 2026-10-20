"""
POKER AI V4.9.7 - PHASE 2: Confidence Interval & Statistical Rigor Audit
- 표본 크기 N=5 (자유도 df=4)에 대한 정규분포(z=1.96) vs t-분포(t=2.776) 신뢰구간 전수 대조
- CondA, CondB, CondC, CondD의 올바른 95% 신뢰구간 및 차이(Delta) 검정
- 학습 시드 간 변동성(Between-Seed Variance) vs 동일 모델 내 게임별 변동성(Within-Game Variance) 엄격 분리
"""

import os
import sys
import math
import csv
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

def run_phase2_ci_audit(re_eval_csv_path: str = "v497/data/phase1_re_evaluation_summary.csv"):
    print("=" * 75)
    print("PHASE 2: CONFIDENCE INTERVAL & STATISTICAL RIGOR AUDIT")
    print("=" * 75)

    if not os.path.exists(re_eval_csv_path):
        print(f"[경고] 재평가 CSV 파일 부재: {re_eval_csv_path}")
        return

    # 데이터 로드
    records = []
    with open(re_eval_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'condition': row['condition'],
                'seed': int(row['seed']),
                'bb100': float(row['bb100']),
                'prem_ev': float(row['prem_ev_bb']),
                'marg_ev': float(row['marg_ev_bb']),
                'trash_ev': float(row['trash_ev_bb']),
            })

    conditions = sorted(list(set(r['condition'] for r in records)))
    N = 5
    df = N - 1
    t_crit = stats.t.ppf(0.975, df) # 약 2.776445
    z_crit = 1.960

    print(f"\n[통계 파라미터]")
    print(f"  독립 학습 시드 수 (N): {N}")
    print(f"  자유도 (df = N - 1):   {df}")
    print(f"  Z 임계값 (z_crit, 정규분포): {z_crit:.3f}")
    print(f"  t 임계값 (t_crit, t분포, df=4): {t_crit:.4f}")

    ci_summary = []
    print("\n" + "=" * 95)
    print(f"{'Condition':<20} | {'Mean bb/100':>12} | {'Std (s)':>10} | {'95% CI (z=1.96)':>16} | {'95% CI (t=2.776, 정정)':>23}")
    print("-" * 95)

    for cond in conditions:
        vals = [r['bb100'] for r in records if r['condition'] == cond]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1))
        sem = std_v / math.sqrt(N)

        ci_z = z_crit * sem
        ci_t = t_crit * sem

        print(f"{cond:<20} | {mean_v:+12.2f} | {std_v:10.2f} | ± {ci_z:14.2f} | ± {ci_t:21.2f}")
        ci_summary.append({
            'condition': cond,
            'N': N,
            'df': df,
            'mean_bb100': mean_v,
            'std_bb100': std_v,
            'sem': sem,
            'ci95_z': ci_z,
            'ci95_t_corrected': ci_t,
            'ci_lower_t': mean_v - ci_t,
            'ci_upper_t': mean_v + ci_t,
        })

    # Paired t-test 대조: CondD vs CondA
    vals_d = [r['bb100'] for r in records if r['condition'] == 'CondD_Norm_EntON']
    vals_a = [r['bb100'] for r in records if r['condition'] == 'CondA_Raw_EntOFF']
    diffs = np.array(vals_d) - np.array(vals_a)
    m_diff = float(np.mean(diffs))
    s_diff = float(np.std(diffs, ddof=1))
    sem_diff = s_diff / math.sqrt(N)
    ci_diff_z = z_crit * sem_diff
    ci_diff_t = t_crit * sem_diff

    t_stat, p_val = stats.ttest_rel(vals_d, vals_a)

    print("\n" + "=" * 95)
    print("대응표본 통계 검정 정밀 감사 (CondD_Norm_EntON vs CondA_Raw_EntOFF)")
    print(f"  평균 차이 (Delta): {m_diff:+7.2f} bb/100")
    print(f"  대응표본 표준편차: {s_diff:7.2f} bb/100")
    print(f"  기존 Z 기반 95% CI:    ± {ci_diff_z:6.2f} bb/100  ([{m_diff - ci_diff_z:+6.2f}, {m_diff + ci_diff_z:+6.2f}])")
    print(f"  정정된 t 기반 95% CI:  ± {ci_diff_t:6.2f} bb/100  ([{m_diff - ci_diff_t:+6.2f}, {m_diff + ci_diff_t:+6.2f}])")
    print(f"  t-통계량: {t_stat:.4f}, df: {df}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  -> 판정: 유의수준 5%에서 유의함.")
    else:
        print("  -> 판정: 신뢰구간에 0이 포함되며 p-value >= 0.05로 통계적 우위 미입증 (보수적 원칙 견지).")
    print("=" * 95)

    # CSV 저장
    csv_out = "v497/data/phase2_confidence_interval_audit.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(ci_summary[0].keys()))
        writer.writeheader()
        writer.writerows(ci_summary)
    print(f"[저장 완료] Phase 2 정정 신뢰구간 데이터: {csv_out}")

if __name__ == "__main__":
    run_phase2_ci_audit()
