"""
POKER AI V5.1.1 - PHASE 1: STATISTICAL INDEPENDENCE & HYPOTHESIS TESTING AUDIT
- V5.1의 의사복제(Pseudoreplication: N=3000 게임 판 수 풀링) 오류 전수 조사
- 독립 학습 단위(시드 N=3)에 기반한 정확한 통계량 재계산:
  * 표본평균 (Sample Mean)
  * 표본표준편차 (Sample Std, ddof=1)
  * 표준오차 (Standard Error: s / sqrt(3))
  * Student-t 95% 신뢰구간 (df = 2, t_crit = 4.3027)
  * 대응표본 t-검정 (Paired t-test, 동일 시드 42, 101, 202)
  * 독립표본 Welch t-검정 (Welch's t-test with Satterthwaite df)
- 원시 데이터를 v511/data/phase1_statistical_integrity_retest.csv에 저장
"""

import os
import sys
import csv
import numpy as np
import scipy.stats as stats

def run_phase1_audit():
    print("=" * 85)
    print("PHASE 1: 통계적 독립성 및 가설 검정 엄밀 재검증 (독립 학습 시드 N=3 기준)")
    print("=" * 85)

    # V5.1 시드별 실제 평가 성적 (bb/100)
    data = {
        "Concat": np.array([36.37, -1414.02, -30.00], dtype=np.float64),
        "Two-Tower": np.array([8.03, 4.48, -30.00], dtype=np.float64),
        "FiLM": np.array([435.65, 66.89, 31.14], dtype=np.float64)
    }

    seeds = [42, 101, 202]
    n_seeds = 3
    df = n_seeds - 1 # df = 2
    t_crit_95 = stats.t.ppf(0.975, df=df) # 4.30265

    summaries = {}
    csv_rows = []

    print(f"\n1. 아키텍처별 엄밀 기술통계량 (단위: bb/100, N=3, Student-t df=2, t_crit={t_crit_95:.4f}):")
    print(f"{'아키텍처':<12} | {'시드별 데이터 (42, 101, 202)':<32} | {'표본평균':<10} | {'표본표준편차':<12} | {'표준오차':<10} | {'정확한 95% CI'}")
    print("-" * 105)

    for name, arr in data.items():
        mean = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        se = float(s / np.sqrt(n_seeds))
        ci_half = float(t_crit_95 * se)
        ci_low = mean - ci_half
        ci_high = mean + ci_half

        summaries[name] = {
            'mean': mean, 's': s, 'se': se, 'ci_half': ci_half, 'ci_low': ci_low, 'ci_high': ci_high
        }

        seed_str = f"[{arr[0]:+7.2f}, {arr[1]:+8.2f}, {arr[2]:+6.2f}]"
        print(f"{name:<12} | {seed_str:<32} | {mean:+9.2f}  | {s:10.2f}   | {se:8.2f}   | [{ci_low:+9.2f}, {ci_high:+9.2f}] (±{ci_half:.2f})")

        csv_rows.append({
            'architecture': name,
            'seed_42': arr[0],
            'seed_101': arr[1],
            'seed_202': arr[2],
            'sample_mean': mean,
            'sample_std': s,
            'standard_error': se,
            't_critical_95': t_crit_95,
            'ci_half_95': ci_half,
            'ci_low_95': ci_low,
            'ci_high_95': ci_high
        })

    # 2. V5.1 기존 보고 오류 vs 정확한 값 비교
    print("\n" + "=" * 85)
    print("2. V5.1 기존 보고 오류 감사 및 정정:")
    print("=" * 85)
    print("  [기존 보고]: FiLM +177.89 ± 223.94 bb/100")
    print(f"  [오류 규명]: 223.94는 95% 신뢰구간이 아니라 '표본표준편차(s = {summaries['FiLM']['s']:.2f})'였음!")
    print(f"  [정정 공식]: N=3에서 SE = s / sqrt(3) = {summaries['FiLM']['se']:.2f} bb/100")
    print(f"               Student-t(df=2) t_crit = {t_crit_95:.4f}")
    print(f"               정확한 95% CI = +177.89 ± {summaries['FiLM']['ci_half']:.2f} bb/100 -> [{summaries['FiLM']['ci_low']:+.2f}, {summaries['FiLM']['ci_high']:+.2f}]")
    print("  -> 따라서 신뢰구간의 하한이 음수(-378.40)를 포함하므로, 3개 시드만으로는 0을 초과하는 유의미한 양의 수익성을 확정할 수 없음!")

    # 3. 가설 검정 재계산 (FiLM vs Concat, Two-Tower vs Concat, FiLM vs Two-Tower)
    print("\n" + "=" * 85)
    print("3. 가설 검정 엄밀 재계산 (N=3 학습 시드 기준):")
    print("=" * 85)

    pairs = [
        ("FiLM vs Concat", data["FiLM"], data["Concat"]),
        ("Two-Tower vs Concat", data["Two-Tower"], data["Concat"]),
        ("FiLM vs Two-Tower", data["FiLM"], data["Two-Tower"])
    ]

    for label, x1, x2 in pairs:
        diff = x1 - x2
        mean_diff = float(np.mean(diff))

        # 1) Paired t-test (동일 시드 42, 101, 202 대응)
        t_paired, p_paired = stats.ttest_rel(x1, x2)

        # 2) Welch's t-test (독립표본, 등분산 가정 안함)
        t_welch, p_welch = stats.ttest_ind(x1, x2, equal_var=False)

        print(f"\n  [{label}] (평균 차이: {mean_diff:+.2f} bb/100)")
        print(f"    * 대응표본 (Paired t-test, df=2) : t = {t_paired:.4f}, p = {p_paired:.4f} -> {'유의미 (p<0.05)' if p_paired < 0.05 else '유의미하지 않음 (p>=0.05)'}")
        print(f"    * 독립표본 (Welch t-test)         : t = {t_welch:.4f}, p = {p_welch:.4f} -> {'유의미 (p<0.05)' if p_welch < 0.05 else '유의미하지 않음 (p>=0.05)'}")

    # 4. V5.1의 의사복제(Pseudoreplication) 원인 규명
    print("\n" + "=" * 85)
    print("4. V5.1 보고서 수치 (t=1.594, p=0.1109) 생성 원인 규명:")
    print("=" * 85)
    print("  - V5.1 코드: ret_a = np.concatenate([r['raw_returns'] for r in m_runs]) (3,000개 핸드 리턴 풀링)")
    print("  - 오류 본질: 독립된 실험 단위인 '학습 시드(N=3)' 대신 게임 진행 핸드(N=3,000)를 독립 표본으로 오인함.")
    print("  - 올바른 결론: 시드 단위 검정 시 Welch p = 0.303, Paired p = 0.270으로, 통계적 유의성은 전혀 존재하지 않음!")

    # CSV 저장
    csv_path = "v511/data/phase1_statistical_integrity_retest.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[저장 완료] Phase 1 통계 재검증 CSV: {csv_path}\n")

if __name__ == '__main__':
    run_phase1_audit()
