"""
POKER AI V4.9.8 - PHASE 4: Frozen Evaluation Across All 20 Checkpoints (4 Conditions x 5 Seeds)
- 동일한 고정 1,000판 스케줄(Frozen Benchmark)에서 20개 전 모델 평가
- 기록 항목:
  * 전체 bb/100
  * Premium / Marginal / Trash EV (bb/hand)
  * AA EV, 72o EV, 3-Bet Defense EV
  * 포지션별 bb/100 (BTN, CO, UTG, SB, BB)
  * 폴드율, 올인율, 하프팟율
- Student's t-분포 (df=4, t_crit=2.7764) 기반 95% CI 산출
- CondD vs CondA 대응표본 t-검정
- 학습 시드 간 변동성(Seed Variance) 정량화
"""

import os
import sys
import json
import csv
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v498.frozen_evaluator import evaluate_model_frozen

def run_phase4_all_seeds():
    print("=" * 80)
    print("PHASE 4: 4대 조건 x 5개 독립 시드 (총 20개 모델) 고정 전수 평가")
    print("=" * 80)

    schedule_path = "v498/data/frozen_schedule_1000.json"
    with open(schedule_path, 'r', encoding='utf-8') as f:
        schedule = json.load(f)

    device = torch.device("cpu")
    SEEDS = [42, 101, 202, 303, 404]
    COND_SPECS = [
        ('CondA_Raw_EntOFF', False),
        ('CondB_Norm_EntOFF', True),
        ('CondC_Raw_EntON', False),
        ('CondD_Norm_EntON', True),
    ]

    all_results = []

    for cprefix, norm in COND_SPECS:
        print(f"\n>>> [평가 진행] {cprefix} (5개 시드)")
        for s in SEEDS:
            mpath = f"v496/models/{cprefix}_seed_{s}.pt"
            if not os.path.exists(mpath):
                print(f"  [경고] 모델 파일 없음: {mpath}")
                continue

            summary, _ = evaluate_model_frozen(
                model_path=mpath,
                schedule=schedule,
                normalize_input=norm,
                device=device
            )

            res = {
                'condition': cprefix,
                'seed': s,
                'model_sha256': summary['model_sha256'][:16],
                'bb100': summary['bb100'],
                'prem_ev_bb': summary['premium_ev_bb'],
                'marg_ev_bb': summary['marginal_ev_bb'],
                'trash_ev_bb': summary['trash_ev_bb'],
                'aa_ev_bb': summary['aa_ev_bb'],
                'seven_two_ev_bb': summary['seven_two_ev_bb'],
                'three_bet_ev_bb': summary['three_bet_defense_ev_bb'],
                'btn_bb100': summary['btn_bb100'],
                'co_bb100': summary['co_bb100'],
                'utg_bb100': summary['utg_bb100'],
                'sb_bb100': summary['sb_bb100'],
                'bb_pos_bb100': summary['bb_pos_bb100'],
                'fold_rate_pct': summary['fold_rate_pct'],
                'allin_rate_pct': summary['allin_rate_pct'],
                'half_rate_pct': summary['half_rate_pct']
            }
            all_results.append(res)
            print(f"  Seed {s:3d} | bb/100: {res['bb100']:+7.2f} | Prem: {res['prem_ev_bb']:+5.2f} BB | Marg: {res['marg_ev_bb']:+5.2f} BB | Trash: {res['trash_ev_bb']:+5.2f} BB | 3-Bet EV: {res['three_bet_ev_bb']:+5.2f} BB")

    # CSV 저장
    csv_raw = "v498/data/phase4_frozen_raw_results.csv"
    with open(csv_raw, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\n[저장 완료] 20개 모델 원시 평가 결과: {csv_raw}")

    # 조건별 통계 요약 (t-분포 95% CI)
    t_crit = stats.t.ppf(0.975, df=4) # 2.7764451
    summary_by_cond = []

    print("\n" + "=" * 80)
    print("PHASE 4 통계 요약 (Student's t-분포 df=4, t_crit=2.7764 적용)")
    print("=" * 80)

    cond_data = defaultdict(list)
    for r in all_results:
        cond_data[r['condition']].append(r)

    for cprefix, _ in COND_SPECS:
        rows = cond_data[cprefix]
        bb100_vals = [r['bb100'] for r in rows]
        prem_vals = [r['prem_ev_bb'] for r in rows]
        marg_vals = [r['marg_ev_bb'] for r in rows]
        trash_vals = [r['trash_ev_bb'] for r in rows]
        three_bet_vals = [r['three_bet_ev_bb'] for r in rows]
        seven_two_vals = [r['seven_two_ev_bb'] for r in rows]

        n = len(bb100_vals)
        mean_bb = np.mean(bb100_vals)
        std_bb = np.std(bb100_vals, ddof=1)
        sem_bb = std_bb / np.sqrt(n)
        ci_half = t_crit * sem_bb

        pos_rate = sum(x > 0 for x in bb100_vals) / n * 100.0

        summary_by_cond.append({
            'condition': cprefix,
            'mean_bb100': mean_bb,
            'std_bb100': std_bb,
            'sem_bb100': sem_bb,
            't_ci95_half': ci_half,
            'ci95_lower': mean_bb - ci_half,
            'ci95_upper': mean_bb + ci_half,
            'positive_rate_pct': pos_rate,
            'mean_prem_ev_bb': np.mean(prem_vals),
            'mean_marg_ev_bb': np.mean(marg_vals),
            'mean_trash_ev_bb': np.mean(trash_vals),
            'mean_three_bet_ev_bb': np.mean(three_bet_vals),
            'mean_seven_two_ev_bb': np.mean(seven_two_vals)
        })

        print(f"[{cprefix}]")
        print(f"  평균 bb/100: {mean_bb:+7.2f} ± {ci_half:.2f} bb/100 (95% CI: [{mean_bb - ci_half:+7.2f}, {mean_bb + ci_half:+7.2f}])")
        print(f"  표준편차 (시드간 변동성): {std_bb:.2f} | 양수 시드 비율: {pos_rate:.1f}% ({sum(x > 0 for x in bb100_vals)}/5)")
        print(f"  핸드별 EV: Premium {np.mean(prem_vals):+6.2f} BB | Marginal {np.mean(marg_vals):+6.2f} BB | Trash {np.mean(trash_vals):+6.2f} BB")
        print(f"  특수 상황: 3-Bet EV {np.mean(three_bet_vals):+6.2f} BB | 72o EV {np.mean(seven_two_vals):+6.2f} BB\n")

    # CondD vs CondA 대응표본 t-검정
    condA_vals = [r['bb100'] for r in cond_data['CondA_Raw_EntOFF']]
    condD_vals = [r['bb100'] for r in cond_data['CondD_Norm_EntON']]
    diffs = np.array(condD_vals) - np.array(condA_vals)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    sem_diff = std_diff / np.sqrt(len(diffs))
    t_stat, p_val = stats.ttest_rel(condD_vals, condA_vals)
    ci_diff_half = t_crit * sem_diff

    print("=" * 80)
    print("CondD (Norm_EntON) vs CondA (Raw_EntOFF) 대응표본 검정 결과:")
    print("=" * 80)
    print(f"  평균 차이 (Delta): {mean_diff:+7.2f} ± {ci_diff_half:.2f} bb/100")
    print(f"  95% 신뢰구간:     [{mean_diff - ci_diff_half:+7.2f}, {mean_diff + ci_diff_half:+7.2f}]")
    print(f"  t-statistic:      {t_stat:.4f}")
    print(f"  p-value:          {p_val:.4f}")
    if p_val < 0.05:
        print("  -> 판정: CondD가 CondA보다 통계적으로 유의미하게 우수함 (p < 0.05).")
    else:
        print("  -> 판정: 차이의 95% 신뢰구간에 0이 포함되므로, 대조군 대비 우위는 보수적 원칙에 따라 '통계적 유의성 미달(미입증)'로 유지.")

    # CSV 저장
    csv_summary = "v498/data/phase4_frozen_condition_summary.csv"
    with open(csv_summary, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_by_cond[0].keys()))
        w.writeheader()
        w.writerows(summary_by_cond)
    print(f"\n[저장 완료] 조건별 통계 요약 CSV: {csv_summary}")

if __name__ == "__main__":
    run_phase4_all_seeds()
