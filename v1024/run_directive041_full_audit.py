#!/usr/bin/env python3
"""
DIRECTIVE 041 Full Independent Holdout Audit Runner
Evaluates:
- Baseline E2E 50k vs Pilot League E2E 70k
Across 5 Independent Opponent Classes (1,000 Hands each under identical seeds):
1. top20 (Symmetric Top 20% Heuristic)
2. postflop_board (Real 7-Card Hand Rank Board Evaluator)
3. allin_catcher (Strict Two Pair+ All-in Catcher)
4. aggressive_3bet (High Frequency 3-Bet & C-Bet)
5. unseen_bluffer (Unseen Postflop Semi-Bluffer)

Generates:
- Raw JSONL logs for every hand and model
- Comprehensive comparison JSON with Paired Difference 95% CI and p-values
"""
import os
import sys
import json
import time
import numpy as np
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v1024.audit_directive041_holdout import evaluate_model_on_holdout

MODEL_50K = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
MODEL_70K = "v1024/models_e2e/PPO_E2E_70k.pt"

OPPONENTS = [
    ("top20", "Symmetric Top 20% Range"),
    ("postflop_board", "7-Card Board Rank Evaluator"),
    ("allin_catcher", "Strict Two-Pair+ All-in Catcher"),
    ("aggressive_3bet", "Aggressive 3-Bet Pressure"),
    ("unseen_bluffer", "Unseen Adaptive Bluffer")
]

def run_paired_audit(num_hands=1000, base_seed=777000):
    print("=" * 80)
    print(f"DIRECTIVE 041: INDEPENDENT 5-WAY LEAGUE HOLDOUT AUDIT ({num_hands} hands / opponent)")
    print(f"Baseline: E2E 50k ({MODEL_50K})")
    print(f"New Pilot: E2E 70k ({MODEL_70K})")
    print("=" * 80)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_hands_per_opp": num_hands,
        "base_seed": base_seed,
        "opponents": {}
    }

    total_diff_pnls = []

    for opp_key, opp_desc in OPPONENTS:
        print(f"\n>>> Evaluating against Opponent: [{opp_key.upper()}] ({opp_desc}) ...")
        t0 = time.time()

        # 1. 50k Baseline
        jsonl_50k = f"v1024/data/directive041_raw_hands_50k_{opp_key}.jsonl"
        res_50k = evaluate_model_on_holdout(
            MODEL_50K, "E2E_50k", opp_key, num_hands=num_hands, base_seed=base_seed, jsonl_out=jsonl_50k
        )

        # 2. 70k New Pilot
        jsonl_70k = f"v1024/data/directive041_raw_hands_70k_{opp_key}.jsonl"
        res_70k = evaluate_model_on_holdout(
            MODEL_70K, "E2E_70k", opp_key, num_hands=num_hands, base_seed=base_seed, jsonl_out=jsonl_70k
        )

        # 3. Paired Difference Statistics (Same seed, hand-by-hand delta)
        pnls_50k = np.array(res_50k["hand_pnls"])
        pnls_70k = np.array(res_70k["hand_pnls"])
        deltas = pnls_70k - pnls_50k
        total_diff_pnls.extend(deltas.tolist())

        mean_delta_bb100 = float(np.mean(deltas) * 100.0)
        std_delta = float(np.std(deltas, ddof=1))
        ci95_delta = 1.96 * (std_delta / np.sqrt(num_hands)) * 100.0
        ci_lower = round(mean_delta_bb100 - ci95_delta, 2)
        ci_upper = round(mean_delta_bb100 + ci95_delta, 2)
        
        # Two-sided paired t-test
        t_stat, p_val = stats.ttest_rel(pnls_70k, pnls_50k)

        elapsed = time.time() - t0
        print(f"  --> Completed in {elapsed:.1f}s")
        print(f"  50k BB/100: {res_50k['bb_per_100']:+.2f} ± {res_50k['ci95_bb']:.2f} | Pot%: {res_50k['pot_bet_pct']}% | HalfPot%: {res_50k['half_pot_bet_pct']}%")
        print(f"  70k BB/100: {res_70k['bb_per_100']:+.2f} ± {res_70k['ci95_bb']:.2f} | Pot%: {res_70k['pot_bet_pct']}% | HalfPot%: {res_70k['half_pot_bet_pct']}%")
        print(f"  Paired Delta: {mean_delta_bb100:+.2f} BB/100 (95% CI: [{ci_lower:+.2f}, {ci_upper:+.2f}], p={p_val:.4e})")

        # Strip long array before storing to JSON
        del res_50k["hand_pnls"]
        del res_70k["hand_pnls"]

        results["opponents"][opp_key] = {
            "description": opp_desc,
            "50k": res_50k,
            "70k": res_70k,
            "paired_delta_bb100": round(mean_delta_bb100, 2),
            "paired_ci95_bb100": round(ci95_delta, 2),
            "paired_ci_range": [ci_lower, ci_upper],
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val)
        }

    # Aggregate All 5,000 Hands Paired Delta
    arr_total_deltas = np.array(total_diff_pnls)
    tot_mean_delta = float(np.mean(arr_total_deltas) * 100.0)
    tot_ci95_delta = 1.96 * (float(np.std(arr_total_deltas, ddof=1)) / np.sqrt(len(arr_total_deltas))) * 100.0
    tot_t_stat, tot_p_val = stats.ttest_1samp(arr_total_deltas, 0.0)

    results["aggregate_5000_hands"] = {
        "total_hands": len(total_diff_pnls),
        "mean_delta_bb100": round(tot_mean_delta, 2),
        "ci95_delta_bb100": round(tot_ci95_delta, 2),
        "ci_range": [round(tot_mean_delta - tot_ci95_delta, 2), round(tot_mean_delta + tot_ci95_delta, 2)],
        "p_value": float(tot_p_val)
    }

    out_path = "v1024/directive041_holdout_audit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80)
    print(f"AUDIT COMPLETE! Saved results to {out_path}")
    print(f"Overall 5,000 Hands Paired Improvement: {tot_mean_delta:+.2f} ± {tot_ci95_delta:.2f} BB/100 (p={tot_p_val:.4e})")
    print("=" * 80)

if __name__ == "__main__":
    run_paired_audit()
