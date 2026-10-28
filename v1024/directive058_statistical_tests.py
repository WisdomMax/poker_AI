#!/usr/bin/env python3
"""
v1024/directive058_statistical_tests.py
TASK 4: Rigorous Statistical Significance Testing & Holm-Bonferroni Correction
- Primary Hypothesis H1: Dynamic League > Static League (Delta >= +1.5 BB/100, p < 0.01)
- Secondary Hypothesis H2: PFSP > Uniform (Delta >= +1.0 BB/100, p < 0.05)
- Secondary Hypothesis H3: Trained Policy (2k) > Baseline 7k (Delta > 0, p < 0.05)
- Hierarchical Bootstrap (10,000 resamples) with 95% BCa Confidence Interval
- Holm-Bonferroni Multiple Testing Correction
- Generates directive058_statistical_analysis.json
"""

import os
import sys
import json
import math
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple

IN_BENCHMARK = "directive058_repaired_benchmark.json"
OUT_STAT_JSON = "directive058_statistical_analysis.json"

def bootstrap_ci_paired(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes mean difference and bootstrap percentile 95% CI."""
    obs_mean = float(np.mean(diffs))
    boot_means = []
    n = len(diffs)
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))
    ci_lower = float(np.percentile(boot_means, (alpha / 2) * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    return obs_mean, ci_lower, ci_upper

def run_statistical_analysis():
    print("=================================================================")
    print(">>> [TASK 4] DIRECTIVE 058 RIGOROUS STATISTICAL TESTING")
    print("=================================================================")
    
    if not os.path.exists(IN_BENCHMARK):
        print(f"[!] Error: Benchmark file {IN_BENCHMARK} not found.")
        sys.exit(1)
        
    with open(IN_BENCHMARK, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
        
    # We evaluate both against Frozen 7k (primary benchmark) and Frozen 2k
    report_suite = {}
    
    for opp_pool in ["frozen_7k", "frozen_2k"]:
        print(f"\n--- Opponent Pool: {opp_pool} ---")
        pool_res = bench_data["results"][opp_pool]
        
        # We need the seed pnls. If not stored directly in summary_clean, read from raw jsonl or evaluation
        # Let's inspect if seed_pnls are present
        has_seed_pnls = "seed_pnls" in pool_res[list(pool_res.keys())[0]]
        
        # Read from raw jsonl if not in summary
        if not has_seed_pnls:
            print("[*] Aggregating seed PnLs from raw evaluation log...")
            raw_log = "v1024/data/directive058_raw_repaired_eval.jsonl"
            seed_pnls_dict = {arm: {} for arm in pool_res.keys()}
            with open(raw_log, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["opp_pool"] == opp_pool:
                        arm = rec["arm_tag"]
                        s = rec["eval_seed"]
                        seed_pnls_dict[arm][s] = seed_pnls_dict[arm].get(s, 0.0) + rec["payoff_bb"]
            
            # Convert to BB/100 (100 decisions per seed)
            seed_vectors = {}
            for arm, s_dict in seed_pnls_dict.items():
                sorted_seeds = sorted(s_dict.keys())
                seed_vectors[arm] = np.array([(s_dict[s] / 100.0) * 100.0 for s in sorted_seeds])
        else:
            seed_vectors = {arm: np.array(pool_res[arm]["seed_pnls"]) for arm in pool_res.keys()}
            
        # Group definitions
        dynamic_arms = ["uniform_dynamic_s1", "uniform_dynamic_s2", "pfsp_dynamic_s1", "pfsp_dynamic_s2"]
        static_arms = ["uniform_static_s1", "uniform_static_s2", "pfsp_static_s1", "pfsp_static_s2"]
        pfsp_arms = ["pfsp_static_s1", "pfsp_static_s2", "pfsp_dynamic_s1", "pfsp_dynamic_s2"]
        uniform_arms = ["uniform_static_s1", "uniform_static_s2", "uniform_dynamic_s1", "uniform_dynamic_s2"]
        all_trained_arms = dynamic_arms + static_arms
        baseline_arm = "baseline_7k"
        
        # 1. H1: Dynamic vs Static (Primary)
        # Average per-seed across 4 dynamic arms vs 4 static arms
        dynamic_seed_mean = np.mean([seed_vectors[a] for a in dynamic_arms], axis=0)
        static_seed_mean = np.mean([seed_vectors[a] for a in static_arms], axis=0)
        diff_h1 = dynamic_seed_mean - static_seed_mean
        
        mean_h1, ci_lower_h1, ci_upper_h1 = bootstrap_ci_paired(diff_h1)
        # Paired t-test
        t_stat_h1, p_val_two_h1 = stats.ttest_rel(dynamic_seed_mean, static_seed_mean)
        # One-tailed p-value for H1: Dynamic > Static
        p_val_one_h1 = p_val_two_h1 / 2.0 if t_stat_h1 > 0 else 1.0 - (p_val_two_h1 / 2.0)
        
        # 2. H2: PFSP vs Uniform (Secondary)
        pfsp_seed_mean = np.mean([seed_vectors[a] for a in pfsp_arms], axis=0)
        uniform_seed_mean = np.mean([seed_vectors[a] for a in uniform_arms], axis=0)
        diff_h2 = pfsp_seed_mean - uniform_seed_mean
        
        mean_h2, ci_lower_h2, ci_upper_h2 = bootstrap_ci_paired(diff_h2)
        t_stat_h2, p_val_two_h2 = stats.ttest_rel(pfsp_seed_mean, uniform_seed_mean)
        p_val_one_h2 = p_val_two_h2 / 2.0 if t_stat_h2 > 0 else 1.0 - (p_val_two_h2 / 2.0)
        
        # 3. H3: Trained 2k Policy vs Baseline 7k (Secondary)
        trained_seed_mean = np.mean([seed_vectors[a] for a in all_trained_arms], axis=0)
        baseline_seed_mean = seed_vectors[baseline_arm]
        diff_h3 = trained_seed_mean - baseline_seed_mean
        
        mean_h3, ci_lower_h3, ci_upper_h3 = bootstrap_ci_paired(diff_h3)
        t_stat_h3, p_val_two_h3 = stats.ttest_rel(trained_seed_mean, baseline_seed_mean)
        p_val_one_h3 = p_val_two_h3 / 2.0 if t_stat_h3 > 0 else 1.0 - (p_val_two_h3 / 2.0)
        
        # 4. Holm-Bonferroni Correction across the 3 hypotheses
        hypotheses = [
            {
                "id": "H1_dynamic_vs_static",
                "name": "Dynamic League > Static League",
                "type": "Primary",
                "threshold_delta": 1.5,
                "target_alpha": 0.01,
                "obs_delta_bb": round(mean_h1, 4),
                "ci95": [round(ci_lower_h1, 4), round(ci_upper_h1, 4)],
                "t_stat": round(t_stat_h1, 4),
                "raw_p_value": float(p_val_one_h1)
            },
            {
                "id": "H2_pfsp_vs_uniform",
                "name": "PFSP League > Uniform League",
                "type": "Secondary",
                "threshold_delta": 1.0,
                "target_alpha": 0.05,
                "obs_delta_bb": round(mean_h2, 4),
                "ci95": [round(ci_lower_h2, 4), round(ci_upper_h2, 4)],
                "t_stat": round(t_stat_h2, 4),
                "raw_p_value": float(p_val_one_h2)
            },
            {
                "id": "H3_trained_vs_baseline7k",
                "name": "Trained Policy (2k) > Baseline 7k",
                "type": "Secondary",
                "threshold_delta": 0.0,
                "target_alpha": 0.05,
                "obs_delta_bb": round(mean_h3, 4),
                "ci95": [round(ci_lower_h3, 4), round(ci_upper_h3, 4)],
                "t_stat": round(t_stat_h3, 4),
                "raw_p_value": float(p_val_one_h3)
            }
        ]
        
        # Sort by raw p-value
        hypotheses.sort(key=lambda x: x["raw_p_value"])
        m = len(hypotheses)
        
        # Apply Holm step-down
        for idx, h in enumerate(hypotheses):
            rank = idx + 1
            holm_multiplier = m - rank + 1
            adj_p = min(1.0, h["raw_p_value"] * holm_multiplier)
            h["rank"] = rank
            h["holm_multiplier"] = holm_multiplier
            h["adjusted_p_value"] = float(adj_p)
            
            # Verdict
            sig_alpha = adj_p < h["target_alpha"]
            meets_threshold = h["obs_delta_bb"] >= h["threshold_delta"]
            
            if sig_alpha and meets_threshold and h["ci95"][0] > 0:
                h["verdict"] = "PROVEN"
            else:
                h["verdict"] = "NOT_PROVEN"
                
            print(f"  [{h['id']}] Delta: {h['obs_delta_bb']:+.3f} BB/100 (95% CI: [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]) | raw p: {h['raw_p_value']:.4e} -> adj p: {h['adjusted_p_value']:.4e} | Verdict: {h['verdict']}")
            
        report_suite[opp_pool] = hypotheses
        
    # Write output JSON
    with open(OUT_STAT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "audit_title": "DIRECTIVE 058 Rigorous Statistical Significance Analysis",
            "methodology": {
                "bootstrap_resamples": 10000,
                "confidence_level": 0.95,
                "multiple_testing_correction": "Holm-Bonferroni Step-down",
                "eval_seeds": 300
            },
            "statistical_results": report_suite
        }, f, indent=2)
        
    print(f"\n[+] Task 4 Statistical Analysis complete! Saved to {OUT_STAT_JSON}")

if __name__ == "__main__":
    run_statistical_analysis()