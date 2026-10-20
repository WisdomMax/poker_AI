import os
import sys
import json
import time
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v1013.run_v1013_replay_and_controller_validation import (
    get_pool_standard, get_pool_aggressive, get_pool_dynamic,
    DeterministicTrueNFSP_Agent, ModelA,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

def evaluate_all_models(models_dict, eval_seeds, hands_per_seed=1000):
    print(f"\n=======================================================")
    print(f"[*] Starting Systematic Replication Evaluation")
    print(f"[*] {len(models_dict)} Models x {len(eval_seeds)} Eval Seeds x {hands_per_seed} Hands")
    print(f"=======================================================")
    
    game = get_5p_fullgame()
    eval_pools = [
        ("Standard", get_pool_standard),
        ("Aggressive", get_pool_aggressive),
        ("Dynamic", get_pool_dynamic)
    ]
    
    model_returns = {m_name: {s: [] for s in eval_seeds} for m_name in models_dict}
    
    t0 = time.time()
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_seed_start = time.time()
        print(f"  -> Evaluating Seed {s_idx}/{len(eval_seeds)}: Seed Base {s_base}...", flush=True)
        for h in range(hands_per_seed):
            seed_h = s_base + h
            seat = h % 5
            p_name, p_gen = eval_pools[h % 3]
            
            # Common deal & opponents for identical cards across all evaluated models
            opps = p_gen(seed_h)
            
            for m_name, model in models_dict.items():
                st = game.new_initial_state()
                advance_chance_nodes(st, np.random.RandomState(seed_h))
                f_set = set()
                dec = 0
                
                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, np.random.RandomState(seed_h))
                        continue
                    cp = st.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                    
                    if cp == seat:
                        inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                        if isinstance(model, ModelA):
                            act = model.act(st, seat, l6, f_set, seed_h, dec)
                        elif isinstance(model, PPONeuralPlayer):
                            act = model.act_eval(inf, l6, rng=np.random.RandomState(seed_h * 100 + seat))
                        else:
                            act = model.select_action(st, cp, l6)
                        dec += 1
                    else:
                        opp_idx = cp if cp < seat else cp - 1
                        opp_bot = opps[opp_idx]
                        if isinstance(opp_bot, DeterministicTrueNFSP_Agent):
                            inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            act = opp_bot.select_action_deterministic(inf_opp, l6, np.random.RandomState(seed_h*101 + cp*17))
                        else:
                            act = opp_bot.select_action(st, cp, l6)
                            
                    if act == 0:
                        f_set.add(cp)
                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                    
                ret_bb = st.returns()[seat] / 100.0
                model_returns[m_name][s_base].append(ret_bb)
        print(f"     [Seed {s_base} completed in {time.time()-t_seed_start:.1f}s]", flush=True)

    summary = {}
    for m_name in models_dict:
        seed_means = [np.mean(model_returns[m_name][s]) * 100.0 for s in eval_seeds]
        all_rets = [r for s in eval_seeds for r in model_returns[m_name][s]]
        
        overall_mean = float(np.mean(seed_means))
        seed_se = float(np.std(seed_means, ddof=1) / np.sqrt(len(eval_seeds)))
        t_crit = stats.t.ppf(0.975, df=len(eval_seeds)-1)
        ci_95 = [float(overall_mean - t_crit * seed_se), float(overall_mean + t_crit * seed_se)]
        
        summary[m_name] = {
            "overall_mean": overall_mean,
            "seed_se": seed_se,
            "ci_95": ci_95,
            "seed_means": seed_means,
            "total_hands": len(all_rets)
        }
        print(f"  * {m_name:12s}: Mean = {overall_mean:+7.2f} bb/100 | SE = {seed_se:5.2f} | 95% CI = [{ci_95[0]:+7.2f}, {ci_95[1]:+7.2f}] | Seeds = {[round(x,1) for x in seed_means]}", flush=True)
        
    return summary, model_returns

def main():
    train_seeds = [101, 202, 303]
    eval_seeds = [90001, 90002, 90003, 90004, 90005]
    
    bp = DeterministicTrueNFSP_Agent()
    eval_models = {
        "Model_A": ModelA(bp, sims=300)
    }
    
    for r_idx in range(1, 4):
        # 50K
        m_50k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_50k.load_state_dict(torch.load(f"v1017/models/model_e_r{r_idx}_50k.pt"))
        m_50k.eval()
        eval_models[f"E_R{r_idx}_50K"] = m_50k
        
        # 100K
        m_100k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_100k.load_state_dict(torch.load(f"v1017/models/model_e_r{r_idx}_100k.pt"))
        m_100k.eval()
        eval_models[f"E_R{r_idx}_100K"] = m_100k
        
    if os.path.exists("v1016/models/model_e_100k.pt"):
        m_v1016 = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m_v1016.load_state_dict(torch.load("v1016/models/model_e_100k.pt"))
        m_v1016.eval()
        eval_models["E_V1016_100K"] = m_v1016
        
    eval_summary, raw_returns = evaluate_all_models(eval_models, eval_seeds, hands_per_seed=1000)
    
    paired_tests = {}
    for r_idx in range(1, 4):
        m50 = f"E_R{r_idx}_50K"
        m100 = f"E_R{r_idx}_100K"
        s50 = eval_summary[m50]["seed_means"]
        s100 = eval_summary[m100]["seed_means"]
        diffs = np.array(s100) - np.array(s50)
        mean_diff = float(np.mean(diffs))
        se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        t_stat, p_val = stats.ttest_rel(s100, s50)
        t_crit = stats.t.ppf(0.975, df=len(diffs)-1)
        ci_diff = [float(mean_diff - t_crit * se_diff), float(mean_diff + t_crit * se_diff)]
        
        paired_tests[f"Run_{r_idx}"] = {
            "mean_50k": eval_summary[m50]["overall_mean"],
            "mean_100k": eval_summary[m100]["overall_mean"],
            "mean_diff": mean_diff,
            "se_diff": se_diff,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "ci_95": ci_diff,
            "seed_diffs": [float(x) for x in diffs]
        }
        print(f"\n[+] Run {r_idx} Scaling Test (E_100K - E_50K):")
        print(f"    - Difference: {mean_diff:+7.2f} ± {se_diff:5.2f} bb/100 (95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}])")
        print(f"    - t-statistic = {t_stat:.3f}, p-value = {p_val:.5f} (df=4)")
        
    pooled_50k = [eval_summary[f"E_R{r}_50K"]["overall_mean"] for r in range(1, 4)]
    pooled_100k = [eval_summary[f"E_R{r}_100K"]["overall_mean"] for r in range(1, 4)]
    meta_diff = np.mean(pooled_100k) - np.mean(pooled_50k)
    print(f"\n[+] Meta-Analysis Across 3 Runs:")
    print(f"    - Mean 50K = {np.mean(pooled_50k):+7.2f} bb/100 (Runs: {[round(x,1) for x in pooled_50k]})")
    print(f"    - Mean 100K = {np.mean(pooled_100k):+7.2f} bb/100 (Runs: {[round(x,1) for x in pooled_100k]})")
    print(f"    - Meta Improvement = {meta_diff:+7.2f} bb/100")
    
    data_matrix = np.array([eval_summary[f"E_R{r}_100K"]["seed_means"] for r in range(1, 4)])
    grand_mean = np.mean(data_matrix)
    ss_total = np.sum((data_matrix - grand_mean)**2)
    ss_run = 5 * np.sum((np.mean(data_matrix, axis=1) - grand_mean)**2)
    ss_seed = 3 * np.sum((np.mean(data_matrix, axis=0) - grand_mean)**2)
    ss_error = ss_total - ss_run - ss_seed
    
    variance_decomp = {
        "ss_total": float(ss_total),
        "ss_run": float(ss_run),
        "ss_seed": float(ss_seed),
        "ss_error": float(ss_error),
        "run_variance_pct": float(ss_run / ss_total * 100.0) if ss_total > 0 else 0.0,
        "seed_variance_pct": float(ss_seed / ss_total * 100.0) if ss_total > 0 else 0.0,
        "residual_pct": float(ss_error / ss_total * 100.0) if ss_total > 0 else 0.0
    }
    print(f"\n[+] Variance Decomposition (100K Models):")
    print(f"    - Training Run Variance (Run-to-Run): {variance_decomp['run_variance_pct']:.2f}%")
    print(f"    - Eval Seed Variance (Card Distribution): {variance_decomp['seed_variance_pct']:.2f}%")
    print(f"    - Residual / Interaction: {variance_decomp['residual_pct']:.2f}%")
    
    output_data = {
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "eval_summary": eval_summary,
        "paired_tests": paired_tests,
        "meta_analysis": {
            "pooled_50k_mean": float(np.mean(pooled_50k)),
            "pooled_100k_mean": float(np.mean(pooled_100k)),
            "meta_diff": float(meta_diff),
            "runs_50k": [float(x) for x in pooled_50k],
            "runs_100k": [float(x) for x in pooled_100k]
        },
        "variance_decomposition": variance_decomp
    }
    
    with open("v1017/data/v1017_replication_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Saved full results to v1017/data/v1017_replication_results.json")

if __name__ == "__main__":
    main()
