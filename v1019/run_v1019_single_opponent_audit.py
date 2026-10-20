import os
import sys
import json
import time
import random
from collections import defaultdict, deque
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
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v1013.run_v1013_replay_and_controller_validation import (
    DeterministicTrueNFSP_Agent, ModelA,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1018.run_v1018_unseen_generalization import (
    AdaptiveHistoryExploiter, DynamicMultiSizingBot, PolarizedBluffValueBot, CheckRaiseTrapBot
)

# =====================================================================
# POOL FACTORIES: Baseline vs Single Opponent Replacement
# =====================================================================
def get_base_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        DeterministicTrueNFSP_Agent()
    ]

def get_adaptive_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        AdaptiveHistoryExploiter(window_size=30)
    ]

def get_multisizing_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        DynamicMultiSizingBot(seed=seed_h * 23)
    ]

def get_polarized_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        PolarizedBluffValueBot(seed=seed_h * 29)
    ]

def get_trap_opponents(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 13),
        DeterministicTightPassiveBot(seed=seed_h * 17),
        DeterministicUltraAggressiveBot(seed=seed_h * 19),
        CheckRaiseTrapBot(seed=seed_h * 31)
    ]

# =====================================================================
# EXPERIMENTAL EVALUATION HARNESS
# =====================================================================
def run_single_opponent_audit(models_dict, eval_seeds, hands_per_cond=600):
    game = get_5p_fullgame()
    conditions = [
        ("Base_Seen", get_base_opponents, "None"),
        ("Single_Adaptive", get_adaptive_opponents, "AdaptiveHistoryExploiter"),
        ("Single_MultiSizing", get_multisizing_opponents, "DynamicMultiSizingBot"),
        ("Single_Polarized", get_polarized_opponents, "PolarizedBluffValueBot"),
        ("Single_Trap", get_trap_opponents, "CheckRaiseTrapBot"),
    ]
    
    # Store returns: cond -> model -> seed -> list of returns
    results = {
        c_name: {m_name: {s: [] for s in eval_seeds} for m_name in models_dict}
        for c_name, _, _ in conditions
    }
    
    # Forensic tracking: big pots, target opponent showdowns
    forensics = {
        c_name: {
            m_name: {
                "all_pots": [],
                "target_opp_showdowns": [], # returns when target opp was involved
                "cr_pots": []               # returns when check-raise occurred
            }
            for m_name in models_dict
        }
        for c_name, _, _ in conditions
    }
    
    print("\n=======================================================")
    print("[*] V10.19 ONE-OPPONENT-AT-A-TIME GENERALIZATION AUDIT")
    print(f"[*] 4 Models x 5 Conditions x {len(eval_seeds)} Seeds x {hands_per_cond} Hands")
    print(f"[*] Total Evaluated: {4 * 5 * len(eval_seeds) * hands_per_cond:,} Model-Hands")
    print("=======================================================")
    
    t0 = time.time()
    
    for s_idx, s_base in enumerate(eval_seeds, 1):
        t_seed = time.time()
        print(f"  -> Progress: Eval Seed {s_idx}/{len(eval_seeds)} (Base {s_base})...", flush=True)
        
        for c_name, p_gen, target_name in conditions:
            for h in range(hands_per_cond):
                seed_h = s_base + h
                seat = h % 5
                opps = p_gen(seed_h)
                
                # Check target opponent seat
                target_seat = (seat + 4) % 5 # Target bot is 4th opponent (index 3 in opps)
                
                for m_name, model in models_dict.items():
                    st = game.new_initial_state()
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                    f_set = set()
                    dec = 0
                    cr_occurred = False
                    
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
                        # Check-raise detection: if action is raise (>1) after a check in same round
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = st.returns()[seat] / 100.0
                    results[c_name][m_name][s_base].append(ret_bb)
                    
                    # Track pot size
                    tot_pot = sum(extract_player_spent(st)) / 100.0
                    forensics[c_name][m_name]["all_pots"].append((tot_pot, ret_bb))
                    
                    # Target opponent showdown: neither hero nor target folded
                    if seat not in f_set and target_seat not in f_set:
                        forensics[c_name][m_name]["target_opp_showdowns"].append(ret_bb)
                        
        print(f"     [Seed {s_base} completed in {time.time()-t_seed:.1f}s]", flush=True)

    # Statistical Compilation
    summary = {}
    base_means = {}
    
    print("\n=======================================================")
    print("[*] SUMMARY BY CONDITION (bb/100 across 5 Eval Seeds)")
    print("=======================================================")
    for c_name, _, target_name in conditions:
        summary[c_name] = {}
        print(f"\n--- [{c_name} (Target: {target_name})] ---")
        for m_name in models_dict:
            seed_means = [np.mean(results[c_name][m_name][s]) * 100.0 for s in eval_seeds]
            mean_val = float(np.mean(seed_means))
            se_val = float(np.std(seed_means, ddof=1) / np.sqrt(len(eval_seeds)))
            t_crit = stats.t.ppf(0.975, df=len(eval_seeds)-1)
            ci = [float(mean_val - t_crit * se_val), float(mean_val + t_crit * se_val)]
            
            summary[c_name][m_name] = {
                "mean": mean_val,
                "se": se_val,
                "ci_95": ci,
                "seed_means": seed_means
            }
            if c_name == "Base_Seen":
                base_means[m_name] = seed_means
            print(f"  * {m_name:12s}: {mean_val:+7.2f} ± {se_val:5.2f} bb/100 | CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}] | Seeds: {[round(x,1) for x in seed_means]}")

    # Causal Impact vs Baseline (Delta EV caused by single replacement)
    print("\n=======================================================")
    print("[*] CAUSAL IMPACT ANALYSIS: Delta EV vs Base_Seen (bb/100)")
    print("=======================================================")
    causal_impacts = {}
    for c_name, _, target_name in conditions[1:]: # Skip Base
        causal_impacts[c_name] = {}
        print(f"\n--- [Effect of Replacing 1 Bot with {target_name}] ---")
        for m_name in models_dict:
            c_seeds = summary[c_name][m_name]["seed_means"]
            b_seeds = base_means[m_name]
            diffs = np.array(c_seeds) - np.array(b_seeds)
            d_mean = float(np.mean(diffs))
            d_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            t_stat, p_val = stats.ttest_rel(c_seeds, b_seeds)
            t_crit = stats.t.ppf(0.975, df=len(diffs)-1)
            ci_d = [float(d_mean - t_crit * d_se), float(d_mean + t_crit * d_se)]
            
            causal_impacts[c_name][m_name] = {
                "delta_mean": d_mean,
                "delta_se": d_se,
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "ci_95": ci_d
            }
            print(f"  * {m_name:12s}: Delta = {d_mean:+7.2f} ± {d_se:5.2f} bb/100 (CI: [{ci_d[0]:+7.2f}, {ci_d[1]:+7.2f}], t={t_stat:.3f}, p={p_val:.4f})")

    # Big Pot Analysis (Top 1% pots)
    big_pot_summary = {}
    for c_name, _, _ in conditions:
        big_pot_summary[c_name] = {}
        for m_name in models_dict:
            pots_rets = forensics[c_name][m_name]["all_pots"]
            if pots_rets:
                pots_rets.sort(key=lambda x: x[0], reverse=True)
                top_1pct_n = max(1, int(len(pots_rets) * 0.01))
                top_1pct = pots_rets[:top_1pct_n]
                top_1pct_profit = sum(x[1] for x in top_1pct) * 100.0 / len(pots_rets)
                big_pot_summary[c_name][m_name] = {
                    "top_1pct_n": top_1pct_n,
                    "top_1pct_ev_bb100": float(top_1pct_profit),
                    "target_showdowns_n": len(forensics[c_name][m_name]["target_opp_showdowns"]),
                    "target_showdown_mean_bb": float(np.mean(forensics[c_name][m_name]["target_opp_showdowns"])) if forensics[c_name][m_name]["target_opp_showdowns"] else 0.0
                }

    return summary, causal_impacts, big_pot_summary

def main():
    eval_seeds = [92001, 92002, 92003, 92004, 92005]
    
    # Load 4 Frozen Models
    bp = DeterministicTrueNFSP_Agent()
    models_dict = {
        "Model_A": ModelA(bp, sims=300)
    }
    
    for r_idx in range(1, 4):
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        path = f"v1017/models/model_e_r{r_idx}_100k.pt"
        m.load_state_dict(torch.load(path))
        m.eval()
        models_dict[f"E_R{r_idx}_100K"] = m
        print(f"[+] Loaded Model E_R{r_idx}_100K from {path}")
        
    summary, causal_impacts, big_pot_summary = run_single_opponent_audit(
        models_dict, eval_seeds, hands_per_cond=600
    )
    
    # Save output data
    output_data = {
        "eval_seeds": eval_seeds,
        "summary": summary,
        "causal_impacts": causal_impacts,
        "big_pot_summary": big_pot_summary
    }
    
    with open("v1019/data/v1019_single_audit_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Results saved to v1019/data/v1019_single_audit_results.json")

if __name__ == "__main__":
    main()
