#!/usr/bin/env python3
"""
v1024/evaluate_directive051_final.py
DIRECTIVE 051: INDEPENDENT CROSS-SEAT BENCHMARK EVALUATION

Compares 4 condition arms on 1,250 hands each (Total 5,000 hands):
1. Baseline 7000h (Starting policy)
2. Production Full Main (2,000 table hands)
3. Production Uniform League (3,300 table hands)
4. Production PFSP League (3,300 table hands)

Controls:
- Independent Holdout Seeds (777000 ~ 777049, completely disjoint from training seeds)
- 50 Seeds x 5 Heroes (A-E) x 5 Seats (0-4) = 1,250 hands per arm
- Fixed 2000h Frozen Opponent Pool (Seats 1-4 relative)
- Paired difference analysis & Seed-level 95% Confidence Intervals
- Strategic behavior metrics: PnL, Winrate, All-in frequency, Showdown rate
"""

import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

BIG_BLIND = 100.0
AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
CHECKPOINT_DIR = "v1024/models_alphago"

def load_strict_model(model_path: str) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[FAIL_FAST] Required model path does not exist: {model_path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        m.load_state_dict(ckpt["model_state_dict"])
    else:
        m.load_state_dict(ckpt)
    m.eval()
    return m

def evaluate_directive051():
    print("==================================================================")
    print("STARTING DIRECTIVE 051 INDEPENDENT HOLDOUT BENCHMARK")
    print("50 Seeds x 5 Heroes x 5 Seats = 1,250 Hands per Condition (Total 5,000 Hands)")
    print("==================================================================")
    
    game = get_5p_fullgame()
    frozen_pool = [f"{a}_2000h" for a in AGENT_NAMES]
    frozen_models = {c: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{c}.pt")) for c in frozen_pool}
    
    model_groups = {
        "Baseline_7000h": {
            a: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES
        },
        "Full_Main_2000h": {
            a: load_strict_model(f"v1024/exp_directive051/production/arm_a_full_main/ckpt_hand_2000/{a}.pt") for a in AGENT_NAMES
        },
        "Uniform_League_3300h": {
            a: load_strict_model(f"v1024/exp_directive051/production/arm_b_uniform/ckpt_hand_3300/{a}.pt") for a in AGENT_NAMES
        },
        "PFSP_League_3300h": {
            a: load_strict_model(f"v1024/exp_directive051/production/arm_c_pfsp/ckpt_hand_3300/{a}.pt") for a in AGENT_NAMES
        }
    }
    
    NUM_SEEDS = 50
    SEED_BASE = 777000 # Independent holdout seeds
    
    results = {}
    raw_logs = []
    seed_grouped_pnl = {grp: defaultdict(list) for grp in model_groups}
    
    t0 = time.time()
    for grp_name, hero_models in model_groups.items():
        print(f"  Evaluating {grp_name} (1,250 hands)...")
        hands_pnl = []
        seat_pnl = defaultdict(list)
        agent_pnl = defaultdict(list)
        allin_count = 0
        total_decisions = 0
        
        for s_idx in range(NUM_SEEDS):
            seed = SEED_BASE + s_idx
            
            for hero_name in AGENT_NAMES:
                hero_m = hero_models[hero_name]
                
                for seat in range(5):
                    rng = np.random.RandomState(seed)
                    st = game.new_initial_state()
                    advance_chance_nodes(st, rng)
                    
                    table_opps = [frozen_pool[(seat + i + 1) % len(frozen_pool)] for i in range(4)]
                    table = []
                    opp_iter = 0
                    for pos in range(5):
                        if pos == seat:
                            table.append(("HERO", hero_name))
                        else:
                            table.append(("OPP", table_opps[opp_iter]))
                            opp_iter += 1
                            
                    while not st.is_terminal():
                        if st.is_chance_node():
                            advance_chance_nodes(st, rng)
                            continue
                        cp = st.current_player()
                        role, ag_id = table[cp]
                        legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                        norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                        
                        m = hero_m if role == "HERO" else frozen_models[ag_id]
                        with torch.no_grad():
                            logits, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                            mask = torch.full_like(logits, -1e9)
                            for a in legal_6: mask[0, a] = 0.0
                            act = int(torch.argmax(logits + mask, dim=-1).item())
                            
                        if role == "HERO":
                            total_decisions += 1
                            if act == 5: # All-in
                                allin_count += 1
                                
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    hero_ret = st.returns()[seat] / BIG_BLIND
                    hands_pnl.append(hero_ret)
                    seat_pnl[seat].append(hero_ret)
                    agent_pnl[hero_name].append(hero_ret)
                    seed_grouped_pnl[grp_name][seed].append(hero_ret)
                    
                    raw_logs.append({
                        "group": grp_name,
                        "seed": seed,
                        "hero_agent": hero_name,
                        "hero_seat": seat,
                        "pnl_bb": float(hero_ret)
                    })
                    
        pnl_arr = np.array(hands_pnl)
        mean_bb100 = float(pnl_arr.mean() * 100.0)
        std_bb100 = float(pnl_arr.std() * 100.0)
        
        # Seed-level aggregation (50 independent observations, each averaging 25 hands)
        seed_means = np.array([np.mean(seed_grouped_pnl[grp_name][s]) * 100.0 for s in range(SEED_BASE, SEED_BASE + NUM_SEEDS)])
        seed_sem = seed_means.std() / np.sqrt(NUM_SEEDS)
        seed_ci95 = [float(seed_means.mean() - 1.96 * seed_sem), float(seed_means.mean() + 1.96 * seed_sem)]
        
        # Seat Breakdown
        seat_summary = {f"Seat_{s}": float(np.mean(seat_pnl[s]) * 100.0) for s in range(5)}
        # Agent Breakdown
        agent_summary = {ag: float(np.mean(agent_pnl[ag]) * 100.0) for ag in AGENT_NAMES}
        
        results[grp_name] = {
            "total_hands": len(pnl_arr),
            "mean_bb100": mean_bb100,
            "std_bb100": std_bb100,
            "seed_level_ci95": seed_ci95,
            "allin_rate_pct": float((allin_count / max(1, total_decisions)) * 100.0),
            "seat_breakdown": seat_summary,
            "agent_breakdown": agent_summary,
            "pnl_array": pnl_arr,
            "seed_means": seed_means
        }
        print(f"    -> {grp_name}: {mean_bb100:+.2f} BB/100 (Seed-level 95% CI: [{seed_ci95[0]:.2f}, {seed_ci95[1]:.2f}]), All-in Rate: {results[grp_name]['allin_rate_pct']:.2f}%")

    # Save raw logs
    raw_path = "v1024/data/directive051_raw_independent_holdout.jsonl"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved 5,000 raw hand logs to {raw_path}")

    # Paired Seed-Level Differences
    base_seed_means = results["Baseline_7000h"]["seed_means"]
    base_pnl_arr = results["Baseline_7000h"]["pnl_array"]
    
    paired_comparisons = {}
    for target in ["Full_Main_2000h", "Uniform_League_3300h", "PFSP_League_3300h"]:
        diff_seeds = results[target]["seed_means"] - base_seed_means
        diff_mean = float(diff_seeds.mean())
        diff_sem = float(diff_seeds.std() / np.sqrt(NUM_SEEDS))
        diff_ci95 = [float(diff_mean - 1.96 * diff_sem), float(diff_mean + 1.96 * diff_sem)]
        
        hand_diffs = (results[target]["pnl_array"] - base_pnl_arr) * 100.0
        identical_hands = int((hand_diffs == 0).sum())
        
        paired_comparisons[f"{target}_vs_Baseline"] = {
            "mean_diff_bb100": diff_mean,
            "seed_ci95": diff_ci95,
            "identical_hands": identical_hands,
            "identical_ratio": float(identical_hands / len(hand_diffs))
        }
        print(f"  Paired {target} vs Baseline: {diff_mean:+.2f} BB/100 (Seed 95% CI: [{diff_ci95[0]:.2f}, {diff_ci95[1]:.2f}]), Identical: {identical_hands}/1250")

    # Head-to-Head: PFSP League vs Uniform League
    pfsp_vs_uni = results["PFSP_League_3300h"]["seed_means"] - results["Uniform_League_3300h"]["seed_means"]
    p_vs_u_mean = float(pfsp_vs_uni.mean())
    p_vs_u_sem = float(pfsp_vs_uni.std() / np.sqrt(NUM_SEEDS))
    p_vs_u_ci95 = [float(p_vs_u_mean - 1.96 * p_vs_u_sem), float(p_vs_u_mean + 1.96 * p_vs_u_sem)]
    
    paired_comparisons["PFSP_vs_Uniform"] = {
        "mean_diff_bb100": p_vs_u_mean,
        "seed_ci95": p_vs_u_ci95
    }
    print(f"\n>>> [CORE RESEARCH QUESTION] PFSP vs Uniform Difference: {p_vs_u_mean:+.2f} BB/100 (Seed 95% CI: [{p_vs_u_ci95[0]:.2f}, {p_vs_u_ci95[1]:.2f}])")

    # Clean non-serializable numpy arrays
    for grp in results:
        del results[grp]["pnl_array"]
        del results[grp]["seed_means"]
        
    final_output = {
        "directive": "DIRECTIVE_051",
        "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_eval_time_seconds": round(time.time() - t0, 2),
        "evaluation_results": results,
        "paired_comparisons": paired_comparisons
    }
    
    out_json = "v1024/directive051_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[DONE] Saved Directive 051 Benchmark to {out_json}")
    return final_output

if __name__ == "__main__":
    evaluate_directive051()
