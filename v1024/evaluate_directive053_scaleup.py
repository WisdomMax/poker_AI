#!/usr/bin/env python3
"""
v1024/evaluate_directive053_scaleup.py
DIRECTIVE 053: EXTENDED 200-SEED INDEPENDENT HOLDOUT BENCHMARK (10,000-HAND SCALE-UP VERIFICATION)

Evaluates:
1. Baseline 7000h
2. Directive052 PFSP 3000h (3k hands scale baseline)
3. Directive053 Uniform GAE 5000h (Scaled 5k hands)
4. Directive053 PFSP GAE 5000h (Scaled 5k hands)

Design:
- 200 Independent Holdout Seeds (555000 ~ 555199, completely disjoint from all training seeds)
- 5 Heroes (Agent A ~ E) x 5 Seats (0 ~ 4) = 5,000 hands per arm (Total 20,000 hands)
- Seed-level paired difference statistics and 95% Confidence Intervals
- Strategic action distribution: Fold, Call, Half-Pot, Pot, All-in frequencies
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

def evaluate_scaleup_holdout():
    print("==================================================================")
    print("STARTING DIRECTIVE 053 EXTENDED 200-SEED INDEPENDENT HOLDOUT")
    print("200 Seeds x 5 Heroes x 5 Seats = 5,000 Hands per Arm (Total 20,000 Hands)")
    print("==================================================================")
    
    game = get_5p_fullgame()
    frozen_pool = [f"{a}_2000h" for a in AGENT_NAMES]
    frozen_models = {c: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{c}.pt")) for c in frozen_pool}
    
    # Target Evaluation Arm Models
    model_groups = {
        "Baseline_7000h": {
            a: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES
        },
        "Directive052_PFSP_3000h": {
            a: load_strict_model(f"v1024/exp_directive052/arm_pfsp_seed_401000/ckpt_hand_3000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive053_Uniform_GAE_5000h": {
            a: load_strict_model(f"v1024/exp_directive053/arm_uniform_gae_5000h/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive053_PFSP_GAE_5000h": {
            a: load_strict_model(f"v1024/exp_directive053/arm_pfsp_gae_5000h/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES
        }
    }
    
    NUM_SEEDS = 200
    SEED_BASE = 555000
    
    results = {}
    raw_logs = []
    seed_grouped_pnl = {grp: defaultdict(list) for grp in model_groups}
    
    t0 = time.time()
    for grp_name, hero_models in model_groups.items():
        print(f"  Evaluating {grp_name} (5,000 hands)...")
        hands_pnl = []
        action_counts = defaultdict(int)
        seat_pnl = defaultdict(list)
        agent_pnl = defaultdict(list)
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
                            action_counts[act] += 1
                            
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
        
        # 200 Seed-level mean aggregation
        seed_means = np.array([np.mean(seed_grouped_pnl[grp_name][s]) * 100.0 for s in range(SEED_BASE, SEED_BASE + NUM_SEEDS)])
        seed_sem = float(seed_means.std() / np.sqrt(NUM_SEEDS))
        seed_ci95 = [float(seed_means.mean() - 1.96 * seed_sem), float(seed_means.mean() + 1.96 * seed_sem)]
        
        action_dist = {f"action_{act}": float((action_counts[act] / max(1, total_decisions)) * 100.0) for act in range(6)}
        
        results[grp_name] = {
            "total_hands": len(pnl_arr),
            "mean_bb100": mean_bb100,
            "std_bb100": std_bb100,
            "seed_mean_bb100": float(seed_means.mean()),
            "seed_ci95": seed_ci95,
            "action_distribution_pct": action_dist,
            "seat_mean_bb100": {f"seat_{s}": round(float(np.mean(v) * 100.0), 2) for s, v in seat_pnl.items()},
            "agent_mean_bb100": {a: round(float(np.mean(v) * 100.0), 2) for a, v in agent_pnl.items()},
            "pnl_array": pnl_arr,
            "seed_means": seed_means
        }
        print(f"    -> {grp_name}: {mean_bb100:+.2f} BB/100 (Seed-level 95% CI: [{seed_ci95[0]:.2f}, {seed_ci95[1]:.2f}]), All-in: {action_dist.get('action_5', 0.0):.2f}%")

    # Save 20,000 raw hand logs
    raw_path = "v1024/data/directive053_raw_scaleup_holdout.jsonl"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved 20,000 raw hand logs to {raw_path}")

    # Paired Seed-Level Differences vs Baseline
    base_seed_means = results["Baseline_7000h"]["seed_means"]
    base_pnl_arr = results["Baseline_7000h"]["pnl_array"]
    paired_diffs = {}
    
    for target in ["Directive052_PFSP_3000h", "Directive053_Uniform_GAE_5000h", "Directive053_PFSP_GAE_5000h"]:
        diff_seeds = results[target]["seed_means"] - base_seed_means
        diff_mean = float(diff_seeds.mean())
        diff_sem = float(diff_seeds.std() / np.sqrt(NUM_SEEDS))
        diff_ci95 = [float(diff_mean - 1.96 * diff_sem), float(diff_mean + 1.96 * diff_sem)]
        
        hand_diffs = (results[target]["pnl_array"] - base_pnl_arr) * 100.0
        identical_hands = int((hand_diffs == 0).sum())
        
        paired_diffs[f"{target}_vs_Baseline"] = {
            "mean_diff_bb100": diff_mean,
            "seed_ci95": diff_ci95,
            "identical_hands": identical_hands,
            "identical_ratio": float(identical_hands / len(hand_diffs))
        }
        print(f"  Paired {target} vs Baseline: {diff_mean:+.2f} BB/100 (Seed 95% CI: [{diff_ci95[0]:.2f}, {diff_ci95[1]:.2f}]), Identical: {identical_hands}/5000")

    # Head-to-Head 1: PFSP GAE 5000h vs Uniform GAE 5000h
    pfsp_vs_uni = results["Directive053_PFSP_GAE_5000h"]["seed_means"] - results["Directive053_Uniform_GAE_5000h"]["seed_means"]
    p_vs_u_mean = float(pfsp_vs_uni.mean())
    p_vs_u_sem = float(pfsp_vs_uni.std() / np.sqrt(NUM_SEEDS))
    p_vs_u_ci95 = [float(p_vs_u_mean - 1.96 * p_vs_u_sem), float(p_vs_u_mean + 1.96 * p_vs_u_sem)]
    
    paired_diffs["PFSP_5000h_vs_Uniform_5000h"] = {
        "mean_diff_bb100": p_vs_u_mean,
        "seed_ci95": p_vs_u_ci95
    }
    print(f"\n>>> [DIRECTIVE 053 CORE VERDICT 1] PFSP GAE 5000h vs Uniform GAE 5000h: {p_vs_u_mean:+.2f} BB/100 (95% CI: [{p_vs_u_ci95[0]:.2f}, {p_vs_u_ci95[1]:.2f}])")

    # Head-to-Head 2: PFSP GAE 5000h vs PFSP 3000h (Scale-Up Gain)
    scale_gain = results["Directive053_PFSP_GAE_5000h"]["seed_means"] - results["Directive052_PFSP_3000h"]["seed_means"]
    s_gain_mean = float(scale_gain.mean())
    s_gain_sem = float(scale_gain.std() / np.sqrt(NUM_SEEDS))
    s_gain_ci95 = [float(s_gain_mean - 1.96 * s_gain_sem), float(s_gain_mean + 1.96 * s_gain_sem)]
    
    paired_diffs["PFSP_5000h_vs_PFSP_3000h"] = {
        "mean_diff_bb100": s_gain_mean,
        "seed_ci95": s_gain_ci95
    }
    print(f">>> [DIRECTIVE 053 CORE VERDICT 2] PFSP Scale-up Gain (5000h vs 3000h): {s_gain_mean:+.2f} BB/100 (95% CI: [{s_gain_ci95[0]:.2f}, {s_gain_ci95[1]:.2f}])")

    for grp in results:
        del results[grp]["pnl_array"]
        del results[grp]["seed_means"]
        
    final_output = {
        "directive": "DIRECTIVE_053",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_eval_time_seconds": round(time.time() - t0, 2),
        "results": results,
        "paired_diffs": paired_diffs
    }
    
    out_json = "v1024/directive053_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[DONE] Saved Directive 053 Benchmark to {out_json}")
    return final_output

if __name__ == "__main__":
    evaluate_scaleup_holdout()
