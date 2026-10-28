#!/usr/bin/env python3
"""
v1024/evaluate_directive054_evolution.py
DIRECTIVE 054 TASK 3 & 5: COMPREHENSIVE LEARNING PROGRESSION & MULTI-SEED INDEPENDENT HOLDOUT BENCHMARK

Evaluates:
1. Baseline 7000h
2. Directive053 Final Models (Seed 701000 Uniform 5k, PFSP 5k)
3. Directive054 Seed 702000 Progression (1k, 3k, 5k for Uniform and PFSP)
4. Directive054 Seed 703000 Progression (1k, 3k, 5k for Uniform and PFSP)

Metrics:
- Seed-level paired differences & 95% CI
- Progression trajectory across training checkpoints (1k -> 3k -> 5k)
- Action distribution shifts (Fold, Call, Raise, All-in)
- Seat-by-seat positional attribution (Seat 0 Button vs Seats 3, 4 Blinds)
- Exact hand-level identical action vs identical payoff ratios
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

def evaluate_directive054_evolution():
    print("==================================================================")
    print("STARTING DIRECTIVE 054 COMPREHENSIVE LEARNING DYNAMICS EVALUATION")
    print("200 Holdout Seeds x 5 Heroes x 5 Seats = 5,000 Hands per Arm")
    print("==================================================================")
    
    game = get_5p_fullgame()
    frozen_pool = [f"{a}_2000h" for a in AGENT_NAMES]
    frozen_models = {c: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{c}.pt")) for c in frozen_pool}
    
    # Target Evaluation Model Groups
    model_groups = {
        "Baseline_7000h": {
            a: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES
        },
        "Directive054_Uniform_702000_1k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_uniform_seed_702000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_Uniform_702000_3k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_uniform_seed_702000/ckpt_hand_3000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_Uniform_702000_5k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_uniform_seed_702000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_PFSP_702000_1k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_pfsp_seed_702000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_PFSP_702000_3k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_pfsp_seed_702000/ckpt_hand_3000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_PFSP_702000_5k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_pfsp_seed_702000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_Uniform_703000_5k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_uniform_seed_703000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES
        },
        "Directive054_PFSP_703000_5k": {
            a: load_strict_model(f"v1024/exp_directive054/arm_pfsp_seed_703000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES
        }
    }
    
    NUM_SEEDS = 200
    SEED_BASE = 555000 # strictly disjoint from all training seeds
    
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
        print(f"    -> {grp_name}: {mean_bb100:+.2f} BB/100 (Seed 95% CI: [{seed_ci95[0]:.2f}, {seed_ci95[1]:.2f}]), All-in: {action_dist.get('action_5', 0.0):.2f}%")

    # Save raw hand logs
    raw_path = "v1024/data/directive054_raw_evolution_holdout.jsonl"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved raw hand logs to {raw_path}")

    # Comparative Paired Analysis
    base_seed_means = results["Baseline_7000h"]["seed_means"]
    base_pnl_arr = results["Baseline_7000h"]["pnl_array"]
    paired_diffs = {}
    
    for grp in model_groups:
        if grp == "Baseline_7000h": continue
        diff_seeds = results[grp]["seed_means"] - base_seed_means
        diff_mean = float(diff_seeds.mean())
        diff_sem = float(diff_seeds.std() / np.sqrt(NUM_SEEDS))
        diff_ci95 = [float(diff_mean - 1.96 * diff_sem), float(diff_mean + 1.96 * diff_sem)]
        
        hand_diffs = (results[grp]["pnl_array"] - base_pnl_arr) * 100.0
        identical_hands = int((hand_diffs == 0).sum())
        
        paired_diffs[f"{grp}_vs_Baseline"] = {
            "mean_diff_bb100": diff_mean,
            "seed_ci95": diff_ci95,
            "identical_hands": identical_hands,
            "identical_ratio": float(identical_hands / len(hand_diffs))
        }

    # Head-to-Head Head-to-Head comparisons
    # 1. Progression: PFSP 702000 (1k vs 3k vs 5k)
    p_1k_means = results["Directive054_PFSP_702000_1k"]["seed_means"]
    p_3k_means = results["Directive054_PFSP_702000_3k"]["seed_means"]
    p_5k_means = results["Directive054_PFSP_702000_5k"]["seed_means"]
    
    diff_3k_1k = p_3k_means - p_1k_means
    diff_5k_3k = p_5k_means - p_3k_means
    diff_5k_1k = p_5k_means - p_1k_means
    
    paired_diffs["PFSP_Progression_3k_vs_1k"] = {
        "mean_diff_bb100": float(diff_3k_1k.mean()),
        "seed_ci95": [float(diff_3k_1k.mean() - 1.96 * (diff_3k_1k.std()/np.sqrt(NUM_SEEDS))), float(diff_3k_1k.mean() + 1.96 * (diff_3k_1k.std()/np.sqrt(NUM_SEEDS)))]
    }
    paired_diffs["PFSP_Progression_5k_vs_3k"] = {
        "mean_diff_bb100": float(diff_5k_3k.mean()),
        "seed_ci95": [float(diff_5k_3k.mean() - 1.96 * (diff_5k_3k.std()/np.sqrt(NUM_SEEDS))), float(diff_5k_3k.mean() + 1.96 * (diff_5k_3k.std()/np.sqrt(NUM_SEEDS)))]
    }
    paired_diffs["PFSP_Progression_5k_vs_1k"] = {
        "mean_diff_bb100": float(diff_5k_1k.mean()),
        "seed_ci95": [float(diff_5k_1k.mean() - 1.96 * (diff_5k_1k.std()/np.sqrt(NUM_SEEDS))), float(diff_5k_1k.mean() + 1.96 * (diff_5k_1k.std()/np.sqrt(NUM_SEEDS)))]
    }

    # 2. PFSP vs Uniform Head-to-Head at 5k (Replicated across Seeds 702000 & 703000)
    pfsp_702_vs_uni_702 = results["Directive054_PFSP_702000_5k"]["seed_means"] - results["Directive054_Uniform_702000_5k"]["seed_means"]
    pfsp_703_vs_uni_703 = results["Directive054_PFSP_703000_5k"]["seed_means"] - results["Directive054_Uniform_703000_5k"]["seed_means"]
    
    paired_diffs["PFSP_vs_Uniform_Seed_702000_5k"] = {
        "mean_diff_bb100": float(pfsp_702_vs_uni_702.mean()),
        "seed_ci95": [float(pfsp_702_vs_uni_702.mean() - 1.96 * (pfsp_702_vs_uni_702.std()/np.sqrt(NUM_SEEDS))), float(pfsp_702_vs_uni_702.mean() + 1.96 * (pfsp_702_vs_uni_702.std()/np.sqrt(NUM_SEEDS)))]
    }
    paired_diffs["PFSP_vs_Uniform_Seed_703000_5k"] = {
        "mean_diff_bb100": float(pfsp_703_vs_uni_703.mean()),
        "seed_ci95": [float(pfsp_703_vs_uni_703.mean() - 1.96 * (pfsp_703_vs_uni_703.std()/np.sqrt(NUM_SEEDS))), float(pfsp_703_vs_uni_703.mean() + 1.96 * (pfsp_703_vs_uni_703.std()/np.sqrt(NUM_SEEDS)))]
    }

    # Combined Multi-Seed PFSP vs Uniform
    combined_pfsp_5k = (results["Directive054_PFSP_702000_5k"]["seed_means"] + results["Directive054_PFSP_703000_5k"]["seed_means"]) / 2.0
    combined_uni_5k = (results["Directive054_Uniform_702000_5k"]["seed_means"] + results["Directive054_Uniform_703000_5k"]["seed_means"]) / 2.0
    comb_diff = combined_pfsp_5k - combined_uni_5k
    paired_diffs["Combined_MultiSeed_PFSP_vs_Uniform_5k"] = {
        "mean_diff_bb100": float(comb_diff.mean()),
        "seed_ci95": [float(comb_diff.mean() - 1.96 * (comb_diff.std()/np.sqrt(NUM_SEEDS))), float(comb_diff.mean() + 1.96 * (comb_diff.std()/np.sqrt(NUM_SEEDS)))]
    }

    for grp in results:
        del results[grp]["pnl_array"]
        del results[grp]["seed_means"]
        
    final_output = {
        "directive": "DIRECTIVE_054",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_eval_time_seconds": round(time.time() - t0, 2),
        "results": results,
        "paired_diffs": paired_diffs
    }
    
    out_json = "v1024/directive054_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[DONE] Saved Directive 054 Benchmark to {out_json}")
    return final_output

if __name__ == "__main__":
    evaluate_directive054_evolution()
