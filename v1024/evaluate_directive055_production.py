#!/usr/bin/env python3
"""
v1024/evaluate_directive055_production.py
DIRECTIVE 055 TASK 4: Final Controlled Evaluation on Unseen Holdout Seeds (888000 ~ 888099).

Evaluates:
1. Baseline 7000h
2. 500h Canary Checkpoints (Uniform 2k, PFSP 2k, Uniform Ext, PFSP Ext)
3. 1,000h Final Checkpoints (Uniform 2k, PFSP 2k, Uniform Ext, PFSP Ext across Seeds 801000, 802000)

Metrics:
- Seed-level 95% Confidence Intervals
- Agent A..E individual BB/100 (Symmetrical verification)
- Seat 0..4 individual BB/100 (Position bias elimination verification)
- Action distributions (Fold, Call, Raise, All-in)
- Paired differences: Corrected PFSP vs Uniform, Extended Pool vs 2k Pool
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from collections import defaultdict, Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

BIG_BLIND = 100.0
AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
CHECKPOINT_DIR = "v1024/models_alphago"
EXP_DIR = "v1024/exp_directive055"

NUM_SEEDS = 100
START_SEED = 888000
SEEDS = [START_SEED + i for i in range(NUM_SEEDS)]

def load_eval_model(path: str) -> PPONeuralPlayer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path missing: {path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    m.eval()
    return m

def evaluate_directive055():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 055 TASK 4] Final Production Evaluation Benchmark")
    print(f"    Holdout Seeds: {NUM_SEEDS} ({START_SEED} ~ {START_SEED + NUM_SEEDS - 1})")
    print("========================================================================")
    
    game = get_5p_fullgame()
    opp_pool = {f"{a}_2000h": load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_2000h.pt")) for a in AGENT_NAMES}
    opp_list = list(opp_pool.keys())
    
    # Define Model Groups to Benchmark
    arm_defs = {
        "Baseline_7000h": {
            "name": "Baseline 7000h",
            "models": {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
        },
        "Uniform_Pool2k_1k_s801": {
            "name": "Uniform 2k 1000h (Seed 801000)",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_uniform_frozen2k_seed_801000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_Pool2k_1k_s802": {
            "name": "Uniform 2k 1000h (Seed 802000)",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_uniform_frozen2k_seed_802000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Pool2k_1k_s801": {
            "name": "Corrected PFSP 2k 1000h (Seed 801000)",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_pfsp_frozen2k_seed_801000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Pool2k_1k_s802": {
            "name": "Corrected PFSP 2k 1000h (Seed 802000)",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_pfsp_frozen2k_seed_802000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_ExtPool_1k_s801": {
            "name": "Uniform ExtendedPool 1000h (Seed 801000)",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_uniform_extended_seed_801000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_ExtPool_1k_s801": {
            "name": "Corrected PFSP ExtendedPool 1000h (Seed 801000)",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_pfsp_extended_seed_801000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Pool2k_500h_Canary": {
            "name": "Corrected PFSP 2k 500h Canary",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_pfsp_frozen2k_seed_801000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_ExtPool_500h_Canary": {
            "name": "Corrected PFSP ExtendedPool 500h Canary",
            "models": {a: load_eval_model(f"{EXP_DIR}/arm_pfsp_extended_seed_801000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        }
    }
    
    out_jsonl = "v1024/data/directive055_raw_evolution_holdout.jsonl"
    os.makedirs("v1024/data", exist_ok=True)
    f_raw = open(out_jsonl, "w", encoding="utf-8")
    
    benchmark_results = {}
    
    for arm_id, arm_info in arm_defs.items():
        print(f"\n--- Benchmarking [{arm_info['name']}] ---")
        t0 = time.time()
        
        models = arm_info["models"]
        hand_pnls = []
        seed_pnls = defaultdict(list)
        seat_pnls = defaultdict(list)
        agent_pnls = defaultdict(list)
        action_counts = Counter()
        allin_pnls = []
        
        for s_idx, seed in enumerate(SEEDS):
            deck_rng = np.random.RandomState(seed * 17 + 101)
            action_rng = np.random.RandomState(seed * 31 + 203)
            
            for hero_name in AGENT_NAMES:
                hero_model = models[hero_name]
                
                for hero_seat in range(5):
                    table_players = [None] * 5
                    table_players[hero_seat] = ("HERO", hero_name, hero_model)
                    
                    opp_idx = 0
                    for pos in range(5):
                        if pos != hero_seat:
                            opp_name = opp_list[opp_idx % len(opp_list)]
                            table_players[pos] = ("OPP", opp_name, opp_pool[opp_name])
                            opp_idx += 1
                            
                    st = game.new_initial_state()
                    advance_chance_nodes(st, deck_rng)
                    
                    had_allin = False
                    while not st.is_terminal():
                        if st.is_chance_node():
                            advance_chance_nodes(st, deck_rng)
                            continue
                        cp = st.current_player()
                        legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                        norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                        
                        p_type, p_id, p_obj = table_players[cp]
                        with torch.no_grad():
                            logits, _ = p_obj(torch.from_numpy(norm_inf).unsqueeze(0))
                            mask = torch.full_like(logits, -1e9)
                            for a in legal_6: mask[0, a] = 0.0
                            probs = torch.softmax(logits + mask, dim=-1).squeeze(0).numpy()
                            act = int(action_rng.choice(len(probs), p=probs))
                            
                        if p_type == "HERO":
                            action_counts[act] += 1
                            if act == 5:
                                had_allin = True
                                
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret_bb = (st.returns()[hero_seat] / BIG_BLIND) * 100.0 # BB/100
                    hand_pnls.append(ret_bb)
                    seed_pnls[seed].append(ret_bb)
                    seat_pnls[hero_seat].append(ret_bb)
                    agent_pnls[hero_name].append(ret_bb)
                    
                    if had_allin:
                        allin_pnls.append(ret_bb)
                        
                    raw_record = {
                        "arm_id": arm_id,
                        "arm_name": arm_info["name"],
                        "seed": seed,
                        "hero_agent": hero_name,
                        "hero_seat": hero_seat,
                        "ret_bb100": round(ret_bb, 2),
                        "had_allin": had_allin
                    }
                    f_raw.write(json.dumps(raw_record) + "\n")
                    
        mean_bb = float(np.mean(hand_pnls))
        std_bb = float(np.std(hand_pnls))
        seed_means = [np.mean(seed_pnls[s]) for s in SEEDS]
        ci_half = 1.96 * float(np.std(seed_means) / np.sqrt(NUM_SEEDS))
        
        tot_acts = sum(action_counts.values())
        act_dist = {
            "Fold": round(action_counts[0] / max(1, tot_acts) * 100, 2),
            "Call": round((action_counts[1] + action_counts[2]) / max(1, tot_acts) * 100, 2),
            "Raise": round((action_counts[3] + action_counts[4]) / max(1, tot_acts) * 100, 2),
            "AllIn": round(action_counts[5] / max(1, tot_acts) * 100, 2)
        }
        
        seat_stats = {pos: round(float(np.mean(seat_pnls[pos])), 2) for pos in range(5)}
        agent_stats = {a: round(float(np.mean(agent_pnls[a])), 2) for a in AGENT_NAMES}
        
        benchmark_results[arm_id] = {
            "name": arm_info["name"],
            "hands": len(hand_pnls),
            "mean_bb100": round(mean_bb, 2),
            "std_bb100": round(std_bb, 2),
            "ci_95": [round(mean_bb - ci_half, 2), round(mean_bb + ci_half, 2)],
            "act_dist": act_dist,
            "allin_rate_pct": act_dist["AllIn"],
            "allin_mean_pnl": round(float(np.mean(allin_pnls)), 2) if allin_pnls else 0.0,
            "seat_stats": seat_stats,
            "agent_stats": agent_stats,
            "elapsed": round(time.time() - t0, 1)
        }
        
        print(f"  Result: {mean_bb:+.2f} BB/100 (95% CI: [{mean_bb - ci_half:+.2f}, {mean_bb + ci_half:+.2f}])")
        print(f"  Seat Stats: {seat_stats}")
        print(f"  Agent Stats: {agent_stats}")
        print(f"  Actions: {act_dist}")
        
    f_raw.close()
    
    # Paired Head-to-Head Comparisons
    def compute_paired(arm_a: str, arm_b: str) -> Dict[str, Any]:
        p_a = benchmark_results[arm_a]["mean_bb100"]
        p_b = benchmark_results[arm_b]["mean_bb100"]
        diff = round(p_a - p_b, 2)
        # pooled SE approximation for paired CI
        se_a = (benchmark_results[arm_a]["ci_95"][1] - benchmark_results[arm_a]["ci_95"][0]) / (2 * 1.96)
        se_b = (benchmark_results[arm_b]["ci_95"][1] - benchmark_results[arm_b]["ci_95"][0]) / (2 * 1.96)
        se_diff = np.sqrt(se_a**2 + se_b**2)
        ci_diff = [round(diff - 1.96 * se_diff, 2), round(diff + 1.96 * se_diff, 2)]
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "mean_diff": diff,
            "ci_95": ci_diff,
            "includes_zero": bool(ci_diff[0] <= 0 <= ci_diff[1])
        }
        
    paired_comparisons = {
        "PFSP_vs_Uniform_Pool2k_s801": compute_paired("PFSP_Pool2k_1k_s801", "Uniform_Pool2k_1k_s801"),
        "PFSP_vs_Uniform_Pool2k_s802": compute_paired("PFSP_Pool2k_1k_s802", "Uniform_Pool2k_1k_s802"),
        "ExtPool_vs_Pool2k_PFSP_s801": compute_paired("PFSP_ExtPool_1k_s801", "PFSP_Pool2k_1k_s801"),
        "ExtPool_vs_Pool2k_Uniform_s801": compute_paired("Uniform_ExtPool_1k_s801", "Uniform_Pool2k_1k_s801"),
        "PFSP_ExtPool_1k_vs_Baseline": compute_paired("PFSP_ExtPool_1k_s801", "Baseline_7000h")
    }
    
    final_output = {
        "benchmark_results": benchmark_results,
        "paired_comparisons": paired_comparisons
    }
    
    with open("v1024/directive055_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print("\nSuccessfully saved benchmark results to v1024/directive055_benchmark_results.json")
    return final_output

if __name__ == "__main__":
    evaluate_directive055()
