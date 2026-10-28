#!/usr/bin/env python3
"""
v1024/evaluate_directive056_final_test.py
DIRECTIVE 056 TASK 5: Final Rigorous Test Set Benchmark on Strictly Separated Seeds (1110000 ~ 1110099).

Evaluates the 2,000-hand scale-up progression:
1. Baseline 7000h (Starting point)
2. PFSP Extended Seed 901000 progression: 500h, 1000h, 1500h, 2000h
3. Uniform Extended Seed 901000: 2000h
4. PFSP Extended Seed 902000: 500h, 2000h
5. Uniform Extended Seed 902000: 2000h

Metrics:
- Seed-level 95% Confidence Intervals
- Direct Paired Differences from seed-level deltas
- Individual Seat 0..4 BB/100 (Verification of zero position bias)
- Individual Agent A..E BB/100
- Action Distributions (Fold, Call, Raise, AllIn)
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
EXP056_DIR = "v1024/exp_directive056"

NUM_SEEDS = 100
START_SEED = 1110000
SEEDS = [START_SEED + i for i in range(NUM_SEEDS)]

def load_eval_model(path: str) -> PPONeuralPlayer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path missing: {path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    m.eval()
    return m

def evaluate_final_test_set():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 056 TASK 5] Final Test Set Evaluation Benchmark")
    print(f"    Strictly Separated Test Seeds: {NUM_SEEDS} ({START_SEED} ~ {START_SEED + NUM_SEEDS - 1})")
    print("========================================================================")
    
    game = get_5p_fullgame()
    opp_pool = {f"{a}_2000h": load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_2000h.pt")) for a in AGENT_NAMES}
    opp_list = list(opp_pool.keys())
    
    test_arms = {
        "Baseline_7000h": {
            "name": "Baseline 7000h",
            "models": {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
        },
        "PFSP_Ext_s901_500h": {
            "name": "PFSP Ext s901 (500h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_pfsp_extended_seed_901000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s901_1000h": {
            "name": "PFSP Ext s901 (1000h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_pfsp_extended_seed_901000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s901_1500h": {
            "name": "PFSP Ext s901 (1500h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_pfsp_extended_seed_901000/ckpt_hand_1500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s901_2000h": {
            "name": "PFSP Ext s901 (2000h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_pfsp_extended_seed_901000/ckpt_hand_2000/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_Ext_s901_2000h": {
            "name": "Uniform Ext s901 (2000h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_uniform_extended_seed_901000/ckpt_hand_2000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s902_500h": {
            "name": "PFSP Ext s902 (500h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_pfsp_extended_seed_902000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s902_2000h": {
            "name": "PFSP Ext s902 (2000h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_pfsp_extended_seed_902000/ckpt_hand_2000/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_Ext_s902_2000h": {
            "name": "Uniform Ext s902 (2000h)",
            "models": {a: load_eval_model(f"{EXP056_DIR}/arm_uniform_extended_seed_902000/ckpt_hand_2000/{a}.pt") for a in AGENT_NAMES}
        }
    }
    
    out_jsonl = "v1024/data/directive056_raw_final_test_holdout.jsonl"
    os.makedirs("v1024/data", exist_ok=True)
    f_raw = open(out_jsonl, "w", encoding="utf-8")
    
    results = {}
    seed_arm_pnls = defaultdict(dict)
    
    for arm_id, arm_info in test_arms.items():
        print(f"\n--- Testing [{arm_info['name']}] ---")
        t0 = time.time()
        
        models = arm_info["models"]
        hand_pnls = []
        seed_pnls = defaultdict(list)
        seat_pnls = defaultdict(list)
        agent_pnls = defaultdict(list)
        action_counts = Counter()
        
        for s_idx, seed in enumerate(SEEDS):
            deck_rng = np.random.RandomState(seed * 23 + 61)
            action_rng = np.random.RandomState(seed * 41 + 83)
            
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
                        
                    ret_bb = (st.returns()[hero_seat] / BIG_BLIND) * 100.0
                    hand_pnls.append(ret_bb)
                    seed_pnls[seed].append(ret_bb)
                    seat_pnls[hero_seat].append(ret_bb)
                    agent_pnls[hero_name].append(ret_bb)
                    
                    raw_rec = {
                        "arm_id": arm_id,
                        "seed": seed,
                        "hero_agent": hero_name,
                        "hero_seat": hero_seat,
                        "ret_bb100": round(ret_bb, 2),
                        "had_allin": had_allin
                    }
                    f_raw.write(json.dumps(raw_rec) + "\n")
                    
        mean_bb = float(np.mean(hand_pnls))
        std_bb = float(np.std(hand_pnls))
        seed_means = [np.mean(seed_pnls[s]) for s in SEEDS]
        ci_half = 1.96 * float(np.std(seed_means) / np.sqrt(NUM_SEEDS))
        
        for s in SEEDS:
            seed_arm_pnls[s][arm_id] = float(np.mean(seed_pnls[s]))
            
        tot_acts = sum(action_counts.values())
        act_dist = {
            "Fold": round(action_counts[0] / max(1, tot_acts) * 100, 2),
            "Call": round((action_counts[1] + action_counts[2]) / max(1, tot_acts) * 100, 2),
            "Raise": round((action_counts[3] + action_counts[4]) / max(1, tot_acts) * 100, 2),
            "AllIn": round(action_counts[5] / max(1, tot_acts) * 100, 2)
        }
        
        results[arm_id] = {
            "name": arm_info["name"],
            "hands": len(hand_pnls),
            "mean_bb100": round(mean_bb, 2),
            "std_bb100": round(std_bb, 2),
            "ci_95": [round(mean_bb - ci_half, 2), round(mean_bb + ci_half, 2)],
            "act_dist": act_dist,
            "seat_stats": {pos: round(float(np.mean(seat_pnls[pos])), 2) for pos in range(5)},
            "agent_stats": {a: round(float(np.mean(agent_pnls[a])), 2) for a in AGENT_NAMES},
            "elapsed": round(time.time() - t0, 1)
        }
        print(f"  Result: {mean_bb:+.2f} BB/100 (95% CI: [{mean_bb - ci_half:+.2f}, {mean_bb + ci_half:+.2f}])")
        print(f"  Seat Stats: {results[arm_id]['seat_stats']}")
        print(f"  Agent Stats: {results[arm_id]['agent_stats']}")
        
    f_raw.close()
    
    # Exact Paired Differences directly from seed-level deltas
    def compute_exact_paired(arm_a: str, arm_b: str) -> Dict[str, Any]:
        deltas = [seed_arm_pnls[s][arm_a] - seed_arm_pnls[s][arm_b] for s in SEEDS]
        mean_delta = float(np.mean(deltas))
        se_delta = float(np.std(deltas) / np.sqrt(NUM_SEEDS))
        ci = [round(mean_delta - 1.96 * se_delta, 2), round(mean_delta + 1.96 * se_delta, 2)]
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "mean_diff": round(mean_delta, 2),
            "ci_95": ci,
            "includes_zero": bool(ci[0] <= 0 <= ci[1])
        }
        
    paired = {
        # Progression: 2000h vs 500h (Seed 901000)
        "PFSP_s901_2000h_vs_500h": compute_exact_paired("PFSP_Ext_s901_2000h", "PFSP_Ext_s901_500h"),
        # Progression: 2000h vs 1000h (Seed 901000)
        "PFSP_s901_2000h_vs_1000h": compute_exact_paired("PFSP_Ext_s901_2000h", "PFSP_Ext_s901_1000h"),
        # Head to Head at 2000h (Seed 901000)
        "PFSP_vs_Uniform_s901_2000h": compute_exact_paired("PFSP_Ext_s901_2000h", "Uniform_Ext_s901_2000h"),
        # Head to Head at 2000h (Seed 902000)
        "PFSP_vs_Uniform_s902_2000h": compute_exact_paired("PFSP_Ext_s902_2000h", "Uniform_Ext_s902_2000h"),
        # Final Scale-up vs Baseline (Seed 901000)
        "PFSP_s901_2000h_vs_Baseline": compute_exact_paired("PFSP_Ext_s901_2000h", "Baseline_7000h"),
        # Final Scale-up vs Baseline (Seed 902000)
        "PFSP_s902_2000h_vs_Baseline": compute_exact_paired("PFSP_Ext_s902_2000h", "Baseline_7000h")
    }
    
    final_output = {
        "benchmark_results": results,
        "exact_paired_comparisons": paired
    }
    
    with open("v1024/directive056_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print("\nSaved Benchmark results to v1024/directive056_benchmark_results.json")
    return final_output

if __name__ == "__main__":
    evaluate_final_test_set()
