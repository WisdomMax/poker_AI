#!/usr/bin/env python3
"""
v1024/evaluate_directive056_task2_catchup.py
DIRECTIVE 056 TASK 2: Comprehensive Catch-up Evaluation of 055 Checkpoints on Unseen Seeds (999000 ~ 999099).

Evaluates previously unbenchmarked 055 checkpoints:
1. Uniform Extended Seed 802000 (500h & 1000h)
2. PFSP Extended Seed 802000 (500h & 1000h)
3. Uniform 2k Seed 801000 (500h) & Seed 802000 (500h)
4. PFSP 2k Seed 802000 (500h)
5. Baseline 7000h & PFSP Extended Seed 801000 (500h & 1000h) for direct comparison

Computes true paired difference confidence intervals directly from seed-level deltas.
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
EXP055_DIR = "v1024/exp_directive055"

NUM_SEEDS = 100
START_SEED = 999000
SEEDS = [START_SEED + i for i in range(NUM_SEEDS)]

def load_eval_model(path: str) -> PPONeuralPlayer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path missing: {path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    m.eval()
    return m

def evaluate_task2_catchup():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 056 TASK 2] 055 Checkpoint Catch-up Evaluation Benchmark")
    print(f"    Holdout Seeds: {NUM_SEEDS} ({START_SEED} ~ {START_SEED + NUM_SEEDS - 1})")
    print("========================================================================")
    
    game = get_5p_fullgame()
    opp_pool = {f"{a}_2000h": load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_2000h.pt")) for a in AGENT_NAMES}
    opp_list = list(opp_pool.keys())
    
    eval_arms = {
        "Baseline_7000h": {
            "name": "Baseline 7000h",
            "models": {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
        },
        "Uniform_Ext_s802_500h": {
            "name": "Uniform Extended s802 (500h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_uniform_extended_seed_802000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_Ext_s802_1k": {
            "name": "Uniform Extended s802 (1000h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_uniform_extended_seed_802000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s802_500h": {
            "name": "PFSP Extended s802 (500h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_pfsp_extended_seed_802000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s802_1k": {
            "name": "PFSP Extended s802 (1000h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_pfsp_extended_seed_802000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s801_500h": {
            "name": "PFSP Extended s801 (500h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_pfsp_extended_seed_801000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_Ext_s801_1k": {
            "name": "PFSP Extended s801 (1000h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_pfsp_extended_seed_801000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_2k_s801_500h": {
            "name": "Uniform 2k s801 (500h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_uniform_frozen2k_seed_801000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_2k_s802_500h": {
            "name": "PFSP 2k s802 (500h)",
            "models": {a: load_eval_model(f"{EXP055_DIR}/arm_pfsp_frozen2k_seed_802000/ckpt_hand_500/{a}.pt") for a in AGENT_NAMES}
        }
    }
    
    out_jsonl = "v1024/data/directive056_raw_task2_catchup_holdout.jsonl"
    os.makedirs("v1024/data", exist_ok=True)
    f_raw = open(out_jsonl, "w", encoding="utf-8")
    
    results = {}
    seed_arm_pnls = defaultdict(dict)
    
    for arm_id, arm_info in eval_arms.items():
        print(f"\n--- Testing [{arm_info['name']}] ---")
        t0 = time.time()
        
        models = arm_info["models"]
        hand_pnls = []
        seed_pnls = defaultdict(list)
        seat_pnls = defaultdict(list)
        agent_pnls = defaultdict(list)
        action_counts = Counter()
        
        for s_idx, seed in enumerate(SEEDS):
            deck_rng = np.random.RandomState(seed * 19 + 53)
            action_rng = np.random.RandomState(seed * 37 + 71)
            
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
        # Check degradation in Seed 801000 (500h vs 1000h)
        "PFSP_Ext_s801_1k_vs_500h": compute_exact_paired("PFSP_Ext_s801_1k", "PFSP_Ext_s801_500h"),
        # Check degradation in Seed 802000 (500h vs 1000h)
        "PFSP_Ext_s802_1k_vs_500h": compute_exact_paired("PFSP_Ext_s802_1k", "PFSP_Ext_s802_500h"),
        "Uniform_Ext_s802_1k_vs_500h": compute_exact_paired("Uniform_Ext_s802_1k", "Uniform_Ext_s802_500h"),
        # Head to Head at 500h
        "PFSP_vs_Uniform_Ext_s802_500h": compute_exact_paired("PFSP_Ext_s802_500h", "Uniform_Ext_s802_500h"),
        # Head to Head at 1000h
        "PFSP_vs_Uniform_Ext_s802_1k": compute_exact_paired("PFSP_Ext_s802_1k", "Uniform_Ext_s802_1k")
    }
    
    final_data = {
        "benchmark_results": results,
        "exact_paired_comparisons": paired
    }
    with open("v1024/directive056_task2_completed_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)
    print("\nSaved Task 2 results to v1024/directive056_task2_completed_eval_results.json")
    return final_data

if __name__ == "__main__":
    evaluate_task2_catchup()
