#!/usr/bin/env python3
"""
v1024/evaluate_directive055_root_cause.py
DIRECTIVE 055 TASK 2: Policy Degradation Root-Cause Diagnostic Benchmark.

Evaluates existing 054 checkpoints across UNSEEN holdout seeds (777000 ~ 777099):
1. Opponent Pool A: Frozen 2k Pool (in-distribution baseline).
2. Opponent Pool B: Frozen 7k Pool (out-of-distribution generalized baseline).
3. Evaluates 9 Models:
   - Baseline 7000h
   - 054 Uniform 702000 (1k, 3k, 5k)
   - 054 PFSP 702000 (1k, 3k, 5k)
   - 054 Uniform 703000 5k & PFSP 703000 5k
4. Hypothesis Tests:
   - H1: Did fixed-seat learning (Seat 4 opponent, Seat 0..3 learners) cause position bias?
   - H2: Did PFSP sign error distort learning progression?
   - H3: Did Agent E (never trained in 054) perform differently from A..D?
   - H4: Does degradation worsen when facing Frozen 7k vs Frozen 2k (narrow pool overfitting)?
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from collections import defaultdict

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

NUM_SEEDS = 100
START_SEED = 777000
SEEDS = [START_SEED + i for i in range(NUM_SEEDS)]

def load_eval_model(path: str) -> PPONeuralPlayer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path missing: {path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    m.eval()
    return m

def run_diagnostic_evaluation() -> Dict[str, Any]:
    print("\n==================================================================")
    print(">>> [DIRECTIVE 055 TASK 2] Root-Cause Isolation Diagnostic Benchmark")
    print(f"    Seeds: {NUM_SEEDS} ({START_SEED} ~ {START_SEED + NUM_SEEDS - 1})")
    print("==================================================================")
    
    game = get_5p_fullgame()
    
    # 1. Load Frozen Opponent Pools
    opp_pool_2k = {f"{a}_2000h": load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_2000h.pt")) for a in AGENT_NAMES}
    opp_pool_7k = {f"{a}_7000h": load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
    
    # 2. Define Evaluation Arms
    arms = {
        "Baseline_7000h": {
            "type": "baseline",
            "models": {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
        },
        "Uniform_702000_1k": {
            "type": "054",
            "models": {a: load_eval_model(f"v1024/exp_directive054/arm_uniform_seed_702000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "Uniform_702000_5k": {
            "type": "054",
            "models": {a: load_eval_model(f"v1024/exp_directive054/arm_uniform_seed_702000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_702000_1k": {
            "type": "054",
            "models": {a: load_eval_model(f"v1024/exp_directive054/arm_pfsp_seed_702000/ckpt_hand_1000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_702000_5k": {
            "type": "054",
            "models": {a: load_eval_model(f"v1024/exp_directive054/arm_pfsp_seed_702000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES}
        },
        "PFSP_703000_5k": {
            "type": "054",
            "models": {a: load_eval_model(f"v1024/exp_directive054/arm_pfsp_seed_703000/ckpt_hand_5000/{a}.pt") for a in AGENT_NAMES}
        }
    }
    
    out_jsonl = "v1024/data/directive055_raw_diagnostic_holdout.jsonl"
    os.makedirs("v1024/data", exist_ok=True)
    f_raw = open(out_jsonl, "w", encoding="utf-8")
    
    results = {}
    
    # Evaluate across both Opponent Pools (2k vs 7k)
    for opp_pool_name, opp_pool in [("Frozen_2k", opp_pool_2k), ("Frozen_7k", opp_pool_7k)]:
        opp_list = list(opp_pool.keys())
        results[opp_pool_name] = {}
        
        for arm_name, arm_cfg in arms.items():
            print(f"\n--- Testing [{arm_name}] vs [{opp_pool_name}] ---")
            t0 = time.time()
            
            hero_models = arm_cfg["models"]
            hand_pnls = []
            seed_pnls = defaultdict(list)
            seat_pnls = defaultdict(list)
            agent_pnls = defaultdict(list)
            
            action_counts = Counter()
            allin_pnl = []
            
            hand_counter = 0
            # 100 seeds x 5 heroes x 5 seats = 2,500 hands per arm/pool combo
            for s_idx, seed in enumerate(SEEDS):
                deck_rng = np.random.RandomState(seed * 13 + 7)
                action_rng = np.random.RandomState(seed * 29 + 11)
                
                for h_hero, hero_name in enumerate(AGENT_NAMES):
                    hero_model = hero_models[hero_name]
                    
                    for hero_seat in range(5):
                        hand_counter += 1
                        
                        # Opponents in other seats
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
                        
                        has_allin = False
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
                                    has_allin = True
                                    
                            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                            
                        ret_bb = (st.returns()[hero_seat] / BIG_BLIND) * 100.0 # BB/100 format
                        hand_pnls.append(ret_bb)
                        seed_pnls[seed].append(ret_bb)
                        seat_pnls[hero_seat].append(ret_bb)
                        agent_pnls[hero_name].append(ret_bb)
                        
                        if has_allin:
                            allin_pnl.append(ret_bb)
                            
                        # Write raw JSONL
                        raw_record = {
                            "pool": opp_pool_name,
                            "arm": arm_name,
                            "seed": seed,
                            "hero_agent": hero_name,
                            "hero_seat": hero_seat,
                            "ret_bb100": round(ret_bb, 2),
                            "had_allin": has_allin
                        }
                        f_raw.write(json.dumps(raw_record) + "\n")
                        
            # Aggregate Statistics
            mean_bb = float(np.mean(hand_pnls))
            std_bb = float(np.std(hand_pnls))
            seed_means = [np.mean(seed_pnls[s]) for s in SEEDS]
            ci_half = 1.96 * float(np.std(seed_means) / np.sqrt(NUM_SEEDS))
            
            tot_acts = sum(action_counts.values())
            act_dist = {
                "Fold": round(action_counts[0] / max(1, tot_acts) * 100, 2),
                "Check": round(action_counts[1] / max(1, tot_acts) * 100, 2),
                "Call": round(action_counts[2] / max(1, tot_acts) * 100, 2),
                "MinRaise": round(action_counts[3] / max(1, tot_acts) * 100, 2),
                "BigRaise": round(action_counts[4] / max(1, tot_acts) * 100, 2),
                "AllIn": round(action_counts[5] / max(1, tot_acts) * 100, 2)
            }
            
            # Seat breakdown
            seat_stats = {pos: round(float(np.mean(seat_pnls[pos])), 2) for pos in range(5)}
            # Agent breakdown (A..D trained, E untrained in 054)
            agent_stats = {a: round(float(np.mean(agent_pnls[a])), 2) for a in AGENT_NAMES}
            
            results[opp_pool_name][arm_name] = {
                "hands": len(hand_pnls),
                "mean_bb100": round(mean_bb, 2),
                "ci_95": [round(mean_bb - ci_half, 2), round(mean_bb + ci_half, 2)],
                "act_dist": act_dist,
                "allin_hands": len(allin_pnl),
                "allin_mean_pnl": round(float(np.mean(allin_pnl)), 2) if allin_pnl else 0.0,
                "seat_stats": seat_stats,
                "agent_stats": agent_stats,
                "elapsed": round(time.time() - t0, 1)
            }
            
            print(f"  Result: {mean_bb:+.2f} BB/100 (95% CI: [{mean_bb - ci_half:+.2f}, {mean_bb + ci_half:+.2f}])")
            print(f"  Seat PnL: 0={seat_stats[0]}, 1={seat_stats[1]}, 2={seat_stats[2]}, 3={seat_stats[3]}, 4={seat_stats[4]}")
            print(f"  Agent PnL: A={agent_stats['Agent_A']}, B={agent_stats['Agent_B']}, C={agent_stats['Agent_C']}, D={agent_stats['Agent_D']}, E={agent_stats['Agent_E']}")
            print(f"  All-in Rate: {act_dist['AllIn']}% | All-in PnL: {results[opp_pool_name][arm_name]['allin_mean_pnl']}")
            
    f_raw.close()
    
    with open("v1024/directive055_diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved diagnostic results to v1024/directive055_diagnostic_results.json")
    return results

if __name__ == "__main__":
    from collections import Counter
    run_diagnostic_evaluation()
