#!/usr/bin/env python3
"""
v1024/evaluate_directive057_benchmark.py
Comprehensive Benchmark Evaluation for DIRECTIVE 057:
1. Development Evaluation (100 Seeds: 1201000~1201099):
   - Progression across 500h -> 1000h -> 1500h -> 2000h for policy degradation tracking.
2. Final Holdout Evaluation (200 Seeds: 1301000~1301199):
   - Strict Holdout testing against Frozen 2k and Frozen 7k (Agent A~E fully symmetric).
   - Paired-difference statistical tests (Static vs Dynamic, PFSP vs Uniform).
   - Fold/Call/Raise/All-in raw action counts verified.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.directive057_production_engine import (
    compute_sha256, get_perfect_latin_square_assignment, BIG_BLIND, AGENT_NAMES, CHECKPOINT_DIR
)

DEV_SEEDS = list(range(1201000, 1201100)) # 100 seeds for development progression
FINAL_SEEDS = list(range(1301000, 1301300)) # 200 seeds for final holdout

def load_eval_model(path: str) -> PPONeuralPlayer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing eval model: {path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m.eval()
    return m

def evaluate_model_roster(
    game,
    hero_models: Dict[str, PPONeuralPlayer], # {agent_name: model}
    opp_models: Dict[str, PPONeuralPlayer],  # {opp_name: model}
    seeds: List[int],
    hands_per_seed: int = 5,
    eval_name: str = "eval",
    raw_log_path: Optional[str] = None
) -> Dict[str, Any]:
    opp_names = sorted(list(opp_models.keys()))
    n_opps = len(opp_names)
    
    seed_pnl_records = []
    agent_pnl_records = {a: [] for a in AGENT_NAMES}
    seat_pnl_records = {s: [] for s in range(5)}
    opp_encounters = {op: 0 for op in opp_names}
    
    action_counts = {
        "fold": 0,    # 0
        "call": 0,    # 1
        "raise": 0,   # 2, 3, 4
        "allin": 0    # 5
    }
    raw_logs = []
    
    hand_global_counter = 0
    
    for s_idx, seed in enumerate(seeds):
        seed_hand_pnls = []
        for h_in_seed in range(1, hands_per_seed + 1):
            hand_global_counter += 1
            
            # Symmetrical opponent rotation
            opp_choice = opp_names[hand_global_counter % n_opps]
            opp_encounters[opp_choice] += 1
            opp_obj = opp_models[opp_choice]
            
            # Independent Card & Action RNG
            deck_seed = (seed * 1000003 + h_in_seed * 10007 + hand_global_counter * 101) & 0xFFFFFFFF
            deck_rng = np.random.RandomState(deck_seed)
            action_rng = np.random.RandomState(deck_seed ^ 0x55555555)
            
            assign = get_perfect_latin_square_assignment(hand_global_counter)
            frozen_seat = assign["frozen_seat"]
            seat_to_agent = assign["seat_to_agent"]
            
            table = [None] * 5
            table[frozen_seat] = ("OPPONENT", opp_choice, opp_obj)
            for st_idx, a_idx in seat_to_agent.items():
                ag_name = AGENT_NAMES[a_idx]
                table[st_idx] = ("HERO", ag_name, hero_models[ag_name])
                
            st = game.new_initial_state()
            advance_chance_nodes(st, deck_rng)
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, deck_rng)
                    continue
                cp = st.current_player()
                legals = True6ActionAdapter.get_legal_6actions(st, cp)
                norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                
                p_type, p_name, p_obj = table[cp]
                with torch.no_grad():
                    logits, _ = p_obj(torch.from_numpy(norm_inf).unsqueeze(0))
                    mask = torch.full_like(logits, -1e9)
                    for a in legals: mask[0, a] = 0.0
                    probs = torch.softmax(logits + mask, dim=-1).squeeze(0).numpy()
                    act = int(action_rng.choice(len(probs), p=probs))
                    
                if p_type == "HERO":
                    if act == 0: action_counts["fold"] += 1
                    elif act == 1: action_counts["call"] += 1
                    elif act in [2, 3, 4]: action_counts["raise"] += 1
                    elif act == 5: action_counts["allin"] += 1
                    
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                
            returns = st.returns()
            for pos, (p_type, p_name, p_obj) in enumerate(table):
                ret_bb = returns[pos] / BIG_BLIND
                seat_pnl_records[pos].append(ret_bb)
                if p_type == "HERO":
                    agent_pnl_records[p_name].append(ret_bb)
                    seed_hand_pnls.append(ret_bb)
                    
            if raw_log_path:
                raw_logs.append({
                    "eval_name": eval_name,
                    "seed": seed,
                    "hand_idx": hand_global_counter,
                    "frozen_seat": frozen_seat,
                    "opponent": opp_choice,
                    "returns_bb": [round(r / BIG_BLIND, 2) for r in returns]
                })
                
        seed_pnl_records.append(float(np.mean(seed_hand_pnls)))
        
    if raw_log_path and raw_logs:
        os.makedirs(os.path.dirname(raw_log_path), exist_ok=True)
        with open(raw_log_path, "a", encoding="utf-8") as f:
            for rl in raw_logs:
                f.write(json.dumps(rl) + "\n")
                
    seed_arr = np.array(seed_pnl_records) * 100.0 # to BB/100
    mean_bb100 = float(np.mean(seed_arr))
    std_bb100 = float(np.std(seed_arr, ddof=1)) if len(seed_arr) > 1 else 0.0
    sem = std_bb100 / np.sqrt(len(seed_arr)) if len(seed_arr) > 1 else 0.0
    ci_95 = [float(mean_bb100 - 1.96 * sem), float(mean_bb100 + 1.96 * sem)]
    
    tot_actions = sum(action_counts.values()) or 1
    act_pcts = {k: round(v / tot_actions * 100.0, 2) for k, v in action_counts.items()}
    
    agent_means = {a: round(float(np.mean(agent_pnl_records[a])) * 100.0, 2) for a in AGENT_NAMES}
    seat_means = {s: round(float(np.mean(seat_pnl_records[s])) * 100.0, 2) for s in range(5)}
    
    return {
        "eval_name": eval_name,
        "total_hands": hand_global_counter,
        "num_seeds": len(seeds),
        "mean_bb100": round(mean_bb100, 2),
        "std_bb100": round(std_bb100, 2),
        "ci_95": [round(ci_95[0], 2), round(ci_95[1], 2)],
        "agent_means_bb100": agent_means,
        "seat_means_bb100": seat_means,
        "action_counts": action_counts,
        "action_pcts": act_pcts,
        "opp_encounters": opp_encounters,
        "seed_level_data": seed_pnl_records
    }


def run_full_benchmark(out_json_path: str = "v1024/directive057_benchmark_results.json"):
    print("\n=================================================================")
    print(">>> [DIRECTIVE 057] RUNNING COMPREHENSIVE BENCHMARK EVALUATION")
    print("=================================================================")
    
    game = get_5p_fullgame()
    raw_log_file = "v1024/data/directive057_raw_benchmark_holdout.jsonl"
    if os.path.exists(raw_log_file):
        os.remove(raw_log_file)
        
    # Baseline 7k opponent pool
    frozen_7k_opps = {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
    frozen_2k_opps = {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_2000h.pt")) for a in AGENT_NAMES}
    
    benchmark_results = {
        "development_evaluation": {},
        "final_holdout_evaluation": {},
        "paired_hypotheses_tests": {}
    }
    
    # Checkpoints to evaluate
    base_exp_dir = "v1024/exp_directive057"
    
    # 1. DEVELOPMENT EVALUATION (Seeds 1201000~1201099, 100 seeds x 5 hands = 500 hands per arm)
    # Track Progression across 500h, 1000h, 1500h, 2000h
    print("\n--- Phase 1: Development Evaluation (Policy Progression Tracking) ---")
    arms_to_track = [
        ("arm_pfsp_dynamic_seed_1001000", [500, 1000, 1500, 2000]),
        ("arm_uniform_dynamic_seed_1001000", [500, 1000, 1500, 2000]),
        ("arm_pfsp_static_seed_1001000", [1000, 2000]),
        ("arm_uniform_static_seed_1001000", [1000, 2000])
    ]
    
    for arm_name, steps in arms_to_track:
        arm_dir = os.path.join(base_exp_dir, arm_name)
        if not os.path.exists(arm_dir):
            continue
        for step in steps:
            ckpt_dir = os.path.join(arm_dir, f"ckpt_hand_{step}")
            if not os.path.exists(ckpt_dir):
                continue
            hero_models = {a: load_eval_model(os.path.join(ckpt_dir, f"{a}.pt")) for a in AGENT_NAMES}
            
            res = evaluate_model_roster(
                game=game,
                hero_models=hero_models,
                opp_models=frozen_7k_opps,
                seeds=DEV_SEEDS,
                hands_per_seed=5,
                eval_name=f"{arm_name}_{step}h_dev",
                raw_log_path=raw_log_file
            )
            key = f"{arm_name}_{step}h"
            benchmark_results["development_evaluation"][key] = res
            print(f"  [DEV] {key:35s} | Mean: {res['mean_bb100']:+7.2f} BB/100 | 95% CI: [{res['ci_95'][0]:+7.2f}, {res['ci_95'][1]:+7.2f}]")

    # 2. FINAL HOLDOUT EVALUATION (Seeds 1301000~1301299, 200 seeds x 5 hands = 1,000 hands per arm)
    print("\n--- Phase 2: Final Holdout Evaluation (Strictly Isolated 200 Seeds) ---")
    final_models = {
        "Baseline_7000h": {a: load_eval_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES},
        "Uniform_Static_2000h_s1": {a: load_eval_model(os.path.join(base_exp_dir, "arm_uniform_static_seed_1001000/ckpt_hand_2000", f"{a}.pt")) for a in AGENT_NAMES},
        "PFSP_Static_2000h_s1": {a: load_eval_model(os.path.join(base_exp_dir, "arm_pfsp_static_seed_1001000/ckpt_hand_2000", f"{a}.pt")) for a in AGENT_NAMES},
        "Uniform_Dynamic_2000h_s1": {a: load_eval_model(os.path.join(base_exp_dir, "arm_uniform_dynamic_seed_1001000/ckpt_hand_2000", f"{a}.pt")) for a in AGENT_NAMES},
        "PFSP_Dynamic_2000h_s1": {a: load_eval_model(os.path.join(base_exp_dir, "arm_pfsp_dynamic_seed_1001000/ckpt_hand_2000", f"{a}.pt")) for a in AGENT_NAMES},
        "PFSP_Dynamic_2000h_s2": {a: load_eval_model(os.path.join(base_exp_dir, "arm_pfsp_dynamic_seed_1002000/ckpt_hand_2000", f"{a}.pt")) for a in AGENT_NAMES},
        "PFSP_Static_2000h_s2": {a: load_eval_model(os.path.join(base_exp_dir, "arm_pfsp_static_seed_1002000/ckpt_hand_2000", f"{a}.pt")) for a in AGENT_NAMES}
    }
    
    for m_name, heroes in final_models.items():
        res_7k = evaluate_model_roster(
            game=game,
            hero_models=heroes,
            opp_models=frozen_7k_opps,
            seeds=FINAL_SEEDS,
            hands_per_seed=5,
            eval_name=f"{m_name}_vs_7k_final",
            raw_log_path=raw_log_file
        )
        res_2k = evaluate_model_roster(
            game=game,
            hero_models=heroes,
            opp_models=frozen_2k_opps,
            seeds=FINAL_SEEDS,
            hands_per_seed=5,
            eval_name=f"{m_name}_vs_2k_final",
            raw_log_path=raw_log_file
        )
        benchmark_results["final_holdout_evaluation"][f"{m_name}_vs_7k"] = res_7k
        benchmark_results["final_holdout_evaluation"][f"{m_name}_vs_2k"] = res_2k
        print(f"  [FINAL] {m_name:25s} vs 7k: {res_7k['mean_bb100']:+7.2f} (CI: {res_7k['ci_95']}) | vs 2k: {res_2k['mean_bb100']:+7.2f} (CI: {res_2k['ci_95']})")

    # 3. DIRECT PAIRED-DIFFERENCE HYPOTHESES TESTS
    print("\n--- Phase 3: Paired Hypotheses Statistical Tests ---")
    def run_paired_diff(name_a, name_b, test_label):
        data_a = np.array(benchmark_results["final_holdout_evaluation"][name_a]["seed_level_data"]) * 100.0
        data_b = np.array(benchmark_results["final_holdout_evaluation"][name_b]["seed_level_data"]) * 100.0
        diff = data_a - data_b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        sem_diff = std_diff / np.sqrt(len(diff))
        ci_95 = [float(mean_diff - 1.96 * sem_diff), float(mean_diff + 1.96 * sem_diff)]
        proven = bool(ci_95[0] > 0 or ci_95[1] < 0)
        
        stat = {
            "test_label": test_label,
            "arm_A": name_a,
            "arm_B": name_b,
            "mean_delta_bb100": round(mean_diff, 2),
            "std_delta": round(std_diff, 2),
            "ci_95": [round(ci_95[0], 2), round(ci_95[1], 2)],
            "zero_included": bool(ci_95[0] <= 0 <= ci_95[1]),
            "verdict": "PROVEN_SIGNIFICANT" if proven else "NOT_PROVEN"
        }
        benchmark_results["paired_hypotheses_tests"][test_label] = stat
        print(f"  [{stat['verdict']}] {test_label}: Delta {stat['mean_delta_bb100']:+7.2f} BB/100 (95% CI: {stat['ci_95']})")

    # Test 1: Dynamic vs Static (PFSP s1)
    run_paired_diff("PFSP_Dynamic_2000h_s1_vs_7k", "PFSP_Static_2000h_s1_vs_7k", "Dynamic_vs_Static_PFSP_s1_vs_7k")
    # Test 2: PFSP vs Uniform (Dynamic s1)
    run_paired_diff("PFSP_Dynamic_2000h_s1_vs_7k", "Uniform_Dynamic_2000h_s1_vs_7k", "PFSP_vs_Uniform_Dynamic_s1_vs_7k")
    # Test 3: Dynamic vs Baseline 7k
    run_paired_diff("PFSP_Dynamic_2000h_s1_vs_7k", "Baseline_7000h_vs_7k", "PFSP_Dynamic_vs_Baseline_7k")

    # Clean seed-level raw data from final json to keep size clean
    for section in ["development_evaluation", "final_holdout_evaluation"]:
        for k in benchmark_results[section]:
            if "seed_level_data" in benchmark_results[section][k]:
                del benchmark_results[section][k]["seed_level_data"]
                
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {out_json_path}")
    print(f"[+] Raw hand log saved to: {raw_log_file}")
    return benchmark_results

if __name__ == "__main__":
    run_full_benchmark()
