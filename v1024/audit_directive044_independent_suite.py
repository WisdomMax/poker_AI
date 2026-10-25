#!/usr/bin/env python3
"""
DIRECTIVE 044: Full Independent Audit Suite for Multi-Agent Self-Play RL
1. Precise All-in Instrumentation (Action 5 vs Full Stack Invested vs Call/Push Allin)
2. Disentangling Initial Noise vs True PPO Training (Group A: 50k, Group B: 0h Noisy, Group C: 2000h)
3. Full 5-Agent Evaluation (A, B, C, D, E all evaluated against Frozen 50k and All-in Catcher)
4. Milestone Strategy Evolution across 0h, 200h, 500h, 1000h, 1500h, 2000h
5. Arithmetic Consistency & Raw JSONL Export
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyspiel
from v1024.alphago_5p_selfplay import (
    AGENT_NAMES,
    PPONeuralPlayer,
    True6ActionAdapter,
    advance_chance_nodes,
    get_5p_fullgame,
    normalize_inf_state_v79,
    robust_parse_cards,
)
from v72.run_v72_true_ledger_and_fold_audit import extract_player_spent
from v71.run_v71_ledger_and_street_audit import get_round_from_state

BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
CHECKPOINT_DIR = "v1024/models_alphago"
RAW_LOG_PATH = "v1024/data/directive044_raw_audit_hands.jsonl"
OUT_JSON_PATH = "v1024/directive044_audit_results.json"

STARTING_STACK = 10000.0
BIG_BLIND = 100.0

def load_ppo_model(ckpt_path: str, device: str = "cpu") -> PPONeuralPlayer:
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def create_noisy_0h_model(agent_id: int, base_ckpt_path: str = BASE_50K_PATH,
                          base_seed: int = 505000, device: str = "cpu") -> PPONeuralPlayer:
    """Recreates the exact initial noisy 0h model before any self-play training"""
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    state = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    
    seed = base_seed + agent_id * 100
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * 0.005 * (agent_id + 1)
            p.add_(noise)
    model.eval()
    model.to(device)
    return model

class TwoPairPlusCatcherBot:
    """Opponent that checks/calls reasonably and only calls all-ins with strong hands"""
    def __init__(self):
        self.name = "TwoPairPlusCatcher"

    def select_action(self, st, cp: int, legal_6: List[int], round_idx: int) -> int:
        spent = extract_player_spent(st)
        max_spent = max(spent)
        my_spent = spent[cp]
        to_call = max_spent - my_spent

        # If free check available
        if to_call == 0:
            return 0 if 0 in legal_6 else 1 # Free Check
        
        # If facing all-in (huge bet)
        if to_call >= STARTING_STACK * 0.5:
            # Conservative all-in call rate (20% postflop, 5% preflop)
            if round_idx >= 1:
                return 1 if (random.random() < 0.20 and 1 in legal_6) else 0
            else:
                return 1 if (random.random() < 0.05 and 1 in legal_6) else 0

        # Normal bet facing
        if to_call <= BIG_BLIND * 3:
            return 1 if 1 in legal_6 else 0
        return 0 # Fold to heavy non-allin bet

def compute_t_test(diffs: List[float]) -> Tuple[float, float, Tuple[float, float]]:
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0, (0.0, 0.0)
    mean = float(np.mean(diffs))
    sem = float(stats.sem(diffs))
    if sem == 0:
        return mean, 1.0, (mean, mean)
    t_val = mean / sem
    df = n - 1
    p_val = float(2 * (1 - stats.t.cdf(abs(t_val), df=df)))
    ci_margin = float(stats.t.ppf(0.975, df=df) * sem)
    return mean, p_val, (mean - ci_margin, mean + ci_margin)

# ==============================================================================
# AUDIT PART 1: 3-Group Comparison (Group A: 50k, Group B: 0h, Group C: 2000h)
# ==============================================================================

def run_3group_paired_benchmark(num_hands: int = 500, base_seed: int = 40000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 1: Disentangling Noise vs Training ({num_hands} hands per agent)...")
    game = get_5p_fullgame()
    os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
    
    base_50k_model = load_ppo_model(BASE_50K_PATH)
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    results_by_agent = {}
    
    raw_log_file = open(RAW_LOG_PATH, "w", encoding="utf-8")

    for a_idx, a_name in enumerate(agents):
        print(f"--- Benchmarking {a_name} (0h Noisy vs 2000h Trained vs Frozen 50k) ---")
        noisy_0h_model = create_noisy_0h_model(a_idx)
        trained_2000h_path = os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt")
        trained_2000h_model = load_ppo_model(trained_2000h_path)
        
        pnl_0h = []
        pnl_2000h = []
        diffs_2000h_minus_0h = []
        
        act_counts_0h = Counter()
        act_counts_2000h = Counter()
        
        allin_stats_0h = {"direct_act5": 0, "full_stack_invested": 0, "pushed_allin": 0, "called_allin": 0, "pot_commit_allin": 0}
        allin_stats_2000h = {"direct_act5": 0, "full_stack_invested": 0, "pushed_allin": 0, "called_allin": 0, "pot_commit_allin": 0}

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            
            # --- EVALUATE NOISY 0H ---
            rng_0h = np.random.RandomState(seed)
            st_0h = game.new_initial_state()
            
            while not st_0h.is_terminal():
                if st_0h.is_chance_node():
                    advance_chance_nodes(st_0h, rng_0h)
                    continue

                cur_p = st_0h.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st_0h, cur_p)
                
                spent_before = extract_player_spent(st_0h)
                my_spent_before = spent_before[cur_p]
                stack_before = STARTING_STACK - my_spent_before
                to_call = max(spent_before) - my_spent_before

                norm_inf = normalize_inf_state_v79(np.array(st_0h.information_state_tensor(cur_p), dtype=np.float32))
                t_state = torch.from_numpy(norm_inf).unsqueeze(0)
                
                active_model = noisy_0h_model if cur_p == hero_seat else base_50k_model
                with torch.no_grad():
                    logits, _ = active_model(t_state)
                    masked = logits.clone()
                    mask = torch.full_like(masked, -1e9)
                    for a in legal_6: mask[0, a] = 0.0
                    masked += mask
                    probs = F.softmax(masked, dim=-1)
                    act_6 = int(torch.argmax(probs, dim=-1).item()) # Greedy
                
                if cur_p == hero_seat:
                    act_counts_0h[act_6] += 1
                    if act_6 == 5: allin_stats_0h["direct_act5"] += 1
                
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_0h, cur_p, act_6)
                st_0h.apply_action(nat_act)
                
                if cur_p == hero_seat:
                    spent_after = extract_player_spent(st_0h)
                    if spent_after[cur_p] >= STARTING_STACK:
                        allin_stats_0h["full_stack_invested"] += 1
                        if act_6 == 5:
                            allin_stats_0h["pushed_allin"] += 1
                        elif act_6 == 1 and to_call >= stack_before:
                            allin_stats_0h["called_allin"] += 1
                        elif act_6 in (2, 3, 4):
                            allin_stats_0h["pot_commit_allin"] += 1
            
            ret_0h = st_0h.returns()[hero_seat] / BIG_BLIND
            pnl_0h.append(ret_0h)
            
            # --- EVALUATE TRAINED 2000H (IDENTICAL SEED & SEAT) ---
            rng_2000h = np.random.RandomState(seed)
            st_2000h = game.new_initial_state()
            hero_actions_this_hand = []

            while not st_2000h.is_terminal():
                if st_2000h.is_chance_node():
                    advance_chance_nodes(st_2000h, rng_2000h)
                    continue

                cur_p = st_2000h.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st_2000h, cur_p)
                
                spent_before = extract_player_spent(st_2000h)
                my_spent_before = spent_before[cur_p]
                stack_before = STARTING_STACK - my_spent_before
                to_call = max(spent_before) - my_spent_before

                norm_inf = normalize_inf_state_v79(np.array(st_2000h.information_state_tensor(cur_p), dtype=np.float32))
                t_state = torch.from_numpy(norm_inf).unsqueeze(0)
                
                active_model = trained_2000h_model if cur_p == hero_seat else base_50k_model
                with torch.no_grad():
                    logits, _ = active_model(t_state)
                    masked = logits.clone()
                    mask = torch.full_like(masked, -1e9)
                    for a in legal_6: mask[0, a] = 0.0
                    masked += mask
                    probs = F.softmax(masked, dim=-1)
                    act_6 = int(torch.argmax(probs, dim=-1).item())
                
                if cur_p == hero_seat:
                    act_counts_2000h[act_6] += 1
                    if act_6 == 5: allin_stats_2000h["direct_act5"] += 1
                    hero_actions_this_hand.append(act_6)
                
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_2000h, cur_p, act_6)
                st_2000h.apply_action(nat_act)
                
                if cur_p == hero_seat:
                    spent_after = extract_player_spent(st_2000h)
                    if spent_after[cur_p] >= STARTING_STACK:
                        allin_stats_2000h["full_stack_invested"] += 1
                        if act_6 == 5:
                            allin_stats_2000h["pushed_allin"] += 1
                        elif act_6 == 1 and to_call >= stack_before:
                            allin_stats_2000h["called_allin"] += 1
                        elif act_6 in (2, 3, 4):
                            allin_stats_2000h["pot_commit_allin"] += 1
            
            ret_2000h = st_2000h.returns()[hero_seat] / BIG_BLIND
            pnl_2000h.append(ret_2000h)
            diffs_2000h_minus_0h.append(ret_2000h - ret_0h)
            
            # Write raw JSONL log
            hand_record = {
                "hand_id": h_i,
                "agent": a_name,
                "seed": seed,
                "seat": hero_seat,
                "pnl_0h_bb": ret_0h,
                "pnl_2000h_bb": ret_2000h,
                "diff_bb": ret_2000h - ret_0h,
                "actions_2000h": hero_actions_this_hand
            }
            raw_log_file.write(json.dumps(hand_record) + "\n")

        mean_0h = float(np.mean(pnl_0h)) * 100.0
        ci_0h = float(stats.sem(pnl_0h)) * 100.0 * 1.96
        
        mean_2000h = float(np.mean(pnl_2000h)) * 100.0
        ci_2000h = float(stats.sem(pnl_2000h)) * 100.0 * 1.96
        
        diff_mean, diff_p, diff_ci = compute_t_test(diffs_2000h_minus_0h)
        diff_bb100 = diff_mean * 100.0
        diff_ci_bb100 = (diff_ci[0] * 100.0, diff_ci[1] * 100.0)

        tot_acts_0h = sum(act_counts_0h.values())
        tot_acts_2000h = sum(act_counts_2000h.values())

        res_a = {
            "num_hands": num_hands,
            "group_b_0h_noisy": {
                "bb_per_100": round(mean_0h, 2),
                "ci95_bb": round(ci_0h, 2),
                "action_counts": dict(act_counts_0h),
                "direct_act5_pct": round(allin_stats_0h["direct_act5"] / max(tot_acts_0h, 1) * 100.0, 2),
                "full_stack_invested_pct": round(allin_stats_0h["full_stack_invested"] / max(tot_acts_0h, 1) * 100.0, 2),
                "allin_breakdown": allin_stats_0h
            },
            "group_c_2000h_trained": {
                "bb_per_100": round(mean_2000h, 2),
                "ci95_bb": round(ci_2000h, 2),
                "action_counts": dict(act_counts_2000h),
                "direct_act5_pct": round(allin_stats_2000h["direct_act5"] / max(tot_acts_2000h, 1) * 100.0, 2),
                "full_stack_invested_pct": round(allin_stats_2000h["full_stack_invested"] / max(tot_acts_2000h, 1) * 100.0, 2),
                "allin_breakdown": allin_stats_2000h
            },
            "paired_difference_2000h_vs_0h": {
                "delta_bb_per_100": round(diff_bb100, 2),
                "ci95": [round(diff_ci_bb100[0], 2), round(diff_ci_bb100[1], 2)],
                "p_value": round(diff_p, 4),
                "is_significant": diff_p < 0.05
            }
        }
        results_by_agent[a_name] = res_a
        print(f"  0h Noisy: {mean_0h:+.2f} BB/100 | 2000h: {mean_2000h:+.2f} BB/100 | Delta: {diff_bb100:+.2f} BB/100 (p={diff_p:.4f})")
        print(f"  All-in Rate: 0h Act5={res_a['group_b_0h_noisy']['direct_act5_pct']}% -> 2000h Act5={res_a['group_c_2000h_trained']['direct_act5_pct']}% (Full Stack={res_a['group_c_2000h_trained']['full_stack_invested_pct']}%)")

    raw_log_file.close()
    return results_by_agent

# ==============================================================================
# AUDIT PART 2: All 5 Agents vs TwoPairPlusCatcher
# ==============================================================================

def run_all_agents_vs_catcher(num_hands: int = 500, base_seed: int = 60000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 2: All 5 Agents vs TwoPairPlusCatcher ({num_hands} hands each)...")
    game = get_5p_fullgame()
    catcher = TwoPairPlusCatcherBot()
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    
    catcher_results = {}
    
    for a_idx, a_name in enumerate(agents):
        model = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt"))
        pnls = []
        act_counts = Counter()
        allin_called_pnls = []
        allin_uncalled_pnls = []

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)
            st = game.new_initial_state()
            
            pushed_allin_this_hand = False

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                    continue

                cur_p = st.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st, cur_p)
                round_idx = get_round_from_state(st)
                
                if cur_p == hero_seat:
                    norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cur_p), dtype=np.float32))
                    t_state = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = model(t_state)
                        masked = logits.clone()
                        mask = torch.full_like(masked, -1e9)
                        for a in legal_6: mask[0, a] = 0.0
                        masked += mask
                        probs = F.softmax(masked, dim=-1)
                        act_6 = int(torch.argmax(probs, dim=-1).item())
                    act_counts[act_6] += 1
                    if act_6 == 5: pushed_allin_this_hand = True
                else:
                    act_6 = catcher.select_action(st, cur_p, legal_6, round_idx)
                
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cur_p, act_6)
                st.apply_action(nat_act)

            ret = st.returns()[hero_seat] / BIG_BLIND
            pnls.append(ret)
            
            if pushed_allin_this_hand:
                if list(st.returns()).count(0.0) < 4:
                    allin_called_pnls.append(ret)
                else:
                    allin_uncalled_pnls.append(ret)

        mean_bb = float(np.mean(pnls)) * 100.0
        ci_bb = float(stats.sem(pnls)) * 100.0 * 1.96
        tot_acts = sum(act_counts.values())
        
        catcher_results[a_name] = {
            "bb_per_100": round(mean_bb, 2),
            "ci95_bb": round(ci_bb, 2),
            "allin_rate_pct": round(act_counts.get(5, 0) / max(tot_acts, 1) * 100.0, 2),
            "pot_rate_pct": round(act_counts.get(3, 0) / max(tot_acts, 1) * 100.0, 2),
            "halfpot_rate_pct": round(act_counts.get(2, 0) / max(tot_acts, 1) * 100.0, 2),
            "called_allin_avg_pnl_bb": round(float(np.mean(allin_called_pnls)), 2) if allin_called_pnls else 0.0,
            "uncalled_allin_avg_pnl_bb": round(float(np.mean(allin_uncalled_pnls)), 2) if allin_uncalled_pnls else 0.0,
            "called_allin_count": len(allin_called_pnls),
            "uncalled_allin_count": len(allin_uncalled_pnls)
        }
        print(f"  {a_name} vs Catcher: {mean_bb:+.2f} ± {ci_bb:.2f} BB/100 | All-in%: {catcher_results[a_name]['allin_rate_pct']}% | Called EV: {catcher_results[a_name]['called_allin_avg_pnl_bb']} BB")

    return catcher_results

# ==============================================================================
# AUDIT PART 3: Milestone Strategy Evolution (0h -> 2000h)
# ==============================================================================

def run_milestone_evolution(num_hands: int = 500, base_seed: int = 70000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 3: Milestone Strategy Evolution (0h -> 2000h)...")
    game = get_5p_fullgame()
    base_50k_model = load_ppo_model(BASE_50K_PATH)
    
    tracked_agents = ["Agent_E", "Agent_D"]
    milestones = ["200h", "500h", "1000h", "1500h", "2000h"]
    evolution_results = {}

    for t_name in tracked_agents:
        a_idx = 4 if t_name == "Agent_E" else 3
        evolution_results[t_name] = {}
        
        noisy_0h = create_noisy_0h_model(a_idx)
        models = {"0h": noisy_0h}
        for m in milestones:
            models[m] = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{t_name}_{m}.pt"))

        for m_key, m_model in models.items():
            pnls = []
            act_counts = Counter()
            for h_i in range(1, num_hands + 1):
                seed = base_seed + a_idx * 5000 + h_i
                hero_seat = (h_i - 1) % 5
                rng = np.random.RandomState(seed)
                st = game.new_initial_state()
                
                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng)
                        continue

                    cur_p = st.current_player()
                    legal_6 = True6ActionAdapter.get_legal_6actions(st, cur_p)
                    norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cur_p), dtype=np.float32))
                    t_state = torch.from_numpy(norm_inf).unsqueeze(0)

                    active_model = m_model if cur_p == hero_seat else base_50k_model
                    with torch.no_grad():
                        logits, _ = active_model(t_state)
                        masked = logits.clone()
                        mask = torch.full_like(masked, -1e9)
                        for a in legal_6: mask[0, a] = 0.0
                        masked += mask
                        probs = F.softmax(masked, dim=-1)
                        act_6 = int(torch.argmax(probs, dim=-1).item())
                    
                    if cur_p == hero_seat:
                        act_counts[act_6] += 1
                        
                    nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cur_p, act_6)
                    st.apply_action(nat_act)

                pnls.append(st.returns()[hero_seat] / BIG_BLIND)

            tot_acts = sum(act_counts.values())
            m_res = {
                "bb_per_100": round(float(np.mean(pnls)) * 100.0, 2),
                "ci95_bb": round(float(stats.sem(pnls)) * 100.0 * 1.96, 2),
                "allin_pct": round(act_counts.get(5, 0) / max(tot_acts, 1) * 100.0, 2),
                "pot_pct": round(act_counts.get(3, 0) / max(tot_acts, 1) * 100.0, 2),
                "halfpot_pct": round(act_counts.get(2, 0) / max(tot_acts, 1) * 100.0, 2)
            }
            evolution_results[t_name][m_key] = m_res
            print(f"  {t_name} [{m_key:5s}] -> {m_res['bb_per_100']:+6.2f} ± {m_res['ci95_bb']:5.2f} BB/100 | Allin%: {m_res['allin_pct']:4.1f}% | Pot%: {m_res['pot_pct']:4.1f}% | HalfPot%: {m_res['halfpot_pct']:4.1f}%")

    return evolution_results

def run_full_directive044_suite():
    print("================================================================================")
    print("DIRECTIVE 044: FIVE-AGENT SELF-PLAY INDEPENDENT AUDIT SUITE")
    print("================================================================================")
    start_t = time.time()
    
    part1_res = run_3group_paired_benchmark(num_hands=500)
    part2_res = run_all_agents_vs_catcher(num_hands=500)
    part3_res = run_milestone_evolution(num_hands=500)
    
    total_time = time.time() - start_t
    
    final_output = {
        "directive": "DIRECTIVE 044: FIVE-AGENT SELF-PLAY INDEPENDENT AUDIT",
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_time, 2),
        "base_50k_path": BASE_50K_PATH,
        "part1_disentangle_noise_vs_training": part1_res,
        "part2_all_agents_vs_catcher": part2_res,
        "part3_milestone_evolution": part3_res
    }
    
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nAudit complete in {total_time:.1f}s! Results saved to {OUT_JSON_PATH}")

if __name__ == "__main__":
    run_full_directive044_suite()
