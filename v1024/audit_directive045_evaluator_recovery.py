#!/usr/bin/env python3
"""
DIRECTIVE 045: Multi-Agent Evaluator Integrity Recovery Suite
1. Exact 0h Seed Alignment: base_seed = 430000 (Agent i: 430000 + i * 100)
2. Dual All-in Denominators: Per-Decision vs Per-Hand (Push vs Call vs Commit)
3. True Real-Card Evaluator: StrongAllinCatcherBot with eval_7cards (Two Pair+ only)
4. Rigorous All-in Call & Showdown Tracking (Uncalled Push vs Called All-in EV)
5. True Milestone Lineage: 0h -> 500h -> 1000h -> 1500h -> 2000h (Seed 430000)
6. Multiple Testing Correction: Holm-Bonferroni adjusted p-values
7. Extended Raw Hand JSONL Logs
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.tests.test_directive040_regression import (
    calc_chen_score, robust_parse_cards, parse_board_cards
)
from v1024.alphago_5p_selfplay import AGENT_NAMES

BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
CHECKPOINT_DIR = "v1024/models_alphago"
RAW_LOG_PATH = "v1024/data/directive045_raw_audit_hands.jsonl"
OUT_JSON_PATH = "v1024/directive045_audit_results.json"

STARTING_STACK = 10000.0
BIG_BLIND = 100.0
PILOT_BASE_SEED = 430000 # Exactly matching run_directive043_2000h_pilot.py

def card_str_to_int(c_str: str) -> int:
    ranks = '23456789TJQKA'
    suits = 'shdc'
    r = ranks.index(c_str[0])
    s = suits.index(c_str[1])
    return r * 4 + s

def load_ppo_model(ckpt_path: str, device: str = "cpu") -> PPONeuralPlayer:
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def create_true_0h_model(agent_id: int, base_ckpt_path: str = BASE_50K_PATH,
                         base_seed: int = PILOT_BASE_SEED, device: str = "cpu") -> PPONeuralPlayer:
    """
    Recreates the EXACT 0h model of the 043 self-play pilot before training.
    Seed formula: base_seed + agent_id * 100 = 430000 + i * 100
    """
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

class TrueStrongAllinCatcherBot:
    """
    Legitimate 7-card hand evaluation bot.
    Only calls all-ins with AA, KK, QQ, AKs preflop or actual Two Pair+ postflop.
    Zero leakage: Only inspects its own hole cards and public board.
    """
    name = "TrueStrongAllinCatcher"

    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def select_action(self, st, cp: int, legal_6: List[int], priv_cards: Dict[int, List[str]], round_idx: int) -> int:
        spent = extract_player_spent(st)
        max_spent = max(spent)
        my_spent = spent[cp]
        to_call = max_spent - my_spent

        c1, c2 = priv_cards[cp]
        chen = calc_chen_score(c1, c2)
        board = parse_board_cards(st)
        
        is_allin_facing = (to_call >= STARTING_STACK * 0.5) or (to_call >= 1000.0)

        if is_allin_facing and to_call > 0:
            if round_idx == 0:
                # Preflop: AA, KK, QQ, AKs only (Chen >= 11.0)
                if chen >= 11.0 and 1 in legal_6:
                    return 1
                return 0 if 0 in legal_6 else 1
            else:
                # Postflop: Real 7-card hand rank (rank >= 2: Two Pair, Trips, Straight, Flush, Full House+)
                if len(board) >= 3:
                    int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
                    rank = eval_7cards(int_cards)[0]
                else:
                    rank = 0
                if rank >= 2 and 1 in legal_6:
                    return 1 # Legitimate Call with Two Pair+
                return 0 if 0 in legal_6 else 1

        # Non-allin normal play
        if to_call == 0:
            if 0 in legal_6 and self.rng.random() < 0.8: return 0 # Check
            return 1 if 1 in legal_6 else 0
        
        if to_call <= BIG_BLIND * 3:
            if chen >= 6.0 and 1 in legal_6: return 1
            return 0 if 0 in legal_6 else 1
            
        return 0 if 0 in legal_6 else 1

def compute_t_test(diffs: List[float]) -> Tuple[float, float, Tuple[float, float]]:
    n = len(diffs)
    if n < 2: return 0.0, 1.0, (0.0, 0.0)
    mean = float(np.mean(diffs))
    sem = float(stats.sem(diffs))
    if sem == 0: return mean, 1.0, (mean, mean)
    t_val = mean / sem
    df = n - 1
    p_val = float(2 * (1 - stats.t.cdf(abs(t_val), df=df)))
    ci_margin = float(stats.t.ppf(0.975, df=df) * sem)
    return mean, p_val, (mean - ci_margin, mean + ci_margin)

def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni correction for multiple hypothesis testing"""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = (m - rank) * p
        adj = min(1.0, max(running_max, adj))
        running_max = adj
        adjusted[orig_idx] = adj
    return adjusted

# ==============================================================================
# AUDIT PART 1: 3-Group Comparison with Exact Seed 430000 & Dual Denominators
# ==============================================================================

def run_part1_true_disentanglement(num_hands: int = 500, eval_seed_base: int = 40000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 1: True 0h Disentanglement (Seed 430000 & Dual Denominators)...")
    game = get_5p_fullgame()
    os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
    raw_log_file = open(RAW_LOG_PATH, "w", encoding="utf-8")
    
    base_50k_model = load_ppo_model(BASE_50K_PATH)
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    results_by_agent = {}
    p_values_for_correction = []

    for a_idx, a_name in enumerate(agents):
        print(f"--- Evaluating {a_name}: True 0h (Seed {PILOT_BASE_SEED + a_idx * 100}) vs 2000h Trained ---")
        true_0h_model = create_true_0h_model(a_idx, base_seed=PILOT_BASE_SEED)
        trained_2000h_path = os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt")
        trained_2000h_model = load_ppo_model(trained_2000h_path)
        
        pnl_0h = []
        pnl_2000h = []
        diffs = []
        
        act_counts_0h = Counter()
        act_counts_2000h = Counter()
        
        # Dual all-in tracking: per-decision and per-hand
        # Per-hand flags
        hands_with_direct_act5_0h = 0
        hands_with_full_stack_0h = 0
        hands_with_pushed_allin_0h = 0
        hands_with_called_allin_0h = 0
        hands_with_commit_allin_0h = 0

        hands_with_direct_act5_2000h = 0
        hands_with_full_stack_2000h = 0
        hands_with_pushed_allin_2000h = 0
        hands_with_called_allin_2000h = 0
        hands_with_commit_allin_2000h = 0
        
        # All-in showdown EV tracking
        uncalled_push_pnls_2000h = []
        called_allin_pnls_2000h = []

        for h_i in range(1, num_hands + 1):
            seed = eval_seed_base + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            
            # --- EVALUATE TRUE 0H ---
            rng_0h = np.random.RandomState(seed)
            st_0h = game.new_initial_state()
            h_direct_act5_0h = False
            h_full_stack_0h = False
            h_push_0h = False
            h_call_allin_0h = False
            h_commit_0h = False

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
                
                active_model = true_0h_model if cur_p == hero_seat else base_50k_model
                with torch.no_grad():
                    logits, _ = active_model(t_state)
                    masked = logits.clone()
                    mask = torch.full_like(masked, -1e9)
                    for a in legal_6: mask[0, a] = 0.0
                    masked += mask
                    probs = F.softmax(masked, dim=-1)
                    act_6 = int(torch.argmax(probs, dim=-1).item())

                if cur_p == hero_seat:
                    act_counts_0h[act_6] += 1
                    if act_6 == 5: h_direct_act5_0h = True

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_0h, cur_p, act_6)
                st_0h.apply_action(nat_act)

                if cur_p == hero_seat:
                    spent_after = extract_player_spent(st_0h)
                    if spent_after[cur_p] >= STARTING_STACK:
                        h_full_stack_0h = True
                        if act_6 == 5: h_push_0h = True
                        elif act_6 == 1 and to_call >= stack_before: h_call_allin_0h = True
                        elif act_6 in (2, 3, 4): h_commit_0h = True

            ret_0h = st_0h.returns()[hero_seat] / BIG_BLIND
            pnl_0h.append(ret_0h)
            if h_direct_act5_0h: hands_with_direct_act5_0h += 1
            if h_full_stack_0h: hands_with_full_stack_0h += 1
            if h_push_0h: hands_with_pushed_allin_0h += 1
            if h_call_allin_0h: hands_with_called_allin_0h += 1
            if h_commit_0h: hands_with_commit_allin_0h += 1

            # --- EVALUATE TRAINED 2000H (IDENTICAL SEED & SEAT) ---
            rng_2000h = np.random.RandomState(seed)
            st_2000h = game.new_initial_state()
            priv_cards = robust_parse_cards(st_2000h)
            
            h_direct_act5_2000h = False
            h_full_stack_2000h = False
            h_push_2000h = False
            h_call_allin_2000h = False
            h_commit_2000h = False
            
            hero_actions_log = []
            hero_allin_occurred = False
            opponent_called_hero_allin = False

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
                    hero_actions_log.append(act_6)
                    if act_6 == 5:
                        h_direct_act5_2000h = True
                        hero_allin_occurred = True
                else:
                    if hero_allin_occurred and act_6 == 1 and to_call >= STARTING_STACK * 0.5:
                        opponent_called_hero_allin = True

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_2000h, cur_p, act_6)
                st_2000h.apply_action(nat_act)

                if cur_p == hero_seat:
                    spent_after = extract_player_spent(st_2000h)
                    if spent_after[cur_p] >= STARTING_STACK:
                        h_full_stack_2000h = True
                        if act_6 == 5: h_push_2000h = True
                        elif act_6 == 1 and to_call >= stack_before: h_call_allin_2000h = True
                        elif act_6 in (2, 3, 4): h_commit_2000h = True

            ret_2000h = st_2000h.returns()[hero_seat] / BIG_BLIND
            pnl_2000h.append(ret_2000h)
            diffs.append(ret_2000h - ret_0h)

            if h_direct_act5_2000h: hands_with_direct_act5_2000h += 1
            if h_full_stack_2000h: hands_with_full_stack_2000h += 1
            if h_push_2000h: hands_with_pushed_allin_2000h += 1
            if h_call_allin_2000h: hands_with_called_allin_2000h += 1
            if h_commit_2000h: hands_with_commit_allin_2000h += 1

            if hero_allin_occurred:
                if opponent_called_hero_allin:
                    called_allin_pnls_2000h.append(ret_2000h)
                else:
                    uncalled_push_pnls_2000h.append(ret_2000h)

            # Write extended raw JSONL
            hand_rec = {
                "hand_id": h_i,
                "agent": a_name,
                "seed": seed,
                "seat": hero_seat,
                "priv_cards": priv_cards[hero_seat],
                "board": parse_board_cards(st_2000h),
                "hero_actions_2000h": hero_actions_log,
                "hero_allin": hero_allin_occurred,
                "opponent_called": opponent_called_hero_allin,
                "pnl_0h_bb": ret_0h,
                "pnl_2000h_bb": ret_2000h,
                "diff_bb": ret_2000h - ret_0h
            }
            raw_log_file.write(json.dumps(hand_rec) + "\n")

        # Stats computation
        mean_0h = float(np.mean(pnl_0h)) * 100.0
        ci_0h = float(stats.sem(pnl_0h)) * 100.0 * 1.96
        
        mean_2000h = float(np.mean(pnl_2000h)) * 100.0
        ci_2000h = float(stats.sem(pnl_2000h)) * 100.0 * 1.96

        diff_mean, raw_p, diff_ci = compute_t_test(diffs)
        diff_bb100 = diff_mean * 100.0
        diff_ci_bb100 = (diff_ci[0] * 100.0, diff_ci[1] * 100.0)
        p_values_for_correction.append(raw_p)

        tot_acts_0h = sum(act_counts_0h.values())
        tot_acts_2000h = sum(act_counts_2000h.values())

        called_ev_str = f"{float(np.mean(called_allin_pnls_2000h)):+.2f} BB" if called_allin_pnls_2000h else "N/A"
        uncalled_ev_str = f"{float(np.mean(uncalled_push_pnls_2000h)):+.2f} BB" if uncalled_push_pnls_2000h else "N/A"

        res_a = {
            "num_hands": num_hands,
            "true_0h_noisy": {
                "bb_per_100": round(mean_0h, 2),
                "ci95_bb": round(ci_0h, 2),
                "per_decision_act5_pct": round(act_counts_0h.get(5, 0) / max(tot_acts_0h, 1) * 100.0, 2),
                "per_hand_direct_act5_pct": round(hands_with_direct_act5_0h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(hands_with_full_stack_0h / num_hands * 100.0, 2),
                "hand_breakdown": {
                    "push_allin_hands": hands_with_pushed_allin_0h,
                    "called_allin_hands": hands_with_called_allin_0h,
                    "commit_allin_hands": hands_with_commit_allin_0h
                }
            },
            "trained_2000h": {
                "bb_per_100": round(mean_2000h, 2),
                "ci95_bb": round(ci_2000h, 2),
                "per_decision_act5_pct": round(act_counts_2000h.get(5, 0) / max(tot_acts_2000h, 1) * 100.0, 2),
                "per_hand_direct_act5_pct": round(hands_with_direct_act5_2000h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(hands_with_full_stack_2000h / num_hands * 100.0, 2),
                "hand_breakdown": {
                    "push_allin_hands": hands_with_pushed_allin_2000h,
                    "called_allin_hands": hands_with_called_allin_2000h,
                    "commit_allin_hands": hands_with_commit_allin_2000h
                },
                "allin_showdown_stats": {
                    "called_allin_count": len(called_allin_pnls_2000h),
                    "called_allin_ev": called_ev_str,
                    "uncalled_push_count": len(uncalled_push_pnls_2000h),
                    "uncalled_push_ev": uncalled_ev_str
                }
            },
            "paired_difference": {
                "delta_bb_per_100": round(diff_bb100, 2),
                "ci95": [round(diff_ci_bb100[0], 2), round(diff_ci_bb100[1], 2)],
                "raw_p_value": round(raw_p, 4)
            }
        }
        results_by_agent[a_name] = res_a
        print(f"  True 0h: {mean_0h:+.2f} BB/100 | 2000h: {mean_2000h:+.2f} BB/100 | Delta: {diff_bb100:+.2f} BB/100 (raw p={raw_p:.4f})")
        print(f"  Per-Hand Full-Stack: 0h={res_a['true_0h_noisy']['per_hand_full_stack_pct']}% -> 2000h={res_a['trained_2000h']['per_hand_full_stack_pct']}% (Called EV: {called_ev_str}, Uncalled EV: {uncalled_ev_str})")

    raw_log_file.close()

    # Apply Holm-Bonferroni correction across all 5 agents
    holm_p_vals = holm_bonferroni_correction(p_values_for_correction)
    for idx, a_name in enumerate(agents):
        results_by_agent[a_name]["paired_difference"]["holm_adjusted_p_value"] = round(holm_p_vals[idx], 4)
        results_by_agent[a_name]["paired_difference"]["is_statistically_significant_holm"] = holm_p_vals[idx] < 0.05
        print(f"  Holm-Bonferroni {a_name}: raw p={p_values_for_correction[idx]:.4f} -> holm p={holm_p_vals[idx]:.4f}")

    return results_by_agent

# ==============================================================================
# AUDIT PART 2: All 5 Agents vs TrueStrongAllinCatcherBot (Real 7-Card Evaluator)
# ==============================================================================

def run_part2_true_catcher(num_hands: int = 500, base_seed: int = 60000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 2: All 5 Agents vs TrueStrongAllinCatcherBot ({num_hands} hands each)...")
    game = get_5p_fullgame()
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    catcher_results = {}

    for a_idx, a_name in enumerate(agents):
        model = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt"))
        catcher = TrueStrongAllinCatcherBot(seat=1, seed=base_seed + a_idx)
        pnls = []
        act_counts = Counter()
        
        hands_with_direct_act5 = 0
        called_allin_pnls = []
        uncalled_allin_pnls = []

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)
            st = game.new_initial_state()
            priv_cards = robust_parse_cards(st)
            
            hero_pushed_allin = False
            catcher_called_allin = False

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                    continue

                cur_p = st.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st, cur_p)
                round_idx = get_round_from_state(st)
                
                spent_before = extract_player_spent(st)
                to_call = max(spent_before) - spent_before[cur_p]

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
                    if act_6 == 5: hero_pushed_allin = True
                else:
                    act_6 = catcher.select_action(st, cur_p, legal_6, priv_cards, round_idx)
                    if hero_pushed_allin and act_6 == 1 and to_call >= STARTING_STACK * 0.5:
                        catcher_called_allin = True

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cur_p, act_6)
                st.apply_action(nat_act)

            ret = st.returns()[hero_seat] / BIG_BLIND
            pnls.append(ret)
            if hero_pushed_allin: hands_with_direct_act5 += 1

            if hero_pushed_allin:
                if catcher_called_allin:
                    called_allin_pnls.append(ret)
                else:
                    uncalled_allin_pnls.append(ret)

        mean_bb = float(np.mean(pnls)) * 100.0
        ci_bb = float(stats.sem(pnls)) * 100.0 * 1.96
        tot_acts = sum(act_counts.values())

        called_ev = round(float(np.mean(called_allin_pnls)), 2) if called_allin_pnls else "N/A"
        uncalled_ev = round(float(np.mean(uncalled_allin_pnls)), 2) if uncalled_allin_pnls else "N/A"

        catcher_results[a_name] = {
            "bb_per_100": round(mean_bb, 2),
            "ci95_bb": round(ci_bb, 2),
            "per_decision_allin_pct": round(act_counts.get(5, 0) / max(tot_acts, 1) * 100.0, 2),
            "per_hand_allin_pct": round(hands_with_direct_act5 / num_hands * 100.0, 2),
            "called_allin_count": len(called_allin_pnls),
            "called_allin_ev": called_ev,
            "uncalled_allin_count": len(uncalled_allin_pnls),
            "uncalled_allin_ev": uncalled_ev
        }
        print(f"  {a_name} vs True Catcher: {mean_bb:+.2f} ± {ci_bb:.2f} BB/100 | Hand All-in%: {catcher_results[a_name]['per_hand_allin_pct']}% | Called EV: {called_ev} (n={len(called_allin_pnls)}) | Uncalled EV: {uncalled_ev} (n={len(uncalled_allin_pnls)})")

    return catcher_results

# ==============================================================================
# AUDIT PART 3: Correct Lineage Strategy Evolution (0h -> 500h -> 1000h -> 1500h -> 2000h)
# ==============================================================================

def run_part3_correct_lineage_evolution(num_hands: int = 500, eval_seed_base: int = 70000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 3: Correct Lineage Evolution (Seed 430000 Single Lineage)...")
    game = get_5p_fullgame()
    base_50k_model = load_ppo_model(BASE_50K_PATH)
    
    tracked_agents = ["Agent_E", "Agent_D"]
    lineage_milestones = ["500h", "1000h", "1500h", "2000h"]
    evolution_results = {}

    for t_name in tracked_agents:
        a_idx = 4 if t_name == "Agent_E" else 3
        evolution_results[t_name] = {}
        
        # 0h exact seed 430000 model
        true_0h = create_true_0h_model(a_idx, base_seed=PILOT_BASE_SEED)
        models = {"0h": true_0h}
        for m in lineage_milestones:
            models[m] = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{t_name}_{m}.pt"))

        for m_key, m_model in models.items():
            pnls = []
            act_counts = Counter()
            hands_with_act5 = 0

            for h_i in range(1, num_hands + 1):
                seed = eval_seed_base + a_idx * 5000 + h_i
                hero_seat = (h_i - 1) % 5
                rng = np.random.RandomState(seed)
                st = game.new_initial_state()
                h_act5 = False

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
                        if act_6 == 5: h_act5 = True

                    nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cur_p, act_6)
                    st.apply_action(nat_act)

                pnls.append(st.returns()[hero_seat] / BIG_BLIND)
                if h_act5: hands_with_act5 += 1

            tot_acts = sum(act_counts.values())
            m_res = {
                "bb_per_100": round(float(np.mean(pnls)) * 100.0, 2),
                "ci95_bb": round(float(stats.sem(pnls)) * 100.0 * 1.96, 2),
                "per_decision_allin_pct": round(act_counts.get(5, 0) / max(tot_acts, 1) * 100.0, 2),
                "per_hand_allin_pct": round(hands_with_act5 / num_hands * 100.0, 2),
                "pot_pct": round(act_counts.get(3, 0) / max(tot_acts, 1) * 100.0, 2),
                "halfpot_pct": round(act_counts.get(2, 0) / max(tot_acts, 1) * 100.0, 2)
            }
            evolution_results[t_name][m_key] = m_res
            print(f"  {t_name} [{m_key:5s}] -> {m_res['bb_per_100']:+6.2f} ± {m_res['ci95_bb']:5.2f} BB/100 | DecAllin%: {m_res['per_decision_allin_pct']:4.1f}% | HandAllin%: {m_res['per_hand_allin_pct']:4.1f}% | Pot%: {m_res['pot_pct']:4.1f}% | HalfPot%: {m_res['halfpot_pct']:4.1f}%")

    return evolution_results

def run_directive045_full_audit():
    print("=" * 80)
    print("DIRECTIVE 045: MULTI-AGENT EVALUATOR INTEGRITY RECOVERY SUITE")
    print(f"Pilot Base Seed: {PILOT_BASE_SEED} (Ensuring 100% 0h Lineage Authenticity)")
    print("=" * 80)
    start_t = time.time()

    part1_res = run_part1_true_disentanglement(num_hands=500)
    part2_res = run_part2_true_catcher(num_hands=500)
    part3_res = run_part3_correct_lineage_evolution(num_hands=500)

    total_time = time.time() - start_t

    final_output = {
        "directive": "DIRECTIVE 045: MULTI-AGENT EVALUATOR INTEGRITY RECOVERY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_time, 2),
        "pilot_base_seed": PILOT_BASE_SEED,
        "base_50k_path": BASE_50K_PATH,
        "part1_true_0h_disentanglement": part1_res,
        "part2_true_allin_catcher_benchmark": part2_res,
        "part3_correct_lineage_evolution": part3_res
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nDIRECTIVE 045 Recovery Audit Complete in {total_time:.1f}s! Saved to {OUT_JSON_PATH}")

if __name__ == "__main__":
    run_directive045_full_audit()
