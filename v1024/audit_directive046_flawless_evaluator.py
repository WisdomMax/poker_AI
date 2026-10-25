#!/usr/bin/env python3
"""
DIRECTIVE 046: Flawless Self-Play Evaluator Engine
1. Enforces advance_chance_nodes() BEFORE calling safe_parse_cards_v46()
2. Full 10-hole-card deduplication check on every single hand
3. Bit-exact 0h initialization matching 043 self-play pilot (seed = 430000 + i * 100)
4. True 7-card evaluator (eval_7cards) StrongAllinCatcherBot (Two Pair+ only)
5. Rigorous All-in Call Ledger & Positive Uncalled EV Tracking
6. Dual Denominators (Per-Decision vs Per-Hand)
7. Single-lineage Milestone Evolution (0h -> 500h -> 1000h -> 1500h -> 2000h)
8. Holm-Bonferroni Multiple Testing Correction
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.tests.test_directive040_regression import (
    calc_chen_score, parse_board_cards
)
from v1024.tests.test_directive046_evaluator_integrity import (
    safe_parse_cards_v46, card_str_to_int
)
from v1024.alphago_5p_selfplay import (
    AGENT_NAMES, IndependentPPOAgent
)

BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
CHECKPOINT_DIR = "v1024/models_alphago"
RAW_LOG_PATH = "v1024/data/directive046_raw_audit_hands.jsonl"
OUT_JSON_PATH = "v1024/directive046_audit_results.json"

STARTING_STACK = 10000.0
BIG_BLIND = 100.0
PILOT_BASE_SEED = 430000

def load_ppo_model(ckpt_path: str, device: str = "cpu") -> PPONeuralPlayer:
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def create_flawless_0h_model(agent_id: int, base_ckpt_path: str = BASE_50K_PATH,
                             base_seed: int = PILOT_BASE_SEED, device: str = "cpu") -> PPONeuralPlayer:
    """Uses exact 043 IndependentPPOAgent initialization sequence"""
    seed = base_seed + agent_id * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    state_dict = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * 0.005 * (agent_id + 1)
            p.add_(noise)
    model.eval()
    model.to(device)
    return model

class FlawlessTwoPairCatcherBot:
    """
    Legitimate 7-card hand evaluation bot.
    - Preflop: Only calls all-ins with AA, KK, QQ, AKs (Chen >= 11.0)
    - Postflop: Evaluates 7 cards using eval_7cards. Only calls all-ins with Two Pair+ (rank >= 2)
    - Free Check (to_call == 0): Check (Action 1 or 0)
    - Normal bets (to_call <= 300): Calls with Chen >= 6.0
    - Otherwise: Folds (Action 0)
    """
    name = "FlawlessTwoPairCatcher"

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

        # Facing All-in / Huge Bet
        if is_allin_facing and to_call > 0:
            if round_idx == 0:
                # Preflop: AA, KK, QQ, AKs only
                if chen >= 11.0 and 1 in legal_6:
                    return 1
                return 0 if 0 in legal_6 else 1
            else:
                # Postflop: Real 7-card hand rank
                if len(board) >= 3:
                    int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
                    rank = eval_7cards(int_cards)[0]
                else:
                    rank = 0
                if rank >= 2 and 1 in legal_6:
                    return 1 # Calls with Two Pair or better
                return 0 if 0 in legal_6 else 1

        # Free Check
        if to_call == 0:
            if 0 in legal_6 and self.rng.random() < 0.8: return 0
            return 1 if 1 in legal_6 else 0
        
        # Moderate Bet
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
# PART 1: True 0h Disentanglement (Seed 430000 & Dual Denominators)
# ==============================================================================

def run_flawless_part1(num_hands: int = 500, eval_seed_base: int = 40000,
                       raw_log_file = None) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 1: Flawless 0h Disentanglement ({num_hands} hands per agent)...")
    game = get_5p_fullgame()
    base_50k_model = load_ppo_model(BASE_50K_PATH)
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    results_by_agent = {}
    p_values = []

    for a_idx, a_name in enumerate(agents):
        print(f"--- Benchmarking {a_name}: Exact 0h (Seed {PILOT_BASE_SEED + a_idx * 100}) vs 2000h Trained ---")
        true_0h_model = create_flawless_0h_model(a_idx)
        trained_2000h_path = os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt")
        trained_2000h_model = load_ppo_model(trained_2000h_path)

        pnl_0h = []
        pnl_2000h = []
        diffs = []
        
        act_counts_0h = Counter()
        act_counts_2000h = Counter()
        
        hands_with_act5_0h = 0
        hands_with_full_stack_0h = 0
        hands_with_act5_2000h = 0
        hands_with_full_stack_2000h = 0

        called_allin_pnls = []
        uncalled_push_pnls = []

        for h_i in range(1, num_hands + 1):
            seed = eval_seed_base + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5

            # --- EVALUATE TRUE 0H ---
            rng_0h = np.random.RandomState(seed)
            st_0h = game.new_initial_state()
            advance_chance_nodes(st_0h, rng_0h) # Advance cards FIRST!
            cards_0h = safe_parse_cards_v46(st_0h) # Verified non-dummy!

            h_act5_0h = False
            h_full_0h = False

            while not st_0h.is_terminal():
                if st_0h.is_chance_node():
                    advance_chance_nodes(st_0h, rng_0h)
                    continue

                cur_p = st_0h.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st_0h, cur_p)
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
                    if act_6 == 5: h_act5_0h = True

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_0h, cur_p, act_6)
                st_0h.apply_action(nat_act)

                if cur_p == hero_seat:
                    if extract_player_spent(st_0h)[cur_p] >= STARTING_STACK:
                        h_full_0h = True

            ret_0h = st_0h.returns()[hero_seat] / BIG_BLIND
            pnl_0h.append(ret_0h)
            if h_act5_0h: hands_with_act5_0h += 1
            if h_full_0h: hands_with_full_stack_0h += 1

            # --- EVALUATE TRAINED 2000H (IDENTICAL SEED & SEAT) ---
            rng_2000h = np.random.RandomState(seed)
            st_2000h = game.new_initial_state()
            advance_chance_nodes(st_2000h, rng_2000h) # Advance cards FIRST!
            cards_2000h = safe_parse_cards_v46(st_2000h) # Verified non-dummy!

            h_act5_2000h = False
            h_full_2000h = False
            hero_pushed_allin = False
            folded_seats_2000h = set()

            hero_acts = []

            while not st_2000h.is_terminal():
                if st_2000h.is_chance_node():
                    advance_chance_nodes(st_2000h, rng_2000h)
                    continue

                cur_p = st_2000h.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st_2000h, cur_p)

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
                    hero_acts.append(act_6)
                    if act_6 == 5:
                        h_act5_2000h = True
                        hero_pushed_allin = True
                
                if act_6 == 0:
                    folded_seats_2000h.add(cur_p)

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_2000h, cur_p, act_6)
                st_2000h.apply_action(nat_act)

                if cur_p == hero_seat:
                    if extract_player_spent(st_2000h)[cur_p] >= STARTING_STACK:
                        h_full_2000h = True

            ret_2000h = st_2000h.returns()[hero_seat] / BIG_BLIND
            pnl_2000h.append(ret_2000h)
            diffs.append(ret_2000h - ret_0h)

            if h_act5_2000h: hands_with_act5_2000h += 1
            if h_full_2000h: hands_with_full_stack_2000h += 1

            opp_called_hero_allin = (len(folded_seats_2000h) < 4)

            if hero_pushed_allin:
                if opp_called_hero_allin:
                    called_allin_pnls.append(ret_2000h)
                else:
                    # Uncalled push MUST be positive (blinds won)
                    assert ret_2000h > 0, f"Uncalled allin had negative return {ret_2000h} BB on hand {h_i}"
                    uncalled_push_pnls.append(ret_2000h)

            if raw_log_file:
                rec = {
                    "hand_id": h_i,
                    "agent": a_name,
                    "seed": seed,
                    "seat": hero_seat,
                    "priv_cards": cards_2000h[hero_seat],
                    "board": parse_board_cards(st_2000h),
                    "hero_actions": hero_acts,
                    "hero_allin": hero_pushed_allin,
                    "opp_called": opp_called_hero_allin,
                    "pnl_0h_bb": ret_0h,
                    "pnl_2000h_bb": ret_2000h,
                    "diff_bb": ret_2000h - ret_0h
                }
                raw_log_file.write(json.dumps(rec) + "\n")

        mean_0h = float(np.mean(pnl_0h)) * 100.0
        ci_0h = float(stats.sem(pnl_0h)) * 100.0 * 1.96
        mean_2000h = float(np.mean(pnl_2000h)) * 100.0
        ci_2000h = float(stats.sem(pnl_2000h)) * 100.0 * 1.96

        diff_mean, raw_p, diff_ci = compute_t_test(diffs)
        diff_bb100 = diff_mean * 100.0
        diff_ci_bb100 = (diff_ci[0] * 100.0, diff_ci[1] * 100.0)
        p_values.append(raw_p)

        tot_acts_0h = sum(act_counts_0h.values())
        tot_acts_2000h = sum(act_counts_2000h.values())

        called_ev_str = f"{float(np.mean(called_allin_pnls)):+.2f} BB" if called_allin_pnls else "N/A"
        uncalled_ev_str = f"{float(np.mean(uncalled_push_pnls)):+.2f} BB" if uncalled_push_pnls else "N/A"

        res_a = {
            "num_hands": num_hands,
            "true_0h_noisy": {
                "bb_per_100": round(mean_0h, 2),
                "ci95_bb": round(ci_0h, 2),
                "per_decision_act5_pct": round(act_counts_0h.get(5, 0) / max(tot_acts_0h, 1) * 100.0, 2),
                "per_hand_direct_act5_pct": round(hands_with_act5_0h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(hands_with_full_stack_0h / num_hands * 100.0, 2)
            },
            "trained_2000h": {
                "bb_per_100": round(mean_2000h, 2),
                "ci95_bb": round(ci_2000h, 2),
                "per_decision_act5_pct": round(act_counts_2000h.get(5, 0) / max(tot_acts_2000h, 1) * 100.0, 2),
                "per_hand_direct_act5_pct": round(hands_with_act5_2000h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(hands_with_full_stack_2000h / num_hands * 100.0, 2),
                "allin_showdown": {
                    "called_count": len(called_allin_pnls),
                    "called_ev": called_ev_str,
                    "uncalled_count": len(uncalled_push_pnls),
                    "uncalled_ev": uncalled_ev_str
                }
            },
            "paired_diff": {
                "delta_bb_per_100": round(diff_bb100, 2),
                "ci95": [round(diff_ci_bb100[0], 2), round(diff_ci_bb100[1], 2)],
                "raw_p_value": round(raw_p, 4)
            }
        }
        results_by_agent[a_name] = res_a
        print(f"  True 0h: {mean_0h:+.2f} BB/100 | 2000h: {mean_2000h:+.2f} BB/100 | Delta: {diff_bb100:+.2f} BB/100 (p={raw_p:.4f})")
        print(f"  Full Stack: 0h={res_a['true_0h_noisy']['per_hand_full_stack_pct']}% -> 2000h={res_a['trained_2000h']['per_hand_full_stack_pct']}% | Called EV: {called_ev_str} | Uncalled EV: {uncalled_ev_str}")

    holm_ps = holm_bonferroni_correction(p_values)
    for idx, a_name in enumerate(agents):
        results_by_agent[a_name]["paired_diff"]["holm_p_value"] = round(holm_ps[idx], 4)
        print(f"  Holm {a_name}: raw p={p_values[idx]:.4f} -> holm p={holm_ps[idx]:.4f}")

    return results_by_agent

# ==============================================================================
# PART 2: All 5 Agents vs FlawlessTwoPairCatcherBot (True 7-Card Evaluator)
# ==============================================================================

def run_flawless_part2(num_hands: int = 500, base_seed: int = 60000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 2: All 5 Agents vs FlawlessTwoPairCatcherBot ({num_hands} hands each)...")
    game = get_5p_fullgame()
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    catcher_results = {}

    for a_idx, a_name in enumerate(agents):
        model = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt"))
        catcher = FlawlessTwoPairCatcherBot(seat=1, seed=base_seed + a_idx)
        pnls = []
        act_counts = Counter()
        hands_with_act5 = 0

        called_allin_pnls = []
        uncalled_allin_pnls = []

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)
            st = game.new_initial_state()
            advance_chance_nodes(st, rng) # Advance cards FIRST!
            cards_map = safe_parse_cards_v46(st) # Verified legitimate cards!

            hero_pushed_allin = False
            folded_seats = set()

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
                    if act_6 == 5: hero_pushed_allin = True
                else:
                    act_6 = catcher.select_action(st, cur_p, legal_6, cards_map, round_idx)
                
                if act_6 == 0:
                    folded_seats.add(cur_p)

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cur_p, act_6)
                st.apply_action(nat_act)

            ret = st.returns()[hero_seat] / BIG_BLIND
            pnls.append(ret)
            if hero_pushed_allin: hands_with_act5 += 1

            catcher_called_allin = (len(folded_seats) < 4)

            if hero_pushed_allin:
                if catcher_called_allin:
                    called_allin_pnls.append(ret)
                else:
                    uncalled_allin_pnls.append(ret)

        mean_bb = float(np.mean(pnls)) * 100.0
        ci_bb = float(stats.sem(pnls)) * 100.0 * 1.96
        tot_acts = sum(act_counts.values())

        called_ev_str = f"{float(np.mean(called_allin_pnls)):+.2f} BB" if called_allin_pnls else "N/A"
        uncalled_ev_str = f"{float(np.mean(uncalled_allin_pnls)):+.2f} BB" if uncalled_allin_pnls else "N/A"

        catcher_results[a_name] = {
            "bb_per_100": round(mean_bb, 2),
            "ci95_bb": round(ci_bb, 2),
            "per_decision_allin_pct": round(act_counts.get(5, 0) / max(tot_acts, 1) * 100.0, 2),
            "per_hand_allin_pct": round(hands_with_act5 / num_hands * 100.0, 2),
            "called_allin_count": len(called_allin_pnls),
            "called_allin_ev": called_ev_str,
            "uncalled_allin_count": len(uncalled_allin_pnls),
            "uncalled_allin_ev": uncalled_ev_str
        }
        print(f"  {a_name} vs Flawless Catcher: {mean_bb:+.2f} ± {ci_bb:.2f} BB/100 | Hand All-in%: {catcher_results[a_name]['per_hand_allin_pct']}% | Called EV: {called_ev_str} (n={len(called_allin_pnls)}) | Uncalled EV: {uncalled_ev_str} (n={len(uncalled_allin_pnls)})")

    return catcher_results

# ==============================================================================
# PART 3: Correct Lineage Evolution (Seed 430000: 0h -> 500h -> ... -> 2000h)
# ==============================================================================

def run_flawless_part3(num_hands: int = 500, eval_seed_base: int = 70000) -> Dict[str, any]:
    print(f"\n>>> Running Audit Part 3: Correct Lineage Evolution (Seed {PILOT_BASE_SEED} Single Lineage)...")
    game = get_5p_fullgame()
    base_50k_model = load_ppo_model(BASE_50K_PATH)
    
    tracked_agents = ["Agent_E", "Agent_D"]
    lineage_milestones = ["500h", "1000h", "1500h", "2000h"]
    evolution_results = {}

    for t_name in tracked_agents:
        a_idx = 4 if t_name == "Agent_E" else 3
        evolution_results[t_name] = {}
        
        true_0h = create_flawless_0h_model(a_idx)
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
                advance_chance_nodes(st, rng) # Advance cards FIRST!
                _ = safe_parse_cards_v46(st)

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

def run_directive046_full_audit():
    print("=" * 80)
    print("DIRECTIVE 046: FLAWLESS SELF-PLAY EVALUATOR AUDIT SUITE")
    print(f"Pilot Base Seed: {PILOT_BASE_SEED}")
    print("=" * 80)
    start_t = time.time()
    
    os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
    raw_log = open(RAW_LOG_PATH, "w", encoding="utf-8")

    part1_res = run_flawless_part1(num_hands=500, raw_log_file=raw_log)
    part2_res = run_flawless_part2(num_hands=500)
    part3_res = run_flawless_part3(num_hands=500)

    raw_log.close()
    total_time = time.time() - start_t

    final_output = {
        "directive": "DIRECTIVE 046: SELF-PLAY EVALUATOR FINAL INTEGRITY GATE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_time, 2),
        "pilot_base_seed": PILOT_BASE_SEED,
        "base_50k_path": BASE_50K_PATH,
        "part1_flawless_0h_disentanglement": part1_res,
        "part2_flawless_catcher_benchmark": part2_res,
        "part3_correct_lineage_evolution": part3_res
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nDIRECTIVE 046 Complete in {total_time:.1f}s! Results saved to {OUT_JSON_PATH}")

if __name__ == "__main__":
    run_directive046_full_audit()
