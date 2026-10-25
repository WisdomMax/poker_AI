#!/usr/bin/env python3
"""
DIRECTIVE 047: Rigorous Independent Evaluation Suite
- Fixes FlawlessTwoPairCatcherBot free-check bug (Action 1 always on to_call==0).
- Uses HandLedgerTracker (Zero false-positive opp_called flags).
- Evaluates 7,000h models against:
  1. FlawlessTwoPairCatcherBot (500 hands each)
  2. Bit-exact 0h Noisy & 2000h Paired Benchmarks (500 hands each)
  3. 7,000h Round-Robin Self-Play League (1,000 hands)
- Generates v1024/directive047_audit_results.json and v1024/data/directive047_raw_audit_hands.jsonl
"""
import sys
import os
import time
import json
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v1024.tests.test_directive046_evaluator_integrity import safe_parse_cards_v46
from v1024.integrity_two_pair_catcher import (
    FlawlessTwoPairCatcherBot, card_str_to_int, parse_board_cards
)
from v1024.integrity_ledger_tracker import HandLedgerTracker, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from v1024.alphago_5p_selfplay import IndependentPPOAgent, AGENT_NAMES

BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
CHECKPOINT_DIR = "v1024/models_alphago"
RAW_AUDIT_JSONL = "v1024/data/directive047_raw_audit_hands.jsonl"
AUDIT_JSON_OUT = "v1024/directive047_audit_results.json"

PILOT_BASE_SEED = 430000

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

def load_ppo_model(path: str) -> PPONeuralPlayer:
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.eval()
    return model

def create_flawless_0h_model(agent_id: int) -> PPONeuralPlayer:
    seed = PILOT_BASE_SEED + agent_id * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    sd = torch.load(BASE_50K_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * 0.005 * (agent_id + 1)
            p.add_(noise)
    model.eval()
    return model

# ==============================================================================
# PART 1: 7,000h Models vs FlawlessTwoPairCatcherBot (Bug-Free, Zero Free Folds)
# ==============================================================================
def run_part1_catcher_evaluation(num_hands: int = 500, base_seed: int = 60000) -> Dict[str, Any]:
    print(f"\n>>> Running Audit Part 1: All 5 Agents (7,000h) vs FlawlessTwoPairCatcherBot ({num_hands} hands each)...")
    game = get_5p_fullgame()
    catcher_results = {}

    for a_idx, a_name in enumerate(AGENT_NAMES):
        ckpt_7000h = os.path.join(CHECKPOINT_DIR, f"{a_name}_7000h.pt")
        model = load_ppo_model(ckpt_7000h)
        catcher = FlawlessTwoPairCatcherBot(seat=1, seed=base_seed + a_idx)

        pnls = []
        act_counts = Counter()
        hands_with_act5 = 0
        called_allin_pnls = []
        uncalled_allin_pnls = []
        false_positive_calls = 0

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)
            st = game.new_initial_state()
            advance_chance_nodes(st, rng)
            cards_map = safe_parse_cards_v46(st)

            tracker = HandLedgerTracker(num_players=5)
            hero_pushed_allin = False

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                    continue

                cp = st.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                round_idx = get_round_from_state(st)

                if cp == hero_seat:
                    raw_inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)
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
                    act_6 = catcher.select_action(st, cp, legal_6, cards_map, round_idx)

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                spent_after = extract_player_spent(st)[cp]
                tracker.record_action(cp, act_6, nat_act, round_idx, spent_after)

            ret_bb = st.returns()[hero_seat] / BIG_BLIND
            pnls.append(ret_bb)
            if hero_pushed_allin: hands_with_act5 += 1

            hero_allin = tracker.is_player_allin(hero_seat)
            opp_called = tracker.was_allin_called(hero_seat)

            if not hero_allin and opp_called:
                false_positive_calls += 1

            if hero_allin:
                if opp_called:
                    called_allin_pnls.append(ret_bb)
                else:
                    uncalled_allin_pnls.append(ret_bb)

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
            "uncalled_allin_ev": uncalled_ev_str,
            "false_positive_calls": false_positive_calls
        }
        print(f"  {a_name} (7000h) vs Flawless Catcher: {mean_bb:+.2f} ± {ci_bb:.2f} BB/100 | All-in%: {catcher_results[a_name]['per_hand_allin_pct']}% | Called EV: {called_ev_str} (n={len(called_allin_pnls)}) | Uncalled EV: {uncalled_ev_str} (n={len(uncalled_allin_pnls)}) | FalsePos: {false_positive_calls}")

    return catcher_results

# ==============================================================================
# PART 2: Paired Benchmark: 7,000h vs 0h (Bit-exact Noisy) & 2,000h
# ==============================================================================
def run_part2_paired_evaluation(num_hands: int = 500, base_seed: int = 40000) -> Dict[str, Any]:
    print(f"\n>>> Running Audit Part 2: Paired Benchmark (7,000h vs 0h & 2,000h across {num_hands} hands)...")
    game = get_5p_fullgame()
    base_model = load_ppo_model(BASE_50K_PATH)
    results = {}
    p_values_0h_7000h = []

    raw_file = open(RAW_AUDIT_JSONL, "w", encoding="utf-8")

    for a_idx, a_name in enumerate(AGENT_NAMES):
        model_0h = create_flawless_0h_model(a_idx)
        model_2000h = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt"))
        model_7000h = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_7000h.pt"))

        pnl_0h = []
        pnl_2000h = []
        pnl_7000h = []
        diffs_0h_7000h = []
        diffs_2000h_7000h = []

        act_cnt_0h = Counter()
        act_cnt_2000h = Counter()
        act_cnt_7000h = Counter()

        act5_hands_0h = 0
        act5_hands_2000h = 0
        act5_hands_7000h = 0

        full_hands_0h = 0
        full_hands_2000h = 0
        full_hands_7000h = 0

        called_allin_pnls_7000h = []
        uncalled_allin_pnls_7000h = []

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)

            # Play hand under 0h, 2000h, 7000h with IDENTICAL initial state & chance cards
            st_init = game.new_initial_state()
            advance_chance_nodes(st_init, rng)
            cards_map = safe_parse_cards_v46(st_init)

            # --- RUN 0h ---
            st_0 = st_init.clone()
            rng_0 = np.random.RandomState(seed)
            h_act5_0 = False
            h_full_0 = False
            while not st_0.is_terminal():
                if st_0.is_chance_node(): advance_chance_nodes(st_0, rng_0); continue
                cp = st_0.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st_0, cp)
                norm_inf = normalize_inf_state_v79(np.array(st_0.information_state_tensor(cp), dtype=np.float32))
                curr_m = model_0h if cp == hero_seat else base_model
                with torch.no_grad():
                    logits, _ = curr_m(torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    masked += mask
                    act = int(torch.argmax(F.softmax(masked, dim=-1), dim=-1).item())
                if cp == hero_seat:
                    act_cnt_0h[act] += 1
                    if act == 5: h_act5_0 = True
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_0, cp, act)
                st_0.apply_action(nat_act)
            ret_0 = st_0.returns()[hero_seat] / BIG_BLIND
            if extract_player_spent(st_0)[hero_seat] >= STARTING_STACK * 0.99: h_full_0 = True
            pnl_0h.append(ret_0)
            if h_act5_0: act5_hands_0h += 1
            if h_full_0: full_hands_0h += 1

            # --- RUN 2000h ---
            st_2 = st_init.clone()
            rng_2 = np.random.RandomState(seed)
            h_act5_2 = False
            h_full_2 = False
            while not st_2.is_terminal():
                if st_2.is_chance_node(): advance_chance_nodes(st_2, rng_2); continue
                cp = st_2.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st_2, cp)
                norm_inf = normalize_inf_state_v79(np.array(st_2.information_state_tensor(cp), dtype=np.float32))
                curr_m = model_2000h if cp == hero_seat else base_model
                with torch.no_grad():
                    logits, _ = curr_m(torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    masked += mask
                    act = int(torch.argmax(F.softmax(masked, dim=-1), dim=-1).item())
                if cp == hero_seat:
                    act_cnt_2000h[act] += 1
                    if act == 5: h_act5_2 = True
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_2, cp, act)
                st_2.apply_action(nat_act)
            ret_2 = st_2.returns()[hero_seat] / BIG_BLIND
            if extract_player_spent(st_2)[hero_seat] >= STARTING_STACK * 0.99: h_full_2 = True
            pnl_2000h.append(ret_2)
            if h_act5_2: act5_hands_2000h += 1
            if h_full_2: full_hands_2000h += 1

            # --- RUN 7000h (with HandLedgerTracker) ---
            st_7 = st_init.clone()
            rng_7 = np.random.RandomState(seed)
            h_act5_7 = False
            h_full_7 = False
            tracker7 = HandLedgerTracker(num_players=5)
            hero_acts_7 = []

            while not st_7.is_terminal():
                if st_7.is_chance_node(): advance_chance_nodes(st_7, rng_7); continue
                cp = st_7.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st_7, cp)
                round_idx = get_round_from_state(st_7)
                norm_inf = normalize_inf_state_v79(np.array(st_7.information_state_tensor(cp), dtype=np.float32))
                curr_m = model_7000h if cp == hero_seat else base_model
                with torch.no_grad():
                    logits, _ = curr_m(torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    masked += mask
                    act = int(torch.argmax(F.softmax(masked, dim=-1), dim=-1).item())
                if cp == hero_seat:
                    act_cnt_7000h[act] += 1
                    hero_acts_7.append(act)
                    if act == 5: h_act5_7 = True
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_7, cp, act)
                st_7.apply_action(nat_act)
                spent_after = extract_player_spent(st_7)[cp]
                tracker7.record_action(cp, act, nat_act, round_idx, spent_after)

            ret_7 = st_7.returns()[hero_seat] / BIG_BLIND
            if extract_player_spent(st_7)[hero_seat] >= STARTING_STACK * 0.99: h_full_7 = True
            pnl_7000h.append(ret_7)
            if h_act5_7: act5_hands_7000h += 1
            if h_full_7: full_hands_7000h += 1

            diff_0_7 = ret_7 - ret_0
            diff_2_7 = ret_7 - ret_2
            diffs_0h_7000h.append(diff_0_7)
            diffs_2000h_7000h.append(diff_2_7)

            hero_allin_7 = tracker7.is_player_allin(hero_seat)
            opp_called_7 = tracker7.was_allin_called(hero_seat)

            if hero_allin_7:
                if opp_called_7:
                    called_allin_pnls_7000h.append(ret_7)
                else:
                    assert ret_7 > 0, f"Uncalled all-in must have positive return on hand {h_i}"
                    uncalled_allin_pnls_7000h.append(ret_7)

            # Write verified raw JSONL record
            raw_rec = {
                "hand_id": h_i,
                "agent": a_name,
                "seed": seed,
                "seat": hero_seat,
                "priv_cards": cards_map[hero_seat],
                "board": parse_board_cards(st_7),
                "hero_actions_7000h": hero_acts_7,
                "hero_allin_7000h": hero_allin_7,
                "opp_called_7000h": opp_called_7,
                "pnl_0h_bb": ret_0,
                "pnl_2000h_bb": ret_2,
                "pnl_7000h_bb": ret_7,
                "diff_0h_to_7000h_bb": diff_0_7,
                "diff_2000h_to_7000h_bb": diff_2_7
            }
            raw_file.write(json.dumps(raw_rec) + "\n")

        mean_0h = float(np.mean(pnl_0h)) * 100.0
        ci_0h = float(stats.sem(pnl_0h)) * 100.0 * 1.96
        mean_2000h = float(np.mean(pnl_2000h)) * 100.0
        ci_2000h = float(stats.sem(pnl_2000h)) * 100.0 * 1.96
        mean_7000h = float(np.mean(pnl_7000h)) * 100.0
        ci_7000h = float(stats.sem(pnl_7000h)) * 100.0 * 1.96

        diff_mean_0_7, raw_p_0_7, diff_ci_0_7 = compute_t_test(diffs_0h_7000h)
        diff_mean_2_7, raw_p_2_7, diff_ci_2_7 = compute_t_test(diffs_2000h_7000h)
        p_values_0h_7000h.append(raw_p_0_7)

        tot_acts_0h = sum(act_cnt_0h.values())
        tot_acts_2000h = sum(act_cnt_2000h.values())
        tot_acts_7000h = sum(act_cnt_7000h.values())

        called_ev_str = f"{float(np.mean(called_allin_pnls_7000h)):+.2f} BB" if called_allin_pnls_7000h else "N/A"
        uncalled_ev_str = f"{float(np.mean(uncalled_allin_pnls_7000h)):+.2f} BB" if uncalled_allin_pnls_7000h else "N/A"

        results[a_name] = {
            "num_hands": num_hands,
            "0h_true_noisy": {
                "bb_per_100": round(mean_0h, 2), "ci95_bb": round(ci_0h, 2),
                "per_hand_allin_pct": round(act5_hands_0h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(full_hands_0h / num_hands * 100.0, 2)
            },
            "2000h_trained": {
                "bb_per_100": round(mean_2000h, 2), "ci95_bb": round(ci_2000h, 2),
                "per_hand_allin_pct": round(act5_hands_2000h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(full_hands_2000h / num_hands * 100.0, 2)
            },
            "7000h_extended": {
                "bb_per_100": round(mean_7000h, 2), "ci95_bb": round(ci_7000h, 2),
                "per_decision_allin_pct": round(act_cnt_7000h.get(5, 0) / max(tot_acts_7000h, 1) * 100.0, 2),
                "per_hand_allin_pct": round(act5_hands_7000h / num_hands * 100.0, 2),
                "per_hand_full_stack_pct": round(full_hands_7000h / num_hands * 100.0, 2),
                "halfpot_pct": round(act_cnt_7000h.get(2, 0) / max(tot_acts_7000h, 1) * 100.0, 2),
                "pot_pct": round(act_cnt_7000h.get(3, 0) / max(tot_acts_7000h, 1) * 100.0, 2),
                "called_allin_count": len(called_allin_pnls_7000h),
                "called_allin_ev": called_ev_str,
                "uncalled_allin_count": len(uncalled_allin_pnls_7000h),
                "uncalled_allin_ev": uncalled_ev_str
            },
            "paired_diff_0h_to_7000h": {
                "delta_bb_per_100": round(diff_mean_0_7 * 100.0, 2),
                "ci95": [round(diff_ci_0_7[0] * 100.0, 2), round(diff_ci_0_7[1] * 100.0, 2)],
                "raw_p_value": round(raw_p_0_7, 4)
            },
            "paired_diff_2000h_to_7000h": {
                "delta_bb_per_100": round(diff_mean_2_7 * 100.0, 2),
                "ci95": [round(diff_ci_2_7[0] * 100.0, 2), round(diff_ci_2_7[1] * 100.0, 2)],
                "raw_p_value": round(raw_p_2_7, 4)
            }
        }
        print(f"  {a_name}: 0h ({mean_0h:+.2f}) -> 2000h ({mean_2000h:+.2f}) -> 7000h ({mean_7000h:+.2f} ± {ci_7000h:.2f} BB/100) | Delta (0h->7000h): {diff_mean_0_7 * 100.0:+.2f} BB (p={raw_p_0_7:.4f})")

    raw_file.close()

    # Apply Holm-Bonferroni correction
    holm_ps = holm_bonferroni_correction(p_values_0h_7000h)
    for idx, a_name in enumerate(AGENT_NAMES):
        results[a_name]["paired_diff_0h_to_7000h"]["holm_p_value"] = round(holm_ps[idx], 4)

    return results

# ==============================================================================
# PART 3: 5-Agent Round-Robin Self-Play League (1,000 hands)
# ==============================================================================
def run_part3_league(num_hands: int = 1000, base_seed: int = 90000) -> Dict[str, Any]:
    print(f"\n>>> Running Audit Part 3: 5-Agent 7,000h Round-Robin League ({num_hands} hands)...")
    game = get_5p_fullgame()
    models = [load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_7000h.pt")) for a_name in AGENT_NAMES]
    
    agent_pnls = defaultdict(list)
    agent_act_counts = defaultdict(Counter)
    agent_allin_hands = defaultdict(int)

    for h_i in range(1, num_hands + 1):
        seed = base_seed + h_i
        rng = np.random.RandomState(seed)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)

        seat_to_agent = {s: (s - (h_i % 5)) % 5 for s in range(5)}
        agent_pushed_allin = defaultdict(bool)

        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng); continue
            cp = st.current_player()
            a_id = seat_to_agent[cp]
            leg = True6ActionAdapter.get_legal_6actions(st, cp)
            norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
            with torch.no_grad():
                logits, _ = models[a_id](torch.from_numpy(norm_inf).unsqueeze(0))
                masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                for a in leg: mask[0, a] = 0.0
                masked += mask
                act = int(torch.argmax(F.softmax(masked, dim=-1), dim=-1).item())
            agent_act_counts[a_id][act] += 1
            if act == 5: agent_pushed_allin[a_id] = True
            nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act)
            st.apply_action(nat_act)

        rets = st.returns()
        assert abs(sum(rets)) < 1e-4, f"League zero-sum violated: {sum(rets)}"
        for a_id in range(5):
            seat = [s for s, aid in seat_to_agent.items() if aid == a_id][0]
            agent_pnls[a_id].append(rets[seat] / BIG_BLIND)
            if agent_pushed_allin[a_id]: agent_allin_hands[a_id] += 1

    league_results = {}
    for a_id, a_name in enumerate(AGENT_NAMES):
        pnls = agent_pnls[a_id]
        tot_acts = sum(agent_act_counts[a_id].values())
        mean_bb = float(np.mean(pnls)) * 100.0
        ci_bb = float(stats.sem(pnls)) * 100.0 * 1.96
        league_results[a_name] = {
            "league_bb_per_100": round(mean_bb, 2),
            "ci95_bb": round(ci_bb, 2),
            "allin_pct": round(agent_allin_hands[a_id] / num_hands * 100.0, 2),
            "halfpot_pct": round(agent_act_counts[a_id].get(2, 0) / max(tot_acts, 1) * 100.0, 2),
            "pot_pct": round(agent_act_counts[a_id].get(3, 0) / max(tot_acts, 1) * 100.0, 2)
        }
        print(f"  League {a_name}: {mean_bb:+.2f} ± {ci_bb:.2f} BB/100 | HalfPot: {league_results[a_name]['halfpot_pct']}% | Pot: {league_results[a_name]['pot_pct']}% | Allin: {league_results[a_name]['allin_pct']}%")

    return league_results

def compute_checkpoint_hashes() -> Dict[str, str]:
    hashes = {}
    for a_name in AGENT_NAMES:
        for ms in [2000, 3000, 4000, 5000, 6000, 7000]:
            p = os.path.join(CHECKPOINT_DIR, f"{a_name}_{ms}h.pt")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                hashes[f"{a_name}_{ms}h"] = h
    return hashes

def main():
    print("=" * 80)
    print("DIRECTIVE 047 COMPREHENSIVE INDEPENDENT EVALUATION")
    print("=" * 80)
    t0 = time.time()

    part1 = run_part1_catcher_evaluation(num_hands=500)
    part2 = run_part2_paired_evaluation(num_hands=500)
    part3 = run_part3_league(num_hands=1000)
    hashes = compute_checkpoint_hashes()

    total_time = time.time() - t0
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_time, 1),
        "pilot_base_seed": PILOT_BASE_SEED,
        "part1_catcher_benchmark": part1,
        "part2_paired_evaluation": part2,
        "part3_league_results": part3,
        "checkpoint_sha256": hashes
    }

    with open(AUDIT_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nAudit complete in {total_time:.1f}s! Exported to {AUDIT_JSON_OUT}")

if __name__ == "__main__":
    main()
