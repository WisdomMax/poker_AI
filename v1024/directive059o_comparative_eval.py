#!/usr/bin/env python3
"""
v1024/directive059o_comparative_eval.py
DIRECTIVE 059-O: Repaired Pot Audit & 9-Arm Cross-Cohort Comparative Evaluation.
Strictly verifies:
  1. Every hand has audit_executed = True (showdown via calculate_pots_exact, foldout via exact calculation).
  2. No broad except hiding errors. Any discrepancy or exception halts immediately with non-zero exit code.
  3. Evaluates 9 arms on 20 dev seeds (3006100-3006119) * 25 hands = 500 table-hands per arm (total 4,500 table-hands).
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import torch
import pyspiel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.directive058_production_engine import get_directive058_balanced_assignment
from v1024.test_directive059f_sidepot_cases import calculate_pots_exact, eval_7card, parse_cards_from_state

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def load_ppo_model(path: str, expected_sha256: str = None) -> PPONeuralPlayer:
    full_path = os.path.join(ROOT_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"[SECURITY ABORT] Missing model file: {full_path}")
    act_hash = compute_sha256(full_path)
    if expected_sha256 is not None and act_hash != expected_sha256:
        raise ValueError(f"[SECURITY ABORT] Hash mismatch for {full_path}! Expected {expected_sha256}, got {act_hash}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

def evaluate_9arms_059o(
    seeds_list,
    hands_per_seed=25,
    output_raw_path=None,
    test_mode=False,
    injected_maldistribution=False,
    injected_calculator_exception=False
):
    """
    Evaluates 9 arms across given seeds with strict independent pot and zero-sum verification.
    """
    manifest_p = os.path.join(ROOT_DIR, "v1024/directive059n_models_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    game = get_5p_fullgame()

    # Pre-load opponents (7k)
    opp_models = {
        name: load_ppo_model(manifest["opponents_7k"][name]["path"], manifest["opponents_7k"][name]["sha256"])
        for name in AGENT_NAMES
    }

    # Pre-load arms
    arms_def = [
        ("Base_2000h_Initial", {name: load_ppo_model(manifest["base_2k"][name]["path"], manifest["base_2k"][name]["sha256"]) for name in AGENT_NAMES}),
        ("J_Static_1000h_s4008000", {name: load_ppo_model(manifest["cohort_j_models"]["Static_1000h_s4008000"][name]["path"], manifest["cohort_j_models"]["Static_1000h_s4008000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("J_Dynamic_1000h_s4008000", {name: load_ppo_model(manifest["cohort_j_models"]["Dynamic_1000h_s4008000"][name]["path"], manifest["cohort_j_models"]["Dynamic_1000h_s4008000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("J_Static_1000h_s4009000", {name: load_ppo_model(manifest["cohort_j_models"]["Static_1000h_s4009000"][name]["path"], manifest["cohort_j_models"]["Static_1000h_s4009000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("J_Dynamic_1000h_s4009000", {name: load_ppo_model(manifest["cohort_j_models"]["Dynamic_1000h_s4009000"][name]["path"], manifest["cohort_j_models"]["Dynamic_1000h_s4009000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("M_Static_1000h_s4010000", {name: load_ppo_model(manifest["cohort_m_models"]["Static_1000h_s4010000"][name]["path"], manifest["cohort_m_models"]["Static_1000h_s4010000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("M_Dynamic_1000h_s4010000", {name: load_ppo_model(manifest["cohort_m_models"]["Dynamic_1000h_s4010000"][name]["path"], manifest["cohort_m_models"]["Dynamic_1000h_s4010000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("M_Static_1000h_s4011000", {name: load_ppo_model(manifest["cohort_m_models"]["Static_1000h_s4011000"][name]["path"], manifest["cohort_m_models"]["Static_1000h_s4011000"][name]["sha256"]) for name in AGENT_NAMES}),
        ("M_Dynamic_1000h_s4011000", {name: load_ppo_model(manifest["cohort_m_models"]["Dynamic_1000h_s4011000"][name]["path"], manifest["cohort_m_models"]["Dynamic_1000h_s4011000"][name]["sha256"]) for name in AGENT_NAMES}),
    ]

    all_raw_records = []
    arm_block_stats = {arm_name: [] for arm_name, _ in arms_def}
    arm_total_stats = {arm_name: {"hero_chips": 0.0, "opp_chips": 0.0, "hands": 0} for arm_name, _ in arms_def}

    total_table_hands = 0
    showdown_count = 0
    foldout_count = 0
    max_pot_discrepancy = 0.0
    max_zero_sum_err = 0.0

    raw_file = open(output_raw_path, "w", encoding="utf-8") if output_raw_path else None

    try:
        for seed_idx, seed in enumerate(seeds_list):
            block_arm_chips = {arm_name: {"hero": 0.0, "opp": 0.0, "hands": 0} for arm_name, _ in arms_def}

            for hand_idx in range(1, hands_per_seed + 1):
                assign = get_directive058_balanced_assignment(hand_idx)
                opp_seat = assign["opp_seat"]
                opp_name = AGENT_NAMES[assign["opp_model_idx"]]
                hero_seat_map = assign["agent_to_seat"]

                # CRN: Each arm plays exact same deck, seats, opponents
                for arm_idx, (arm_name, hero_models) in enumerate(arms_def):
                    table = [None] * 5
                    table[opp_seat] = ("OPPONENT", opp_name, opp_models[opp_name])
                    for h_name, s_idx_table in hero_seat_map.items():
                        table[s_idx_table] = ("HERO", h_name, hero_models[h_name])

                    # Strictly separated RNG seeds
                    deck_seed = (seed * 1000003 + hand_idx * 10007) & 0xFFFFFFFF
                    deck_rng = np.random.RandomState(deck_seed)
                    action_rng = np.random.RandomState((deck_seed ^ 0x55555555) + arm_idx * 101)

                    s = game.new_initial_state()
                    advance_chance_nodes(s, deck_rng)

                    actions_history = []
                    folded = [False] * 5

                    while not s.is_terminal():
                        if s.is_chance_node():
                            advance_chance_nodes(s, deck_rng)
                            continue

                        cp = s.current_player()
                        p_type, p_name, p_model = table[cp]
                        legals = True6ActionAdapter.get_legal_6actions(s, cp)
                        raw_inf = np.array(s.information_state_tensor(cp), dtype=np.float32)
                        norm_inf = normalize_inf_state_v79(raw_inf)

                        with torch.no_grad():
                            logits, _ = p_model(torch.from_numpy(norm_inf).unsqueeze(0))
                            mask = torch.full_like(logits, -1e9)
                            for a in legals:
                                mask[0, a] = 0.0
                            probs = torch.softmax(logits + mask, dim=-1).squeeze(0).numpy()

                        act = int(action_rng.choice(len(probs), p=probs))
                        if act not in legals:
                            raise RuntimeError(f"[SECURITY ABORT] Illegal action {act} chosen! Legal actions: {legals}")
                        if act == 0:
                            folded[cp] = True
                        engine_act = True6ActionAdapter.map_6action_to_engine_amount(s, cp, act)
                        s.apply_action(engine_act)
                        actions_history.append((cp, p_name, act, engine_act))

                    assert s.is_terminal(), "State must be terminal"
                    engine_rets = list(s.returns())

                    # Injection tests for integrity verification
                    if injected_maldistribution:
                        # Zero-sum preserving maldistribution (+10 to P0, -10 to P1)
                        engine_rets[0] += 10.0
                        engine_rets[1] -= 10.0

                    # 1. Zero-sum verification
                    err = abs(sum(engine_rets))
                    if err > max_zero_sum_err:
                        max_zero_sum_err = err
                    if err > 1e-4:
                        raise AssertionError(f"[ABORT] Hand {total_table_hands}: Zero-sum failed: sum={sum(engine_rets)}")

                    # 2. Extract state details & card verification
                    spents = extract_player_spent(s)
                    s_str = s.to_string()
                    holes, board = parse_cards_from_state(s_str)

                    all_deal_cards = []
                    for hc in holes:
                        all_deal_cards.extend(hc)
                    all_deal_cards.extend(board)
                    if len(all_deal_cards) != len(set(all_deal_cards)):
                        raise RuntimeError(f"[SECURITY ABORT] Hand {total_table_hands}: Duplicate card detected! Cards: {all_deal_cards}")

                    # 3. Independent Pot & Payout Audit (Strictly Executed Every Hand)
                    audit_executed = False
                    calc_rets = [0.0] * 5

                    if injected_calculator_exception:
                        raise ValueError("[INJECTED_CALCULATOR_EXCEPTION] Independent audit crashed!")

                    active_players = [p for p in range(5) if not folded[p]]
                    if len(active_players) == 1:
                        # Foldout: Single survivor wins all spents from others
                        foldout_count += 1
                        winner = active_players[0]
                        calc_rets = [-spents[p] for p in range(5)]
                        calc_rets[winner] = sum(spents) - spents[winner]
                        audit_executed = True
                    elif len(active_players) > 1:
                        # Showdown: Exact side-pot distribution using 7-card evaluator
                        showdown_count += 1
                        scores = [None] * 5
                        for p in active_players:
                            scores[p] = eval_7card(holes[p] + board)
                        calc_rets = calculate_pots_exact(spents, folded, scores)
                        audit_executed = True
                    else:
                        raise RuntimeError(f"[ABORT] Hand {total_table_hands}: No active players at showdown!")

                    assert audit_executed, f"[AUDIT ERROR] Hand {total_table_hands}: audit_executed is False!"

                    discrepancy = max(abs(e - c) for e, c in zip(engine_rets, calc_rets))
                    if discrepancy > max_pot_discrepancy:
                        max_pot_discrepancy = discrepancy

                    if discrepancy > 1e-4:
                        error_dump = {
                            "hand": total_table_hands,
                            "arm": arm_name,
                            "spents": spents,
                            "folded": folded,
                            "holes": holes,
                            "board": board,
                            "engine_rets": engine_rets,
                            "calc_rets": calc_rets,
                            "discrepancy": discrepancy
                        }
                        dump_path = f"v1024/reports/pot_discrepancy_abort_hand_{total_table_hands}.json"
                        with open(dump_path, "w", encoding="utf-8") as f:
                            json.dump(error_dump, f, indent=2)
                        raise AssertionError(f"[AUDIT ABORT] Hand {total_table_hands}: Sidepot mismatch! diff={discrepancy}, dumped to {dump_path}")

                    # 4 Hero players participate per table hand
                    hero_chips_hand = sum(float(engine_rets[s_i]) for h_name, s_i in hero_seat_map.items()) / 4.0
                    opp_chip_hand = float(engine_rets[opp_seat])

                    arm_total_stats[arm_name]["hero_chips"] += hero_chips_hand
                    arm_total_stats[arm_name]["opp_chips"] += opp_chip_hand
                    arm_total_stats[arm_name]["hands"] += 1

                    block_arm_chips[arm_name]["hero"] += hero_chips_hand
                    block_arm_chips[arm_name]["opp"] += opp_chip_hand
                    block_arm_chips[arm_name]["hands"] += 1

                    total_table_hands += 1

                    rec = {
                        "table_hand": total_table_hands,
                        "seed": seed,
                        "hand_idx": hand_idx,
                        "arm": arm_name,
                        "opp_seat": opp_seat,
                        "hero_seat_map": hero_seat_map,
                        "hero_chip_avg": hero_chips_hand,
                        "opp_chip": opp_chip_hand,
                        "spents": spents,
                        "folded": folded,
                        "audit_executed": audit_executed,
                        "engine_rets": engine_rets,
                        "calc_rets": calc_rets,
                        "discrepancy": discrepancy
                    }
                    if raw_file:
                        raw_file.write(json.dumps(rec) + "\n")
                    if test_mode:
                        all_raw_records.append(rec)

            for arm_name, _ in arms_def:
                h_c = block_arm_chips[arm_name]["hero"]
                hands = block_arm_chips[arm_name]["hands"]
                bb100 = (h_c / (hands * 100.0)) * 100.0
                arm_block_stats[arm_name].append(bb100)

    finally:
        if raw_file:
            raw_file.close()

    # Summarize stats
    arms_summary = {}
    for arm_name, _ in arms_def:
        h_chips = arm_total_stats[arm_name]["hero_chips"]
        o_chips = arm_total_stats[arm_name]["opp_chips"]
        h_cnt = arm_total_stats[arm_name]["hands"]
        hero_bb100 = (h_chips / (h_cnt * 100.0)) * 100.0
        opp_bb100 = (o_chips / (h_cnt * 100.0)) * 100.0

        b_vals = np.array(arm_block_stats[arm_name])
        within_se = float(np.std(b_vals, ddof=1) / np.sqrt(len(b_vals))) if len(b_vals) > 1 else 0.0
        within_sd = float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else 0.0

        arms_summary[arm_name] = {
            "hero_bb100": float(hero_bb100),
            "opp_bb100": float(opp_bb100),
            "within_eval_se_bb100": within_se,
            "within_eval_sd_bb100": within_sd,
            "table_hands": h_cnt
        }

    # Paired comparisons
    base_b = np.array(arm_block_stats["Base_2000h_Initial"])
    pairs = [
        ("cohort_j_seed_4008000", "cohort_j", 4008000, "J_Static_1000h_s4008000", "J_Dynamic_1000h_s4008000"),
        ("cohort_j_seed_4009000", "cohort_j", 4009000, "J_Static_1000h_s4009000", "J_Dynamic_1000h_s4009000"),
        ("cohort_m_seed_4010000", "cohort_m", 4010000, "M_Static_1000h_s4010000", "M_Dynamic_1000h_s4010000"),
        ("cohort_m_seed_4011000", "cohort_m", 4011000, "M_Static_1000h_s4011000", "M_Dynamic_1000h_s4011000"),
    ]

    comparisons_summary = {}
    for p_id, cohort, seed, stat_arm, dyn_arm in pairs:
        s_b = np.array(arm_block_stats[stat_arm])
        d_b = np.array(arm_block_stats[dyn_arm])
        diff_dyn_stat = d_b - s_b
        delta_ds = float(np.mean(diff_dyn_stat))
        paired_se_ds = float(np.std(diff_dyn_stat, ddof=1) / np.sqrt(len(diff_dyn_stat))) if len(diff_dyn_stat) > 1 else 0.0

        diff_stat_base = s_b - base_b
        prog_stat = float(np.mean(diff_stat_base))
        paired_se_stat_base = float(np.std(diff_stat_base, ddof=1) / np.sqrt(len(diff_stat_base))) if len(diff_stat_base) > 1 else 0.0

        diff_dyn_base = d_b - base_b
        prog_dyn = float(np.mean(diff_dyn_base))
        paired_se_dyn_base = float(np.std(diff_dyn_base, ddof=1) / np.sqrt(len(diff_dyn_base))) if len(diff_dyn_base) > 1 else 0.0

        comparisons_summary[p_id] = {
            "cohort": cohort,
            "seed": seed,
            "delta_dynamic_minus_static": delta_ds,
            "paired_se_dynamic_minus_static": paired_se_ds,
            "progress_static_vs_base": prog_stat,
            "paired_se_static_vs_base": paired_se_stat_base,
            "progress_dynamic_vs_base": prog_dyn,
            "paired_se_dynamic_vs_base": paired_se_dyn_base
        }

    return {
        "directive": "059-O",
        "eval_table_hands_per_arm": hands_per_seed * len(seeds_list),
        "total_eval_table_hands": total_table_hands,
        "dev_seeds_count": len(seeds_list),
        "showdown_count": showdown_count,
        "foldout_count": foldout_count,
        "max_pot_discrepancy": float(max_pot_discrepancy),
        "max_zero_sum_error": float(max_zero_sum_err),
        "arms_results": arms_summary,
        "comparisons": comparisons_summary,
        "all_records": all_raw_records if test_mode else []
    }

if __name__ == "__main__":
    dev_seeds = [3006100 + i for i in range(20)]
    print(f"Starting DIRECTIVE 059-O full comparative evaluation: 9 arms, 20 seeds (3006100-3006119)...")
    res = evaluate_9arms_059o(
        dev_seeds,
        hands_per_seed=25,
        output_raw_path="v1024/data/directive059o_comparative_raw.jsonl"
    )
    with open("v1024/reports/directive059o_comparative_evidence.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("DIRECTIVE 059-O comparative evaluation complete. Saved to v1024/reports/directive059o_comparative_evidence.json")
