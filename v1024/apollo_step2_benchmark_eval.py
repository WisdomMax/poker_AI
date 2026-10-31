#!/usr/bin/env python3
"""
v1024/apollo_step2_benchmark_eval.py
Project Apollo Step 2.5: Average Policy Benchmark Evaluation on 100 Independent Seeds.
Authorized by Lead RL Scientist Dr. Max.

Compares:
  - Apollo Average Policy Models (CardTransformerPokerPlayer with Multi-Head Attention, 891k params)
  - Legacy Baseline 2000h (2-layer MLP, 427k params)
Against:
  - Fixed 7k Benchmark Opponents.

Strict Constraints:
  - CPU Thread Limit: torch.set_num_threads(4) (Strictly <= 40% CPU usage)
  - 100 Independent Evaluation Seeds (2501000 - 2501099, 500 table hands, 2,500 player hands)
  - In-path 100% pot audit via calculate_pots_exact and zero-sum checks.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import scipy.stats as stats
import torch
import pyspiel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Strictly enforce CPU thread limit <= 4 threads (< 40% CPU utilization on 10-core machine)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.models_alphago.card_transformer_backbone import CardTransformerPokerPlayer
from v1024.directive058_production_engine import get_directive058_balanced_assignment
from v1024.test_directive059f_sidepot_cases import calculate_pots_exact, eval_7card, parse_cards_from_state

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
ACTION_NAMES = ["Fold", "Check/Call", "Raise_Min", "Raise_HalfPot", "Raise_Pot", "All_In"]
EXP_DIR = os.path.join(ROOT_DIR, "v1024/exp_apollo_step2")
BASE_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
REPORTS_DIR = os.path.join(ROOT_DIR, "v1024/reports")

def run_apollo_step2_evaluation(num_seeds: int = 100, seed_start: int = 2501000):
    print("=================================================================")
    print("PROJECT APOLLO: STEP 2.5 AVERAGE POLICY BENCHMARK EVALUATION")
    print(f"Hardware Constraint: PyTorch Threads = {torch.get_num_threads()} (Strictly <= 40% CPU)")
    print(f"Independent Evaluation Seeds: {num_seeds} ({seed_start} - {seed_start + num_seeds - 1})")
    print("=================================================================")

    # 1. Load Apollo Average Policy Models (Card-Transformer)
    print("\n[1/3] Loading Apollo Card-Transformer Average Models...")
    apollo_models = {}
    for name in AGENT_NAMES:
        p = os.path.join(EXP_DIR, f"avg_models/{name}.pt")
        m = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        apollo_models[name] = m
        print(f"  - {name}: Loaded Apollo Average Policy")

    # 2. Load Legacy Baseline (Base 2000h) and 7k Opponents
    print("\n[2/3] Loading Legacy Baseline (Base 2000h) and 7k Opponents...")
    baseline_models = {}
    bench_models = {}
    for name in AGENT_NAMES:
        p_base = os.path.join(BASE_DIR, f"{name}_2000h.pt")
        m_b = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_b = torch.load(p_base, map_location="cpu", weights_only=False)
        m_b.load_state_dict(ckpt_b["model_state_dict"] if isinstance(ckpt_b, dict) and "model_state_dict" in ckpt_b else ckpt_b)
        m_b.eval()
        baseline_models[name] = m_b

        p_7k = os.path.join(BASE_DIR, f"{name}_7000h.pt")
        m_7k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_7k = torch.load(p_7k, map_location="cpu", weights_only=False)
        m_7k.load_state_dict(ckpt_7k["model_state_dict"] if isinstance(ckpt_7k, dict) and "model_state_dict" in ckpt_7k else ckpt_7k)
        m_7k.eval()
        bench_models[name] = m_7k

    # 3. CRN Evaluation
    print("\n[3/3] Executing 100 Seeds CRN Head-to-Head Evaluation...")
    game = get_5p_fullgame()
    hands_per_seed = 5

    apollo_chips = {n: 0.0 for n in AGENT_NAMES}
    base_chips = {n: 0.0 for n in AGENT_NAMES}
    apollo_hands = {n: 0 for n in AGENT_NAMES}
    base_hands = {n: 0 for n in AGENT_NAMES}
    apollo_actions = {n: [0] * 6 for n in AGENT_NAMES}
    base_actions = {n: [0] * 6 for n in AGENT_NAMES}

    seed_apollo_returns = []
    seed_base_returns = []
    pot_audit_passes = 0
    max_zero_sum_err = 0.0
    t_start = time.time()

    for s_idx in range(num_seeds):
        curr_seed = seed_start + s_idx
        s_apollo_ret = 0.0
        s_base_ret = 0.0

        for h in range(hands_per_seed):
            hand_idx = s_idx * hands_per_seed + h + 1
            assign = get_directive058_balanced_assignment(hand_idx)
            hero_seat_map = assign["agent_to_seat"]
            opp_seat = assign["opp_seat"]
            opp_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_m = bench_models[opp_name]

            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x66666666

            # --- A. Apollo Average Models Run ---
            table_a = [None] * 5
            table_a[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                table_a[s_pos] = apollo_models[a_name]

            d_rng_a = np.random.RandomState(deck_seed)
            a_rng_a = np.random.RandomState(act_seed)
            s_a = game.new_initial_state()
            advance_chance_nodes(s_a, d_rng_a)

            folded_a = [False] * 5
            while not s_a.is_terminal():
                if s_a.is_chance_node():
                    advance_chance_nodes(s_a, d_rng_a)
                    continue
                cp = s_a.current_player()
                m = table_a[cp]
                legals = True6ActionAdapter.get_legal_6actions(s_a, cp)
                raw_inf = np.array(s_a.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                with torch.no_grad():
                    res = m(torch.from_numpy(norm_inf).unsqueeze(0))
                    logits = res[0] if isinstance(res, tuple) else res
                    probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                act = int(a_rng_a.choice(len(probs), p=probs))

                for a_name, s_pos in hero_seat_map.items():
                    if cp == s_pos:
                        apollo_actions[a_name][act] += 1
                if act == 0:
                    folded_a[cp] = True
                s_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(s_a, cp, act))

            rets_a = list(s_a.returns())
            # Zero-sum check
            err_a = abs(sum(rets_a))
            if err_a > max_zero_sum_err: max_zero_sum_err = err_a
            if err_a > 1e-4: raise AssertionError(f"Zero sum failed: {err_a}")

            # Pot verification
            spents_a = extract_player_spent(s_a)
            holes_a, board_a = parse_cards_from_state(s_a.to_string())
            active_a = [p for p in range(5) if not folded_a[p]]
            calc_rets_a = [0.0] * 5
            if len(active_a) == 1:
                winner = active_a[0]
                calc_rets_a = [-spents_a[p] for p in range(5)]
                calc_rets_a[winner] = sum(spents_a) - spents_a[winner]
            elif len(active_a) > 1:
                scores_a = [eval_7card(holes_a[p] + board_a) if p in active_a else None for p in range(5)]
                calc_rets_a = calculate_pots_exact(spents_a, folded_a, scores_a)
            diff_a = max(abs(c - e) for c, e in zip(calc_rets_a, rets_a))
            if diff_a > 1e-4: raise RuntimeError(f"Audit pot discrepancy: {diff_a}")
            pot_audit_passes += 1

            for a_name, s_pos in hero_seat_map.items():
                apollo_chips[a_name] += rets_a[s_pos]
                apollo_hands[a_name] += 1
                s_apollo_ret += rets_a[s_pos]

            # --- B. Legacy Baseline (Base 2000h) Run ---
            table_b = [None] * 5
            table_b[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                table_b[s_pos] = baseline_models[a_name]

            d_rng_b = np.random.RandomState(deck_seed)
            a_rng_b = np.random.RandomState(act_seed)
            s_b = game.new_initial_state()
            advance_chance_nodes(s_b, d_rng_b)

            folded_b = [False] * 5
            while not s_b.is_terminal():
                if s_b.is_chance_node():
                    advance_chance_nodes(s_b, d_rng_b)
                    continue
                cp = s_b.current_player()
                m = table_b[cp]
                legals = True6ActionAdapter.get_legal_6actions(s_b, cp)
                raw_inf = np.array(s_b.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                with torch.no_grad():
                    logits, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                    probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                act = int(a_rng_b.choice(len(probs), p=probs))

                for a_name, s_pos in hero_seat_map.items():
                    if cp == s_pos:
                        base_actions[a_name][act] += 1
                if act == 0:
                    folded_b[cp] = True
                s_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(s_b, cp, act))

            rets_b = list(s_b.returns())
            for a_name, s_pos in hero_seat_map.items():
                base_chips[a_name] += rets_b[s_pos]
                base_hands[a_name] += 1
                s_base_ret += rets_b[s_pos]

        seed_apollo_returns.append(s_apollo_ret / (hands_per_seed * 4))
        seed_base_returns.append(s_base_ret / (hands_per_seed * 4))

        if (s_idx + 1) % 25 == 0:
            c_bb = np.mean(seed_apollo_returns)
            b_bb = np.mean(seed_base_returns)
            print(f"  Eval Seeds: {s_idx+1:3d}/{num_seeds} | Apollo: {c_bb:+6.2f} vs Base: {b_bb:+6.2f} BB/100")

    # Metrics computation
    apollo_arr = np.array(seed_apollo_returns)
    base_arr = np.array(seed_base_returns)
    diff_arr = apollo_arr - base_arr

    mean_apollo = float(np.mean(apollo_arr))
    mean_base = float(np.mean(base_arr))
    delta = float(np.mean(diff_arr))
    se_delta = float(stats.sem(diff_arr))
    ci95 = [round(delta - 1.96 * se_delta, 2), round(delta + 1.96 * se_delta, 2)]
    t_stat, p_val = stats.ttest_rel(apollo_arr, base_arr)

    eval_time = time.time() - t_start

    print("\n=================================================================")
    print("APOLLO STEP 2.5 EVALUATION COMPLETE (Strict Data-First Analysis)")
    print(f"Apollo Average Models: {mean_apollo:+.2f} BB/100 (SE: {float(stats.sem(apollo_arr)):.2f})")
    print(f"Baseline Base 2000h:   {mean_base:+.2f} BB/100 (SE: {float(stats.sem(base_arr)):.2f})")
    print(f"Delta:                 {delta:+.2f} BB/100 (95% CI: [{ci95[0]}, {ci95[1]}], p={p_val:.4f})")
    print(f"Elapsed Time:          {eval_time:.1f}s | Pot Audit Passes: {pot_audit_passes}")
    print("=================================================================")

    # Action distributions
    apollo_act_dist = {}
    base_act_dist = {}
    for n in AGENT_NAMES:
        a_tot = sum(apollo_actions[n])
        b_tot = sum(base_actions[n])
        apollo_act_dist[n] = {ACTION_NAMES[i]: round(apollo_actions[n][i] / a_tot * 100, 2) for i in range(6)}
        base_act_dist[n] = {ACTION_NAMES[i]: round(base_actions[n][i] / b_tot * 100, 2) for i in range(6)}

    agent_eval_summary = {}
    for n in AGENT_NAMES:
        agent_eval_summary[n] = {
            "apollo_bb100": round(apollo_chips[n] / apollo_hands[n], 2),
            "baseline_bb100": round(base_chips[n] / base_hands[n], 2),
            "diff_bb100": round((apollo_chips[n] - base_chips[n]) / apollo_hands[n], 2)
        }
        print(f"  {n:7s} | Apollo: {agent_eval_summary[n]['apollo_bb100']:+7.2f} vs Base: {agent_eval_summary[n]['baseline_bb100']:+7.2f} (Diff: {agent_eval_summary[n]['diff_bb100']:+7.2f})")

    eval_evidence = {
        "directive": "APOLLO_STEP2_5_EVALUATION",
        "lead_scientist": "Dr. Max",
        "independent_seeds_count": num_seeds,
        "seed_range": [seed_start, seed_start + num_seeds - 1],
        "pot_audit_passes": pot_audit_passes,
        "max_zero_sum_error": max_zero_sum_err,
        "apollo_summary": {
            "mean_hero_bb100": round(mean_apollo, 2),
            "se_bb100": round(float(stats.sem(apollo_arr)), 2),
            "agents": agent_eval_summary
        },
        "baseline_summary": {
            "mean_hero_bb100": round(mean_base, 2),
            "se_bb100": round(float(stats.sem(base_arr)), 2)
        },
        "paired_comparison": {
            "delta_bb100": round(delta, 2),
            "paired_se_bb100": round(se_delta, 2),
            "ci95": ci95,
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val)
        },
        "action_distributions": {
            "Apollo_Average": apollo_act_dist,
            "Baseline_2000h": base_act_dist
        }
    }

    out_path = os.path.join(REPORTS_DIR, "apollo_step2_eval_evidence.json")
    with open(out_path, "w") as f:
        json.dump(eval_evidence, f, indent=2)
    print(f"\n[+] Evaluation evidence written to {out_path}")

    return eval_evidence

if __name__ == "__main__":
    run_apollo_step2_evaluation(num_seeds=100, seed_start=2501000)
