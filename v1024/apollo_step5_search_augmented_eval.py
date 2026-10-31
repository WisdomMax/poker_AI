#!/usr/bin/env python3
"""
v1024/apollo_step5_search_augmented_eval.py
Project Apollo Step 5: Search-Augmented Reinforcement Learning & Mixed Strategy Causal Study.
Authorized by Principal RL Research Director Dr. Max.

Hypothesis:
  1. The -1,274 BB/100 catastrophe in Step 4 was caused by deterministic Greedy Argmax breaking mixed strategy.
  2. Restoring Mixed Strategy (T=0.8 Softmax) will drastically cut exploitability.
  3. Augmenting policy with 1-Step Lookahead Subgame Resolving (Q(s,a) = R + gamma*V(s')) will achieve positive alpha.

Causal Study Design (3 Modes on Exact Same 100 Sealed Seeds vs 4 7k Anchors):
  Mode A: Raw Greedy Argmax (Step 4 baseline failure)
  Mode B: Temperature Softmax Mixed Strategy (T=0.8)
  Mode C: 1-Step Lookahead Search-Augmented (Policy logits + alpha * Q_search)

Strict Hardware Constraint: torch.set_num_threads(4) (CPU <= 40%).
"""

import os
import sys
import copy
import json
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
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

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
ACTION_NAMES = ["Fold", "Check/Call", "Raise_Min", "Raise_HalfPot", "Raise_Pot", "All_In"]
MODEL_PATH = os.path.join(ROOT_DIR, "v1024/exp_apollo_step4/main_champion_avg.pt")
BENCH_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
REPORTS_DIR = os.path.join(ROOT_DIR, "v1024/reports")

class SearchAugmentedHero:
    def __init__(self, model_path: str):
        self.model = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        self.model.eval()

    def select_action(self, state, hero_seat, legals, mask, mode="search", d_rng=None, a_rng=None):
        raw_inf = np.array(state.information_state_tensor(hero_seat), dtype=np.float32)
        norm_inf = normalize_inf_state_v79(raw_inf)
        t_obs = torch.from_numpy(norm_inf).unsqueeze(0)

        with torch.no_grad():
            logits, val = self.model(t_obs)
            logits = logits.squeeze(0).numpy()
            t_val = val.item()

        # Apply legal mask
        logits = logits + mask

        if mode == "argmax":
            # Mode A: Deterministic Greedy Argmax
            return int(np.argmax(logits))

        elif mode == "mixed":
            # Mode B: Temperature Softmax Mixed Strategy (T=0.8)
            T = 0.8
            scaled_logits = logits / T
            # Numerical stability
            scaled_logits = scaled_logits - np.max(scaled_logits[legals])
            exp_l = np.exp(scaled_logits) * (mask == 0.0)
            probs = exp_l / np.sum(exp_l)
            act = int(a_rng.choice(6, p=probs))
            return act

        elif mode == "search":
            # Mode C: 1-Step Lookahead Subgame Resolving
            # Evaluate Q(s, a) by expanding each legal action 1 step
            q_values = np.full(6, -1e9, dtype=np.float32)

            for act in legals:
                # Clone state for forward simulation
                s_sim = state.clone()
                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_sim, hero_seat, act)
                s_sim.apply_action(engine_act)

                # Advance chance nodes if any
                if not s_sim.is_terminal() and s_sim.is_chance_node():
                    d_rng_sim = np.random.RandomState(d_rng.randint(0, 2**31 - 1))
                    advance_chance_nodes(s_sim, d_rng_sim)

                if s_sim.is_terminal():
                    # Direct immediate payoff
                    ret = s_sim.returns()[hero_seat]
                    q_val = ret / 100.0 # BB unit
                else:
                    # Rollout 1 step: evaluate next state's value function
                    cp_next = s_sim.current_player()
                    if cp_next == hero_seat:
                        # Hero moves again: read Hero's V(s')
                        raw_next = np.array(s_sim.information_state_tensor(hero_seat), dtype=np.float32)
                        norm_next = normalize_inf_state_v79(raw_next)
                        with torch.no_grad():
                            _, v_next = self.model(torch.from_numpy(norm_next).unsqueeze(0))
                        q_val = v_next.item()
                    else:
                        # Opponent's turn: zero-sum approximation (-V_opp or current baseline)
                        raw_next = np.array(s_sim.information_state_tensor(hero_seat), dtype=np.float32)
                        norm_next = normalize_inf_state_v79(raw_next)
                        with torch.no_grad():
                            _, v_next = self.model(torch.from_numpy(norm_next).unsqueeze(0))
                        # Discounted future expectation
                        q_val = 0.95 * v_next.item()

                q_values[act] = q_val

            # Combine Policy Logits with Search Q-values (AlphaZero / ReBeL style)
            # Normalize Q across legal actions
            q_legal = q_values[legals]
            q_std = np.std(q_legal) + 1e-6
            q_norm = (q_legal - np.mean(q_legal)) / q_std

            search_logits = np.copy(logits)
            search_logits[legals] += 1.5 * q_norm # Search augmentation weight alpha = 1.5

            # Mixed Softmax Sampling over Search Logits
            T = 0.8
            scaled_logits = search_logits / T
            scaled_logits = scaled_logits - np.max(scaled_logits[legals])
            exp_l = np.exp(scaled_logits) * (mask == 0.0)
            probs = exp_l / np.sum(exp_l)
            act = int(a_rng.choice(6, p=probs))
            return act

        return int(legals[0])

def run_causal_study():
    print("=================================================================")
    print("PROJECT APOLLO: STEP 5 (SEARCH-AUGMENTED RL CAUSAL STUDY)")
    print("Investigating the Root Cause of Step 4 Failure and Search Resolving")
    print("Comparison: Mode A (Argmax) vs Mode B (Mixed T=0.8) vs Mode C (1-Step Search)")
    print(f"Hardware Constraint: PyTorch Threads = {torch.get_num_threads()} (Strictly <= 40% CPU)")
    print("Evaluation Setup: 100 Sealed Seeds x 50 Hands = 5,000 Hands vs 4 7k Anchors")
    print("=================================================================")

    hero = SearchAugmentedHero(MODEL_PATH)
    game = get_5p_fullgame()

    # Load 4 7k Anchor Opponents (Agent_A, C, D, E)
    anchor_opps = {}
    for name in ["Agent_A", "Agent_C", "Agent_D", "Agent_E"]:
        p_7k = os.path.join(BENCH_DIR, f"{name}_7000h.pt")
        m_7k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_7k = torch.load(p_7k, map_location="cpu", weights_only=False)
        m_7k.load_state_dict(ckpt_7k["model_state_dict"] if isinstance(ckpt_7k, dict) and "model_state_dict" in ckpt_7k else ckpt_7k)
        m_7k.eval()
        anchor_opps[name] = m_7k

    anchor_list = [anchor_opps["Agent_A"], anchor_opps["Agent_C"], anchor_opps["Agent_D"], anchor_opps["Agent_E"]]

    num_seeds = 100
    hands_per_seed = 50
    seed_start = 2501000

    results_a = [] # Argmax
    results_b = [] # Mixed Softmax
    results_c = [] # Search Augmented

    t_start = time.time()

    for s_idx in range(num_seeds):
        curr_seed = seed_start + s_idx
        chips_a = 0.0
        chips_b = 0.0
        chips_c = 0.0

        for h in range(hands_per_seed):
            hero_seat = h % 5
            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x77777777

            # -------------------------------------------------------------
            # Run Mode A (Argmax)
            # -------------------------------------------------------------
            d_rng_a = np.random.RandomState(deck_seed)
            a_rng_a = np.random.RandomState(act_seed)
            s_a = game.new_initial_state()
            advance_chance_nodes(s_a, d_rng_a)

            while not s_a.is_terminal():
                if s_a.is_chance_node():
                    advance_chance_nodes(s_a, d_rng_a)
                    continue
                cp = s_a.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_a, cp)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                if cp == hero_seat:
                    act_6 = hero.select_action(s_a, hero_seat, legals, mask, mode="argmax")
                else:
                    opp_idx = cp if cp < hero_seat else cp - 1
                    raw_inf = np.array(s_a.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)
                    t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        res = anchor_list[opp_idx](t_obs)
                        logits = res[0] if isinstance(res, tuple) else res
                        probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_a, cp, act_6)
                s_a.apply_action(engine_act)

            chips_a += s_a.returns()[hero_seat]

            # -------------------------------------------------------------
            # Run Mode B (Mixed Strategy T=0.8)
            # -------------------------------------------------------------
            d_rng_b = np.random.RandomState(deck_seed)
            a_rng_b = np.random.RandomState(act_seed)
            s_b = game.new_initial_state()
            advance_chance_nodes(s_b, d_rng_b)

            while not s_b.is_terminal():
                if s_b.is_chance_node():
                    advance_chance_nodes(s_b, d_rng_b)
                    continue
                cp = s_b.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_b, cp)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                if cp == hero_seat:
                    act_6 = hero.select_action(s_b, hero_seat, legals, mask, mode="mixed", a_rng=a_rng_b)
                else:
                    opp_idx = cp if cp < hero_seat else cp - 1
                    raw_inf = np.array(s_b.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)
                    t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        res = anchor_list[opp_idx](t_obs)
                        logits = res[0] if isinstance(res, tuple) else res
                        probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_b, cp, act_6)
                s_b.apply_action(engine_act)

            chips_b += s_b.returns()[hero_seat]

            # -------------------------------------------------------------
            # Run Mode C (1-Step Search Augmented)
            # -------------------------------------------------------------
            d_rng_c = np.random.RandomState(deck_seed)
            a_rng_c = np.random.RandomState(act_seed)
            s_c = game.new_initial_state()
            advance_chance_nodes(s_c, d_rng_c)

            while not s_c.is_terminal():
                if s_c.is_chance_node():
                    advance_chance_nodes(s_c, d_rng_c)
                    continue
                cp = s_c.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_c, cp)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                if cp == hero_seat:
                    act_6 = hero.select_action(s_c, hero_seat, legals, mask, mode="search", d_rng=d_rng_c, a_rng=a_rng_c)
                else:
                    opp_idx = cp if cp < hero_seat else cp - 1
                    raw_inf = np.array(s_c.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)
                    t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        res = anchor_list[opp_idx](t_obs)
                        logits = res[0] if isinstance(res, tuple) else res
                        probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_c, cp, act_6)
                s_c.apply_action(engine_act)

            chips_c += s_c.returns()[hero_seat]

        bb_a = chips_a / hands_per_seed
        bb_b = chips_b / hands_per_seed
        bb_c = chips_c / hands_per_seed

        results_a.append(bb_a)
        results_b.append(bb_b)
        results_c.append(bb_c)

        if (s_idx + 1) % 25 == 0 or (s_idx + 1) == num_seeds:
            elapsed = time.time() - t_start
            print(f"  Seeds: {s_idx + 1:3d}/{num_seeds} ({elapsed:4.1f}s) | Argmax: {np.mean(results_a):+7.2f} | Mixed: {np.mean(results_b):+7.2f} | Search: {np.mean(results_c):+7.2f} BB/100")

    # Statistical Analysis
    mean_a = float(np.mean(results_a))
    se_a = float(stats.sem(results_a))

    mean_b = float(np.mean(results_b))
    se_b = float(stats.sem(results_b))

    mean_c = float(np.mean(results_c))
    se_c = float(stats.sem(results_c))

    # Paired comparisons
    diff_ba = np.array(results_b) - np.array(results_a)
    delta_ba = float(np.mean(diff_ba))
    p_ba = float(stats.ttest_rel(results_b, results_a).pvalue)

    diff_cb = np.array(results_c) - np.array(results_b)
    delta_cb = float(np.mean(diff_cb))
    p_cb = float(stats.ttest_rel(results_c, results_b).pvalue)

    diff_ca = np.array(results_c) - np.array(results_a)
    delta_ca = float(np.mean(diff_ca))
    p_ca = float(stats.ttest_rel(results_c, results_a).pvalue)

    print("\n" + "=" * 65)
    print("APOLLO STEP 5 CAUSAL STUDY BENCHMARK RESULTS")
    print(f"Mode A (Greedy Argmax):     {mean_a:+7.2f} BB/100 (SE: {se_a:.2f})")
    print(f"Mode B (Mixed Strategy):    {mean_b:+7.2f} BB/100 (SE: {se_b:.2f}) [Delta vs A: {delta_ba:+7.2f}, p={p_ba:.4f}]")
    print(f"Mode C (Search-Augmented):  {mean_c:+7.2f} BB/100 (SE: {se_c:.2f}) [Delta vs A: {delta_ca:+7.2f}, p={p_ca:.4f}]")
    print(f"Search Gain over Mixed:     {delta_cb:+7.2f} BB/100 (p={p_cb:.4f})")
    print("=" * 65)

    evidence = {
        "benchmark": "100_sealed_seeds_3way_causal_comparison",
        "num_seeds": num_seeds,
        "hands_per_seed": hands_per_seed,
        "mode_a_argmax": {"bb100": mean_a, "se": se_a},
        "mode_b_mixed": {"bb100": mean_b, "se": se_b, "delta_vs_a": delta_ba, "p_val": p_ba},
        "mode_c_search": {"bb100": mean_c, "se": se_c, "delta_vs_a": delta_ca, "p_val": p_ca, "delta_vs_b": delta_cb, "p_val_vs_b": p_cb}
    }

    out_path = os.path.join(REPORTS_DIR, "apollo_step5_search_evidence.json")
    with open(out_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\n[+] Evidence saved to {out_path}")

if __name__ == "__main__":
    run_causal_study()
