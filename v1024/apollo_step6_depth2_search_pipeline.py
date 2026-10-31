#!/usr/bin/env python3
"""
v1024/apollo_step6_depth2_search_pipeline.py
Project Apollo Step 6: Depth-2 Subgame Search & AlphaZero Search-to-Policy Distillation.
Authorized by Principal RL Research Director Dr. Max.

Key Innovations:
  1. Depth-2 Subgame Search Engine:
     - Hero simulates action a1.
     - Simulates opponent's conditional counter-response using opponent models.
     - Evaluates resulting state V(s'') to compute deep Q(s, a).
  2. AlphaZero Search-to-Policy Distillation:
     - Trains Actor with Cross-Entropy against improved search distribution pi_search.
  3. Hardware Constraint: Strictly PyTorch threads = 4 (CPU <= 40%).
  4. 100 Sealed Seeds Comparative Benchmark: Depth-1 Search vs Depth-2 Search.
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
import torch.optim as optim
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
from v1024.test_directive059f_sidepot_cases import calculate_pots_exact, eval_7card, parse_cards_from_state

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
ACTION_NAMES = ["Fold", "Check/Call", "Raise_Min", "Raise_HalfPot", "Raise_Pot", "All_In"]
INIT_WEIGHTS = os.path.join(ROOT_DIR, "v1024/exp_apollo_step4/main_champion_avg.pt")
EXP_DIR = os.path.join(ROOT_DIR, "v1024/exp_apollo_step6")
BENCH_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
REPORTS_DIR = os.path.join(ROOT_DIR, "v1024/reports")

# -------------------------------------------------------------
# Depth-2 Subgame Search Resolver
# -------------------------------------------------------------
def depth2_subgame_resolve(state, hero_seat, legals, mask, hero_model, opp_models, d_rng, a_rng, depth=2, alpha=2.0, T=0.8):
    raw_inf = np.array(state.information_state_tensor(hero_seat), dtype=np.float32)
    norm_inf = normalize_inf_state_v79(raw_inf)
    t_obs = torch.from_numpy(norm_inf).unsqueeze(0)

    with torch.no_grad():
        logits, _ = hero_model(t_obs)
        logits = logits.squeeze(0).numpy() + mask

    q_values = np.full(6, -1e9, dtype=np.float32)

    for a1 in legals:
        # Step 1: Simulate Hero Action
        s1 = state.clone()
        eng_a1 = True6ActionAdapter.map_6action_to_engine_amount(s1, hero_seat, a1)
        s1.apply_action(eng_a1)

        if not s1.is_terminal() and s1.is_chance_node():
            d_sim = np.random.RandomState(d_rng.randint(0, 2**31 - 1))
            advance_chance_nodes(s1, d_sim)

        if s1.is_terminal():
            q_values[a1] = s1.returns()[hero_seat] / 100.0
            continue

        if depth == 1:
            # Depth 1: immediate value function
            raw_next = np.array(s1.information_state_tensor(hero_seat), dtype=np.float32)
            norm_next = normalize_inf_state_v79(raw_next)
            with torch.no_grad():
                _, v_next = hero_model(torch.from_numpy(norm_next).unsqueeze(0))
            q_values[a1] = v_next.item()
            continue

        # Step 2: Depth 2 - Rollout Opponent Counter-Action
        cp = s1.current_player()
        if cp == hero_seat:
            raw_next = np.array(s1.information_state_tensor(hero_seat), dtype=np.float32)
            norm_next = normalize_inf_state_v79(raw_next)
            with torch.no_grad():
                _, v_next = hero_model(torch.from_numpy(norm_next).unsqueeze(0))
            q_values[a1] = v_next.item()
        else:
            opp_idx = cp if cp < hero_seat else cp - 1
            opp_m = opp_models[opp_idx]
            opp_legals = True6ActionAdapter.get_legal_6actions(s1, cp)
            opp_mask = np.full(6, -1e9, dtype=np.float32)
            for oa in opp_legals: opp_mask[oa] = 0.0

            raw_opp = np.array(s1.information_state_tensor(cp), dtype=np.float32)
            norm_opp = normalize_inf_state_v79(raw_opp)
            with torch.no_grad():
                res = opp_m(torch.from_numpy(norm_opp).unsqueeze(0))
                opp_logits = res[0] if isinstance(res, tuple) else res
                opp_probs = torch.softmax(opp_logits + torch.from_numpy(opp_mask).unsqueeze(0), dim=-1).squeeze(0).numpy()

            opp_act = int(np.argmax(opp_probs))
            eng_opp = True6ActionAdapter.map_6action_to_engine_amount(s1, cp, opp_act)
            s1.apply_action(eng_opp)

            if not s1.is_terminal() and s1.is_chance_node():
                d_sim = np.random.RandomState(d_rng.randint(0, 2**31 - 1))
                advance_chance_nodes(s1, d_sim)

            if s1.is_terminal():
                q_values[a1] = s1.returns()[hero_seat] / 100.0
            else:
                raw_next = np.array(s1.information_state_tensor(hero_seat), dtype=np.float32)
                norm_next = normalize_inf_state_v79(raw_next)
                with torch.no_grad():
                    _, v_next = hero_model(torch.from_numpy(norm_next).unsqueeze(0))
                q_values[a1] = 0.95 * v_next.item()

    # Normalize Q across legal actions
    q_legal = q_values[legals]
    q_std = np.std(q_legal) + 1e-6
    q_norm = (q_legal - np.mean(q_legal)) / q_std

    search_logits = np.copy(logits)
    search_logits[legals] += alpha * q_norm

    # Temperature Softmax
    scaled_logits = (search_logits - np.max(search_logits[legals])) / T
    exp_l = np.exp(scaled_logits) * (mask == 0.0)
    pi_search = exp_l / np.sum(exp_l)

    act = int(a_rng.choice(6, p=pi_search))
    return act, pi_search

# -------------------------------------------------------------
# Training and Evaluation Orchestration
# -------------------------------------------------------------
def run_step6():
    print("=================================================================")
    print("PROJECT APOLLO: STEP 6 (DEPTH-2 SEARCH & POLICY DISTILLATION)")
    print("AlphaZero Search-to-Policy Distillation & Depth-2 Counter-Resolving")
    print(f"Hardware Constraint: PyTorch Threads = {torch.get_num_threads()} (Strictly <= 40% CPU)")
    print("=================================================================")

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load Anchor Models
    print("\n[1/4] Loading 4 7k Anchor Opponents...")
    anchor_opps = []
    for name in ["Agent_A", "Agent_C", "Agent_D", "Agent_E"]:
        p_7k = os.path.join(BENCH_DIR, f"{name}_7000h.pt")
        m_7k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_7k = torch.load(p_7k, map_location="cpu", weights_only=False)
        m_7k.load_state_dict(ckpt_7k["model_state_dict"] if isinstance(ckpt_7k, dict) and "model_state_dict" in ckpt_7k else ckpt_7k)
        m_7k.eval()
        anchor_opps.append(m_7k)

    # 2. Initialize Hero Model
    print("\n[2/4] Initializing Hero Model from Step 4 Average Weights...")
    hero_model = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(INIT_WEIGHTS, map_location="cpu", weights_only=False)
    hero_model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    hero_model.train()

    optimizer = optim.AdamW(hero_model.parameters(), lr=0.0001, weight_decay=1e-4)
    game = get_5p_fullgame()

    # 3. AlphaZero Search Distillation Training (3,000 Hands)
    train_hands = 3000
    print(f"\n[3/4] Commencing {train_hands} Hands AlphaZero Search Distillation Training...")
    t_start = time.time()

    distill_states = []
    distill_pi_targets = []
    distill_rewards = []
    distill_masks = []
    hero_chips_total = 0.0

    seed = 9010000
    np.random.seed(seed)
    torch.manual_seed(seed)

    for hand in range(1, train_hands + 1):
        hero_seat = (hand - 1) % 5
        deck_seed = (seed * 1000003 + hand * 10007) & 0xFFFFFFFF
        act_seed = deck_seed ^ 0x33333333
        d_rng = np.random.RandomState(deck_seed)
        a_rng = np.random.RandomState(act_seed)

        s = game.new_initial_state()
        advance_chance_nodes(s, d_rng)

        hand_states = []
        hand_pi_targets = []
        hand_masks = []

        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, d_rng)
                continue

            cp = s.current_player()
            legals = True6ActionAdapter.get_legal_6actions(s, cp)
            mask = np.full(6, -1e9, dtype=np.float32)
            for a in legals: mask[a] = 0.0

            if cp == hero_seat:
                # Run Depth-2 Subgame Resolving
                act_6, pi_search = depth2_subgame_resolve(
                    s, hero_seat, legals, mask, hero_model, anchor_opps, d_rng, a_rng, depth=2, alpha=2.0, T=0.8
                )
                raw_inf = np.array(s.information_state_tensor(hero_seat), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)

                hand_states.append(norm_inf)
                hand_pi_targets.append(pi_search)
                hand_masks.append(mask)
            else:
                opp_idx = cp if cp < hero_seat else cp - 1
                raw_opp = np.array(s.information_state_tensor(cp), dtype=np.float32)
                norm_opp = normalize_inf_state_v79(raw_opp)
                with torch.no_grad():
                    res = anchor_opps[opp_idx](torch.from_numpy(norm_opp).unsqueeze(0))
                    opp_logits = res[0] if isinstance(res, tuple) else res
                    opp_probs = torch.softmax(opp_logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                act_6 = int(np.argmax(opp_probs))

            engine_act = True6ActionAdapter.map_6action_to_engine_amount(s, cp, act_6)
            s.apply_action(engine_act)

        # Returns & Reward Attribution
        hero_payoff = s.returns()[hero_seat]
        hero_chips_total += hero_payoff
        rew_bb = hero_payoff / 100.0

        n_steps = len(hand_states)
        for i, (s_vec, pi_t, msk) in enumerate(zip(hand_states, hand_pi_targets, hand_masks)):
            distill_states.append(s_vec)
            distill_pi_targets.append(pi_t)
            distill_masks.append(msk)
            distill_rewards.append(rew_bb if i == n_steps - 1 else 0.0)

        # Distillation Update Every 100 Hands
        if hand % 100 == 0 and len(distill_states) >= 64:
            b_states = torch.from_numpy(np.array(distill_states, dtype=np.float32))
            b_targets = torch.from_numpy(np.array(distill_pi_targets, dtype=np.float32))
            b_masks = torch.from_numpy(np.array(distill_masks, dtype=np.float32))
            b_rewards = torch.tensor(distill_rewards, dtype=torch.float32)

            for _ in range(2):
                logits, vals = hero_model(b_states)
                vals = vals.squeeze(-1)
                log_probs = torch.log_softmax(logits + b_masks, dim=-1)

                # Cross-entropy loss against improved search distribution pi_search (AlphaZero policy distillation)
                policy_distill_loss = -(b_targets * log_probs).sum(dim=-1).mean()
                val_loss = nn.MSELoss()(vals, b_rewards)
                loss = policy_distill_loss + 0.5 * val_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(hero_model.parameters(), 1.0)
                optimizer.step()

            distill_states.clear()
            distill_pi_targets.clear()
            distill_masks.clear()
            distill_rewards.clear()

        if hand % 500 == 0:
            elapsed = time.time() - t_start
            hps = hand / elapsed
            bb100 = hero_chips_total / hand
            print(f"  Hand {hand:4d}/{train_hands} | HPS: {hps:4.1f} | Hero BB/100: {bb100:+7.2f}")

    train_elapsed = time.time() - t_start
    print(f"\n[+] AlphaZero Distillation Training Complete in {train_elapsed:.1f}s.")
    hero_model.eval()

    # Save Step 6 Model
    torch.save(hero_model.state_dict(), os.path.join(EXP_DIR, "apollo_step6_distilled_hero.pt"))

    # 4. Independent 100 Sealed Seeds Evaluation: Depth-1 vs Depth-2 Search
    print("\n" + "=" * 65)
    print("STARTING APOLLO STEP 6 CONFIRMATORY BENCHMARK EVALUATION (100 Seeds)")
    print("Direct Head-to-Head: Depth-1 Search vs Depth-2 Search (Exact Same Seeds)")
    print("=================================================================")

    num_seeds = 100
    hands_per_seed = 50
    seed_start = 2501000

    results_depth1 = []
    results_depth2 = []

    eval_start = time.time()

    for s_idx in range(num_seeds):
        curr_seed = seed_start + s_idx
        chips_d1 = 0.0
        chips_d2 = 0.0

        for h in range(hands_per_seed):
            hero_seat = h % 5
            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x99999999

            # --- A. Depth-1 Search Hand ---
            d_rng1 = np.random.RandomState(deck_seed)
            a_rng1 = np.random.RandomState(act_seed)
            s_d1 = game.new_initial_state()
            advance_chance_nodes(s_d1, d_rng1)

            while not s_d1.is_terminal():
                if s_d1.is_chance_node():
                    advance_chance_nodes(s_d1, d_rng1)
                    continue
                cp = s_d1.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_d1, cp)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                if cp == hero_seat:
                    act_6, _ = depth2_subgame_resolve(s_d1, hero_seat, legals, mask, hero_model, anchor_opps, d_rng1, a_rng1, depth=1)
                else:
                    opp_idx = cp if cp < hero_seat else cp - 1
                    raw_opp = np.array(s_d1.information_state_tensor(cp), dtype=np.float32)
                    norm_opp = normalize_inf_state_v79(raw_opp)
                    with torch.no_grad():
                        res = anchor_opps[opp_idx](torch.from_numpy(norm_opp).unsqueeze(0))
                        opp_logits = res[0] if isinstance(res, tuple) else res
                        opp_probs = torch.softmax(opp_logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(opp_probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_d1, cp, act_6)
                s_d1.apply_action(engine_act)

            chips_d1 += s_d1.returns()[hero_seat]

            # --- B. Depth-2 Search Hand (Exact Same Cards & Seed) ---
            d_rng2 = np.random.RandomState(deck_seed)
            a_rng2 = np.random.RandomState(act_seed)
            s_d2 = game.new_initial_state()
            advance_chance_nodes(s_d2, d_rng2)

            while not s_d2.is_terminal():
                if s_d2.is_chance_node():
                    advance_chance_nodes(s_d2, d_rng2)
                    continue
                cp = s_d2.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_d2, cp)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                if cp == hero_seat:
                    act_6, _ = depth2_subgame_resolve(s_d2, hero_seat, legals, mask, hero_model, anchor_opps, d_rng2, a_rng2, depth=2)
                else:
                    opp_idx = cp if cp < hero_seat else cp - 1
                    raw_opp = np.array(s_d2.information_state_tensor(cp), dtype=np.float32)
                    norm_opp = normalize_inf_state_v79(raw_opp)
                    with torch.no_grad():
                        res = anchor_opps[opp_idx](torch.from_numpy(norm_opp).unsqueeze(0))
                        opp_logits = res[0] if isinstance(res, tuple) else res
                        opp_probs = torch.softmax(opp_logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(opp_probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_d2, cp, act_6)
                s_d2.apply_action(engine_act)

            chips_d2 += s_d2.returns()[hero_seat]

        bb_d1 = chips_d1 / hands_per_seed
        bb_d2 = chips_d2 / hands_per_seed
        results_depth1.append(bb_d1)
        results_depth2.append(bb_d2)

        if (s_idx + 1) % 25 == 0 or (s_idx + 1) == num_seeds:
            elapsed = time.time() - eval_start
            print(f"  Seeds: {s_idx + 1:3d}/{num_seeds} ({elapsed:4.1f}s) | Depth-1: {np.mean(results_depth1):+7.2f} vs Depth-2: {np.mean(results_depth2):+7.2f} BB/100")

    mean_d1 = float(np.mean(results_depth1))
    se_d1 = float(stats.sem(results_depth1))
    mean_d2 = float(np.mean(results_depth2))
    se_d2 = float(stats.sem(results_depth2))

    diffs = np.array(results_depth2) - np.array(results_depth1)
    delta = float(np.mean(diffs))
    se_delta = float(stats.sem(diffs))
    ci95 = [float(delta - 1.984 * se_delta), float(delta + 1.984 * se_delta)]
    t_stat, p_val = stats.ttest_rel(results_depth2, results_depth1)

    print("\n" + "=" * 65)
    print("APOLLO STEP 6 BENCHMARK RESULTS")
    print(f"Depth-1 Search (Step 5):   {mean_d1:+7.2f} BB/100 (SE: {se_d1:.2f})")
    print(f"Depth-2 Search (Step 6):   {mean_d2:+7.2f} BB/100 (SE: {se_d2:.2f})")
    print(f"Depth Search Gain (Delta): {delta:+7.2f} BB/100 (95% CI: [{ci95[0]:.2f}, {ci95[1]:.2f}], p={p_val:.4f})")
    print("=" * 65)

    evidence = {
        "benchmark": "100_sealed_seeds_depth1_vs_depth2",
        "num_seeds": num_seeds,
        "hands_per_seed": hands_per_seed,
        "depth1_search_bb100": mean_d1,
        "depth1_se": se_d1,
        "depth2_search_bb100": mean_d2,
        "depth2_se": se_d2,
        "delta_gain": delta,
        "ci95": ci95,
        "p_value": float(p_val),
        "statistical_significance": bool(p_val < 0.05 and delta > 0)
    }

    out_path = os.path.join(REPORTS_DIR, "apollo_step6_depth2_evidence.json")
    with open(out_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\n[+] Evidence saved to {out_path}")

if __name__ == "__main__":
    run_step6()
