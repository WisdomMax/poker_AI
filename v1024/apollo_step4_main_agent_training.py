#!/usr/bin/env python3
"""
v1024/apollo_step4_main_agent_training.py
Project Apollo Step 4: Dedicated Main Champion League Training & Benchmark Evaluation.
Authorized by Principal RL Research Director Dr. Max.

Key Architectural Principles:
  1. Elimination of Intra-League Cannibalism (형제 착취 완전 박멸):
     - Only ONE learning agent (Main Champion initialized from Step 3 Agent_B).
     - Hero occupies 1 seat (rotating 0..4 every hand).
     - Remaining 4 seats are occupied by diverse fixed benchmark opponents (Pool of 10: 7k Anchors + Base 2000h).
  2. Model: CardTransformerPokerPlayer (52-card Attention + ResNet, 891k params).
  3. Hardware Constraint: PyTorch threads = 4 (strictly <= 40% CPU on 10-core machine).
  4. NFSP Average Policy EMA (decay = 0.995) to track GTO Nash Equilibrium.
  5. 100% in-loop strict pot verification (calculate_pots_exact) and zero-sum checks.
  6. Linked 100 independent sealed seeds benchmark evaluation (2501000 - 2501099).
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

# Hardware Resource Constraint: Strictly bound PyTorch to 4 threads (Mac 10-core <= 40%)
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
INIT_WEIGHTS = os.path.join(ROOT_DIR, "v1024/exp_apollo_step3/final_avg_models/Agent_B.pt")
EXP_DIR = os.path.join(ROOT_DIR, "v1024/exp_apollo_step4")
BENCH_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
REPORTS_DIR = os.path.join(ROOT_DIR, "v1024/reports")

class ApolloMainChampion:
    def __init__(self, init_path: str, lr: float = 0.00015, ema_decay: float = 0.995):
        # Best Response Model (Active Learner)
        self.br_model = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        self.br_model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        self.br_model.train()

        # Average Policy Model (NFSP)
        self.avg_model = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        self.avg_model.load_state_dict(self.br_model.state_dict())
        self.avg_model.eval()

        self.optimizer = optim.AdamW(self.br_model.parameters(), lr=lr, weight_decay=1e-4)
        self.ema_decay = ema_decay

        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []
        self.values = []
        self.masks = []

        self.last_value_loss = 0.0
        self.last_policy_loss = 0.0
        self.current_entropy = 1.0

    def update(self, ppo_epochs=2, clip_eps=0.2):
        if len(self.states) < 64:
            return

        b_states = torch.from_numpy(np.array(self.states, dtype=np.float32))
        b_actions = torch.tensor(self.actions, dtype=torch.long)
        b_rewards = torch.tensor(self.rewards, dtype=torch.float32)
        b_old_logp = torch.tensor(self.old_log_probs, dtype=torch.float32)
        b_values = torch.tensor(self.values, dtype=torch.float32)
        b_masks = torch.from_numpy(np.array(self.masks, dtype=np.float32))

        adv = b_rewards - b_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_vloss = 0.0
        total_ploss = 0.0

        for _ in range(ppo_epochs):
            logits, vals = self.br_model(b_states)
            vals = vals.squeeze(-1)
            probs = torch.softmax(logits + b_masks, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logp = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - b_old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            p_loss = -torch.min(surr1, surr2).mean()

            v_loss = nn.MSELoss()(vals, b_rewards)
            loss = p_loss + 0.5 * v_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.br_model.parameters(), 1.0)
            self.optimizer.step()

            total_vloss += v_loss.item()
            total_ploss += p_loss.item()

        self.last_value_loss = total_vloss / ppo_epochs
        self.last_policy_loss = total_ploss / ppo_epochs

        # EMA update for Average Policy
        with torch.no_grad():
            for p_avg, p_br in zip(self.avg_model.parameters(), self.br_model.parameters()):
                p_avg.data.mul_(self.ema_decay).add_(p_br.data, alpha=1.0 - self.ema_decay)

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.old_log_probs.clear()
        self.values.clear()
        self.masks.clear()

def run_step4_main_league():
    total_hands = 10000
    seed = 8010000

    print("=================================================================")
    print("PROJECT APOLLO: STEP 4 (MAIN CHAMPION DEDICATED LEAGUE)")
    print("Model: CardTransformerPokerPlayer (Initialized from Step 3 Agent_B)")
    print("Table: 1 Hero vs 4 Diverse Fixed Benchmarks (7k + Base 2000h)")
    print(f"Hardware Constraint: PyTorch Threads = {torch.get_num_threads()} (Strictly <= 40% CPU)")
    print(f"Total Table Hands: {total_hands} | Seed: {seed}")
    print("=================================================================")

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Initialize Main Champion
    print("\n[1/3] Initializing Main Champion from Step 3 Agent_B...")
    hero = ApolloMainChampion(INIT_WEIGHTS, lr=0.00015, ema_decay=0.995)
    print(f"  - Main Champion initialized successfully from {INIT_WEIGHTS}")

    # 2. Build Diverse Fixed Opponent Pool (10 Models: 5 from 7k + 5 from Base 2000h)
    print("\n[2/3] Building Diverse Fixed Opponent Pool (10 Models)...")
    opponent_pool = []
    
    # 7k Anchors
    for name in AGENT_NAMES:
        p_7k = os.path.join(BENCH_DIR, f"{name}_7000h.pt")
        m_7k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_7k = torch.load(p_7k, map_location="cpu", weights_only=False)
        m_7k.load_state_dict(ckpt_7k["model_state_dict"] if isinstance(ckpt_7k, dict) and "model_state_dict" in ckpt_7k else ckpt_7k)
        m_7k.eval()
        opponent_pool.append((f"7k_{name}", m_7k))

    # Base 2000h
    for name in AGENT_NAMES:
        p_base = os.path.join(BENCH_DIR, f"{name}_2000h.pt")
        m_b = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_b = torch.load(p_base, map_location="cpu", weights_only=False)
        m_b.load_state_dict(ckpt_b["model_state_dict"] if isinstance(ckpt_b, dict) and "model_state_dict" in ckpt_b else ckpt_b)
        m_b.eval()
        opponent_pool.append((f"Base2k_{name}", m_b))

    print(f"  - Loaded {len(opponent_pool)} diverse benchmark opponents into league pool.")

    # 3. Main Dedicated League Training Loop
    game = get_5p_fullgame()
    matchmaking_rng = np.random.RandomState(seed ^ 0xCCCCCCCC)
    t_start = time.time()
    hero_chips_total = 0.0
    pot_audit_passes = 0
    max_zero_sum_err = 0.0

    print(f"\n[3/3] Commencing {total_hands} Table-Hand Dedicated Main Champion Training...")

    for hand in range(1, total_hands + 1):
        # Hero seat rotates 0..4
        hero_seat = (hand - 1) % 5
        
        # Sample 4 distinct opponents from pool of 10
        opp_indices = matchmaking_rng.choice(len(opponent_pool), size=4, replace=False)
        
        table_models = [None] * 5
        table_models[hero_seat] = "HERO"
        opp_ptr = 0
        for s in range(5):
            if s != hero_seat:
                table_models[s] = opponent_pool[opp_indices[opp_ptr]][1]
                opp_ptr += 1

        deck_seed = (seed * 1000003 + hand * 10007) & 0xFFFFFFFF
        act_seed = deck_seed ^ 0x55555555
        deck_rng = np.random.RandomState(deck_seed)
        action_rng = np.random.RandomState(act_seed)

        s = game.new_initial_state()
        advance_chance_nodes(s, deck_rng)

        folded = [False] * 5
        hand_hero_states = []
        hand_hero_actions = []
        hand_hero_logps = []
        hand_hero_vals = []
        hand_hero_masks = []

        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, deck_rng)
                continue

            cp = s.current_player()
            legals = True6ActionAdapter.get_legal_6actions(s, cp)
            raw_inf = np.array(s.information_state_tensor(cp), dtype=np.float32)
            norm_inf = normalize_inf_state_v79(raw_inf)

            mask = np.full(6, -1e9, dtype=np.float32)
            for a in legals: mask[a] = 0.0

            if cp == hero_seat:
                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    logits, val = hero.br_model(t_obs)
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()

                act = int(action_rng.choice(len(probs), p=probs))
                log_p = float(np.log(probs[act] + 1e-12))
                v_est = float(val.item())

                # Entropy tracking
                ent = float(-np.sum(probs[legals] * np.log(probs[legals] + 1e-12)))
                hero.current_entropy = ent

                hand_hero_states.append(norm_inf)
                hand_hero_actions.append(act)
                hand_hero_logps.append(log_p)
                hand_hero_vals.append(v_est)
                hand_hero_masks.append(mask)
            else:
                opp_m = table_models[cp]
                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    res = opp_m(t_obs)
                    logits = res[0] if isinstance(res, tuple) else res
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                act = int(action_rng.choice(len(probs), p=probs))

            if act == 0:
                folded[cp] = True
            engine_act = True6ActionAdapter.map_6action_to_engine_amount(s, cp, act)
            s.apply_action(engine_act)

        assert s.is_terminal()
        returns = list(s.returns())

        # Zero sum check
        err = abs(sum(returns))
        if err > max_zero_sum_err: max_zero_sum_err = err
        assert err < 1e-4, f"Zero sum failed: {err}"

        # Independent Pot Verification
        spents = extract_player_spent(s)
        holes, board = parse_cards_from_state(s.to_string())
        active_players = [p for p in range(5) if not folded[p]]
        calc_rets = [0.0] * 5

        if len(active_players) == 1:
            winner = active_players[0]
            calc_rets = [-spents[p] for p in range(5)]
            calc_rets[winner] = sum(spents) - spents[winner]
        elif len(active_players) > 1:
            scores = [None] * 5
            for p in active_players:
                scores[p] = eval_7card(holes[p] + board)
            calc_rets = calculate_pots_exact(spents, folded, scores)

        pot_diff = max(abs(calc_rets[p] - returns[p]) for p in range(5))
        assert pot_diff < 1e-4, f"Hand {hand} pot audit failed: diff={pot_diff}"
        pot_audit_passes += 1

        hero_payoff = returns[hero_seat]
        hero_chips_total += hero_payoff
        reward_bb = hero_payoff / 100.0

        n_steps = len(hand_hero_states)
        for i, (s_vec, a_6, lp, val, msk) in enumerate(zip(hand_hero_states, hand_hero_actions, hand_hero_logps, hand_hero_vals, hand_hero_masks)):
            step_rew = reward_bb if i == n_steps - 1 else 0.0
            hero.states.append(s_vec)
            hero.actions.append(a_6)
            hero.old_log_probs.append(lp)
            hero.values.append(val)
            hero.masks.append(msk)
            hero.rewards.append(step_rew)

        # Update every 200 hands
        if hand % 200 == 0:
            hero.update(ppo_epochs=2, clip_eps=0.2)

        # Progress log
        if hand % 1000 == 0:
            elapsed = time.time() - t_start
            hps = hand / elapsed
            hero_bb100 = hero_chips_total / hand
            print(f"  Hand {hand:5d}/{total_hands} | HPS: {hps:5.1f} | Hero BB/100: {hero_bb100:+7.2f} | Ent: {hero.current_entropy:.2f} | VLoss: {hero.last_value_loss:.2f}")

    train_elapsed = time.time() - t_start
    print(f"\n[+] Apollo Step 4 Main Champion Training Complete in {train_elapsed:.1f}s.")

    # Save Checkpoints
    final_hero_path = os.path.join(EXP_DIR, "main_champion_final.pt")
    avg_hero_path = os.path.join(EXP_DIR, "main_champion_avg.pt")
    torch.save(hero.br_model.state_dict(), final_hero_path)
    torch.save(hero.avg_model.state_dict(), avg_hero_path)
    print(f"  - Saved Main Champion Final Policy: {final_hero_path}")
    print(f"  - Saved Main Champion Average Policy: {avg_hero_path}")

    # Training Evidence
    train_evidence = {
        "status": "COMPLETED",
        "total_hands": total_hands,
        "elapsed_seconds": round(train_elapsed, 2),
        "pot_audit_passes": pot_audit_passes,
        "max_zero_sum_error": max_zero_sum_err,
        "final_hero_bb100": round(hero_chips_total / total_hands, 2),
        "final_entropy": round(hero.current_entropy, 3),
        "final_vloss": round(hero.last_value_loss, 2)
    }
    with open(os.path.join(REPORTS_DIR, "apollo_step4_training_evidence.json"), "w") as f:
        json.dump(train_evidence, f, indent=2)

    # 4. Confirmatory 100 Independent Sealed Seeds Benchmark Evaluation
    print("\n=================================================================")
    print("STARTING APOLLO STEP 4 INDEPENDENT BENCHMARK EVALUATION (100 Seeds)")
    print("Apollo Main Champion (Avg Policy) vs 7k External Anchors (4 Opponents)")
    print("vs Baseline Base 2000h Agent_B (Exact Same 100 Seeds & Positions)")
    print("=================================================================")

    run_step4_evaluation(hero.avg_model, num_seeds=100, seed_start=2501000)

def run_step4_evaluation(hero_avg_model, num_seeds: int = 100, seed_start: int = 2501000):
    game = get_5p_fullgame()
    hands_per_seed = 50 # 50 hands per seed * 100 seeds = 5,000 hands of deep evaluation

    # Load Baseline Base 2000h Agent_B
    base_b_path = os.path.join(BENCH_DIR, "Agent_B_2000h.pt")
    base_model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt_b = torch.load(base_b_path, map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt_b["model_state_dict"] if isinstance(ckpt_b, dict) and "model_state_dict" in ckpt_b else ckpt_b)
    base_model.eval()

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

    seed_apollo_bbs = []
    seed_base_bbs = []

    for s_idx in range(num_seeds):
        curr_seed = seed_start + s_idx
        apollo_seed_chips = 0.0
        base_seed_chips = 0.0

        for h in range(hands_per_seed):
            hero_seat = h % 5
            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x66666666

            # A. Apollo Main Champion Hand
            table_a = [None] * 5
            table_a[hero_seat] = hero_avg_model
            opp_ptr = 0
            for s_pos in range(5):
                if s_pos != hero_seat:
                    table_a[s_pos] = anchor_list[opp_ptr]
                    opp_ptr += 1

            d_rng_a = np.random.RandomState(deck_seed)
            a_rng_a = np.random.RandomState(act_seed)
            s_a = game.new_initial_state()
            advance_chance_nodes(s_a, d_rng_a)

            while not s_a.is_terminal():
                if s_a.is_chance_node():
                    advance_chance_nodes(s_a, d_rng_a)
                    continue
                cur_p = s_a.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_a, cur_p)
                raw_inf = np.array(s_a.information_state_tensor(cur_p), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)

                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                if cur_p == hero_seat:
                    # Argmax greedy action from Transformer
                    t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = hero_avg_model(t_obs)
                        t_mask = torch.from_numpy(mask).unsqueeze(0)
                        probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(probs))
                else:
                    t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        res = table_a[cur_p](t_obs)
                        logits = res[0] if isinstance(res, tuple) else res
                        t_mask = torch.from_numpy(mask).unsqueeze(0)
                        probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                    act_6 = int(np.argmax(probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_a, cur_p, act_6)
                s_a.apply_action(engine_act)

            apollo_seed_chips += s_a.returns()[hero_seat]

            # B. Baseline Base 2000h Agent_B Hand (Exact Same Cards & Matchup)
            table_b = [None] * 5
            table_b[hero_seat] = base_model
            opp_ptr = 0
            for s_pos in range(5):
                if s_pos != hero_seat:
                    table_b[s_pos] = anchor_list[opp_ptr]
                    opp_ptr += 1

            d_rng_b = np.random.RandomState(deck_seed)
            a_rng_b = np.random.RandomState(act_seed)
            s_b = game.new_initial_state()
            advance_chance_nodes(s_b, d_rng_b)

            while not s_b.is_terminal():
                if s_b.is_chance_node():
                    advance_chance_nodes(s_b, d_rng_b)
                    continue
                cur_p = s_b.current_player()
                legals = True6ActionAdapter.get_legal_6actions(s_b, cur_p)
                raw_inf = np.array(s_b.information_state_tensor(cur_p), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)

                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    res = table_b[cur_p](t_obs)
                    logits = res[0] if isinstance(res, tuple) else res
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                act_6 = int(np.argmax(probs))

                engine_act = True6ActionAdapter.map_6action_to_engine_amount(s_b, cur_p, act_6)
                s_b.apply_action(engine_act)

            base_seed_chips += s_b.returns()[hero_seat]

        apollo_bb = apollo_seed_chips / hands_per_seed
        base_bb = base_seed_chips / hands_per_seed
        seed_apollo_bbs.append(apollo_bb)
        seed_base_bbs.append(base_bb)

        if (s_idx + 1) % 25 == 0 or (s_idx + 1) == num_seeds:
            cur_ap = np.mean(seed_apollo_bbs)
            cur_ba = np.mean(seed_base_bbs)
            print(f"  Eval Seeds: {s_idx + 1:3d}/{num_seeds} | Apollo Champion: {cur_ap:+7.2f} vs Base: {cur_ba:+7.2f} BB/100")

    apollo_mean = float(np.mean(seed_apollo_bbs))
    apollo_se = float(stats.sem(seed_apollo_bbs))
    base_mean = float(np.mean(seed_base_bbs))
    base_se = float(stats.sem(seed_base_bbs))

    diffs = np.array(seed_apollo_bbs) - np.array(seed_base_bbs)
    delta_mean = float(np.mean(diffs))
    delta_se = float(stats.sem(diffs))
    ci95 = [float(delta_mean - 1.984 * delta_se), float(delta_mean + 1.984 * delta_se)]
    t_stat, p_val = stats.ttest_rel(seed_apollo_bbs, seed_base_bbs)

    print("\n" + "=" * 65)
    print("APOLLO STEP 4 BENCHMARK RESULTS")
    print(f"Apollo Main Champion:  {apollo_mean:+7.2f} BB/100 (SE: {apollo_se:.2f})")
    print(f"Baseline Base 2000h:   {base_mean:+7.2f} BB/100 (SE: {base_se:.2f})")
    print(f"Delta:                 {delta_mean:+7.2f} BB/100 (95% CI: [{ci95[0]:.2f}, {ci95[1]:.2f}], p={p_val:.4f})")
    print("=" * 65)

    eval_evidence = {
        "benchmark": "100_sealed_seeds_vs_7k_anchors",
        "num_seeds": num_seeds,
        "hands_per_seed": hands_per_seed,
        "total_hands_evaluated": num_seeds * hands_per_seed,
        "apollo_main_champion_bb100": apollo_mean,
        "apollo_main_champion_se": apollo_se,
        "baseline_bb100": base_mean,
        "baseline_se": base_se,
        "delta_bb100": delta_mean,
        "ci95": ci95,
        "p_value": float(p_val),
        "statistical_significance": bool(p_val < 0.05 and delta_mean > 0),
        "target_exceeded_100bb": bool(apollo_mean >= 100.0)
    }

    evidence_path = os.path.join(REPORTS_DIR, "apollo_step4_eval_evidence.json")
    with open(evidence_path, "w") as f:
        json.dump(eval_evidence, f, indent=2)
    print(f"\n[+] Step 4 Evaluation evidence written to {evidence_path}")
    return eval_evidence

if __name__ == "__main__":
    run_step4_main_league()
