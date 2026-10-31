#!/usr/bin/env python3
"""
v1024/apollo_step3_scaleup_training.py
Project Apollo Step 3: 20,000 Table-Hand Large-Scale NFSP & Average Policy Training.
Authorized by Lead RL Scientist Dr. Max.

Key Specifications:
  1. Inherits weights from Step 2 Average Policy models (exp_apollo_step2/avg_models).
  2. Scale: 20,000 table hands (80,000 active hero hands) with seed 7010000.
  3. Hardware Constraint: Strictly forces torch.set_num_threads(4) for CPU usage <= 40%.
  4. Hyperparameters:
     - lr = 0.0002 (stable convergence)
     - EMA Decay = 0.998 (long-term Nash equilibrium convergence over 20k hands)
     - Target Entropy = 0.75
  5. 100% in-loop strict pot verification (calculate_pots_exact) and zero-sum checks.
  6. Linked 100 independent sealed seeds benchmark evaluation (2501000 - 2501099).
"""

import os
import sys
import json
import time
import hashlib
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
from v1024.directive058_production_engine import get_directive058_balanced_assignment
from v1024.test_directive059f_sidepot_cases import calculate_pots_exact, eval_7card, parse_cards_from_state

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
ACTION_NAMES = ["Fold", "Check/Call", "Raise_Min", "Raise_HalfPot", "Raise_Pot", "All_In"]
INIT_CKPT_DIR = os.path.join(ROOT_DIR, "v1024/exp_apollo_step2/avg_models")
EXP_DIR = os.path.join(ROOT_DIR, "v1024/exp_apollo_step3")
BENCH_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
REPORTS_DIR = os.path.join(ROOT_DIR, "v1024/reports")

class ApolloStep3Agent:
    def __init__(self, name: str, init_path: str, lr: float = 0.0002, ema_decay: float = 0.998):
        self.name = name
        # Best Response Model (Active Learner)
        self.br_model = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        self.br_model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        self.optimizer = optim.Adam(self.br_model.parameters(), lr=lr, eps=1e-5)

        # Average Policy Model (EMA Nash Tracker)
        self.avg_model = CardTransformerPokerPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        self.avg_model.load_state_dict(self.br_model.state_dict())
        self.avg_model.eval()
        self.ema_decay = ema_decay

        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.rew_buf = []
        self.legal_masks = []

        self.total_chips = 0.0
        self.hands_played = 0
        self.current_entropy = 1.0
        self.target_entropy = 0.75
        self.beta_entropy = 0.02
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0

    def clear_buffer(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.val_buf.clear()
        self.rew_buf.clear()
        self.legal_masks.clear()

    def update_ema_average_policy(self):
        with torch.no_grad():
            for avg_p, br_p in zip(self.avg_model.parameters(), self.br_model.parameters()):
                avg_p.data.mul_(self.ema_decay).add_(br_p.data, alpha=1.0 - self.ema_decay)

    def update_beta_entropy(self) -> float:
        diff = self.target_entropy - self.current_entropy
        beta = 0.02 * np.exp(2.0 * diff)
        self.beta_entropy = float(np.clip(beta, 0.01, 0.20))
        return self.beta_entropy

def run_apollo_step3_pipeline(total_hands: int = 20000, seed: int = 7010000):
    print("=================================================================")
    print("PROJECT APOLLO: STEP 3 (20,000 TABLE-HAND SCALE-UP)")
    print("Model: CardTransformerPokerPlayer (Self-Attention + ResNet)")
    print(f"Hardware Constraint: PyTorch Threads = {torch.get_num_threads()} (Strictly <= 40% CPU)")
    print(f"Total Table Hands: {total_hands} | Seed: {seed}")
    print("=================================================================")

    os.makedirs(EXP_DIR, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Initialize Agents with Step 2 weights
    print("\n[1/3] Loading Step 2 Card-Transformer Initial Weights...")
    agents = {}
    for name in AGENT_NAMES:
        p_init = os.path.join(INIT_CKPT_DIR, f"{name}.pt")
        agents[name] = ApolloStep3Agent(name, p_init, lr=0.0002, ema_decay=0.998)
        print(f"  - {name}: Initialized from {p_init}")

    # 2. Load 7k External Anchor Opponents
    print("\n[2/3] Loading 7k External Anchor Opponents...")
    bench_models = {}
    for name in AGENT_NAMES:
        p = os.path.join(BENCH_DIR, f"{name}_7000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        bench_models[name] = m

    game = get_5p_fullgame()
    matchmaking_rng = np.random.RandomState(seed ^ 0xAAAAAAAA)
    batch_size = 200  # PPO update interval
    pot_audit_passes = 0
    max_zero_sum_err = 0.0
    t_start = time.time()

    print("\n[3/3] Commencing 20,000 Table-Hand NFSP Scale-Up Training...")

    for hand in range(1, total_hands + 1):
        assign = get_directive058_balanced_assignment(hand)
        hero_seat_map = assign["agent_to_seat"]
        opp_seat = assign["opp_seat"]

        # Matchmaking: 55% Self-Play with Average Policy, 45% External 7k Anchor
        p_mode = matchmaking_rng.rand()
        if p_mode < 0.55:
            opp_agent_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_key = f"avg_{opp_agent_name}"
            opp_model = agents[opp_agent_name].avg_model
        else:
            opp_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_key = f"7k_{opp_name}"
            opp_model = bench_models[opp_name]

        table = [None] * 5
        table[opp_seat] = ("OPPONENT", opp_key, opp_model)
        for h_name, s_idx in hero_seat_map.items():
            table[s_idx] = ("HERO", h_name, agents[h_name])

        deck_seed = (seed * 1000003 + hand * 10007) & 0xFFFFFFFF
        deck_rng = np.random.RandomState(deck_seed)
        action_rng = np.random.RandomState(deck_seed ^ 0x66666666)

        s = game.new_initial_state()
        advance_chance_nodes(s, deck_rng)

        folded = [False] * 5
        hand_steps = {h_name: [] for h_name in hero_seat_map.keys()}

        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, deck_rng)
                continue

            cp = s.current_player()
            p_type, p_id, p_obj = table[cp]
            legals = True6ActionAdapter.get_legal_6actions(s, cp)
            raw_inf = np.array(s.information_state_tensor(cp), dtype=np.float32)
            norm_inf = normalize_inf_state_v79(raw_inf)

            mask = np.full(6, -1e9, dtype=np.float32)
            for a in legals: mask[a] = 0.0

            if p_type == "HERO":
                agent = p_obj
                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    logits, val = agent.br_model(t_obs)
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                
                act = int(action_rng.choice(len(probs), p=probs))
                log_p = float(np.log(probs[act] + 1e-12))
                v_est = float(val.item())

                hand_steps[p_id].append({
                    "obs": norm_inf,
                    "act": act,
                    "log_p": log_p,
                    "val": v_est,
                    "mask": mask
                })
            else:
                opp_m = p_obj
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
        engine_rets = list(s.returns())

        # Zero sum check
        err = abs(sum(engine_rets))
        if err > max_zero_sum_err: max_zero_sum_err = err
        if err > 1e-4: raise AssertionError(f"Zero sum failed: {err}")

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

        diff = max(abs(c - e) for c, e in zip(calc_rets, engine_rets))
        if diff > 1e-4:
            raise RuntimeError(f"Audit pot discrepancy: diff={diff}")
        pot_audit_passes += 1

        # Reward Buffering with Street-level attribution
        for h_name, steps in hand_steps.items():
            s_idx = hero_seat_map[h_name]
            net_chips = float(engine_rets[s_idx])
            agents[h_name].total_chips += net_chips
            agents[h_name].hands_played += 1
            reward_bb = net_chips / 100.0

            n_steps = len(steps)
            for i, step in enumerate(steps):
                step_rew = reward_bb if i == n_steps - 1 else 0.0
                agents[h_name].obs_buf.append(step["obs"])
                agents[h_name].act_buf.append(step["act"])
                agents[h_name].logp_buf.append(step["log_p"])
                agents[h_name].val_buf.append(step["val"])
                agents[h_name].rew_buf.append(step_rew)
                agents[h_name].legal_masks.append(step["mask"])

        # PPO Update
        if hand % batch_size == 0:
            for a_name, agent in agents.items():
                if len(agent.obs_buf) < 10:
                    continue

                b_obs = torch.tensor(np.array(agent.obs_buf), dtype=torch.float32)
                b_act = torch.tensor(agent.act_buf, dtype=torch.long)
                b_old_logp = torch.tensor(agent.logp_buf, dtype=torch.float32)
                b_old_val = torch.tensor(agent.val_buf, dtype=torch.float32)
                b_rew = np.array(agent.rew_buf, dtype=np.float32)
                b_masks = torch.tensor(np.array(agent.legal_masks), dtype=torch.float32)

                b_ret = np.zeros_like(b_rew)
                running = 0.0
                for t in reversed(range(len(b_rew))):
                    running = b_rew[t] + 0.99 * running
                    b_ret[t] = running
                b_ret_t = torch.tensor(b_ret, dtype=torch.float32)
                b_adv = b_ret_t - b_old_val
                if len(b_adv) > 1 and b_adv.std() > 1e-6:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                agent.br_model.train()
                clip_eps = 0.2
                for _ in range(4):
                    agent.optimizer.zero_grad()
                    logits, val = agent.br_model(b_obs)
                    masked_logits = logits + b_masks
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    new_logp = dist.log_prob(b_act)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_logp - b_old_logp)
                    surr1 = ratios * b_adv
                    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    val_clipped = b_old_val + torch.clamp(val.squeeze(-1) - b_old_val, -clip_eps, clip_eps)
                    v_loss1 = (val.squeeze(-1) - b_ret_t) ** 2
                    v_loss2 = (val_clipped - b_ret_t) ** 2
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                    beta_ent = agent.update_beta_entropy()
                    total_loss = policy_loss + 0.5 * value_loss - beta_ent * entropy
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(agent.br_model.parameters(), max_norm=0.5)
                    agent.optimizer.step()

                agent.current_entropy = float(entropy.item())
                agent.last_policy_loss = float(policy_loss.item())
                agent.last_value_loss = float(value_loss.item())
                agent.clear_buffer()

                # Update EMA Average Policy
                agent.update_ema_average_policy()

            if hand % 1000 == 0:
                ent_str = " ".join([f"{n[-1]}:{agents[n].current_entropy:.2f}" for n in AGENT_NAMES])
                vloss_str = " ".join([f"{n[-1]}:{agents[n].last_value_loss:.1f}" for n in AGENT_NAMES])
                rate = hand / (time.time() - t_start)
                print(f"  Hand {hand:5d}/{total_hands} | HPS: {rate:4.1f} | Ent: {ent_str} | VLoss: {vloss_str}")

    train_elapsed = time.time() - t_start
    print(f"\n[+] Apollo Step 3 Scale-Up Training Complete in {train_elapsed:.1f}s.")

    # Save Checkpoints
    ckpt_dir = os.path.join(EXP_DIR, "final_avg_models")
    os.makedirs(ckpt_dir, exist_ok=True)
    for name, agent in agents.items():
        p_avg = os.path.join(ckpt_dir, f"{name}.pt")
        torch.save({"model_state_dict": agent.avg_model.state_dict()}, p_avg)
        print(f"  - Saved 20k-Hand {name} Average Policy to {p_avg}")

    train_evidence = {
        "directive": "APOLLO_STEP3_SCALEUP",
        "lead_scientist": "Dr. Max",
        "total_table_hands": total_hands,
        "pot_audit_passes": pot_audit_passes,
        "max_zero_sum_error": max_zero_sum_err,
        "train_elapsed_sec": round(train_elapsed, 2),
        "final_entropies": {n: round(agents[n].current_entropy, 3) for n in AGENT_NAMES},
        "final_vlosses": {n: round(agents[n].last_value_loss, 2) for n in AGENT_NAMES}
    }
    with open(os.path.join(REPORTS_DIR, "apollo_step3_training_evidence.json"), "w") as f:
        json.dump(train_evidence, f, indent=2)

    # 4. Confirmatory 100 Independent Sealed Seeds Benchmark Evaluation
    print("\n=================================================================")
    print("STARTING APOLLO STEP 3 INDEPENDENT BENCHMARK EVALUATION (100 Seeds)")
    print("Apollo 20k Average Models vs Baseline Base 2000h (Opponent: 7k)")
    print("=================================================================")

    eval_evidence = run_step3_evaluation(agents, num_seeds=100, seed_start=2501000)
    return eval_evidence

def run_step3_evaluation(agents: dict, num_seeds: int = 100, seed_start: int = 2501000):
    game = get_5p_fullgame()
    hands_per_seed = 5

    # Load Baseline Models (Base 2000h)
    baseline_models = {}
    bench_models = {}
    for name in AGENT_NAMES:
        p_base = os.path.join(BENCH_DIR, f"{name}_2000h.pt")
        m_b = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_b = torch.load(p_base, map_location="cpu", weights_only=False)
        m_b.load_state_dict(ckpt_b["model_state_dict"] if isinstance(ckpt_b, dict) and "model_state_dict" in ckpt_b else ckpt_b)
        m_b.eval()
        baseline_models[name] = m_b

        p_7k = os.path.join(BENCH_DIR, f"{name}_7000h.pt")
        m_7k = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt_7k = torch.load(p_7k, map_location="cpu", weights_only=False)
        m_7k.load_state_dict(ckpt_7k["model_state_dict"] if isinstance(ckpt_7k, dict) and "model_state_dict" in ckpt_7k else ckpt_7k)
        m_7k.eval()
        bench_models[name] = m_7k

    apollo_chips = {n: 0.0 for n in AGENT_NAMES}
    base_chips = {n: 0.0 for n in AGENT_NAMES}
    apollo_hands = {n: 0 for n in AGENT_NAMES}
    base_hands = {n: 0 for n in AGENT_NAMES}
    apollo_actions = {n: [0] * 6 for n in AGENT_NAMES}
    base_actions = {n: [0] * 6 for n in AGENT_NAMES}

    seed_apollo_returns = []
    seed_base_returns = []

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

            # A. Apollo 20k Average Models Run
            table_a = [None] * 5
            table_a[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                table_a[s_pos] = agents[a_name].avg_model

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
            for a_name, s_pos in hero_seat_map.items():
                apollo_chips[a_name] += rets_a[s_pos]
                apollo_hands[a_name] += 1
                s_apollo_ret += rets_a[s_pos]

            # B. Baseline Base 2000h Run
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
            print(f"  Eval Seeds: {s_idx+1:3d}/{num_seeds} | Apollo 20k: {c_bb:+6.2f} vs Base: {b_bb:+6.2f} BB/100")

    apollo_arr = np.array(seed_apollo_returns)
    base_arr = np.array(seed_base_returns)
    diff_arr = apollo_arr - base_arr

    mean_apollo = float(np.mean(apollo_arr))
    mean_base = float(np.mean(base_arr))
    delta = float(np.mean(diff_arr))
    se_delta = float(stats.sem(diff_arr))
    ci95 = [round(delta - 1.96 * se_delta, 2), round(delta + 1.96 * se_delta, 2)]
    t_stat, p_val = stats.ttest_rel(apollo_arr, base_arr)

    print("\n=================================================================")
    print("APOLLO STEP 3 BENCHMARK RESULTS")
    print(f"Apollo 20k Average Models: {mean_apollo:+.2f} BB/100 (SE: {float(stats.sem(apollo_arr)):.2f})")
    print(f"Baseline Base 2000h:       {mean_base:+.2f} BB/100 (SE: {float(stats.sem(base_arr)):.2f})")
    print(f"Delta:                     {delta:+.2f} BB/100 (95% CI: [{ci95[0]}, {ci95[1]}], p={p_val:.4f})")
    print("=================================================================")

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
        "directive": "APOLLO_STEP3_EVALUATION",
        "lead_scientist": "Dr. Max",
        "independent_seeds_count": num_seeds,
        "seed_range": [seed_start, seed_start + num_seeds - 1],
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
            "Apollo_20k_Average": apollo_act_dist,
            "Baseline_2000h": base_act_dist
        }
    }

    out_path = os.path.join(REPORTS_DIR, "apollo_step3_eval_evidence.json")
    with open(out_path, "w") as f:
        json.dump(eval_evidence, f, indent=2)
    print(f"\n[+] Step 3 Evaluation evidence written to {out_path}")

    return eval_evidence

if __name__ == "__main__":
    run_apollo_step3_pipeline(total_hands=20000, seed=7010000)
