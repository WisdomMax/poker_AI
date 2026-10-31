#!/usr/bin/env python3
"""
v1024/directive063_champion_scaleup.py
DIRECTIVE 063: Champion Distillation & Exploitability Minimization Scale-up (5,000h).
Authorized by Lead RL Scientist Dr. Max.

Key Specifications:
  1. Inherits weights from DIRECTIVE 062 3,000h champions (ckpt_hand_3000).
  2. Targeted Exploiter & Elite League Matchmaking:
     - 50% Active learner matchups
     - 35% Top Elite Snapshots (Agent_D 3000h, Agent_A 3000h, Base 2000h Champions)
     - 15% Historical Archive & Anchor Snapshots
  3. Fine-tuning Hyperparameters:
     - lr = 0.0003
     - H_target = 0.75 (adaptive entropy floor)
     - BB reward scaling (/100.0)
  4. Scale-up: 5,000 table hands (20,000 active hero hands) with seed 5040000.
  5. Checkpoints saved every 1,000 table hands.
  6. Strict in-loop independent pot verification (calculate_pots_exact) and zero-sum checks.
  7. Linked 200 sealed seeds benchmark evaluation (1501000 - 1501199) vs Base 2000h.
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

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.directive058_production_engine import get_directive058_balanced_assignment
from v1024.test_directive059f_sidepot_cases import calculate_pots_exact, eval_7card, parse_cards_from_state

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
ACTION_NAMES = ["Fold", "Check/Call", "Raise_Min", "Raise_HalfPot", "Raise_Pot", "All_In"]
CKPT_062_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive062/ckpt_hand_3000")
BASE_2000_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
EXP_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive063")

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class AlphaLeagueLearner:
    def __init__(self, name: str, init_model_path: str, lr: float = 0.0003, target_entropy: float = 0.75):
        self.name = name
        self.model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(init_model_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.rew_buf = []
        self.legal_masks = []
        
        self.total_chips = 0.0
        self.hands_played = 0
        self.update_count = 0
        
        self.current_entropy = 1.0
        self.target_entropy = target_entropy
        self.beta_entropy = 0.01
        
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_clip_frac = 0.0

    def clear_buffer(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.val_buf.clear()
        self.rew_buf.clear()
        self.legal_masks.clear()

    def update_beta_entropy(self) -> float:
        diff = self.target_entropy - self.current_entropy
        beta = 0.01 * np.exp(2.0 * diff)
        self.beta_entropy = float(np.clip(beta, 0.01, 0.20))
        return self.beta_entropy

def run_directive063_pipeline(total_hands: int = 5000, seed: int = 5040000):
    print("=================================================================")
    print("DIRECTIVE 063: CHAMPION SCALE-UP & EXPLOITABILITY MINIMIZATION (5,000h)")
    print(f"Lead Scientist: Dr. Max | Total Hands: {total_hands} | Seed: {seed}")
    print("=================================================================")

    os.makedirs(EXP_DIR, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Initialize Active Learners with DIRECTIVE 062 3,000h weights
    print("[1/3] Loading DIRECTIVE 062 3,000h Champion Weights...")
    learners = {}
    init_hashes = {}
    for name in AGENT_NAMES:
        path = os.path.join(CKPT_062_DIR, f"{name}.pt")
        init_hashes[name] = compute_sha256(path)
        learners[name] = AlphaLeagueLearner(name, path, lr=0.0003, target_entropy=0.75)
        print(f"  - {name}: loaded from {path} (SHA: {init_hashes[name][:12]}...)")

    # 2. Build Elite and Historical League Pools
    print("[2/3] Building Targeted Elite & League Pools...")
    elite_pool = {}
    # Top performers from 062 and Base 2000h
    for e_name in ["Agent_A", "Agent_D"]:
        p_062 = os.path.join(CKPT_062_DIR, f"{e_name}.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p_062, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        m.eval()
        elite_pool[f"elite_062_{e_name}"] = m

    for b_name in ["Agent_A", "Agent_D", "Agent_E"]:
        p_base = os.path.join(BASE_2000_DIR, f"{b_name}_2000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p_base, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        elite_pool[f"elite_base_{b_name}"] = m

    # General Archive & Anchors
    archive_pool = {}
    for name in AGENT_NAMES:
        p_base = os.path.join(BASE_2000_DIR, f"{name}_2000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p_base, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        archive_pool[f"archive_base_{name}"] = m

    # Load 062 historical checkpoints into archive
    for h_step in [1000, 2000]:
        h_dir = os.path.join(ROOT_DIR, f"v1024/exp_directive062/ckpt_hand_{h_step}")
        if os.path.exists(h_dir):
            for name in AGENT_NAMES:
                p_h = os.path.join(h_dir, f"{name}.pt")
                if os.path.exists(p_h):
                    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
                    ckpt = torch.load(p_h, map_location="cpu", weights_only=False)
                    m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                    m.eval()
                    archive_pool[f"archive_062_h{h_step}_{name}"] = m

    print(f"  - Elite Opponent Pool: {len(elite_pool)} models")
    print(f"  - Archive & Anchor Pool: {len(archive_pool)} models")

    game = get_5p_fullgame()
    matchmaking_rng = np.random.RandomState(seed ^ 0x77777777)

    # Logging trackers
    update_records = []
    checkpoint_records = []
    total_foldouts = 0
    total_showdowns = 0
    pot_audit_passes = 0
    max_zero_sum_err = 0.0

    t_start = time.time()
    batch_size = 250  # PPO update interval

    print("\n[3/3] Commencing 5,000 Table-Hand AlphaStar League Training...")

    for hand in range(1, total_hands + 1):
        assign = get_directive058_balanced_assignment(hand)
        hero_seat_map = assign["agent_to_seat"]
        opp_seat = assign["opp_seat"]

        # Matchmaking: 50% Active, 35% Elite, 15% Archive
        p_mode = matchmaking_rng.rand()
        if p_mode < 0.50:
            opp_agent_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_key = f"active_{opp_agent_name}"
            opp_model = learners[opp_agent_name].model
        elif p_mode < 0.85:
            e_keys = list(elite_pool.keys())
            opp_key = elite_pool_choice = e_keys[matchmaking_rng.randint(len(e_keys))]
            opp_model = elite_pool[opp_key]
        else:
            a_keys = list(archive_pool.keys())
            opp_key = a_keys[matchmaking_rng.randint(len(a_keys))]
            opp_model = archive_pool[opp_key]

        table = [None] * 5
        table[opp_seat] = ("OPPONENT", opp_key, opp_model)
        for h_name, s_idx in hero_seat_map.items():
            table[s_idx] = ("HERO", h_name, learners[h_name])

        # Deterministic separated RNGs
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
                learner = p_obj
                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    logits, val = learner.model(t_obs)
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
                    logits, _ = opp_m(t_obs)
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                act = int(action_rng.choice(len(probs), p=probs))

            if act == 0:
                folded[cp] = True
            engine_act = True6ActionAdapter.map_6action_to_engine_amount(s, cp, act)
            s.apply_action(engine_act)

        assert s.is_terminal(), "Game must be terminal"
        engine_rets = list(s.returns())

        # 1. Zero-sum verification
        err = abs(sum(engine_rets))
        if err > max_zero_sum_err: max_zero_sum_err = err
        if err > 1e-4: raise AssertionError(f"Zero sum failed: {err}")

        # 2. Card verification
        spents = extract_player_spent(s)
        holes, board = parse_cards_from_state(s.to_string())
        all_deal_cards = []
        for hc in holes: all_deal_cards.extend(hc)
        all_deal_cards.extend(board)
        if len(all_deal_cards) != len(set(all_deal_cards)):
            raise RuntimeError(f"Duplicate card detected: {all_deal_cards}")

        # 3. In-path Independent Pot Verification
        active_players = [p for p in range(5) if not folded[p]]
        audit_executed = False
        calc_rets = [0.0] * 5

        if len(active_players) == 1:
            total_foldouts += 1
            winner = active_players[0]
            calc_rets = [-spents[p] for p in range(5)]
            calc_rets[winner] = sum(spents) - spents[winner]
            audit_executed = True
        elif len(active_players) > 1:
            total_showdowns += 1
            scores = [None] * 5
            for p in active_players:
                scores[p] = eval_7card(holes[p] + board)
            calc_rets = calculate_pots_exact(spents, folded, scores)
            audit_executed = True

        if audit_executed:
            diff = max(abs(c - e) for c, e in zip(calc_rets, engine_rets))
            if diff > 1e-4:
                raise RuntimeError(f"Audit pot discrepancy: diff={diff}, calc={calc_rets}, engine={engine_rets}")
            pot_audit_passes += 1

        # 4. Trajectory Recording with BB Scaling
        for h_name, steps in hand_steps.items():
            s_idx = hero_seat_map[h_name]
            net_chips = float(engine_rets[s_idx])
            learners[h_name].total_chips += net_chips
            learners[h_name].hands_played += 1
            reward_bb = net_chips / 100.0  # 100 chips = 1 BB

            n_steps = len(steps)
            for i, step in enumerate(steps):
                step_rew = reward_bb if i == n_steps - 1 else 0.0
                learners[h_name].obs_buf.append(step["obs"])
                learners[h_name].act_buf.append(step["act"])
                learners[h_name].logp_buf.append(step["log_p"])
                learners[h_name].val_buf.append(step["val"])
                learners[h_name].rew_buf.append(step_rew)
                learners[h_name].legal_masks.append(step["mask"])

        # PPO Update interval
        if hand % batch_size == 0:
            for l_name, learner in learners.items():
                if len(learner.obs_buf) < 10:
                    continue

                b_obs = torch.tensor(np.array(learner.obs_buf), dtype=torch.float32)
                b_act = torch.tensor(learner.act_buf, dtype=torch.long)
                b_old_logp = torch.tensor(learner.logp_buf, dtype=torch.float32)
                b_old_val = torch.tensor(learner.val_buf, dtype=torch.float32)
                b_rew = np.array(learner.rew_buf, dtype=np.float32)
                b_masks = torch.tensor(np.array(learner.legal_masks), dtype=torch.float32)

                # Return calculation
                b_ret = np.zeros_like(b_rew)
                running = 0.0
                for t in reversed(range(len(b_rew))):
                    running = b_rew[t] + 0.99 * running
                    b_ret[t] = running
                b_ret_t = torch.tensor(b_ret, dtype=torch.float32)
                b_adv = b_ret_t - b_old_val
                if len(b_adv) > 1 and b_adv.std() > 1e-6:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                # PPO Epochs
                learner.model.train()
                clip_eps = 0.2
                for _ in range(4):
                    learner.optimizer.zero_grad()
                    logits, val = learner.model(b_obs)
                    masked_logits = logits + b_masks
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    new_logp = dist.log_prob(b_act)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_logp - b_old_logp)
                    surr1 = ratios * b_adv
                    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value clipping
                    val_clipped = b_old_val + torch.clamp(val.squeeze(-1) - b_old_val, -clip_eps, clip_eps)
                    v_loss1 = (val.squeeze(-1) - b_ret_t) ** 2
                    v_loss2 = (val_clipped - b_ret_t) ** 2
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                    beta_ent = learner.update_beta_entropy()
                    total_loss = policy_loss + 0.5 * value_loss - beta_ent * entropy
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(learner.model.parameters(), max_norm=0.5)
                    learner.optimizer.step()

                learner.current_entropy = float(entropy.item())
                learner.last_policy_loss = float(policy_loss.item())
                learner.last_value_loss = float(value_loss.item())
                learner.clear_buffer()
                learner.update_count += 1

            ent_str = " ".join([f"{n[-1]}:{learners[n].current_entropy:.2f}" for n in AGENT_NAMES])
            vloss_str = " ".join([f"{n[-1]}:{learners[n].last_value_loss:.1f}" for n in AGENT_NAMES])
            rate = hand / (time.time() - t_start)
            print(f"  Hand {hand:5d}/{total_hands} | HPS: {rate:4.1f} | Ent: {ent_str} | VLoss: {vloss_str}")

        # League Snapshot Archive & Checkpoint save every 1,000 table hands
        if hand % 1000 == 0:
            ckpt_sub = os.path.join(EXP_DIR, f"ckpt_hand_{hand}")
            os.makedirs(ckpt_sub, exist_ok=True)
            for name, learner in learners.items():
                p = os.path.join(ckpt_sub, f"{name}.pt")
                torch.save({"model_state_dict": learner.model.state_dict()}, p)
                
                # Add snapshot to archive pool
                snap = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
                snap.load_state_dict(learner.model.state_dict())
                snap.eval()
                archive_pool[f"archive_063_h{hand}_{name}"] = snap
            print(f"[+] [ALPHA LEAGUE] Checkpoint saved at hand {hand}. Total Archive Pool: {len(archive_pool)} models.")

    train_elapsed = time.time() - t_start
    print(f"\n[+] DIRECTIVE 063 Training Complete in {train_elapsed:.1f}s.")

    # Save training evidence
    training_evidence = {
        "directive": "DIRECTIVE_063",
        "lead_scientist": "Dr. Max",
        "total_table_hands": total_hands,
        "active_hero_hands": total_hands * 4,
        "seed": seed,
        "learning_rate": 0.0003,
        "target_entropy": 0.75,
        "total_foldouts": total_foldouts,
        "total_showdowns": total_showdowns,
        "pot_audit_passes": pot_audit_passes,
        "max_zero_sum_error": max_zero_sum_err,
        "final_entropies": {n: learners[n].current_entropy for n in AGENT_NAMES},
        "final_value_losses": {n: learners[n].last_value_loss for n in AGENT_NAMES},
        "archive_pool_size": len(archive_pool),
        "elite_pool_size": len(elite_pool)
    }
    with open(os.path.join(ROOT_DIR, "v1024/reports/directive063_training_evidence.json"), "w") as f:
        json.dump(training_evidence, f, indent=2)

    # 4. Confirmatory 200 Sealed Seeds Evaluation
    print("\n=================================================================")
    print("STARTING DIRECTIVE 063 BENCHMARK EVALUATION (200 Sealed Seeds)")
    print("Candidate: Directive063_Champion_5000h vs Baseline: Base_2000h")
    print("=================================================================")

    eval_results = run_directive063_sealed_eval(learners)
    return eval_results

def run_directive063_sealed_eval(candidate_learners: dict, num_seeds: int = 200, seed_start: int = 1501000):
    game = get_5p_fullgame()
    hands_per_seed = 5

    # Load Baseline Models (Base 2000h)
    baseline_models = {}
    for name in AGENT_NAMES:
        p = os.path.join(BASE_2000_DIR, f"{name}_2000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        baseline_models[name] = m

    # Benchmark Opponents (Fixed Benchmark 7k)
    bench_models = {}
    bench_dir = os.path.join(ROOT_DIR, "v1024/models_alphago")
    for name in AGENT_NAMES:
        p = os.path.join(bench_dir, f"{name}_7000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        bench_models[name] = m

    cand_chips = {n: 0.0 for n in AGENT_NAMES}
    base_chips = {n: 0.0 for n in AGENT_NAMES}
    cand_hands = {n: 0 for n in AGENT_NAMES}
    base_hands = {n: 0 for n in AGENT_NAMES}

    cand_actions = {n: [0] * 6 for n in AGENT_NAMES}
    base_actions = {n: [0] * 6 for n in AGENT_NAMES}

    seed_cand_deltas = []
    seed_base_deltas = []

    for s_idx in range(num_seeds):
        curr_seed = seed_start + s_idx
        s_cand_ret = 0.0
        s_base_ret = 0.0

        for h in range(hands_per_seed):
            hand_idx = s_idx * hands_per_seed + h + 1
            assign = get_directive058_balanced_assignment(hand_idx)
            hero_seat_map = assign["agent_to_seat"]
            opp_seat = assign["opp_seat"]
            opp_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_m = bench_models[opp_name]

            # Common random seeds
            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x66666666

            # 1. Candidate play
            cand_table = [None] * 5
            cand_table[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                cand_table[s_pos] = candidate_learners[a_name].model

            c_deck_rng = np.random.RandomState(deck_seed)
            c_act_rng = np.random.RandomState(act_seed)
            s_c = game.new_initial_state()
            advance_chance_nodes(s_c, c_deck_rng)

            c_folded = [False] * 5
            while not s_c.is_terminal():
                if s_c.is_chance_node():
                    advance_chance_nodes(s_c, c_deck_rng)
                    continue
                cp = s_c.current_player()
                m = cand_table[cp]
                legals = True6ActionAdapter.get_legal_6actions(s_c, cp)
                raw_inf = np.array(s_c.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                with torch.no_grad():
                    logits, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                    probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                act = int(c_act_rng.choice(len(probs), p=probs))

                for a_name, s_pos in hero_seat_map.items():
                    if cp == s_pos:
                        cand_actions[a_name][act] += 1
                if act == 0:
                    c_folded[cp] = True
                s_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(s_c, cp, act))

            c_rets = list(s_c.returns())
            for a_name, s_pos in hero_seat_map.items():
                cand_chips[a_name] += c_rets[s_pos]
                cand_hands[a_name] += 1
                s_cand_ret += c_rets[s_pos]

            # 2. Baseline play
            base_table = [None] * 5
            base_table[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                base_table[s_pos] = baseline_models[a_name]

            b_deck_rng = np.random.RandomState(deck_seed)
            b_act_rng = np.random.RandomState(act_seed)
            s_b = game.new_initial_state()
            advance_chance_nodes(s_b, b_deck_rng)

            b_folded = [False] * 5
            while not s_b.is_terminal():
                if s_b.is_chance_node():
                    advance_chance_nodes(s_b, b_deck_rng)
                    continue
                cp = s_b.current_player()
                m = base_table[cp]
                legals = True6ActionAdapter.get_legal_6actions(s_b, cp)
                raw_inf = np.array(s_b.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                with torch.no_grad():
                    logits, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                    probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                act = int(b_act_rng.choice(len(probs), p=probs))

                for a_name, s_pos in hero_seat_map.items():
                    if cp == s_pos:
                        base_actions[a_name][act] += 1
                if act == 0:
                    b_folded[cp] = True
                s_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(s_b, cp, act))

            b_rets = list(s_b.returns())
            for a_name, s_pos in hero_seat_map.items():
                base_chips[a_name] += b_rets[s_pos]
                base_hands[a_name] += 1
                s_base_ret += b_rets[s_pos]

        seed_cand_deltas.append(s_cand_ret / (hands_per_seed * 4))
        seed_base_deltas.append(s_base_ret / (hands_per_seed * 4))

        if (s_idx + 1) % 50 == 0:
            c_bb = np.mean(seed_cand_deltas)
            b_bb = np.mean(seed_base_deltas)
            print(f"  Eval Seeds: {s_idx+1:3d}/{num_seeds} | Cand: {c_bb:+6.2f} vs Base: {b_bb:+6.2f} BB/100")

    # Metrics computation
    cand_bb100_arr = np.array(seed_cand_deltas)
    base_bb100_arr = np.array(seed_base_deltas)
    diff_arr = cand_bb100_arr - base_bb100_arr

    mean_cand = float(np.mean(cand_bb100_arr))
    mean_base = float(np.mean(base_bb100_arr))
    delta = float(np.mean(diff_arr))
    se_delta = float(stats.sem(diff_arr))
    ci95 = [round(delta - 1.96 * se_delta, 2), round(delta + 1.96 * se_delta, 2)]

    t_stat, p_val = stats.ttest_rel(cand_bb100_arr, base_bb100_arr)

    print("\n=================================================================")
    print("DIRECTIVE 063 CHAMPION SCALE-UP COMPLETE")
    print(f"Candidate (5000h): {mean_cand:+.2f} BB/100")
    print(f"Baseline  (2000h): {mean_base:+.2f} BB/100")
    print(f"Delta:            {delta:+.2f} BB/100 (95% CI: [{ci95[0]}, {ci95[1]}], p={p_val:.4f})")
    print("=================================================================")

    # Action distribution summary
    cand_act_dist = {}
    base_act_dist = {}
    for n in AGENT_NAMES:
        c_tot = sum(cand_actions[n])
        b_tot = sum(base_actions[n])
        cand_act_dist[n] = {ACTION_NAMES[i]: round(cand_actions[n][i] / c_tot * 100, 2) for i in range(6)}
        base_act_dist[n] = {ACTION_NAMES[i]: round(base_actions[n][i] / b_tot * 100, 2) for i in range(6)}

    agent_eval_summary = {}
    for n in AGENT_NAMES:
        agent_eval_summary[n] = {
            "cand_bb100": round(cand_chips[n] / cand_hands[n], 2),
            "base_bb100": round(base_chips[n] / base_hands[n], 2),
            "hands": cand_hands[n]
        }

    eval_evidence = {
        "directive": "DIRECTIVE_063",
        "lead_scientist": "Dr. Max",
        "total_eval_table_hands": num_seeds * hands_per_seed,
        "sealed_seeds_count": num_seeds,
        "candidate_summary": {
            "mean_hero_bb100": round(mean_cand, 2),
            "se_bb100": round(float(stats.sem(cand_bb100_arr)), 2),
            "agents": agent_eval_summary
        },
        "baseline_summary": {
            "mean_hero_bb100": round(mean_base, 2),
            "se_bb100": round(float(stats.sem(base_bb100_arr)), 2)
        },
        "paired_comparison": {
            "delta_bb100": round(delta, 2),
            "paired_se_bb100": round(se_delta, 2),
            "ci95": ci95,
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "statistically_significant_outperformance": bool(p_val < 0.05 and delta > 0),
            "statistically_significant_degradation": bool(p_val < 0.05 and delta < 0)
        },
        "action_distributions": {
            "Candidate_063_5000h": cand_act_dist,
            "Baseline_2000h": base_act_dist
        }
    }

    out_path = os.path.join(ROOT_DIR, "v1024/reports/directive063_evaluation_evidence.json")
    with open(out_path, "w") as f:
        json.dump(eval_evidence, f, indent=2)
    print(f"[+] Evaluation evidence written to {out_path}")

    return eval_evidence

if __name__ == "__main__":
    run_directive063_pipeline(total_hands=5000, seed=5040000)
