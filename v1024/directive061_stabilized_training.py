#!/usr/bin/env python3
"""
v1024/directive061_stabilized_training.py
DIRECTIVE 061: Stabilized PPO Training with Adaptive Entropy Floor & Reward Scaling.
Authorized by Lead RL Scientist Dr. Max.

Key Innovations:
  1. Reward Scaling to BB units: R_step = net_chips / (100.0 * len(steps))
  2. Adaptive Entropy Floor: beta_ent = clamp(0.01 * exp(2.0 * (0.80 - H_current)), 0.01, 0.20)
  3. Value Clipping & Normalized Advantage
  4. 1,000 Table-Hands (4,000 Hero-Hands) from clean Base 2000h models
  5. In-loop strict pot verification (calculate_pots_exact) and zero-sum check
  6. Automatic linked confirmatory evaluation on 200 sealed seeds (1501000 - 1501199)
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
MODELS_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
EXP_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive061")

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class StabilizedLearnerContext:
    def __init__(self, name: str, init_model_path: str, lr: float = 0.0003):
        self.name = name
        self.model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(init_model_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # Buffers
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.rew_buf = []
        self.legal_masks = []
        
        self.total_chips = 0.0
        self.hands_played = 0
        self.update_count = 0
        
        # Adaptive Entropy Controller
        self.current_entropy = 1.0
        self.target_entropy = 0.80
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

    def compute_adaptive_beta(self) -> float:
        """
        Max's Adaptive Entropy Floor formula:
        beta = clamp(0.01 * exp(2.0 * (0.80 - H_current)), 0.01, 0.20)
        """
        diff = self.target_entropy - self.current_entropy
        beta = 0.01 * np.exp(2.0 * diff)
        self.beta_entropy = float(np.clip(beta, 0.01, 0.20))
        return self.beta_entropy

def run_directive061_pipeline(total_hands: int = 1000, seed: int = 5020000):
    print("=================================================================")
    print("DIRECTIVE 061: STABILIZED PPO TRAINING & ADAPTIVE ENTROPY ENGINE")
    print(f"Lead Scientist: Dr. Max | Total Hands: {total_hands} | Seed: {seed}")
    print("=================================================================")

    os.makedirs(EXP_DIR, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    game = get_5p_fullgame()

    # 1. Rollback: Load fresh Clean Base 2000h models
    learners = {}
    for a in AGENT_NAMES:
        p = os.path.join(MODELS_DIR, f"{a}_2000h.pt")
        learners[a] = StabilizedLearnerContext(a, p)

    # 2. Opponents (7k)
    opp_pool = {}
    for a in AGENT_NAMES:
        p = os.path.join(MODELS_DIR, f"{a}_7000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=True)
        m.eval()
        opp_pool[f"opp_7k_{a}"] = m

    f_safe = open(os.path.join(EXP_DIR, "safety_verifications.jsonl"), "w", encoding="utf-8")
    f_ppo = open(os.path.join(EXP_DIR, "ppo_updates.jsonl"), "w", encoding="utf-8")

    t0 = time.time()
    total_showdowns = 0
    total_foldouts = 0
    max_pot_discrepancy = 0.0
    max_zero_sum_err = 0.0

    try:
        for hand in range(1, total_hands + 1):
            assign = get_directive058_balanced_assignment(hand)
            opp_seat = assign["opp_seat"]
            opp_agent_idx = assign["opp_model_idx"]
            opp_name = AGENT_NAMES[opp_agent_idx]
            hero_seat_map = assign["agent_to_seat"]
            rest_hero = assign["rest_hero"]

            # Select opponent from pool
            opp_keys = list(opp_pool.keys())
            opp_key = opp_keys[opp_agent_idx % len(opp_keys)]
            opp_model = opp_pool[opp_key]

            # Setup table
            table = [None] * 5
            table[opp_seat] = ("OPPONENT", opp_key, opp_model)
            for h_name, s_idx in hero_seat_map.items():
                table[s_idx] = ("HERO", h_name, learners[h_name])

            # Deterministic separated RNGs
            deck_seed = (seed * 1000003 + hand * 10007) & 0xFFFFFFFF
            deck_rng = np.random.RandomState(deck_seed)
            action_rng = np.random.RandomState(deck_seed ^ 0x44444444)

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
            if err > max_zero_sum_err:
                max_zero_sum_err = err
            if err > 1e-4:
                raise AssertionError(f"[ABORT] Hand {hand}: Zero-sum failed: sum={sum(engine_rets)}")

            # 2. Extract state & card verification
            spents = extract_player_spent(s)
            holes, board = parse_cards_from_state(s.to_string())
            all_deal_cards = []
            for hc in holes: all_deal_cards.extend(hc)
            all_deal_cards.extend(board)
            if len(all_deal_cards) != len(set(all_deal_cards)):
                raise RuntimeError(f"[ABORT] Hand {hand}: Duplicate card detected! Cards: {all_deal_cards}")

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
            else:
                raise RuntimeError(f"[ABORT] Hand {hand}: No active players at showdown!")

            assert audit_executed, f"[AUDIT ERROR] Hand {hand}: audit_executed is False!"
            discrepancy = max(abs(e - c) for e, c in zip(engine_rets, calc_rets))
            if discrepancy > max_pot_discrepancy:
                max_pot_discrepancy = discrepancy
            if discrepancy > 1e-4:
                raise AssertionError(f"[AUDIT ABORT] Hand {hand}: Sidepot mismatch! diff={discrepancy}")

            # 4. Reward Normalization to BB: R_step = net_chips / (100.0 * len(steps))
            for h_name, s_idx in hero_seat_map.items():
                net_chips = float(engine_rets[s_idx])
                learner = learners[h_name]
                learner.total_chips += net_chips
                learner.hands_played += 1
                
                steps = hand_steps[h_name]
                if steps:
                    # Normalized to BB unit (1 BB = 100 chips)
                    step_bb_reward = (net_chips / 100.0) / len(steps)
                    for st in steps:
                        learner.obs_buf.append(st["obs"])
                        learner.act_buf.append(st["act"])
                        learner.logp_buf.append(st["log_p"])
                        learner.val_buf.append(st["val"])
                        learner.rew_buf.append(step_bb_reward)
                        learner.legal_masks.append(st["mask"])

            # 5. PPO Update every 10 hands with Adaptive Entropy Floor
            if hand % 10 == 0:
                for h_name, learner in learners.items():
                    if len(learner.obs_buf) >= 10:
                        obs_t = torch.tensor(np.array(learner.obs_buf), dtype=torch.float32)
                        act_t = torch.tensor(learner.act_buf, dtype=torch.long)
                        old_logp_t = torch.tensor(learner.logp_buf, dtype=torch.float32)
                        old_val_t = torch.tensor(learner.val_buf, dtype=torch.float32)
                        rew_t = torch.tensor(learner.rew_buf, dtype=torch.float32)
                        masks_t = torch.tensor(np.array(learner.legal_masks), dtype=torch.float32)

                        # Advantage computation with standardization
                        adv_t = rew_t - old_val_t
                        if len(adv_t) > 1:
                            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                        # Dynamic Adaptive Entropy Bonus Coefficient
                        beta_ent = learner.compute_adaptive_beta()

                        learner.model.train()
                        for _ in range(4):
                            logits, values = learner.model(obs_t)
                            values = values.squeeze(-1)
                            probs = torch.softmax(logits + masks_t, dim=-1)
                            dist = torch.distributions.Categorical(probs=probs + 1e-12)
                            new_logp = dist.log_prob(act_t)
                            entropy = dist.entropy().mean()

                            # Policy Loss with PPO Clipping
                            ratio = torch.exp(new_logp - old_logp_t)
                            surr1 = ratio * adv_t
                            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_t
                            policy_loss = -torch.min(surr1, surr2).mean()

                            # Value Loss with Value Clipping (Max's standard)
                            val_clipped = old_val_t + torch.clamp(values - old_val_t, -0.2, 0.2)
                            vf_loss1 = (values - rew_t) ** 2
                            vf_loss2 = (val_clipped - rew_t) ** 2
                            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                            # Combined Objective with Adaptive Entropy Floor
                            total_loss = policy_loss + 0.5 * value_loss - beta_ent * entropy

                            learner.optimizer.zero_grad()
                            total_loss.backward()
                            nn.utils.clip_grad_norm_(learner.model.parameters(), 0.5)
                            learner.optimizer.step()

                        learner.model.eval()
                        learner.update_count += 1
                        clip_frac = float(((ratio - 1.0).abs() > 0.2).float().mean().item())
                        learner.current_entropy = float(entropy.item())
                        learner.last_policy_loss = float(policy_loss.item())
                        learner.last_value_loss = float(value_loss.item())
                        learner.last_clip_frac = clip_frac

                        f_ppo.write(json.dumps({
                            "hand": hand,
                            "agent": h_name,
                            "update": learner.update_count,
                            "policy_loss": learner.last_policy_loss,
                            "value_loss": learner.last_value_loss,
                            "entropy": learner.current_entropy,
                            "beta_entropy": beta_ent,
                            "clip_frac": clip_frac
                        }) + "\n")

                        learner.clear_buffer()

            # 6. Checkpoints and League Expansion at 250, 500, 750, 1000
            if hand in [250, 500, 750, total_hands]:
                ckpt_dir = os.path.join(EXP_DIR, f"ckpt_hand_{hand}")
                os.makedirs(ckpt_dir, exist_ok=True)
                for h_name, learner in learners.items():
                    torch.save({
                        "model_state_dict": learner.model.state_dict(),
                        "optimizer_state_dict": learner.optimizer.state_dict(),
                        "hand": hand,
                        "chips": learner.total_chips,
                        "entropy": learner.current_entropy,
                        "beta_entropy": learner.beta_entropy
                    }, os.path.join(ckpt_dir, f"{h_name}.pt"))
                
                # Expand Opponent League Pool
                for h_name, learner in learners.items():
                    snap_m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
                    snap_m.load_state_dict(learner.model.state_dict())
                    snap_m.eval()
                    opp_pool[f"snap_{hand}h_{h_name}"] = snap_m

                print(f"[+] [CKPT & LEAGUE] Checkpoint saved at hand {hand}. League Pool: {len(opp_pool)} models.")

            if hand % 100 == 0 or hand == total_hands:
                hps = hand / (time.time() - t0)
                ents = [f"{l.name[-1]}:{l.current_entropy:.2f}" for l in learners.values()]
                print(f"  Hand {hand:4d}/{total_hands} | HPS: {hps:4.1f} | Entropies: {' '.join(ents)}")

    finally:
        f_safe.close()
        f_ppo.close()

    # Training Evidence
    train_evidence = {
        "directive": "DIRECTIVE_061",
        "lead_scientist": "Dr. Max",
        "total_table_hands": total_hands,
        "total_hero_hands": sum(l.hands_played for l in learners.values()),
        "total_showdowns": total_showdowns,
        "total_foldouts": total_foldouts,
        "max_pot_discrepancy": float(max_pot_discrepancy),
        "max_zero_sum_error": float(max_zero_sum_err),
        "learners_summary": {}
    }

    for h_name, learner in learners.items():
        bb100 = (learner.total_chips / (learner.hands_played * 100.0)) * 100.0
        train_evidence["learners_summary"][h_name] = {
            "hands_played": learner.hands_played,
            "total_chips": learner.total_chips,
            "hero_bb100": float(bb100),
            "final_entropy": learner.current_entropy,
            "entropy_floor_preserved": bool(learner.current_entropy >= 0.50),
            "final_beta_entropy": learner.beta_entropy,
            "final_policy_loss": learner.last_policy_loss,
            "final_value_loss": learner.last_value_loss,
            "final_clip_frac": learner.last_clip_frac
        }

    train_ev_p = os.path.join(ROOT_DIR, "v1024/reports/directive061_training_evidence.json")
    with open(train_ev_p, "w", encoding="utf-8") as f:
        json.dump(train_evidence, f, indent=2)
    print(f"\n[+] DIRECTIVE 061 Training Complete. Evidence saved to {train_ev_p}")

    # =========================================================================
    # 7. Linked Confirmatory Benchmark Evaluation on 200 Sealed Seeds
    # =========================================================================
    print("\n=================================================================")
    print("STARTING LINKED BENCHMARK EVALUATION (200 Sealed Seeds: 1501000..1501199)")
    print("Candidate: 061_1000h vs Baseline: Base_2000h (Opponents: 7000h)")
    print("=================================================================")

    sealed_seeds = [1501000 + i for i in range(200)]
    hands_per_seed = 5

    cand_models = {a: learners[a].model for a in AGENT_NAMES}
    for m in cand_models.values(): m.eval()

    base_models = {}
    for a in AGENT_NAMES:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(os.path.join(MODELS_DIR, f"{a}_2000h.pt"), map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=True)
        m.eval()
        base_models[a] = m

    opp_models_eval = {}
    for a in AGENT_NAMES:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(os.path.join(MODELS_DIR, f"{a}_7000h.pt"), map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=True)
        m.eval()
        opp_models_eval[a] = m

    eval_arms = [
        ("Candidate_061_1000h", cand_models),
        ("Baseline_2000h", base_models)
    ]

    arm_hand_results = {name: [] for name, _ in eval_arms}
    arm_agent_chips = {name: {a: 0.0 for a in AGENT_NAMES} for name, _ in eval_arms}
    arm_agent_hands = {name: {a: 0 for a in AGENT_NAMES} for name, _ in eval_arms}
    arm_action_counts = {name: {a: [0]*6 for a in AGENT_NAMES} for name, _ in eval_arms}

    eval_table_hands = 0
    eval_max_pot_disc = 0.0
    eval_max_zero_err = 0.0

    t_eval0 = time.time()
    for s_idx, seed_val in enumerate(sealed_seeds):
        for h_in_seed in range(1, hands_per_seed + 1):
            eval_table_hands += 1
            assign = get_directive058_balanced_assignment(eval_table_hands)
            opp_seat = assign["opp_seat"]
            opp_name = AGENT_NAMES[assign["opp_model_idx"]]
            hero_seat_map = assign["agent_to_seat"]

            deck_seed = (seed_val * 1000003 + h_in_seed * 10007) & 0xFFFFFFFF

            for arm_idx, (arm_name, hero_models) in enumerate(eval_arms):
                table = [None] * 5
                table[opp_seat] = ("OPPONENT", opp_name, opp_models_eval[opp_name])
                for h_name, s_idx_table in hero_seat_map.items():
                    table[s_idx_table] = ("HERO", h_name, hero_models[h_name])

                deck_rng = np.random.RandomState(deck_seed)
                action_rng = np.random.RandomState((deck_seed ^ 0x99999999) + arm_idx * 101)

                s = game.new_initial_state()
                advance_chance_nodes(s, deck_rng)

                folded = [False] * 5

                while not s.is_terminal():
                    if s.is_chance_node():
                        advance_chance_nodes(s, deck_rng)
                        continue

                    cp = s.current_player()
                    p_type, p_id, p_model = table[cp]
                    legals = True6ActionAdapter.get_legal_6actions(s, cp)
                    raw_inf = np.array(s.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)

                    with torch.no_grad():
                        logits, _ = p_model(torch.from_numpy(norm_inf).unsqueeze(0))
                        mask = torch.full_like(logits, -1e9)
                        for a in legals: mask[0, a] = 0.0
                        probs = torch.softmax(logits + mask, dim=-1).squeeze(0).numpy()

                    act = int(action_rng.choice(len(probs), p=probs))
                    if p_type == "HERO":
                        arm_action_counts[arm_name][p_id][act] += 1

                    if act == 0:
                        folded[cp] = True
                    engine_act = True6ActionAdapter.map_6action_to_engine_amount(s, cp, act)
                    s.apply_action(engine_act)

                engine_rets = list(s.returns())

                # Pot audit in eval
                spents = extract_player_spent(s)
                holes, board = parse_cards_from_state(s.to_string())
                active_players = [p for p in range(5) if not folded[p]]
                if len(active_players) == 1:
                    winner = active_players[0]
                    calc_rets = [-spents[p] for p in range(5)]
                    calc_rets[winner] = sum(spents) - spents[winner]
                else:
                    scores = [None] * 5
                    for p in active_players:
                        scores[p] = eval_7card(holes[p] + board)
                    calc_rets = calculate_pots_exact(spents, folded, scores)

                disc = max(abs(e - c) for e, c in zip(engine_rets, calc_rets))
                if disc > eval_max_pot_disc: eval_max_pot_disc = disc

                hero_sum_hand = sum(float(engine_rets[s_idx_table]) for h_name, s_idx_table in hero_seat_map.items())
                for h_name, s_idx_table in hero_seat_map.items():
                    arm_agent_chips[arm_name][h_name] += float(engine_rets[s_idx_table])
                    arm_agent_hands[arm_name][h_name] += 1

                avg_hero_bb = (hero_sum_hand / 4.0) / 100.0
                arm_hand_results[arm_name].append(avg_hero_bb)

        if (s_idx + 1) % 50 == 0 or (s_idx + 1) == len(sealed_seeds):
            cand_bb100 = np.mean(arm_hand_results["Candidate_061_1000h"]) * 100.0
            base_bb100 = np.mean(arm_hand_results["Baseline_2000h"]) * 100.0
            print(f"  Eval Seeds: {s_idx+1:3d}/{len(sealed_seeds)} | Cand: {cand_bb100:+6.2f} vs Base: {base_bb100:+6.2f} BB/100")

    # Statistical Evaluation Analysis
    cand_rets = np.array(arm_hand_results["Candidate_061_1000h"]) * 100.0
    base_rets = np.array(arm_hand_results["Baseline_2000h"]) * 100.0
    paired_diffs = cand_rets - base_rets

    delta_mean = float(np.mean(paired_diffs))
    paired_se = float(np.std(paired_diffs, ddof=1) / np.sqrt(len(paired_diffs)))
    t_stat, p_val = stats.ttest_rel(cand_rets, base_rets)
    ci95_low = delta_mean - 1.96 * paired_se
    ci95_high = delta_mean + 1.96 * paired_se

    action_dists = {}
    for arm_name, _ in eval_arms:
        action_dists[arm_name] = {}
        for a in AGENT_NAMES:
            tot = sum(arm_action_counts[arm_name][a])
            action_dists[arm_name][a] = {
                ACTION_NAMES[i]: round((arm_action_counts[arm_name][a][i] / tot) * 100.0, 2) if tot > 0 else 0.0
                for i in range(6)
            }

    agent_eval_scores = {}
    for arm_name, _ in eval_arms:
        agent_eval_scores[arm_name] = {}
        for a in AGENT_NAMES:
            c = arm_agent_chips[arm_name][a]
            h = arm_agent_hands[arm_name][a]
            agent_eval_scores[arm_name][a] = {
                "hands": h,
                "chips": c,
                "bb100": round((c / (h * 100.0)) * 100.0, 2)
            }

    eval_evidence = {
        "directive": "DIRECTIVE_061",
        "lead_scientist": "Dr. Max",
        "total_eval_table_hands": eval_table_hands,
        "sealed_seeds_count": len(sealed_seeds),
        "candidate_summary": {
            "mean_hero_bb100": round(float(np.mean(cand_rets)), 2),
            "se_bb100": round(float(np.std(cand_rets, ddof=1) / np.sqrt(len(cand_rets))), 2),
            "agents": agent_eval_scores["Candidate_061_1000h"]
        },
        "baseline_summary": {
            "mean_hero_bb100": round(float(np.mean(base_rets)), 2),
            "se_bb100": round(float(np.std(base_rets, ddof=1) / np.sqrt(len(base_rets))), 2),
            "agents": agent_eval_scores["Baseline_2000h"]
        },
        "paired_comparison": {
            "delta_bb100": round(delta_mean, 2),
            "paired_se_bb100": round(paired_se, 2),
            "ci95": [round(ci95_low, 2), round(ci95_high, 2)],
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "statistically_significant_improvement": bool(p_val < 0.05 and delta_mean > 0),
            "statistically_significant_degradation": bool(p_val < 0.05 and delta_mean < 0)
        },
        "action_distributions": action_dists
    }

    eval_ev_p = os.path.join(ROOT_DIR, "v1024/reports/directive061_evaluation_evidence.json")
    with open(eval_ev_p, "w", encoding="utf-8") as f:
        json.dump(eval_evidence, f, indent=2)

    print("\n=================================================================")
    print("DIRECTIVE 061 FULL PIPELINE COMPLETE")
    print(f"Candidate: {eval_evidence['candidate_summary']['mean_hero_bb100']} BB/100")
    print(f"Baseline:  {eval_evidence['baseline_summary']['mean_hero_bb100']} BB/100")
    print(f"Delta:     {delta_mean:+.2f} BB/100 (95% CI: [{ci95_low:+.2f}, {ci95_high:+.2f}], p={p_val:.4f})")
    print("=================================================================")
    return train_evidence, eval_evidence

if __name__ == "__main__":
    run_directive061_pipeline(total_hands=1000, seed=5020000)
