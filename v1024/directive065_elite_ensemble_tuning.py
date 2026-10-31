#!/usr/bin/env python3
"""
v1024/directive065_elite_ensemble_tuning.py
DIRECTIVE 065: Final Elite Ensemble & Targeted Exploiter Fine-Tuning.
Authorized by Lead RL Scientist Dr. Max.

Key Specifications:
  1. Elite Preservation:
     - Freeze and preserve proven top performers from DIRECTIVE 064:
       Agent_A (+62.34 BB), Agent_B (+252 BB leak defense), Agent_C (+140.39 BB ace), Agent_D (+91.00 BB).
  2. Targeted Exploiter Tuning on Agent_E (500 table hands):
     - Agent_E plays 1:4 against frozen 7k benchmark anchors.
     - PPO fine-tuning with lr = 0.0002, target_entropy = 0.70 to extinguish min-raise vulnerability.
     - 100% strict in-path pot verification (calculate_pots_exact) and zero-sum verification.
  3. Grand Finale Benchmark Evaluation:
     - Full 200 sealed seeds (1501000 - 1501199, 1,000 table hands, 5,000 player hands).
     - Head-to-head CRN comparison against Base 2000h.
     - Target: Outperform baseline (Delta BB/100 > 0).
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
BASE_2000_DIR = os.path.join(ROOT_DIR, "v1024/models_alphago")
REMEDIED_064_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive064/remedied_models")
EXP_065_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive065")

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class AgentETargetedTuner:
    def __init__(self, init_model_path: str, lr: float = 0.0002, target_entropy: float = 0.70):
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
        self.current_entropy = 1.0
        self.target_entropy = target_entropy
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

    def update_beta_entropy(self) -> float:
        diff = self.target_entropy - self.current_entropy
        beta = 0.02 * np.exp(2.0 * diff)
        self.beta_entropy = float(np.clip(beta, 0.01, 0.20))
        return self.beta_entropy

def run_directive065_pipeline():
    print("=================================================================")
    print("DIRECTIVE 065: FINAL ELITE ENSEMBLE & TARGETED FINE-TUNING")
    print("Lead RL Scientist: Dr. Max | Scientific Charter Compliant")
    print("=================================================================")

    os.makedirs(EXP_065_DIR, exist_ok=True)
    seed = 5060000
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Elite Preservation (Agents A, B, C, D)
    print("\n[1/3] Freezing and Preserving Elite Champions from DIRECTIVE 064...")
    elite_roster = {}
    for name in ["Agent_A", "Agent_B", "Agent_C", "Agent_D"]:
        p = os.path.join(REMEDIED_064_DIR, f"{name}.pt")
        sha = compute_sha256(p)
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        elite_roster[name] = m
        # Also copy to 065 dir for frozen preservation
        p_dst = os.path.join(EXP_065_DIR, f"{name}.pt")
        torch.save({"model_state_dict": m.state_dict()}, p_dst)
        print(f"  - {name}: Frozen preserved (SHA: {sha[:12]}...) -> Saved to {p_dst}")

    # 2. Targeted Tuning on Agent_E against 7k Anchors (500 hands)
    print("\n[2/3] Targeted Fine-Tuning on Agent_E (500 table hands vs 7k Anchors)...")
    p_init_e = os.path.join(REMEDIED_064_DIR, "Agent_E.pt")
    tuner_e = AgentETargetedTuner(p_init_e, lr=0.0002, target_entropy=0.70)

    # Load 4 distinct 7k benchmark opponents
    bench_opponents = []
    for opp_name in ["Agent_A", "Agent_B", "Agent_C", "Agent_D"]:
        p_opp = os.path.join(BASE_2000_DIR, f"{opp_name}_7000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p_opp, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        bench_opponents.append((opp_name, m))

    game = get_5p_fullgame()
    total_tuning_hands = 500
    batch_size = 100
    t_start = time.time()
    pot_audit_passes = 0
    max_zero_sum_err = 0.0

    for hand in range(1, total_tuning_hands + 1):
        # Agent_E rotates positions across all 5 seats
        e_seat = (hand - 1) % 5
        table = [None] * 5
        table[e_seat] = ("HERO", tuner_e)
        opp_idx = 0
        for s_idx in range(5):
            if s_idx != e_seat:
                table[s_idx] = ("OPPONENT", bench_opponents[opp_idx][1])
                opp_idx += 1

        deck_seed = (seed * 1000003 + hand * 10007) & 0xFFFFFFFF
        deck_rng = np.random.RandomState(deck_seed)
        action_rng = np.random.RandomState(deck_seed ^ 0x66666666)

        s = game.new_initial_state()
        advance_chance_nodes(s, deck_rng)

        folded = [False] * 5
        e_steps = []

        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, deck_rng)
                continue

            cp = s.current_player()
            p_type, p_obj = table[cp]
            legals = True6ActionAdapter.get_legal_6actions(s, cp)
            raw_inf = np.array(s.information_state_tensor(cp), dtype=np.float32)
            norm_inf = normalize_inf_state_v79(raw_inf)

            mask = np.full(6, -1e9, dtype=np.float32)
            for a in legals: mask[a] = 0.0

            if p_type == "HERO":
                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    logits, val = tuner_e.model(t_obs)
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                act = int(action_rng.choice(len(probs), p=probs))
                log_p = float(np.log(probs[act] + 1e-12))
                v_est = float(val.item())

                e_steps.append({
                    "obs": norm_inf,
                    "act": act,
                    "log_p": log_p,
                    "val": v_est,
                    "mask": mask
                })
            else:
                t_obs = torch.from_numpy(norm_inf).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = p_obj(t_obs)
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    probs = torch.softmax(logits + t_mask, dim=-1).squeeze(0).numpy()
                act = int(action_rng.choice(len(probs), p=probs))

            if act == 0:
                folded[cp] = True
            engine_act = True6ActionAdapter.map_6action_to_engine_amount(s, cp, act)
            s.apply_action(engine_act)

        assert s.is_terminal()
        engine_rets = list(s.returns())

        # Zero-sum check
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

        # Buffering for Agent_E
        net_chips = float(engine_rets[e_seat])
        tuner_e.total_chips += net_chips
        tuner_e.hands_played += 1
        reward_bb = net_chips / 100.0

        n_steps = len(e_steps)
        for i, step in enumerate(e_steps):
            step_rew = reward_bb if i == n_steps - 1 else 0.0
            tuner_e.obs_buf.append(step["obs"])
            tuner_e.act_buf.append(step["act"])
            tuner_e.logp_buf.append(step["log_p"])
            tuner_e.val_buf.append(step["val"])
            tuner_e.rew_buf.append(step_rew)
            tuner_e.legal_masks.append(step["mask"])

        # PPO Update
        if hand % batch_size == 0 and len(tuner_e.obs_buf) >= 10:
            b_obs = torch.tensor(np.array(tuner_e.obs_buf), dtype=torch.float32)
            b_act = torch.tensor(tuner_e.act_buf, dtype=torch.long)
            b_old_logp = torch.tensor(tuner_e.logp_buf, dtype=torch.float32)
            b_old_val = torch.tensor(tuner_e.val_buf, dtype=torch.float32)
            b_rew = np.array(tuner_e.rew_buf, dtype=np.float32)
            b_masks = torch.tensor(np.array(tuner_e.legal_masks), dtype=torch.float32)

            b_ret = np.zeros_like(b_rew)
            running = 0.0
            for t in reversed(range(len(b_rew))):
                running = b_rew[t] + 0.99 * running
                b_ret[t] = running
            b_ret_t = torch.tensor(b_ret, dtype=torch.float32)
            b_adv = b_ret_t - b_old_val
            if len(b_adv) > 1 and b_adv.std() > 1e-6:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            tuner_e.model.train()
            clip_eps = 0.2
            for _ in range(4):
                tuner_e.optimizer.zero_grad()
                logits, val = tuner_e.model(b_obs)
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

                beta_ent = tuner_e.update_beta_entropy()
                total_loss = policy_loss + 0.5 * value_loss - beta_ent * entropy
                total_loss.backward()
                nn.utils.clip_grad_norm_(tuner_e.model.parameters(), max_norm=0.5)
                tuner_e.optimizer.step()

            tuner_e.current_entropy = float(entropy.item())
            tuner_e.last_policy_loss = float(policy_loss.item())
            tuner_e.last_value_loss = float(value_loss.item())
            tuner_e.clear_buffer()

            rate = hand / (time.time() - t_start)
            print(f"  Tuning Hand {hand:3d}/{total_tuning_hands} | HPS: {rate:4.1f} | Ent: {tuner_e.current_entropy:.2f} | VLoss: {tuner_e.last_value_loss:.1f} | NetBB: {tuner_e.total_chips/100.0:+.1f}")

    # Save fine-tuned Agent_E
    p_final_e = os.path.join(EXP_065_DIR, "Agent_E.pt")
    torch.save({"model_state_dict": tuner_e.model.state_dict()}, p_final_e)
    tuner_e.model.eval()
    elite_roster["Agent_E"] = tuner_e.model
    print(f"\n[+] Agent_E Fine-Tuning Complete. Champion saved to {p_final_e}")

    tuning_evidence = {
        "directive": "DIRECTIVE_065_TUNING",
        "lead_scientist": "Dr. Max",
        "total_tuning_hands": total_tuning_hands,
        "pot_audit_passes": pot_audit_passes,
        "max_zero_sum_error": max_zero_sum_err,
        "final_entropy": tuner_e.current_entropy,
        "final_value_loss": tuner_e.last_value_loss,
        "tuning_net_chips": tuner_e.total_chips
    }
    with open(os.path.join(ROOT_DIR, "v1024/reports/directive065_tuning_evidence.json"), "w") as f:
        json.dump(tuning_evidence, f, indent=2)

    # 3. GRAND FINALE BENCHMARK EVALUATION (200 Sealed Seeds vs Base 2000h)
    print("\n=================================================================")
    print("[3/3] COMMENCING GRAND FINALE BENCHMARK EVALUATION (200 Sealed Seeds)")
    print("Candidate Dream Team (A, B, C, D 064 + E 065) vs Base 2000h")
    print("=================================================================")

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
    for name in AGENT_NAMES:
        p = os.path.join(BASE_2000_DIR, f"{name}_7000h.pt")
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
    num_seeds = 200
    seed_start = 1501000
    hands_per_seed = 5

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

            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x66666666

            # Candidate Dream Team
            cand_table = [None] * 5
            cand_table[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                cand_table[s_pos] = elite_roster[a_name]

            c_deck_rng = np.random.RandomState(deck_seed)
            c_act_rng = np.random.RandomState(act_seed)
            s_c = game.new_initial_state()
            advance_chance_nodes(s_c, c_deck_rng)

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
                s_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(s_c, cp, act))

            c_rets = list(s_c.returns())
            for a_name, s_pos in hero_seat_map.items():
                cand_chips[a_name] += c_rets[s_pos]
                cand_hands[a_name] += 1
                s_cand_ret += c_rets[s_pos]

            # Baseline 2000h
            base_table = [None] * 5
            base_table[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                base_table[s_pos] = baseline_models[a_name]

            b_deck_rng = np.random.RandomState(deck_seed)
            b_act_rng = np.random.RandomState(act_seed)
            s_b = game.new_initial_state()
            advance_chance_nodes(s_b, b_deck_rng)

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
    print("DIRECTIVE 065 GRAND FINALE BENCHMARK COMPLETE")
    print(f"Candidate Dream Team: {mean_cand:+.2f} BB/100")
    print(f"Baseline Base 2000h:  {mean_base:+.2f} BB/100")
    print(f"Delta:                {delta:+.2f} BB/100 (95% CI: [{ci95[0]}, {ci95[1]}], p={p_val:.4f})")
    print("=================================================================")

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
            "net_diff_bb100": round((cand_chips[n] - base_chips[n]) / cand_hands[n], 2)
        }
        print(f"  {n:7s} | Cand: {agent_eval_summary[n]['cand_bb100']:+7.2f} vs Base: {agent_eval_summary[n]['base_bb100']:+7.2f} (Diff: {agent_eval_summary[n]['net_diff_bb100']:+7.2f})")

    final_evidence = {
        "directive": "DIRECTIVE_065",
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
            "Candidate_DreamTeam": cand_act_dist,
            "Baseline_2000h": base_act_dist
        }
    }

    out_path = os.path.join(ROOT_DIR, "v1024/reports/directive065_final_evidence.json")
    with open(out_path, "w") as f:
        json.dump(final_evidence, f, indent=2)
    print(f"\n[+] Final evidence written to {out_path}")

    return final_evidence

if __name__ == "__main__":
    run_directive065_pipeline()
