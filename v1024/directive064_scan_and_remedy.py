#!/usr/bin/env python3
"""
v1024/directive064_scan_and_remedy.py
DIRECTIVE 064: Checkpoint Trajectory Scan & Anti-Echo Chamber League Remediation (1,500h).
Authorized by Lead RL Scientist Dr. Max.

Key Specifications:
  Task 1: Trajectory Scan
    - Evaluate ckpt_hand_1000, 2000, 3000, 4000, 5000 on 200 sealed seeds (1501000 - 1501199).
    - Identify peak performance epoch (Model Selection).
    - Save results to directive064_scan_evidence.json.
  
  Task 2: Anti-Echo Chamber Remediation (1,500h)
    - Inherit weights from the peak checkpoint identified in Task 1.
    - Matchmaking: 45% Fixed External 7k Anchors (punish Agent_E's min-raise exploit),
                   40% Active Learners, 15% Elite Archives.
    - Adaptive entropy floor raised to H_target = 0.85 to unfreeze Action distribution.
    - Scale: 1,500 table hands (6,000 active hero hands) with seed 5050000.
    - Confirmatory 200 sealed seeds benchmark evaluation.
    - Save results to directive064_remedy_evidence.json.
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
EXP_063_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive063")
EXP_064_DIR = os.path.join(ROOT_DIR, "v1024/exp_directive064")

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def eval_checkpoint_on_sealed_seeds(ckpt_dir: str, num_seeds: int = 200, seed_start: int = 1501000):
    game = get_5p_fullgame()
    hands_per_seed = 5

    models = {}
    for name in AGENT_NAMES:
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if not os.path.exists(p):
            p = os.path.join(ckpt_dir, f"{name}_2000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        models[name] = m

    bench_models = {}
    for name in AGENT_NAMES:
        p = os.path.join(BASE_2000_DIR, f"{name}_7000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        bench_models[name] = m

    agent_chips = {n: 0.0 for n in AGENT_NAMES}
    agent_hands = {n: 0 for n in AGENT_NAMES}
    seed_returns = []

    for s_idx in range(num_seeds):
        curr_seed = seed_start + s_idx
        s_ret = 0.0

        for h in range(hands_per_seed):
            hand_idx = s_idx * hands_per_seed + h + 1
            assign = get_directive058_balanced_assignment(hand_idx)
            hero_seat_map = assign["agent_to_seat"]
            opp_seat = assign["opp_seat"]
            opp_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_m = bench_models[opp_name]

            deck_seed = (curr_seed * 1000003 + h * 10007) & 0xFFFFFFFF
            act_seed = deck_seed ^ 0x66666666

            table = [None] * 5
            table[opp_seat] = opp_m
            for a_name, s_pos in hero_seat_map.items():
                table[s_pos] = models[a_name]

            c_deck_rng = np.random.RandomState(deck_seed)
            c_act_rng = np.random.RandomState(act_seed)
            s_c = game.new_initial_state()
            advance_chance_nodes(s_c, c_deck_rng)

            while not s_c.is_terminal():
                if s_c.is_chance_node():
                    advance_chance_nodes(s_c, c_deck_rng)
                    continue
                cp = s_c.current_player()
                m = table[cp]
                legals = True6ActionAdapter.get_legal_6actions(s_c, cp)
                raw_inf = np.array(s_c.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf)
                mask = np.full(6, -1e9, dtype=np.float32)
                for a in legals: mask[a] = 0.0

                with torch.no_grad():
                    logits, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                    probs = torch.softmax(logits + torch.from_numpy(mask).unsqueeze(0), dim=-1).squeeze(0).numpy()
                act = int(c_act_rng.choice(len(probs), p=probs))
                s_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(s_c, cp, act))

            c_rets = list(s_c.returns())
            for a_name, s_pos in hero_seat_map.items():
                agent_chips[a_name] += c_rets[s_pos]
                agent_hands[a_name] += 1
                s_ret += c_rets[s_pos]

        seed_returns.append(s_ret / (hands_per_seed * 4))

    mean_bb = float(np.mean(seed_returns))
    se_bb = float(stats.sem(seed_returns))
    agent_summary = {n: round(agent_chips[n] / agent_hands[n], 2) for n in AGENT_NAMES}
    return {
        "mean_bb100": round(mean_bb, 2),
        "se_bb100": round(se_bb, 2),
        "agents": agent_summary,
        "seed_returns": seed_returns
    }

class AlphaLeagueRemedyLearner:
    def __init__(self, name: str, init_model_path: str, lr: float = 0.0003, target_entropy: float = 0.85):
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
        self.beta_entropy = float(np.clip(beta, 0.02, 0.25))
        return self.beta_entropy

def run_directive064_pipeline():
    print("=================================================================")
    print("DIRECTIVE 064: TRAJECTORY SCAN & ANTI-ECHO REMEDIATION")
    print("Lead RL Scientist: Dr. Max | Scientific Charter Compliant")
    print("=================================================================")

    os.makedirs(EXP_064_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # TASK 1: CHECKPOINT TRAJECTORY SCAN
    # -----------------------------------------------------------------
    print("\n--- [TASK 1] Scanning Exp 063 Checkpoint Trajectory (1k to 5k) ---")
    scan_steps = [1000, 2000, 3000, 4000, 5000]
    scan_results = {}
    best_step = None
    best_bb = -999999.0

    for step in scan_steps:
        sub_dir = os.path.join(EXP_063_DIR, f"ckpt_hand_{step}")
        if not os.path.exists(sub_dir):
            continue
        print(f"  Scanning ckpt_hand_{step} across 200 sealed seeds...", end="", flush=True)
        res = eval_checkpoint_on_sealed_seeds(sub_dir)
        scan_results[f"ckpt_{step}"] = res
        print(f" -> Mean: {res['mean_bb100']:+6.2f} BB/100 (SE: {res['se_bb100']:.2f}) | Agents: {res['agents']}")
        if res["mean_bb100"] > best_bb:
            best_bb = res["mean_bb100"]
            best_step = step

    print(f"\n[+] Trajectory Scan Complete. Peak Checkpoint Identified: ckpt_hand_{best_step} ({best_bb:+6.2f} BB/100)")
    
    with open(os.path.join(ROOT_DIR, "v1024/reports/directive064_scan_evidence.json"), "w") as f:
        json.dump({
            "directive": "DIRECTIVE_064_TASK1",
            "lead_scientist": "Dr. Max",
            "peak_checkpoint": f"ckpt_hand_{best_step}",
            "peak_mean_bb100": best_bb,
            "trajectory_scan": scan_results
        }, f, indent=2)

    # -----------------------------------------------------------------
    # TASK 2: ANTI-ECHO CHAMBER REMEDIATION TRAINING (1,500h)
    # -----------------------------------------------------------------
    total_remedy_hands = 1500
    seed = 5050000
    torch.manual_seed(seed)
    np.random.seed(seed)

    peak_dir = os.path.join(EXP_063_DIR, f"ckpt_hand_{best_step}")
    print(f"\n--- [TASK 2] Starting Anti-Echo Remedy Training from {peak_dir} (1,500h) ---")
    
    learners = {}
    for name in AGENT_NAMES:
        path = os.path.join(peak_dir, f"{name}.pt")
        learners[name] = AlphaLeagueRemedyLearner(name, path, lr=0.0003, target_entropy=0.85)

    # Build Anti-Echo Pools: 45% External 7k Anchors, 40% Active, 15% Elite 062/063
    anchor_7k_pool = {}
    for name in AGENT_NAMES:
        p = os.path.join(BASE_2000_DIR, f"{name}_7000h.pt")
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        anchor_7k_pool[f"7k_{name}"] = m

    elite_pool = {}
    for name in ["Agent_A", "Agent_D"]:
        p_062 = os.path.join(ROOT_DIR, f"v1024/exp_directive062/ckpt_hand_3000/{name}.pt")
        if os.path.exists(p_062):
            m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
            ckpt = torch.load(p_062, map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
            m.eval()
            elite_pool[f"elite_062_{name}"] = m

    game = get_5p_fullgame()
    matchmaking_rng = np.random.RandomState(seed ^ 0x88888888)
    batch_size = 250
    t_start = time.time()
    pot_audit_passes = 0
    max_zero_sum_err = 0.0

    for hand in range(1, total_remedy_hands + 1):
        assign = get_directive058_balanced_assignment(hand)
        hero_seat_map = assign["agent_to_seat"]
        opp_seat = assign["opp_seat"]

        # 45% 7k Anchors (punishes min-raise exploit), 40% Active, 15% Elite
        p_mode = matchmaking_rng.rand()
        if p_mode < 0.45:
            a_keys = list(anchor_7k_pool.keys())
            opp_key = a_keys[matchmaking_rng.randint(len(a_keys))]
            opp_model = anchor_7k_pool[opp_key]
        elif p_mode < 0.85:
            opp_agent_name = AGENT_NAMES[assign["opp_model_idx"]]
            opp_key = f"active_{opp_agent_name}"
            opp_model = learners[opp_agent_name].model
        else:
            e_keys = list(elite_pool.keys())
            opp_key = e_keys[matchmaking_rng.randint(len(e_keys))]
            opp_model = elite_pool[opp_key]

        table = [None] * 5
        table[opp_seat] = ("OPPONENT", opp_key, opp_model)
        for h_name, s_idx in hero_seat_map.items():
            table[s_idx] = ("HERO", h_name, learners[h_name])

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

        assert s.is_terminal()
        engine_rets = list(s.returns())

        # Zero sum check
        err = abs(sum(engine_rets))
        if err > max_zero_sum_err: max_zero_sum_err = err
        if err > 1e-4: raise AssertionError(f"Zero sum failed: {err}")

        # In-path Independent Pot Verification
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

        # Trajectory Buffering
        for h_name, steps in hand_steps.items():
            s_idx = hero_seat_map[h_name]
            net_chips = float(engine_rets[s_idx])
            learners[h_name].total_chips += net_chips
            learners[h_name].hands_played += 1
            reward_bb = net_chips / 100.0

            n_steps = len(steps)
            for i, step in enumerate(steps):
                step_rew = reward_bb if i == n_steps - 1 else 0.0
                learners[h_name].obs_buf.append(step["obs"])
                learners[h_name].act_buf.append(step["act"])
                learners[h_name].logp_buf.append(step["log_p"])
                learners[h_name].val_buf.append(step["val"])
                learners[h_name].rew_buf.append(step_rew)
                learners[h_name].legal_masks.append(step["mask"])

        # PPO Update
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

                b_ret = np.zeros_like(b_rew)
                running = 0.0
                for t in reversed(range(len(b_rew))):
                    running = b_rew[t] + 0.99 * running
                    b_ret[t] = running
                b_ret_t = torch.tensor(b_ret, dtype=torch.float32)
                b_adv = b_ret_t - b_old_val
                if len(b_adv) > 1 and b_adv.std() > 1e-6:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

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
            print(f"  Remedy Hand {hand:4d}/{total_remedy_hands} | HPS: {rate:4.1f} | Ent: {ent_str} | VLoss: {vloss_str}")

    # Save final remedied models
    remedy_ckpt_dir = os.path.join(EXP_064_DIR, "remedied_models")
    os.makedirs(remedy_ckpt_dir, exist_ok=True)
    for name, learner in learners.items():
        torch.save({"model_state_dict": learner.model.state_dict()}, os.path.join(remedy_ckpt_dir, f"{name}.pt"))
    print(f"\n[+] Anti-Echo Remedy Training Complete. Checkpoints saved to {remedy_ckpt_dir}")

    # -----------------------------------------------------------------
    # FINAL EVALUATION: 200 Sealed Seeds Benchmark vs Base 2000h
    # -----------------------------------------------------------------
    print("\n--- Running Final Evaluation: Remedied Models vs Base 2000h on 200 Sealed Seeds ---")
    final_res = eval_checkpoint_on_sealed_seeds(remedy_ckpt_dir)
    base_res = eval_checkpoint_on_sealed_seeds(BASE_2000_DIR, seed_start=1501000)

    cand_arr = np.array(final_res["seed_returns"])
    base_arr = np.array(base_res["seed_returns"])
    diff_arr = cand_arr - base_arr

    delta = float(np.mean(diff_arr))
    se_delta = float(stats.sem(diff_arr))
    ci95 = [round(delta - 1.96 * se_delta, 2), round(delta + 1.96 * se_delta, 2)]
    t_stat, p_val = stats.ttest_rel(cand_arr, base_arr)

    print("\n=================================================================")
    print("DIRECTIVE 064 REMEDIATION FINAL EVALUATION COMPLETE")
    print(f"Candidate (Remedied): {final_res['mean_bb100']:+.2f} BB/100")
    print(f"Baseline  (Base2000): {base_res['mean_bb100']:+.2f} BB/100")
    print(f"Delta:                {delta:+.2f} BB/100 (95% CI: [{ci95[0]}, {ci95[1]}], p={p_val:.4f})")
    print(f"Agent Performance:    {final_res['agents']}")
    print("=================================================================")

    remedy_evidence = {
        "directive": "DIRECTIVE_064",
        "lead_scientist": "Dr. Max",
        "initial_peak_checkpoint": f"ckpt_hand_{best_step}",
        "total_remedy_table_hands": total_remedy_hands,
        "active_hero_hands": total_remedy_hands * 4,
        "pot_audit_passes": pot_audit_passes,
        "max_zero_sum_error": max_zero_sum_err,
        "final_evaluation": {
            "candidate_bb100": final_res["mean_bb100"],
            "candidate_se": final_res["se_bb100"],
            "baseline_bb100": base_res["mean_bb100"],
            "baseline_se": base_res["se_bb100"],
            "delta_bb100": round(delta, 2),
            "paired_se": round(se_delta, 2),
            "ci95": ci95,
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "agents_candidate": final_res["agents"],
            "agents_baseline": base_res["agents"]
        }
    }

    with open(os.path.join(ROOT_DIR, "v1024/reports/directive064_remedy_evidence.json"), "w") as f:
        json.dump(remedy_evidence, f, indent=2)
    print(f"[+] Remedy evidence written to reports/directive064_remedy_evidence.json")

if __name__ == "__main__":
    run_directive064_pipeline()
