#!/usr/bin/env python3
"""
v1024/directive056_production_engine.py
DIRECTIVE 056: BALANCED PRODUCTION REINFORCEMENT LEARNING ENGINE

Architectural Upgrades:
1. Perfectly Balanced Symmetrical Latin Square Seat Rotation:
   - 25-hand base integer cycle.
   - Across 1,000 hands: every agent (A, B, C, D, E) gets EXACTLY 160 hands in every seat (0, 1, 2, 3, 4).
   - Exactly 200 sit-outs per agent. Zero position bias, zero player bias.
2. Cumulative Process CPU Time / Wall Time Throttle Controller:
   - Measures (cumulative_cpu_time / elapsed_wall_time) * 100%.
   - Active proportional sleep feedback maintains total session average CPU <= 70.0%.
3. Production Corrected AlphaStar PFSP Sampler:
   - Prioritizes hard opponents: weight = exp(+opp_net_bb / T).
4. Exact Categorical KL & Policy Entropy Metrics:
   - Legal-action distribution true categorical KL and entropy pre/post update.
5. Bit-Exact Pause/Resume State Serialization.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

BIG_BLIND = 100.0
AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
CHECKPOINT_DIR = "v1024/models_alphago"

def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file for SHA256: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_perfect_latin_square_assignment(hand_idx: int) -> Dict[str, Any]:
    """
    Computes mathematically perfect symmetrical Latin square seat assignments for hand_idx (1-based).
    Base cycle: 25 hands.
    """
    h0 = hand_idx - 1
    c25 = h0 % 25
    r = c25 % 5       # Frozen opponent seat 0..4
    k = c25 // 5      # Step 0..4
    
    frozen_seat = r
    sit_out_agent = (r + k) % 5
    avail_seats = [s for s in range(5) if s != frozen_seat]
    
    seat_to_agent = {}
    for s_idx, seat in enumerate(avail_seats):
        agent = (r + k + s_idx + 1) % 5
        seat_to_agent[seat] = agent
        
    return {
        "frozen_seat": frozen_seat,
        "sit_out_agent": sit_out_agent,
        "seat_to_agent": seat_to_agent # seat -> agent_idx (0..4)
    }

class CorrectedAlphaStarPFSPSampler:
    """Prioritizes hard opponents who defeated the learner (higher opp_net_bb => higher probability)."""
    def __init__(
        self,
        checkpoints: List[str],
        temperature: float = 10.0,
        exploration_floor: float = 0.1,
        clip_bound: float = 25.0
    ):
        self.checkpoints = list(checkpoints)
        self.temperature = float(temperature)
        self.exploration_floor = float(exploration_floor)
        self.clip_bound = float(clip_bound)
        self.history = {c: {"games": 0, "net_bb": 0.0} for c in self.checkpoints}
        self.selection_log = []

    def record_match(self, opponent: str, opponent_net_bb: float):
        if opponent in self.history:
            self.history[opponent]["games"] += 1
            self.history[opponent]["net_bb"] += float(opponent_net_bb)

    def get_probabilities(self) -> Dict[str, float]:
        n = len(self.checkpoints)
        if n == 0:
            return {}
        raw_w = []
        for c in self.checkpoints:
            h = self.history[c]
            if h["games"] == 0:
                raw_w.append(1.0)
            else:
                avg_opp_bb = h["net_bb"] / h["games"]
                clipped = float(np.clip(avg_opp_bb, -self.clip_bound, self.clip_bound))
                raw_w.append(float(np.exp(clipped / self.temperature)))
                
        tot_w = sum(raw_w)
        uniform_p = 1.0 / n
        probs = {}
        for c, w in zip(self.checkpoints, raw_w):
            p = (1.0 - self.exploration_floor) * (w / tot_w) + self.exploration_floor * uniform_p
            probs[c] = float(p)
            
        tot_p = sum(probs.values())
        norm = {k: v / tot_p for k, v in probs.items()}
        assert abs(sum(norm.values()) - 1.0) < 1e-6, "PFSP sum != 1.0"
        return norm

class UniformHistoricalSampler:
    """Baseline uniform opponent sampler."""
    def __init__(self, opponent_pool: List[str]):
        self.opponent_pool = list(opponent_pool)
        self.selection_log = []
        self.history = {c: {"games": 0, "net_bb": 0.0} for c in self.opponent_pool}
        
    def get_probabilities(self) -> Dict[str, float]:
        n = len(self.opponent_pool)
        return {c: 1.0 / n for c in self.opponent_pool}
        
    def record_match(self, opp: str, opponent_net_bb: float):
        if opp in self.history:
            self.history[opp]["games"] += 1
            self.history[opp]["net_bb"] += float(opponent_net_bb)

class CumulativeProcessCPUController:
    """Controls Process CPU Time / Elapsed Wall Time <= target_cpu_pct (70.0%)."""
    def __init__(self, target_cpu_pct: float = 70.0, log_path: Optional[str] = None):
        self.proc = psutil.Process()
        self.target_cpu_pct = float(target_cpu_pct)
        self.log_path = log_path
        self.current_sleep_s = 0.005
        self.t_start = time.time()
        c_init = self.proc.cpu_times()
        self.cpu_start = c_init.user + c_init.system
        self.history = []

    def throttle(self, hand_idx: int):
        now = time.time()
        elapsed_wall = max(0.001, now - self.t_start)
        c_now = self.proc.cpu_times()
        elapsed_cpu = (c_now.user + c_now.system) - self.cpu_start
        cumulative_ratio = (elapsed_cpu / elapsed_wall) * 100.0
        
        # Proportional Feedback Control
        if cumulative_ratio > self.target_cpu_pct:
            delta = (cumulative_ratio - self.target_cpu_pct) / 10.0
            self.current_sleep_s = min(0.08, self.current_sleep_s + 0.002 * max(0.5, delta))
        elif cumulative_ratio < self.target_cpu_pct - 15.0:
            self.current_sleep_s = max(0.0005, self.current_sleep_s - 0.001)
            
        time.sleep(self.current_sleep_s)
        
        if hand_idx % 50 == 0:
            entry = {
                "hand": hand_idx,
                "timestamp": round(now, 2),
                "cumulative_cpu_ratio": round(cumulative_ratio, 2),
                "elapsed_cpu_s": round(elapsed_cpu, 2),
                "elapsed_wall_s": round(elapsed_wall, 2),
                "target_cpu": self.target_cpu_pct,
                "sleep_duration_s": round(self.current_sleep_s, 5),
                "mem_mb": round(self.proc.memory_info().rss / 1024 / 1024, 2)
            }
            self.history.append(entry)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

class Directive056PPOAgent:
    """Enhanced PPO Agent with Exact Categorical KL, Policy Entropy, and NaN Protection."""
    def __init__(
        self,
        name: str,
        init_path: str,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.name = name
        self.model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256, lr=lr, clip_eps=clip_eps)
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"[FAIL_FAST] Required model path missing: {init_path}")
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        self.model.train()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.old_dist_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.legal_masks = []
        
        self.step_count = 0
        self.gradient_steps = 0
        self.total_pnl_bb = 0.0

    def select_action(self, norm_inf: np.ndarray, legal_actions: List[int], action_rng: np.random.RandomState):
        self.step_count += 1
        inf_tensor = torch.from_numpy(norm_inf).unsqueeze(0)
        with torch.no_grad():
            logits, val = self.model(inf_tensor)
            mask = torch.full_like(logits, -1e9)
            for a in legal_actions:
                mask[0, a] = 0.0
            masked_logits = logits + mask
            probs_t = torch.softmax(masked_logits, dim=-1)
            probs = probs_t.squeeze(0).numpy()
            
            action = int(action_rng.choice(len(probs), p=probs))
            dist = torch.distributions.Categorical(probs_t)
            log_prob = dist.log_prob(torch.tensor(action)).item()
            v_est = float(val.item())
            
        self.states.append(norm_inf)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.old_dist_probs.append(probs)
        self.values.append(v_est)
        self.legal_masks.append(legal_actions)
        return action, log_prob

    def record_hand_reward(self, hand_return_bb: float):
        self.total_pnl_bb += hand_return_bb
        num_new_steps = len(self.states) - len(self.rewards)
        if num_new_steps > 0:
            step_rewards = [0.0] * (num_new_steps - 1) + [hand_return_bb]
            self.rewards.extend(step_rewards)
            step_dones = [0.0] * (num_new_steps - 1) + [1.0]
            self.dones.extend(step_dones)

    def ppo_update(self) -> Optional[Dict[str, float]]:
        if len(self.states) < 1:
            return None
            
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_lp_t = torch.tensor(self.old_log_probs, dtype=torch.float32)
        old_probs_t = torch.tensor(np.array(self.old_dist_probs), dtype=torch.float32)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32)
        values_t = torch.tensor(self.values, dtype=torch.float32)
        dones_t = torch.tensor(self.dones, dtype=torch.float32)
        
        # Corrected GAE with strict (1 - done) hand boundary masking
        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            next_val = values_t[t + 1] if t + 1 < len(values_t) else 0.0
            non_terminal = 1.0 - dones_t[t].item()
            delta = rewards_t[t] + self.gamma * non_terminal * next_val - values_t[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)
            
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = adv_t + values_t
        if adv_t.std() > 1e-6:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            
        self.optimizer.zero_grad()
        logits, v_pred = self.model(states_t)
        
        mask = torch.full_like(logits, -1e9)
        for i, legals in enumerate(self.legal_masks):
            for a in legals:
                mask[i, a] = 0.0
        masked_logits = logits + mask
        
        new_dist = torch.distributions.Categorical(torch.softmax(masked_logits, dim=-1))
        new_lp = new_dist.log_prob(actions_t)
        
        ratio = torch.exp(new_lp - old_lp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(v_pred.squeeze(-1), returns_t)
        
        new_probs = torch.softmax(masked_logits, dim=-1)
        entropy = -(new_probs * torch.log(new_probs + 1e-12)).sum(dim=-1).mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"[FATAL_NUMERICAL_ANOMALY] Loss became NaN/Inf in agent {self.name}")
            
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).item()
        self.optimizer.step()
        self.gradient_steps += 1
        
        with torch.no_grad():
            post_logits, post_v = self.model(states_t)
            post_masked = post_logits + mask
            post_probs = torch.softmax(post_masked, dim=-1)
            kl_div = (old_probs_t * (torch.log(old_probs_t + 1e-12) - torch.log(post_probs + 1e-12))).sum(dim=-1).mean().item()
            post_dist = torch.distributions.Categorical(post_probs)
            post_lp = post_dist.log_prob(actions_t)
            post_ratio = torch.exp(post_lp - old_lp_t)
            clip_fraction = float(((post_ratio < 1.0 - self.clip_eps) | (post_ratio > 1.0 + self.clip_eps)).float().mean().item())
            
        metrics = {
            "policy_loss": round(float(policy_loss.item()), 5),
            "value_loss": round(float(value_loss.item()), 5),
            "entropy": round(float(entropy.item()), 5),
            "grad_norm": round(float(grad_norm), 5),
            "categorical_kl": round(float(kl_div), 6),
            "clip_fraction": round(float(clip_fraction), 5),
            "buffer_size": len(self.states)
        }
        
        self.states.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.old_dist_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.legal_masks.clear()
        
        return metrics

    def save_checkpoint(self, path: str):
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "gradient_steps": self.gradient_steps,
            "total_pnl_bb": self.total_pnl_bb,
            "states": self.states,
            "actions": self.actions,
            "old_log_probs": self.old_log_probs,
            "old_dist_probs": self.old_dist_probs,
            "rewards": self.rewards,
            "values": self.values,
            "dones": self.dones,
            "legal_masks": self.legal_masks
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step_count = ckpt["step_count"]
        self.gradient_steps = ckpt["gradient_steps"]
        self.total_pnl_bb = ckpt["total_pnl_bb"]
        self.states = list(ckpt.get("states", []))
        self.actions = list(ckpt.get("actions", []))
        self.old_log_probs = list(ckpt.get("old_log_probs", []))
        self.old_dist_probs = list(ckpt.get("old_dist_probs", []))
        self.rewards = list(ckpt.get("rewards", []))
        self.values = list(ckpt.get("values", []))
        self.dones = list(ckpt.get("dones", []))
        self.legal_masks = list(ckpt.get("legal_masks", []))

def run_directive056_arm(
    arm_name: str,
    league_type: str, # "uniform" or "pfsp"
    opponent_pool_type: str, # "extended" or "frozen2k"
    total_hands: int,
    seed_base: int,
    out_dir: str,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f">>> [DIRECTIVE 056 RUNNER] {arm_name}")
    print(f"    League: {league_type} | Pool: {opponent_pool_type} | Hands: {total_hands}")
    print(f"    Seed: {seed_base} | Out: {out_dir}")
    print(f"=======================================================")
    
    os.makedirs(out_dir, exist_ok=True)
    game = get_5p_fullgame()
    
    learners = {a: Directive056PPOAgent(a, os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
    
    if opponent_pool_type == "frozen2k":
        opp_ckpts = [f"{a}_2000h" for a in AGENT_NAMES]
    else: # "extended": 2k + 7k baseline pool
        opp_ckpts = [f"{a}_2000h" for a in AGENT_NAMES] + [f"{a}_7000h" for a in AGENT_NAMES]
        
    frozen_models = {}
    for c in opp_ckpts:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{c}.pt"), map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        m.eval()
        frozen_models[c] = m
        
    if league_type == "uniform":
        sampler = UniformHistoricalSampler(opp_ckpts)
    else:
        sampler = CorrectedAlphaStarPFSPSampler(opp_ckpts, temperature=10.0, exploration_floor=0.1)
        
    rng_master = np.random.RandomState(seed_base)
    torch.manual_seed(seed_base)
    start_hand = 1
    
    if resume_from and os.path.exists(resume_from):
        print(f"  Resuming from checkpoint: {resume_from}")
        for a in AGENT_NAMES:
            learners[a].load_checkpoint(os.path.join(resume_from, f"{a}.pt"))
        meta = torch.load(os.path.join(resume_from, "meta_state.pt"), weights_only=False)
        start_hand = meta["hand"] + 1
        rng_master.set_state(meta["rng_master"])
        torch.set_rng_state(meta["torch_rng"])
        if hasattr(sampler, "history") and "sampler_history" in meta:
            sampler.history.update(meta["sampler_history"])
        if hasattr(sampler, "selection_log") and "sampler_selection" in meta:
            sampler.selection_log = list(meta["sampler_selection"])
        print(f"  -> Successfully resumed at Hand {start_hand}")
        
    cpu_monitor = CumulativeProcessCPUController(
        target_cpu_pct=70.0,
        log_path=os.path.join(out_dir, "cpu_metrics.jsonl")
    )
    
    ppo_logs = []
    if resume_from and os.path.exists(os.path.join(out_dir, "ppo_updates.jsonl")):
        with open(os.path.join(out_dir, "ppo_updates.jsonl"), "r", encoding="utf-8") as f:
            ppo_logs = [json.loads(line) for line in f if line.strip()]
    t0 = time.time()
    
    for h_i in range(start_hand, total_hands + 1):
        deck_rng = np.random.RandomState(seed_base + h_i * 10007)
        action_rng = np.random.RandomState(seed_base + h_i * 30011)
        
        # 1. Opponent Sampling
        opp_probs = sampler.get_probabilities()
        opp_names = list(opp_probs.keys())
        opp_p_vals = list(opp_probs.values())
        norm_p = np.array(opp_p_vals) / sum(opp_p_vals)
        chosen_opp = str(rng_master.choice(opp_names, p=norm_p))
        sampler.selection_log.append((h_i, chosen_opp))
        
        # 2. Perfectly Balanced Symmetrical Latin Square Assignment
        assign = get_perfect_latin_square_assignment(h_i)
        frozen_seat = assign["frozen_seat"]
        seat_to_agent = assign["seat_to_agent"]
        
        active_table = [None] * 5
        active_table[frozen_seat] = ("FROZEN", chosen_opp, frozen_models[chosen_opp])
        for seat_idx, a_idx in seat_to_agent.items():
            l_name = AGENT_NAMES[a_idx]
            active_table[seat_idx] = ("LEARNER", l_name, learners[l_name])
            
        # 3. Play Hand
        st = game.new_initial_state()
        advance_chance_nodes(st, deck_rng)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, deck_rng)
                continue
            cp = st.current_player()
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
            
            p_type, p_id, p_obj = active_table[cp]
            if p_type == "LEARNER":
                act, _ = p_obj.select_action(norm_inf, legal_6, action_rng)
            else:
                with torch.no_grad():
                    logits, _ = p_obj(torch.from_numpy(norm_inf).unsqueeze(0))
                    mask = torch.full_like(logits, -1e9)
                    for a in legal_6: mask[0, a] = 0.0
                    probs = torch.softmax(logits + mask, dim=-1).squeeze(0).numpy()
                    act = int(action_rng.choice(len(probs), p=probs))
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            
        # 4. Resolve Returns
        returns = st.returns()
        for pos, (p_type, p_id, p_obj) in enumerate(active_table):
            ret_bb = returns[pos] / BIG_BLIND
            if p_type == "LEARNER":
                p_obj.record_hand_reward(ret_bb)
                
        opp_net_bb = returns[frozen_seat] / BIG_BLIND
        sampler.record_match(chosen_opp, opp_net_bb)
        
        # 5. PPO Updates every 5 hands for active learners
        if h_i % 5 == 0:
            for a in AGENT_NAMES:
                up_res = learners[a].ppo_update()
                if up_res:
                    up_res["hand"] = h_i
                    up_res["agent"] = a
                    ppo_logs.append(up_res)
                    
        # 6. Periodic Checkpoints at 25h (test), 500h, 1000h, 1500h, 2000h
        if h_i == 25 or h_i % 500 == 0 or h_i == total_hands:
            ckpt_dir = os.path.join(out_dir, f"ckpt_hand_{h_i}")
            os.makedirs(ckpt_dir, exist_ok=True)
            for a in AGENT_NAMES:
                learners[a].save_checkpoint(os.path.join(ckpt_dir, f"{a}.pt"))
            meta = {
                "hand": h_i,
                "rng_master": rng_master.get_state(),
                "torch_rng": torch.get_rng_state(),
                "sampler_history": dict(sampler.history),
                "sampler_selection": list(sampler.selection_log[-500:])
            }
            torch.save(meta, os.path.join(ckpt_dir, "meta_state.pt"))
            
        # 7. CPU Throttling
        cpu_monitor.throttle(h_i)
        
        if h_i % 200 == 0 or h_i == total_hands:
            elapsed = time.time() - t0
            hps = h_i / max(0.1, elapsed)
            total_learner_pnl = sum(learners[a].total_pnl_bb for a in AGENT_NAMES)
            c_ratio = cpu_monitor.history[-1]["cumulative_cpu_ratio"] if cpu_monitor.history else 0.0
            print(f"  Hand {h_i:5d}/{total_hands} | HPS: {hps:.1f} | CPU Ratio: {c_ratio:.1f}% | Total PnL: {total_learner_pnl:+.2f} BB | Updates: {len(ppo_logs)}")
            
    with open(os.path.join(out_dir, "ppo_updates.jsonl"), "w", encoding="utf-8") as f:
        for entry in ppo_logs:
            f.write(json.dumps(entry) + "\n")
            
    final_cpu_ratio = cpu_monitor.history[-1]["cumulative_cpu_ratio"] if cpu_monitor.history else 0.0
    summary = {
        "arm_name": arm_name,
        "league_type": league_type,
        "opponent_pool_type": opponent_pool_type,
        "total_hands": total_hands,
        "seed_base": seed_base,
        "elapsed_seconds": round(time.time() - t0, 2),
        "cumulative_cpu_ratio_pct": final_cpu_ratio,
        "total_ppo_updates": len(ppo_logs),
        "learners_pnl": {a: round(learners[a].total_pnl_bb, 2) for a in AGENT_NAMES},
        "sum_learners_pnl": round(sum(learners[a].total_pnl_bb for a in AGENT_NAMES), 2),
        "sampler_final_probs": sampler.get_probabilities()
    }
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f">>> [DONE] {arm_name} finished in {summary['elapsed_seconds']}s (CPU Ratio: {final_cpu_ratio:.1f}%). Total PnL: {summary['sum_learners_pnl']} BB")
    return summary
