#!/usr/bin/env python3
"""
v1024/directive057_production_engine.py
DIRECTIVE 057: CONTROLLED DYNAMIC LEAGUE EXPANSION PRODUCTION RL ENGINE

Key Architectural Features:
1. Static vs Dynamic League Opponent Pool:
   - Base Pool: Frozen 2k (5 models: Agent A~E) + Frozen 7k (5 models: Agent A~E).
   - Dynamic Pool: Freezes and adds 5-agent learner snapshots at 500h, 1000h, 1500h, 2000h.
   - Enforces immutability: Snapshots are strictly frozen and cryptographically hashed.
2. Production Corrected PFSP with Exploration Floor:
   - Priority weight = exp(+opp_net_bb / T) where opp_net_bb > 0 indicates stronger opponent.
   - Minimum exploration floor epsilon = 0.05 ensures every opponent has positive selection probability.
   - Real-time opponent tracking: games, learner-view net BB, selection probability, logged to JSONL.
3. Perfectly Symmetrical Latin Square Seat Rotation:
   - 25-hand integer cycle ensures 100.000% uniform seat allocation (160 hands/seat per 1,000h).
4. Cumulative Process CPU Controller:
   - Maintains (Cumulative CPU / Wall Time) <= 70.0% with dynamic PID throttle sleep.
5. Independent Dual RNG (CRN):
   - Deck RNG and Action RNG strictly decoupled.
6. Bit-Exact Pause/Resume Serialization.
"""

import os
import sys
import json
import time
import hashlib
import shutil
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

class LeagueOpponentPool:
    """
    Manages Static or Dynamic Opponent Pool for DIRECTIVE 057.
    """
    def __init__(
        self,
        base_models: List[Dict[str, Any]], # list of {"id": str, "path": str, "meta": dict}
        mode: str = "static",              # "static" or "dynamic"
        sampler_type: str = "uniform",     # "uniform" or "pfsp"
        temperature: float = 10.0,
        min_exploration_prob: float = 0.05,
        clip_bound: float = 25.0,
        history_log_path: Optional[str] = None
    ):
        self.mode = mode.lower()
        self.sampler_type = sampler_type.lower()
        self.temperature = float(temperature)
        self.min_exploration_prob = float(min_exploration_prob)
        self.clip_bound = float(clip_bound)
        self.history_log_path = history_log_path
        
        self.opponents: Dict[str, Dict[str, Any]] = {}
        for bm in base_models:
            self._register_opponent(bm["id"], bm["path"], bm.get("meta", {}))
            
        self.selection_history = []

    def _register_opponent(self, opp_id: str, model_path: str, meta: Dict[str, Any]):
        sha = compute_sha256(model_path)
        self.opponents[opp_id] = {
            "id": opp_id,
            "path": model_path,
            "sha256": sha,
            "meta": meta,
            "added_time": time.time(),
            "games_played": 0,
            "learner_net_bb": 0.0,  # learner net profit against this opponent
            "opponent_net_bb": 0.0  # opponent net profit against learner
        }

    def add_snapshot(self, snapshot_id: str, model_path: str, meta: Dict[str, Any]):
        """Dynamically adds a frozen snapshot to the league pool."""
        if self.mode != "dynamic":
            raise RuntimeError(f"Cannot add snapshot in {self.mode} league mode!")
        if snapshot_id in self.opponents:
            return
        self._register_opponent(snapshot_id, model_path, meta)

    def record_result(self, opp_id: str, learner_net_bb: float, opponent_net_bb: float):
        if opp_id in self.opponents:
            opp = self.opponents[opp_id]
            opp["games_played"] += 1
            opp["learner_net_bb"] += float(learner_net_bb)
            opp["opponent_net_bb"] += float(opponent_net_bb)

    def get_selection_probabilities(self) -> Dict[str, float]:
        n = len(self.opponents)
        if n == 0:
            return {}
            
        opp_ids = sorted(list(self.opponents.keys()))
        uniform_p = 1.0 / n
        
        if self.sampler_type == "uniform":
            return {oid: uniform_p for oid in opp_ids}
            
        # PFSP Sampler: Prioritize opponents that defeat learner (higher opp["opponent_net_bb"] / games)
        raw_w = []
        for oid in opp_ids:
            opp = self.opponents[oid]
            g = opp["games_played"]
            if g == 0:
                raw_w.append(1.0)
            else:
                avg_opp_bb = opp["opponent_net_bb"] / g
                clipped = float(np.clip(avg_opp_bb, -self.clip_bound, self.clip_bound))
                raw_w.append(float(np.exp(clipped / self.temperature)))
                
        tot_w = sum(raw_w)
        probs = {}
        for oid, w in zip(opp_ids, raw_w):
            # Apply exploration floor
            p = (1.0 - self.min_exploration_prob) * (w / tot_w) + self.min_exploration_prob * uniform_p
            probs[oid] = float(p)
            
        tot_p = sum(probs.values())
        norm = {k: v / tot_p for k, v in probs.items()}
        assert abs(sum(norm.values()) - 1.0) < 1e-6, "League probabilities sum != 1.0"
        return norm

    def sample_opponent(self, rng: np.random.RandomState, hand_idx: int) -> Tuple[str, str]:
        """Returns (opp_id, model_path)."""
        probs_dict = self.get_selection_probabilities()
        opp_ids = sorted(list(probs_dict.keys()))
        probs = [probs_dict[oid] for oid in opp_ids]
        
        chosen_id = str(rng.choice(opp_ids, p=probs))
        chosen_path = self.opponents[chosen_id]["path"]
        
        log_entry = {
            "hand": hand_idx,
            "chosen_opponent": chosen_id,
            "pool_size": len(self.opponents),
            "sampler_type": self.sampler_type,
            "mode": self.mode,
            "chosen_prob": round(probs_dict[chosen_id], 5)
        }
        self.selection_history.append(log_entry)
        if self.history_log_path:
            with open(self.history_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        return chosen_id, chosen_path

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

class Directive057PPOAgent:
    """Production PPO Agent with Exact Categorical KL, Policy Entropy, and NaN Protection."""
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
        if len(self.states) > len(self.rewards):
            diff = len(self.states) - len(self.rewards)
            for i in range(diff):
                is_last = (i == diff - 1)
                self.rewards.append(hand_return_bb if is_last else 0.0)
                self.dones.append(is_last)

    def update_policy(self, k_epochs: int = 4, entropy_coef: float = 0.01) -> Optional[Dict[str, float]]:
        if len(self.states) < 4:
            return None
            
        states_t = torch.from_numpy(np.array(self.states, dtype=np.float32))
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(self.old_log_probs, dtype=torch.float32)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32)
        values_t = torch.tensor(self.values, dtype=torch.float32)
        dones_t = torch.tensor(self.dones, dtype=torch.float32)
        old_dist_probs_t = torch.from_numpy(np.array(self.old_dist_probs, dtype=np.float32))
        
        # GAE Advantage Estimation
        advantages = torch.zeros_like(rewards_t)
        last_gae = 0.0
        n_steps = len(rewards_t)
        for t in reversed(range(n_steps)):
            next_val = values_t[t + 1] if t + 1 < n_steps else 0.0
            non_terminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_val * non_terminal - values_t[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            last_gae = advantages[t].item()
            
        returns = advantages + values_t
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std
        
        # Initial policy distribution pre-update
        with torch.no_grad():
            init_logits, _ = self.model(states_t)
            for idx, legals in enumerate(self.legal_masks):
                m = torch.full((6,), -1e9)
                m[legals] = 0.0
                init_logits[idx] += m
            init_probs = torch.softmax(init_logits, dim=-1)
            init_entropy = -torch.sum(init_probs * torch.log(init_probs + 1e-12), dim=-1).mean().item()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        grad_norm_accum = 0.0
        
        for _ in range(k_epochs):
            logits, val = self.model(states_t)
            for idx, legals in enumerate(self.legal_masks):
                m = torch.full((6,), -1e9)
                m[legals] = 0.0
                logits[idx] += m
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            
            ratios = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratios * norm_advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * norm_advantages
            pol_loss = -torch.min(surr1, surr2).mean()
            
            val_loss = F.mse_loss(val.squeeze(-1), returns)
            ent_loss = -dist.entropy().mean()
            
            loss = pol_loss + 0.5 * val_loss + entropy_coef * ent_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss_accum += loss.item()
            policy_loss_accum += pol_loss.item()
            value_loss_accum += val_loss.item()
            grad_norm_accum += gnorm.item()
            self.gradient_steps += 1
            
        # Post-update Categorical KL and Policy Entropy
        with torch.no_grad():
            post_logits, _ = self.model(states_t)
            for idx, legals in enumerate(self.legal_masks):
                m = torch.full((6,), -1e9)
                m[legals] = 0.0
                post_logits[idx] += m
            post_probs = torch.softmax(post_logits, dim=-1)
            post_dist = torch.distributions.Categorical(post_probs)
            post_entropy = post_dist.entropy().mean().item()
            
            # Exact Categorical KL Divergence: KL(P_init || P_post)
            kl_div = torch.sum(
                init_probs * (torch.log(init_probs + 1e-12) - torch.log(post_probs + 1e-12)),
                dim=-1
            ).mean().item()
            
            # Action ratio
            post_new_log_probs = post_dist.log_prob(actions_t)
            post_ratios = torch.exp(post_new_log_probs - old_log_probs_t)
            clip_fraction = (torch.abs(post_ratios - 1.0) > self.clip_eps).float().mean().item()

        metrics = {
            "agent": self.name,
            "gradient_steps": self.gradient_steps,
            "loss": total_loss_accum / k_epochs,
            "policy_loss": policy_loss_accum / k_epochs,
            "value_loss": value_loss_accum / k_epochs,
            "grad_norm": grad_norm_accum / k_epochs,
            "entropy_pre": init_entropy,
            "entropy_post": post_entropy,
            "kl_div": kl_div,
            "clip_fraction": clip_fraction,
            "num_decisions": len(self.states)
        }
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.old_dist_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.legal_masks.clear()
        
        return metrics

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "agent_name": self.name,
            "step_count": self.step_count,
            "gradient_steps": self.gradient_steps,
            "total_pnl_bb": self.total_pnl_bb,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "meta": extra_meta or {}
        }
        torch.save(ckpt, path)
        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "agent_name": self.name,
                "step_count": self.step_count,
                "gradient_steps": self.gradient_steps,
                "total_pnl_bb": self.total_pnl_bb,
                "sha256": compute_sha256(path),
                "timestamp": time.time(),
                "extra": extra_meta or {}
            }, f, indent=2)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[FAIL_FAST] Checkpoint path missing: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step_count = ckpt.get("step_count", 0)
        self.gradient_steps = ckpt.get("gradient_steps", 0)
        self.total_pnl_bb = ckpt.get("total_pnl_bb", 0.0)


class Directive057ProductionSession:
    """
    Executes a complete training arm under DIRECTIVE 057 rules:
    - 5-agent learner simultaneous PPO
    - Symmetrical Latin Square seat rotation
    - Static vs Dynamic League Opponent Pool
    - Cumulative CPU Time / Wall Time <= 70.0%
    - 500h Periodic Snapshot Ingestion (Dynamic League)
    - Checkpoint saving & state serialization
    """
    def __init__(
        self,
        arm_name: str,
        league_mode: str,       # "static" or "dynamic"
        sampler_type: str,      # "uniform" or "pfsp"
        total_hands: int,
        seed_base: int,
        out_dir: str,
        resume_from: Optional[str] = None
    ):
        self.arm_name = arm_name
        self.league_mode = league_mode.lower()
        self.sampler_type = sampler_type.lower()
        self.total_hands = total_hands
        self.seed_base = seed_base
        self.out_dir = out_dir
        self.resume_from = resume_from
        
        os.makedirs(self.out_dir, exist_ok=True)
        self.snapshot_base_dir = os.path.join(self.out_dir, "snapshots")
        os.makedirs(self.snapshot_base_dir, exist_ok=True)
        
        self.game = get_5p_fullgame()
        self.learners = {
            a: Directive057PPOAgent(a, os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt"))
            for a in AGENT_NAMES
        }
        
        # 1. Base Opponent Pool: Frozen 2k (5) + Frozen 7k (5) = 10 models
        base_models = []
        for a in AGENT_NAMES:
            base_models.append({
                "id": f"frozen_2k_{a}",
                "path": os.path.join(CHECKPOINT_DIR, f"{a}_2000h.pt"),
                "meta": {"source": "frozen_2k", "agent": a}
            })
            base_models.append({
                "id": f"frozen_7k_{a}",
                "path": os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt"),
                "meta": {"source": "frozen_7k", "agent": a}
            })
            
        self.league_pool = LeagueOpponentPool(
            base_models=base_models,
            mode=self.league_mode,
            sampler_type=self.sampler_type,
            temperature=10.0,
            min_exploration_prob=0.05,
            history_log_path=os.path.join(self.out_dir, "league_selection_history.jsonl")
        )
        
        # In-memory opponent player instances cache
        self.opp_models_cache: Dict[str, PPONeuralPlayer] = {}
        for bm in base_models:
            self._load_cached_opponent_model(bm["id"], bm["path"])
            
        self.cpu_controller = CumulativeProcessCPUController(
            target_cpu_pct=70.0,
            log_path=os.path.join(self.out_dir, "cpu_metrics.jsonl")
        )
        
        self.rng_master = np.random.RandomState(seed_base)
        torch.manual_seed(seed_base)
        self.start_hand = 1
        self.ppo_logs = []
        
        if self.resume_from and os.path.exists(self.resume_from):
            self._resume_state()

    def _load_cached_opponent_model(self, opp_id: str, path: str):
        if opp_id not in self.opp_models_cache:
            m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            m.eval()
            self.opp_models_cache[opp_id] = m

    def _resume_state(self):
        print(f"[*] Resuming state from {self.resume_from}...")
        for a in AGENT_NAMES:
            self.learners[a].load_checkpoint(os.path.join(self.resume_from, f"{a}.pt"))
            
        meta = torch.load(os.path.join(self.resume_from, "meta_state.pt"), weights_only=False)
        self.start_hand = meta["hand"] + 1
        self.rng_master.set_state(meta["rng_master"])
        torch.set_rng_state(meta["torch_rng"])
        
        # Restore league pool
        if "league_opponents" in meta:
            for opp_id, opp_info in meta["league_opponents"].items():
                if opp_id not in self.league_pool.opponents:
                    self.league_pool.add_snapshot(opp_id, opp_info["path"], opp_info["meta"])
                    self._load_cached_opponent_model(opp_id, opp_info["path"])
                # restore stats
                self.league_pool.opponents[opp_id]["games_played"] = opp_info["games_played"]
                self.league_pool.opponents[opp_id]["learner_net_bb"] = opp_info["learner_net_bb"]
                self.league_pool.opponents[opp_id]["opponent_net_bb"] = opp_info["opponent_net_bb"]
                
        # Restore PPO logs
        log_file = os.path.join(self.out_dir, "ppo_updates.jsonl")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                self.ppo_logs = [json.loads(line) for line in f if line.strip()]
        print(f"[*] Resumed at hand {self.start_hand}. Total opponents in pool: {len(self.league_pool.opponents)}")

    def _create_and_ingest_snapshot(self, hand_idx: int):
        """Creates an immutable frozen snapshot of learners and adds them to league pool."""
        snap_dir = os.path.join(self.snapshot_base_dir, f"snap_{hand_idx}h")
        os.makedirs(snap_dir, exist_ok=True)
        print(f"\n[+] [DYNAMIC LEAGUE] Freezing 5-learner snapshot at Hand {hand_idx} -> {snap_dir}")
        
        for a in AGENT_NAMES:
            snap_path = os.path.join(snap_dir, f"learner_{a}.pt")
            # Save raw model weights
            torch.save(self.learners[a].model.state_dict(), snap_path)
            snap_id = f"snap_{hand_idx}h_{a}"
            
            meta = {
                "source": "dynamic_snapshot",
                "hand_created": hand_idx,
                "agent": a,
                "sha256": compute_sha256(snap_path)
            }
            self.league_pool.add_snapshot(snap_id, snap_path, meta)
            self._load_cached_opponent_model(snap_id, snap_path)
            
        print(f"    League Pool expanded to {len(self.league_pool.opponents)} total opponents.")

    def run(self) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f">>> [DIRECTIVE 057] Starting Session: {self.arm_name}")
        print(f"    Mode: {self.league_mode} | Sampler: {self.sampler_type} | Hands: {self.total_hands}")
        print(f"    Seed: {self.seed_base} | Directory: {self.out_dir}")
        print(f"=======================================================")
        
        t0 = time.time()
        
        for h_i in range(self.start_hand, self.total_hands + 1):
            deck_rng = np.random.RandomState((self.seed_base * 1000003 + h_i * 10007) & 0xFFFFFFFF)
            action_rng = np.random.RandomState((self.seed_base * 3000017 + h_i * 30011) & 0xFFFFFFFF)
            
            # 1. Opponent Sampling from League Pool
            opp_id, opp_path = self.league_pool.sample_opponent(self.rng_master, h_i)
            opp_model = self.opp_models_cache[opp_id]
            
            # 2. Latin Square Symmetrical Seat Rotation
            assign = get_perfect_latin_square_assignment(h_i)
            frozen_seat = assign["frozen_seat"]
            seat_to_agent = assign["seat_to_agent"]
            
            active_table = [None] * 5
            active_table[frozen_seat] = ("OPPONENT", opp_id, opp_model)
            for seat_idx, a_idx in seat_to_agent.items():
                l_name = AGENT_NAMES[a_idx]
                active_table[seat_idx] = ("LEARNER", l_name, self.learners[l_name])
                
            # 3. Play Hand
            st = self.game.new_initial_state()
            advance_chance_nodes(st, deck_rng)
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, deck_rng)
                    continue
                cp = st.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                
                p_type, p_name, p_obj = active_table[cp]
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
                
            # 4. Resolve Returns and Track Net BB
            returns = st.returns()
            for pos, (p_type, p_name, p_obj) in enumerate(active_table):
                ret_bb = returns[pos] / BIG_BLIND
                if p_type == "LEARNER":
                    p_obj.record_hand_reward(ret_bb)
                    
            opp_net_bb = returns[frozen_seat] / BIG_BLIND
            learner_avg_bb = -opp_net_bb / 4.0 # 4 learners combined vs 1 frozen
            self.league_pool.record_result(opp_id, learner_avg_bb, opp_net_bb)
            
            # 5. PPO Updates every 5 hands for all 5 learners
            if h_i % 5 == 0:
                for a in AGENT_NAMES:
                    up_res = self.learners[a].update_policy()
                    if up_res:
                        up_res["hand"] = h_i
                        up_res["agent"] = a
                        self.ppo_logs.append(up_res)
                        
            # 6. Dynamic League Snapshot Addition at 500h, 1000h, 1500h
            if self.league_mode == "dynamic" and (h_i % 500 == 0) and h_i < self.total_hands:
                self._create_and_ingest_snapshot(h_i)
                
            # 7. Checkpointing at 25h (test), 500h, 1000h, 1500h, 2000h
            if h_i == 25 or h_i % 500 == 0 or h_i == self.total_hands:
                ckpt_dir = os.path.join(self.out_dir, f"ckpt_hand_{h_i}")
                os.makedirs(ckpt_dir, exist_ok=True)
                for a in AGENT_NAMES:
                    self.learners[a].save_checkpoint(os.path.join(ckpt_dir, f"{a}.pt"))
                meta = {
                    "hand": h_i,
                    "rng_master": self.rng_master.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "league_opponents": {
                        oid: {
                            "path": o["path"],
                            "meta": o["meta"],
                            "games_played": o["games_played"],
                            "learner_net_bb": o["learner_net_bb"],
                            "opponent_net_bb": o["opponent_net_bb"]
                        } for oid, o in self.league_pool.opponents.items()
                    }
                }
                torch.save(meta, os.path.join(ckpt_dir, "meta_state.pt"))
                
            # 8. CPU Throttling (Wall-time / CPU ratio control)
            self.cpu_controller.throttle(h_i)
            
            if h_i % 200 == 0 or h_i == self.total_hands:
                elapsed = time.time() - t0
                hps = h_i / max(0.1, elapsed)
                total_pnl = sum(self.learners[a].total_pnl_bb for a in AGENT_NAMES)
                c_ratio = self.cpu_controller.history[-1]["cumulative_cpu_ratio"] if self.cpu_controller.history else 0.0
                print(f"  Hand {h_i:5d}/{self.total_hands} | HPS: {hps:.1f} | CPU Ratio: {c_ratio:.1f}% | Total PnL: {total_pnl:+.2f} BB | Opponents: {len(self.league_pool.opponents)}")
                
        # Save PPO logs
        with open(os.path.join(self.out_dir, "ppo_updates.jsonl"), "w", encoding="utf-8") as f:
            for entry in self.ppo_logs:
                f.write(json.dumps(entry) + "\n")
                
        final_cpu_ratio = self.cpu_controller.history[-1]["cumulative_cpu_ratio"] if self.cpu_controller.history else 0.0
        summary = {
            "arm_name": self.arm_name,
            "league_mode": self.league_mode,
            "sampler_type": self.sampler_type,
            "total_hands": self.total_hands,
            "seed_base": self.seed_base,
            "elapsed_seconds": round(time.time() - t0, 2),
            "final_cpu_ratio_pct": final_cpu_ratio,
            "total_ppo_updates": len(self.ppo_logs),
            "total_opponents_in_pool": len(self.league_pool.opponents),
            "learners_pnl": {a: round(self.learners[a].total_pnl_bb, 2) for a in AGENT_NAMES},
            "sum_learners_pnl": round(sum(self.learners[a].total_pnl_bb for a in AGENT_NAMES), 2),
            "final_probabilities": self.league_pool.get_selection_probabilities()
        }
        with open(os.path.join(self.out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        print(f">>> [DONE] {self.arm_name} finished in {summary['elapsed_seconds']}s (CPU Ratio: {final_cpu_ratio:.1f}%). Total PnL: {summary['sum_learners_pnl']} BB")
        return summary
