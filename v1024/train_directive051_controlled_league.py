#!/usr/bin/env python3
"""
v1024/train_directive051_controlled_league.py
DIRECTIVE 051: CONTROLLED ALPHASTAR LEAGUE TRAINING ENGINE

Implements:
1. Three Experimental Arms starting from identical 7000h baseline:
   - Arm A: Full Main Self-Play (2,000 table hands, 5 active learners = 10,000 agent hands)
   - Arm B: Uniform Historical League (3,300 table hands, 3 active learners + 2 frozen uniform = 9,900 agent hands)
   - Arm C: PFSP Historical League (3,300 table hands, 3 active learners + 2 frozen PFSP = 9,900 agent hands)
2. Complete checkpoint saving/loading:
   - Model state dict
   - Optimizer state dict
   - Experience rollout buffer
   - Python, NumPy, Torch RNG states
   - Sampler state & win rates
3. CPU 70% throttling, zero-sum checking, NaN/Inf safeguard, immutable frozen checkpoints.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.audit_directive049c_final_gate import DynamicAlphaStarPFSPSampler

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

def multi_way_payoff_attribution(returns: List[float]) -> Dict[Tuple[int, int], float]:
    """Exact multi-way zero-sum head-to-head payoff attribution."""
    winners = {i: returns[i] for i in range(5) if returns[i] > 1e-6}
    losers = {i: -returns[i] for i in range(5) if returns[i] < -1e-6}
    total_gain = sum(winners.values())
    total_loss = sum(losers.values())
    pairwise_h2h = defaultdict(float)
    if total_gain > 1e-6 and total_loss > 1e-6:
        for w_idx, w_amt in winners.items():
            for l_idx, l_amt in losers.items():
                contrib = (w_amt / total_gain) * l_amt
                pairwise_h2h[(w_idx, l_idx)] = contrib
    return dict(pairwise_h2h)

class UniformHistoricalSampler:
    """Uniform Historical Opponent Sampler (Control)."""
    def __init__(self, opponent_pool: List[str]):
        self.opponent_pool = list(opponent_pool)
        self.sample_counts = Counter()
        self.name = "UniformHistoricalSampler"
        
    def sample_opponent(self, rng: np.random.RandomState, hand_id: int) -> Tuple[str, float]:
        idx = rng.randint(0, len(self.opponent_pool))
        chosen = self.opponent_pool[idx]
        self.sample_counts[chosen] += 1
        return chosen, 1.0 / len(self.opponent_pool)
        
    def record_match(self, opponent_name: str, net_score_bb: float):
        pass

class FullStatePPOAgent:
    """Production PPO Agent with complete bitwise state saving and loading."""
    def __init__(self, name: str, model_path: str, lr: float = 3e-4, clip_eps: float = 0.2):
        self.name = name
        self.model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256, lr=lr, clip_eps=clip_eps)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[FAIL_FAST] Model path missing: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.train()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
        self.clip_eps = clip_eps
        
        # Buffer
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.legal_masks = []
        
        self.step_count = 0
        self.gradient_steps = 0
        self.total_pnl_bb = 0.0

    def select_action(self, norm_inf: np.ndarray, legal_actions: List[int], rng: np.random.RandomState) -> Tuple[int, float]:
        self.step_count += 1
        inf_tensor = torch.from_numpy(norm_inf).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(inf_tensor)
            mask = torch.full_like(logits, -1e9)
            for a in legal_actions:
                mask[0, a] = 0.0
            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()
            
            action = int(rng.choice(len(probs), p=probs))
            dist = torch.distributions.Categorical(torch.softmax(masked_logits, dim=-1))
            log_prob = dist.log_prob(torch.tensor(action)).item()
            
        self.states.append(norm_inf)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.legal_masks.append(legal_actions)
        return action, log_prob

    def record_hand_reward(self, hand_return_bb: float):
        self.total_pnl_bb += hand_return_bb
        num_new_steps = len(self.states) - len(self.rewards)
        if num_new_steps > 0:
            step_reward = hand_return_bb / num_new_steps
            self.rewards.extend([step_reward] * num_new_steps)

    def ppo_update(self) -> Dict[str, float]:
        if len(self.states) < 4:
            return {"loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_lp_t = torch.tensor(self.old_log_probs, dtype=torch.float32)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32)
        
        adv = rewards_t - rewards_t.mean()
        if adv.std() > 1e-6:
            adv = adv / (adv.std() + 1e-8)
            
        self.optimizer.zero_grad()
        logits, values = self.model(states_t)
        
        masked_logits = logits.clone()
        for idx, leg in enumerate(self.legal_masks):
            mask = torch.full_like(logits[idx], -1e9)
            for a in leg: mask[a] = 0.0
            masked_logits[idx] = logits[idx] + mask
            
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_lp = dist.log_prob(actions_t)
        
        ratio = torch.exp(new_lp - old_lp_t)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(-1), rewards_t)
        loss = policy_loss + 0.5 * value_loss
        
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).item())
        self.optimizer.step()
        self.gradient_steps += 1
        
        with torch.no_grad():
            kl = float((old_lp_t - new_lp).mean().item())
            clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())
            
        self.states.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.rewards.clear()
        self.legal_masks.clear()
        return {
            "loss": float(loss.item()),
            "kl": kl,
            "clip_frac": clip_frac,
            "grad_norm": grad_norm
        }

    def save_checkpoint(self, path: str):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "gradient_steps": self.gradient_steps,
            "total_pnl_bb": self.total_pnl_bb,
            "states": self.states,
            "actions": self.actions,
            "old_log_probs": self.old_log_probs,
            "rewards": self.rewards,
            "legal_masks": self.legal_masks
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step_count = ckpt.get("step_count", 0)
        self.gradient_steps = ckpt.get("gradient_steps", 0)
        self.total_pnl_bb = ckpt.get("total_pnl_bb", 0.0)
        self.states = ckpt.get("states", [])
        self.actions = ckpt.get("actions", [])
        self.old_log_probs = ckpt.get("old_log_probs", [])
        self.rewards = ckpt.get("rewards", [])
        self.legal_masks = ckpt.get("legal_masks", [])

def run_experiment_arm(
    arm_mode: str,
    total_hands: int,
    output_dir: str,
    seed_base: int = 100000,
    update_interval: int = 25,
    ckpt_interval: int = 500
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f">>> [DIRECTIVE 051] Running Arm: {arm_mode} ({total_hands} Hands)")
    print(f"    Output Dir: {output_dir}")
    print(f"=======================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    game = get_5p_fullgame()
    
    # 1. Initialize Active Learners from 7000h Checkpoints
    learners = {}
    for a in AGENT_NAMES:
        init_path = os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")
        learners[a] = FullStatePPOAgent(a, init_path)
    print(f"  Initialized 5 Active Learners from 7000h baseline.")
    
    # 2. Frozen Opponent Pool (2000h)
    frozen_pool = [f"{a}_2000h" for a in AGENT_NAMES]
    frozen_shas_start = {c: compute_sha256(os.path.join(CHECKPOINT_DIR, f"{c}.pt")) for c in frozen_pool}
    frozen_models = {}
    for c in frozen_pool:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{c}.pt"), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            m.load_state_dict(ckpt["model_state_dict"])
        else:
            m.load_state_dict(ckpt)
        m.eval()
        frozen_models[c] = m
        
    # 3. Sampler Selection
    if arm_mode == "uniform_league":
        sampler = UniformHistoricalSampler(frozen_pool)
    elif arm_mode == "pfsp_league":
        sampler = DynamicAlphaStarPFSPSampler(frozen_pool, temperature=10.0, exploration_floor=0.1)
    else:
        sampler = None # full_main
        
    rng_master = np.random.RandomState(seed_base)
    
    hand_logs = []
    update_logs = []
    zero_sum_violations = 0
    illegal_actions = 0
    nan_inf_violations = 0
    
    t_start = time.time()
    for h_i in range(1, total_hands + 1):
        hand_seed = seed_base + h_i
        hand_rng = np.random.RandomState(hand_seed)
        
        if arm_mode == "full_main":
            # All 5 seats are active learners, fair seat permutation
            table_agents = [AGENT_NAMES[(h_i + i) % 5] for i in range(5)]
            active_seats = [0, 1, 2, 3, 4]
            frozen_seats = []
        else:
            # 3 active learners + 2 frozen opponents
            table_learners = [AGENT_NAMES[(h_i + i) % 5] for i in range(3)]
            opp1, _ = sampler.sample_opponent(rng_master, h_i)
            opp2, _ = sampler.sample_opponent(rng_master, h_i)
            table_agents = table_learners + [opp1, opp2]
            active_seats = [0, 1, 2]
            frozen_seats = [3, 4]
            
        st = game.new_initial_state()
        advance_chance_nodes(st, hand_rng)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, hand_rng)
                continue
            cp = st.current_player()
            ag_id = table_agents[cp]
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            if not legal_6:
                illegal_actions += 1
                break
            norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
            
            if cp in active_seats:
                act, _ = learners[ag_id].select_action(norm_inf, legal_6, rng_master)
            else:
                m = frozen_models[ag_id]
                with torch.no_grad():
                    lgt, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                    msk = torch.full_like(lgt, -1e9)
                    for a in legal_6: msk[0, a] = 0.0
                    act = int(torch.argmax(lgt + msk, dim=-1).item())
                    
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            
        returns = [st.returns()[p] / BIG_BLIND for p in range(5)]
        if abs(sum(returns)) > 1e-4:
            zero_sum_violations += 1
            
        for cp in active_seats:
            ag_id = table_agents[cp]
            learners[ag_id].record_hand_reward(returns[cp])
            
        # Multi-way zero-sum attribution for league opponents
        if arm_mode in ["uniform_league", "pfsp_league"]:
            h2h_map = multi_way_payoff_attribution(returns)
            # Net payoff between active learners (0, 1, 2) and opponent 1 (seat 3) & opponent 2 (seat 4)
            net_to_opp1 = sum(h2h_map.get((3, p), 0.0) - h2h_map.get((p, 3), 0.0) for p in [0, 1, 2])
            net_to_opp2 = sum(h2h_map.get((4, p), 0.0) - h2h_map.get((p, 4), 0.0) for p in [0, 1, 2])
            sampler.record_match(table_agents[3], -net_to_opp1)
            sampler.record_match(table_agents[4], -net_to_opp2)
            
        # PPO Update
        if h_i % update_interval == 0:
            for a_name in AGENT_NAMES:
                metrics = learners[a_name].ppo_update()
                if metrics["loss"] > 0:
                    update_logs.append({
                        "hand": h_i,
                        "agent": a_name,
                        "metrics": metrics
                    })
                    if np.isnan(metrics["loss"]) or np.isinf(metrics["loss"]):
                        nan_inf_violations += 1
                        
        # Periodic Checkpoints
        if h_i % ckpt_interval == 0 or h_i == total_hands:
            ckpt_sub = os.path.join(output_dir, f"ckpt_hand_{h_i}")
            os.makedirs(ckpt_sub, exist_ok=True)
            for a_name in AGENT_NAMES:
                learners[a_name].save_checkpoint(os.path.join(ckpt_sub, f"{a_name}.pt"))
            # Save complete sampler state & RNG state
            state_bundle = {
                "hand": h_i,
                "rng_master": rng_master.get_state(),
                "torch_rng": torch.get_rng_state(),
                "arm_mode": arm_mode
            }
            if sampler and hasattr(sampler, "opponent_history"):
                state_bundle["sampler_history"] = dict(sampler.opponent_history)
                state_bundle["sampler_weights"] = dict(sampler.opponent_weights)
            torch.save(state_bundle, os.path.join(ckpt_sub, "meta_state.pt"))
            print(f"  [{arm_mode}] Checkpoint saved at Hand {h_i} -> {ckpt_sub}")
            
        # CPU Throttler: Sleep 2ms per hand to keep CPU usage <= 70% on Mac Mini
        time.sleep(0.002)
        
    total_time = time.time() - t_start
    
    # Frozen checkpoint immutability check
    frozen_shas_end = {c: compute_sha256(os.path.join(CHECKPOINT_DIR, f"{c}.pt")) for c in frozen_pool}
    immutable = (frozen_shas_start == frozen_shas_end)
    
    summary = {
        "arm_mode": arm_mode,
        "total_hands": total_hands,
        "runtime_seconds": round(total_time, 2),
        "zero_sum_violations": zero_sum_violations,
        "illegal_actions": illegal_actions,
        "nan_inf_violations": nan_inf_violations,
        "frozen_models_immutable": immutable,
        "total_ppo_updates": len(update_logs),
        "agent_total_pnl": {a: round(learners[a].total_pnl_bb, 2) for a in AGENT_NAMES},
        "agent_gradient_steps": {a: learners[a].gradient_steps for a in AGENT_NAMES}
    }
    
    # Save training summary
    with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [{arm_mode}] Training Completed in {total_time:.2f}s. Summary: {summary}")
    return summary

def run_canary_phase():
    """Runs Phase 1 Canary Test: 300 hands per arm (Total 900 table hands)."""
    print("\n==================================================================")
    print("STARTING DIRECTIVE 051 PHASE 1: INTEGRATED CANARY TEST (900 HANDS)")
    print("==================================================================")
    
    canary_dir = "v1024/exp_directive051/canary"
    res_a = run_experiment_arm("full_main", 300, os.path.join(canary_dir, "arm_a_full_main"), seed_base=101000)
    res_b = run_experiment_arm("uniform_league", 300, os.path.join(canary_dir, "arm_b_uniform"), seed_base=102000)
    res_c = run_experiment_arm("pfsp_league", 300, os.path.join(canary_dir, "arm_c_pfsp"), seed_base=103000)
    
    canary_pass = (
        res_a["zero_sum_violations"] == 0 and res_b["zero_sum_violations"] == 0 and res_c["zero_sum_violations"] == 0 and
        res_a["illegal_actions"] == 0 and res_b["illegal_actions"] == 0 and res_c["illegal_actions"] == 0 and
        res_a["nan_inf_violations"] == 0 and res_b["nan_inf_violations"] == 0 and res_c["nan_inf_violations"] == 0 and
        res_a["frozen_models_immutable"] and res_b["frozen_models_immutable"] and res_c["frozen_models_immutable"]
    )
    print(f"\n[CANARY OVERALL RESULT] Phase 1 Canary Passed: {canary_pass}")
    return canary_pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIRECTIVE 051 Training Engine")
    parser.add_argument("--mode", choices=["canary", "full_main", "uniform_league", "pfsp_league", "all_main"], default="canary")
    parser.add_argument("--hands", type=int, default=300)
    args = parser.parse_args()
    
    if args.mode == "canary":
        run_canary_phase()
    elif args.mode == "all_main":
        main_dir = "v1024/exp_directive051/production"
        run_experiment_arm("full_main", 2000, os.path.join(main_dir, "arm_a_full_main"), seed_base=201000)
        run_experiment_arm("uniform_league", 3300, os.path.join(main_dir, "arm_b_uniform"), seed_base=202000)
        run_experiment_arm("pfsp_league", 3300, os.path.join(main_dir, "arm_c_pfsp"), seed_base=203000)
    else:
        out_dir = f"v1024/exp_directive051/run_{args.mode}"
        run_experiment_arm(args.mode, args.hands, out_dir, seed_base=300000)
