"""
DIRECTIVE 043: Universal Poker AlphaGo
5-Player Multi-Agent Self-Play Reinforcement Learning Engine

Architecture:
- 5 Independent PPO Agents (Agent A, B, C, D, E)
- Independent Neural Models, Optimizers, and Experience Buffers
- Absolute Input Normalization Integrity (normalize_inf_state_v79 applied uniformly)
- Multi-Agent Zero-Sum Invariant Checked Every Hand: sum(returns) == 0.0
- Simultaneous Independent Gradient Updates for all 5 Agents
- Real-time Tracking of Action Distributions, Losses, and Strategic Shifts
"""
import sys
import os
import time
import random
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.tests.test_directive040_regression import robust_parse_cards, parse_board_cards
from v1024.integrity_poker_metrics import HandIntegrityTracker

STARTING_STACK = 10000.0 # $100 BB

AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]

class IndependentPPOAgent:
    """완전히 독립적인 신경망 가중치, Optimizer, 버퍼를 가진 강화학습 에이전트"""
    def __init__(self, agent_id: int, name: str, base_ckpt_path: str = None,
                 lr: float = 0.0003, clip_eps: float = 0.2, beta_entropy: float = 0.035, seed: int = 42):
        self.agent_id = agent_id
        self.name = name
        self.clip_eps = clip_eps
        self.beta_entropy = beta_entropy
        self.device = torch.device("cpu")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        if base_ckpt_path and os.path.exists(base_ckpt_path):
            state_dict = torch.load(base_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            # Add unique small symmetry-breaking perturbation per agent
            with torch.no_grad():
                for p in self.model.parameters():
                    noise = torch.randn_like(p) * 0.005 * (agent_id + 1)
                    p.add_(noise)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rollout_buffer = [] # (norm_inf, act, log_prob, val, legal_mask)
        
        # Statistics
        self.action_counts = Counter()
        self.street_action_counts = defaultdict(Counter)
        self.entropy_history = []
        self.loss_history = []
        self.gradient_update_count = 0
        self.total_pnl_bb = 0.0

    def select_action(self, norm_inf: np.ndarray, legal_actions: List[int], round_idx: int) -> Tuple[int, float, float, np.ndarray]:
        """
        반드시 정규화된 norm_inf 상태를 입력받아 행동, log_prob, value 산출
        (DIRECTIVE 042 정규화 불일치 결함 원천 해결)
        """
        self.model.eval()
        t_state = torch.from_numpy(norm_inf).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, val = self.model(t_state)
            mask = torch.full_like(logits, -1e9)
            for a in legal_actions:
                mask[0, a] = 0.0
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            probs_arr = probs.squeeze(0).cpu().numpy()
            
        act_int = int(action.item())
        self.action_counts[act_int] += 1
        self.street_action_counts[round_idx][act_int] += 1
        
        probs_safe = np.maximum(probs_arr, 1e-8)
        ent = float(-np.sum(probs_safe * np.log(probs_safe)))
        self.entropy_history.append(ent)
        
        return act_int, float(log_prob.item()), float(val.item()), probs_arr

    def buffer_transition(self, norm_inf: np.ndarray, action: int, log_prob: float,
                          ret_bb: float, value: float, legal_actions: List[int]):
        """할인된 보상과 Advantage를 계산하여 독립 버퍼에 적재"""
        r_norm = ret_bb / 100.0 # Reward scaling
        adv = r_norm - value
        mask = [1 if k in legal_actions else 0 for k in range(6)]
        self.rollout_buffer.append((norm_inf, action, log_prob, r_norm, adv, mask))

    def update_policy(self, batch_size: int = 64, epochs: int = 4) -> Dict[str, float]:
        """
        독립 PPO 정책 업데이트 수행
        norm_inf 상태를 일관되게 사용하여 Importance Sampling Ratio 무결성 보장
        """
        if len(self.rollout_buffer) < batch_size:
            return {}
            
        buf = self.rollout_buffer
        states, actions, old_lps, returns, advs, masks = zip(*buf)
        s_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        a_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_lp_t = torch.tensor(old_lps, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)
        adv_t = torch.tensor(advs, dtype=torch.float32).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        mask_t = torch.tensor(np.array(masks), dtype=torch.bool).to(self.device)
        
        ds = len(buf)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        clip_fractions = []
        ratios_list = []
        steps = 0

        self.model.train()
        for _ in range(epochs):
            perm = np.random.permutation(ds)
            for b_start in range(0, ds - batch_size + 1, batch_size):
                idx = perm[b_start:b_start + batch_size]
                logits, values = self.model(s_t[idx])
                masked_logits = logits.clone()
                masked_logits[~mask_t[idx]] = -1e9
                probs = F.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lps = dist.log_prob(a_t[idx])
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_lps - old_lp_t[idx])
                ratios_list.extend(ratios.detach().cpu().numpy().tolist())
                surr1 = ratios * adv_t[idx]
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, ret_t[idx])

                # Clipping fraction tracking
                with torch.no_grad():
                    clipped = (ratios < (1.0 - self.clip_eps)) | (ratios > (1.0 + self.clip_eps))
                    clip_fractions.append(float(clipped.float().mean().item()))

                loss = actor_loss + 0.5 * critic_loss - self.beta_entropy * entropy

                self.optimizer.zero_grad()
                loss.backward()
                cur_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                total_grad_norm += float(cur_norm.item())
                steps += 1

        self.rollout_buffer.clear()
        self.gradient_update_count += 1
        
        res = {
            "actor_loss": total_actor_loss / max(steps, 1),
            "critic_loss": total_critic_loss / max(steps, 1),
            "entropy": total_entropy / max(steps, 1),
            "grad_norm": total_grad_norm / max(steps, 1),
            "clip_frac": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "mean_ratio": float(np.mean(ratios_list)) if ratios_list else 1.0
        }
        self.loss_history.append(res)
        return res

class MultiAgentPokerAlphaGoEngine:
    """5인 동시 자가대국 강화학습 관리 엔진"""
    def __init__(self, base_ckpt_path: str = "v1024/models_e2e/PPO_E2E_Final_50k.pt",
                 log_file: str = "v1024/alphago_5p_training.log", base_seed: int = 505000):
        self.game = get_5p_fullgame()
        self.log_file = log_file
        self.base_seed = base_seed
        self.agents = [
            IndependentPPOAgent(
                agent_id=i, name=AGENT_NAMES[i],
                base_ckpt_path=base_ckpt_path,
                lr=0.0003, seed=base_seed + i * 100
            ) for i in range(5)
        ]
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs("v1024/models_alphago", exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== DIRECTIVE 043: UNIVERSAL POKER ALPHAGO 5-AGENT SELF-PLAY LOG ===\n")

    def log(self, msg: str):
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{ts} {msg}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_selfplay_pilot(self, num_hands: int = 2000, checkpoint_interval: int = 500,
                           update_batch_hands: int = 50) -> Dict[str, any]:
        self.log(f"Starting Multi-Agent Self-Play Training for {num_hands} hands...")
        self.log(f"Participating Agents: {[a.name for a in self.agents]}")
        start_time = time.time()
        
        hands_data = []

        for h_idx in range(1, num_hands + 1):
            h_seed = self.base_seed + h_idx
            chance_rng = np.random.RandomState(h_seed)

            st = self.game.new_initial_state()
            advance_chance_nodes(st, chance_rng)

            # Seat rotation: Agent i sits at seat (i + h_idx) % 5
            # Ensures zero positional bias over time
            seat_to_agent = {s: (s - (h_idx % 5)) % 5 for s in range(5)}
            agent_to_seat = {a_id: s for s, a_id in seat_to_agent.items()}

            priv_cards = robust_parse_cards(st)
            trajectories = {a_id: [] for a_id in range(5)}
            folded_seats = set()

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, chance_rng)
                    continue

                cp = st.current_player()
                agent_id = seat_to_agent[cp]
                agent = self.agents[agent_id]

                legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                round_idx = get_round_from_state(st)

                # Information State Extraction & Strict Normalization
                raw_inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                norm_inf = normalize_inf_state_v79(raw_inf) # 100% Normalized State Used Everywhere

                act_6, log_prob, val, probs = agent.select_action(norm_inf, legal_6, round_idx)
                trajectories[agent_id].append((norm_inf, act_6, log_prob, val, legal_6))

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                if act_6 == 0:
                    folded_seats.add(cp)
                st.apply_action(nat_act)

            # Hand Finished: Settle returns
            returns_chips = st.returns()
            
            # Strict Zero-Sum Invariant Check (sum == 0.0)
            tot_return = sum(returns_chips)
            assert abs(tot_return) < 1e-4, f"Hand {h_idx}: Multi-agent zero-sum violated! sum={tot_return}"

            # Buffer trajectories for all 5 agents
            hand_agent_pnls = {}
            for a_id in range(5):
                seat = agent_to_seat[a_id]
                pnl_dollars = returns_chips[seat]
                pnl_bb = pnl_dollars / 100.0
                self.agents[a_id].total_pnl_bb += pnl_bb
                hand_agent_pnls[AGENT_NAMES[a_id]] = pnl_bb

                traj = trajectories[a_id]
                n_steps = len(traj)
                for i in range(n_steps):
                    s_t, a_t, lp_t, v_t, leg_t = traj[i]
                    R_t = pnl_bb * (0.99 ** (n_steps - 1 - i))
                    self.agents[a_id].buffer_transition(s_t, a_t, lp_t, R_t, v_t, leg_t)

            hands_data.append(hand_agent_pnls)

            # Synchronous Periodic Update across ALL 5 Agents
            if h_idx % update_batch_hands == 0:
                update_metrics = {}
                for a in self.agents:
                    m = a.update_policy(batch_size=64, epochs=4)
                    if m: update_metrics[a.name] = m
                
                # Throttle CPU <= 70%
                time.sleep(0.001)

            # Real-time Logging every 100 hands
            if h_idx % 100 == 0:
                self.log(f"--- Hand [{h_idx:,} / {num_hands:,}] Summary ---")
                for a in self.agents:
                    tot_acts = sum(a.action_counts.values())
                    pot_pct = (a.action_counts[3] / tot_acts * 100.0) if tot_acts > 0 else 0.0
                    half_pct = (a.action_counts[2] / tot_acts * 100.0) if tot_acts > 0 else 0.0
                    allin_pct = (a.action_counts[5] / tot_acts * 100.0) if tot_acts > 0 else 0.0
                    avg_ent = np.mean(a.entropy_history[-200:]) if a.entropy_history else 0.0
                    self.log(f"  {a.name} (Updates: {a.gradient_update_count}): PnL: {a.total_pnl_bb:+.1f} BB | "
                             f"Ent: {avg_ent:.3f} | Fold {a.action_counts[0]}, Call {a.action_counts[1]}, "
                             f"HalfPot {a.action_counts[2]} ({half_pct:.1f}%), Pot {a.action_counts[3]} ({pot_pct:.1f}%), "
                             f"Allin {a.action_counts[5]} ({allin_pct:.1f}%)")

            # Checkpoints every checkpoint_interval hands
            if h_idx % checkpoint_interval == 0:
                self.log(f"--> [MILESTONE {h_idx} Hands] Saving Independent Checkpoints for all 5 Agents...")
                for a in self.agents:
                    ckpt_path = f"v1024/models_alphago/{a.name}_{h_idx}h.pt"
                    torch.save(a.model.state_dict(), ckpt_path)
                self.log(f"--> Saved all 5 checkpoints at {h_idx} hands.")

        total_elapsed = time.time() - start_time
        self.log("=" * 70)
        self.log(f"DIRECTIVE 043: 5-PLAYER SELF-PLAY PILOT COMPLETED IN {total_elapsed:.1f}s")
        self.log("=" * 70)
        return {
            "num_hands": num_hands,
            "elapsed_sec": total_elapsed,
            "final_agent_pnls": {a.name: a.total_pnl_bb for a in self.agents},
            "gradient_updates": {a.name: a.gradient_update_count for a in self.agents}
        }
