"""
POKER AI V4.3 - PHASE 5: OpenSpiel Leduc Poker 환경 구축 및 아키텍처 확장
- 6장 카드 (J, Q, K 각 2장), 2라운드 베팅, 1장 공개 커뮤니티 카드
- 30차원 정보집합 벡터, 3개 액션 (0: Fold, 1: Call/Check, 2: Raise)
- Legal Action Masking 내장 PyTorch 신경망
- OpenSpiel 정책 평가 호환 어댑터
"""

import sys
import os
import copy
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python import policy
from open_spiel.python.utils.reservoir_buffer import ReservoirBuffer


# =====================================================================
# 1. Leduc 신경망 아키텍처 (Legal Action Masking 지원)
# =====================================================================

class LeducQNetwork(nn.Module):
    def __init__(self, state_dim: int = 30, num_actions: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_vals = self.net(x)
        if legal_mask is not None:
            q_vals = q_vals.masked_fill(legal_mask == 0, -1e9)
        return q_vals


class LeducAveragePolicyNet(nn.Module):
    def __init__(self, state_dim: int = 30, num_actions: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(x)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        return logits

    def get_action_probs(self, x: torch.Tensor, legal_actions: List[int]) -> np.ndarray:
        with torch.no_grad():
            mask = torch.zeros(3, device=x.device)
            mask[legal_actions] = 1.0
            logits = self.forward(x.unsqueeze(0), mask.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            # 정규화
            sub_sum = sum(probs[a] for a in legal_actions)
            final_p = np.zeros(3, dtype=np.float32)
            if sub_sum > 0:
                for a in legal_actions:
                    final_p[a] = probs[a] / sub_sum
            else:
                for a in legal_actions:
                    final_p[a] = 1.0 / len(legal_actions)
            return final_p


class LeducCircularReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, legal_mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, legal_mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# =====================================================================
# 2. Leduc NFSP 에이전트 클래스
# =====================================================================

class LeducNFSPAgent:
    def __init__(
        self,
        player_id: int,
        state_dim: int = 30,
        num_actions: int = 3,
        hidden_dim: int = 128,
        anticipatory_param: float = 0.1,
        rl_lr: float = 0.005,
        sl_lr: float = 0.002,
        rl_capacity: int = 100000,
        sl_capacity: int = 200000,
        batch_size: int = 128,
        target_update: int = 200,
        eps_start: float = 0.08,
        eps_end: float = 0.005,
        eps_decay: int = 100000,
        device: Optional[torch.device] = None
    ):
        self.player_id = player_id
        self.num_actions = num_actions
        self.eta = anticipatory_param
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or torch.device("cpu")

        # 1. Best Response Q-Network
        self.q_net = LeducQNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.rl_optimizer = optim.Adam(self.q_net.parameters(), lr=rl_lr)
        self.rl_buffer = LeducCircularReplayBuffer(rl_capacity)

        # 2. Average Policy Network
        self.avg_net = LeducAveragePolicyNet(state_dim, num_actions, hidden_dim).to(self.device)
        self.sl_optimizer = optim.Adam(self.avg_net.parameters(), lr=sl_lr)
        self.sl_buffer = ReservoirBuffer(sl_capacity)

        # 3. Exploration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.total_steps = 0
        self.current_mode = 'best_response'

    def sample_episode_mode(self):
        self.current_mode = 'best_response' if random.random() < self.eta else 'average_policy'

    def get_epsilon(self) -> float:
        if self.total_steps >= self.eps_decay: return self.eps_end
        return self.eps_start + (self.total_steps / self.eps_decay) * (self.eps_end - self.eps_start)

    def act(self, info_state_vec: List[float], legal_actions: List[int], is_evaluation: bool = False) -> Tuple[int, np.ndarray]:
        st = torch.tensor(info_state_vec, dtype=torch.float32, device=self.device)

        if is_evaluation or self.current_mode == 'average_policy':
            probs = self.avg_net.get_action_probs(st, legal_actions)
            p_arr = np.array([probs[a] for a in ACTIONS_3], dtype=np.float64)
            p_arr /= np.sum(p_arr)
            action = int(np.random.choice(ACTIONS_3, p=p_arr))
            return action, probs
        else:
            # Best Response (DQN epsilon-greedy)
            eps = 0.0 if is_evaluation else self.get_epsilon()
            mask = torch.zeros(self.num_actions, device=self.device)
            mask[legal_actions] = 1.0

            with torch.no_grad():
                q = self.q_net(st.unsqueeze(0), mask.unsqueeze(0)).squeeze(0).cpu().numpy()

            best_act = max(legal_actions, key=lambda a: q[a])
            probs = np.zeros(self.num_actions, dtype=np.float32)
            for a in legal_actions:
                probs[a] = eps / len(legal_actions)
            probs[best_act] += (1.0 - eps)

            if random.random() < eps:
                action = random.choice(legal_actions)
            else:
                action = best_act
            return action, probs

    def train_step(self):
        self.total_steps += 1
        rl_loss_v, sl_loss_v = 0.0, 0.0

        # A. RL Update
        if len(self.rl_buffer) >= self.batch_size:
            batch = self.rl_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, masks = zip(*batch)

            b_st = torch.stack(states).to(self.device)
            b_ac = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            b_rw = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            b_nst = torch.stack(next_states).to(self.device)
            b_dn = torch.tensor(dones, dtype=torch.float32, device=self.device)
            b_mk = torch.stack(masks).to(self.device)

            curr_q = self.q_net(b_st).gather(1, b_ac).squeeze(1)
            with torch.no_grad():
                # Target Q with next state legal action masking
                next_q_all = self.target_q_net(b_nst, b_mk)
                max_next_q = next_q_all.max(1)[0]
                target_q = b_rw + (1.0 - b_dn) * max_next_q

            rl_loss = nn.MSELoss()(curr_q, target_q)
            self.rl_optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.rl_optimizer.step()
            rl_loss_v = rl_loss.item()

            if self.total_steps % self.target_update == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        # B. SL Update
        if len(self.sl_buffer) >= self.batch_size:
            sl_samples = self.sl_buffer.sample(self.batch_size)
            sl_st, sl_pr, sl_mk = zip(*sl_samples)

            b_st = torch.stack(sl_st).to(self.device)
            b_pr = torch.tensor(np.array(sl_pr), dtype=torch.float32, device=self.device)
            b_mk = torch.stack(sl_mk).to(self.device)

            logits = self.avg_net(b_st, b_mk)
            log_p = torch.log_softmax(logits, dim=-1)
            sl_loss = -torch.mean(torch.sum(b_pr * log_p, dim=-1))

            self.sl_optimizer.zero_grad()
            sl_loss.backward()
            nn.utils.clip_grad_norm_(self.avg_net.parameters(), 1.0)
            self.sl_optimizer.step()
            sl_loss_v = sl_loss.item()

        return rl_loss_v, sl_loss_v


ACTIONS_3 = [0, 1, 2] # 0=Fold, 1=Call/Check, 2=Raise


# =====================================================================
# 3. OpenSpiel 평가 인터페이스 어댑터
# =====================================================================

class LeducNFSPPolicyWrapper(policy.Policy):
    """DeepMind OpenSpiel Exploitability 측정을 위한 Policy 객체 어댑터"""
    def __init__(self, game, agent0: LeducNFSPAgent, agent1: LeducNFSPAgent):
        super().__init__(game, [0, 1])
        self.agent0 = agent0
        self.agent1 = agent1

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        info_vec = state.information_state_tensor(player_id)
        legal_acts = state.legal_actions(player_id)
        agent = self.agent0 if player_id == 0 else self.agent1
        st = torch.tensor(info_vec, dtype=torch.float32, device=agent.device)
        probs = agent.avg_net.get_action_probs(st, legal_acts)
        tot = sum(probs[a] for a in legal_acts)
        if tot > 0:
            return {a: float(probs[a] / tot) for a in legal_acts}
        else:
            return {a: float(1.0 / len(legal_acts)) for a in legal_acts}
