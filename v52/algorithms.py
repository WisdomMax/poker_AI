"""
POKER AI V5.2 - PHASE 1: ALGORITHMS IMPLEMENTATION & SPECIFICATION
- 3대 자기 대국(Self-Play) 알고리즘 통합 인터페이스:
  1. Model A: PPO Self-Play (PyTorch On-Policy PPO)
  2. Model B: External Sampling MCCFR (C++ pyspiel.ExternalSamplingMCCFRSolver)
  3. Model C: NFSP (Neural Fictitious Self-Play: DQN Best Response + SL Average Policy)
- 불완전정보 원칙 100% 준수 (상대 비공개 카드/미래 정보 누출 0%)
"""

import os
import sys
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyspiel
from open_spiel.python import policy

# =====================================================================
# 1. MODEL A: PPO Self-Play Policy & Agent
# =====================================================================
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Actor head orthogonal init
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        raw_logits = self.actor(x)
        masked_logits = raw_logits.clone()
        masked_logits[~mask] = -1e9
        probs = F.softmax(masked_logits, dim=-1)
        val = self.critic(x)
        return raw_logits, masked_logits, probs, val


class PPOSelfPlayPolicy(policy.Policy):
    """OpenSpiel Exploitability 계산을 위한 PPO Policy Wrapper"""
    def __init__(self, game: pyspiel.Game, model: PPOActorCritic, device: torch.device):
        super().__init__(game, list(range(game.num_players())))
        self.model = model
        self.device = device

    def action_probabilities(self, state, player_id=None):
        legal = state.legal_actions(state.current_player())
        if not legal:
            return {}
        info_vec = np.array(state.information_state_tensor(state.current_player()), dtype=np.float32)
        x = torch.tensor(info_vec, device=self.device).unsqueeze(0)
        mask = torch.zeros(state.num_distinct_actions(), dtype=torch.bool, device=self.device)
        mask[legal] = True

        self.model.eval()
        with torch.no_grad():
            _, _, probs, _ = self.model(x, mask.unsqueeze(0))
            p_arr = probs.squeeze(0).cpu().numpy()

        return {a: float(p_arr[a]) for a in legal}


# =====================================================================
# 2. MODEL C: NFSP (Neural Fictitious Self-Play) PyTorch Implementation
# =====================================================================
class ReservoirBuffer:
    """Average Strategy 모방 학습을 위한 저수지 샘플링(Reservoir Sampling) 버퍼"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.total_seen = 0

    def add(self, item):
        self.total_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.data[idx] = item

    def sample(self, batch_size: int):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)


class NFSPAgent:
    """NFSP 에이전트: DQN Best-Response + Supervised Average Strategy"""
    def __init__(
        self,
        player_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr_rl: float = 0.01,
        lr_sl: float = 0.005,
        eta: float = 0.1, # Best Response 실행 확률
        epsilon_start: float = 0.2,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        device: torch.device = torch.device("cpu")
    ):
        self.player_id = player_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eta = eta
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.device = device

        # 1. Best Response Q-Network (RL)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.q_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=lr_rl)

        # 2. Average Policy Network (SL)
        self.sl_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.opt_sl = optim.Adam(self.sl_net.parameters(), lr=lr_sl)

        # 버퍼
        self.rl_buffer = deque(maxlen=20000)
        self.sl_buffer = ReservoirBuffer(capacity=20000)

        self.sample_mode = "best_response" # 'best_response' or 'average_policy'

    def select_action(self, state, legal_actions: list, is_evaluation: bool = False):
        info_vec = np.array(state.information_state_tensor(self.player_id), dtype=np.float32)
        x = torch.tensor(info_vec, device=self.device).unsqueeze(0)

        if is_evaluation:
            # 평가 시에는 학습된 평균 전략(Average Policy)의 Softmax 사용
            self.sl_net.eval()
            with torch.no_grad():
                logits = self.sl_net(x)
                mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                mask[legal_actions] = True
                logits[0, ~mask] = -1e9
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            return int(np.random.choice(self.action_dim, p=probs))

        # 학습 시: 확률 eta로 Best Response, 1-eta로 Average Policy
        if random.random() < self.eta:
            self.sample_mode = "best_response"
            # epsilon-greedy Q
            if random.random() < self.epsilon:
                act = random.choice(legal_actions)
            else:
                self.q_net.eval()
                with torch.no_grad():
                    q_vals = self.q_net(x).squeeze(0)
                    mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                    mask[legal_actions] = True
                    q_vals[~mask] = -1e9
                    act = int(q_vals.argmax().item())

            # Best Response 행동은 SL 버퍼에 저장 대상
            self.sl_buffer.add((info_vec, act, legal_actions))
        else:
            self.sample_mode = "average_policy"
            self.sl_net.eval()
            with torch.no_grad():
                logits = self.sl_net(x)
                mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                mask[legal_actions] = True
                logits[0, ~mask] = -1e9
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            act = int(np.random.choice(self.action_dim, p=probs))

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return act

    def update(self, batch_size: int = 64, gamma: float = 0.99):
        # 1. Update RL (DQN)
        if len(self.rl_buffer) >= batch_size:
            batch = random.sample(self.rl_buffer, batch_size)
            states, acts, rews, next_states, next_legals, dones = zip(*batch)

            b_s = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            b_a = torch.tensor(acts, dtype=torch.long, device=self.device).unsqueeze(1)
            b_r = torch.tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(1)
            b_ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            b_d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            self.q_net.train()
            cur_q = self.q_net(b_s).gather(1, b_a)

            with torch.no_grad():
                next_q_all = self.q_target(b_ns)
                # Next legal mask
                for idx, leg in enumerate(next_legals):
                    if len(leg) > 0:
                        mask = torch.ones(self.action_dim, dtype=torch.bool)
                        mask[leg] = False
                        next_q_all[idx, mask] = -1e9
                max_next_q = next_q_all.max(dim=1, keepdim=True)[0]
                target_q = b_r + (1.0 - b_d) * gamma * max_next_q

            loss_q = F.mse_loss(cur_q, target_q)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()

        # 2. Update SL (Average Policy)
        if len(self.sl_buffer) >= batch_size:
            batch_sl = self.sl_buffer.sample(batch_size)
            states_sl, acts_sl, _ = zip(*batch_sl)
            b_ssl = torch.tensor(np.array(states_sl), dtype=torch.float32, device=self.device)
            b_asl = torch.tensor(acts_sl, dtype=torch.long, device=self.device)

            self.sl_net.train()
            logits_sl = self.sl_net(b_ssl)
            loss_sl = F.cross_entropy(logits_sl, b_asl)

            self.opt_sl.zero_grad()
            loss_sl.backward()
            self.opt_sl.step()


class NFSPAveragePolicy(policy.Policy):
    """OpenSpiel Exploitability 계산을 위한 NFSP Average Policy Wrapper"""
    def __init__(self, game: pyspiel.Game, agents: List[NFSPAgent]):
        super().__init__(game, list(range(game.num_players())))
        self.agents = agents

    def action_probabilities(self, state, player_id=None):
        legal = state.legal_actions(state.current_player())
        if not legal:
            return {}
        p_id = state.current_player()
        ag = self.agents[p_id]
        info_vec = np.array(state.information_state_tensor(p_id), dtype=np.float32)
        x = torch.tensor(info_vec, device=ag.device).unsqueeze(0)

        ag.sl_net.eval()
        with torch.no_grad():
            logits = ag.sl_net(x)
            mask = torch.zeros(ag.action_dim, dtype=torch.bool, device=ag.device)
            mask[legal] = True
            logits[0, ~mask] = -1e9
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {a: float(probs[a]) for a in legal}
