"""
POKER AI V4.1 - DeepMind NFSP (Neural Fictitious Self-Play) PyTorch 구현
- Heinrich & Silver (2016) 논문 기반
- Best Response Learning (DQN with Target Network & Replay Buffer)
- Average Strategy Learning (Supervised Learning with Reservoir Buffer)
- Anticipatory Parameter eta 로 BR과 AVG 전략 스위칭
"""

import sys
import os
import random
import copy
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from open_spiel.python.utils.reservoir_buffer import ReservoirBuffer
from kuhn_engine_v3 import PASS, BET, ACTIONS, CARD_NAMES


class QNetwork(nn.Module):
    """Best Response를 위한 DQN Q-Network"""
    def __init__(self, state_dim: int = 7, num_actions: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AveragePolicyNetwork(nn.Module):
    """평균 정책을 위한 지도학습 분류기 네트워크"""
    def __init__(self, state_dim: int = 7, num_actions: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    def get_action_probs(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = self.forward(x).cpu().numpy()
            probs = probs / np.sum(probs)
            return probs


def encode_kuhn_state(card: int, history: str, device: torch.device) -> torch.Tensor:
    """7차원 원-핫 상태 벡터: 카드(3) + 히스토리(4)"""
    card_vec = [0.0, 0.0, 0.0]
    card_vec[card - 1] = 1.0

    hist_map = {"": 0, "p": 1, "b": 2, "pb": 3}
    hist_vec = [0.0, 0.0, 0.0, 0.0]
    hist_idx = hist_map.get(history, 0)
    hist_vec[hist_idx] = 1.0

    feat = card_vec + hist_vec
    return torch.tensor(feat, dtype=torch.float32, device=device)


class CircularReplayBuffer:
    """DQN 경험 재현 버퍼 (FIFO)"""
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class NFSPAgent:
    """
    Kuhn Poker를 위한 DeepMind NFSP 에이전트
    """
    def __init__(
        self,
        player_id: int,
        state_dim: int = 7,
        num_actions: int = 2,
        hidden_dim: int = 64,
        anticipatory_param: float = 0.1, # eta
        rl_learning_rate: float = 0.01,
        sl_learning_rate: float = 0.005,
        rl_buffer_capacity: int = 50000,
        sl_buffer_capacity: int = 200000,
        batch_size: int = 128,
        target_update_interval: int = 300,
        epsilon_start: float = 0.08,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 20000,
        device: Optional[torch.device] = None
    ):
        self.player_id = player_id
        self.num_actions = num_actions
        self.eta = anticipatory_param
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device or torch.device("cpu")

        # 1. Best Response Q-Network & Optimizer
        self.q_net = QNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.rl_optimizer = optim.Adam(self.q_net.parameters(), lr=rl_learning_rate)
        self.rl_buffer = CircularReplayBuffer(rl_buffer_capacity)

        # 2. Average Policy Network & Optimizer
        self.avg_policy_net = AveragePolicyNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.sl_optimizer = optim.Adam(self.avg_policy_net.parameters(), lr=sl_learning_rate)
        self.sl_buffer = ReservoirBuffer(sl_buffer_capacity)

        # 3. Epsilon Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        # 에피소드 모드: 'best_response' or 'average_policy'
        self.current_mode = 'best_response'

    def sample_episode_mode(self):
        """매 에피소드 시작 시 eta 확률로 BR, 1-eta 확률로 AVG 모드 선택"""
        if random.random() < self.eta:
            self.current_mode = 'best_response'
        else:
            self.current_mode = 'average_policy'

    def get_epsilon(self) -> float:
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = self.total_steps / self.epsilon_decay_steps
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, card: int, history: str, is_evaluation: bool = False) -> int:
        state_tensor = encode_kuhn_state(card, history, self.device)

        if is_evaluation or self.current_mode == 'average_policy':
            # 평균 정책 네트워크 기반 행동
            probs = self.avg_policy_net.get_action_probs(state_tensor)
            return int(np.random.choice(ACTIONS, p=probs))
        else:
            # Best Response (DQN epsilon-greedy)
            eps = 0.0 if is_evaluation else self.get_epsilon()
            if random.random() < eps:
                return random.choice(ACTIONS)
            with torch.no_grad():
                q_vals = self.q_net(state_tensor).cpu().numpy()
                return int(np.argmax(q_vals))

    def train_step(self):
        """RL (DQN) 및 SL (Average Policy) 1단계 역전파 학습"""
        self.total_steps += 1

        # A. Train RL (DQN)
        rl_loss_val = 0.0
        if len(self.rl_buffer) >= self.batch_size:
            batch = self.rl_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            b_states = torch.stack(states).to(self.device)
            b_actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            b_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            b_next_states = torch.stack(next_states).to(self.device)
            b_dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            curr_q = self.q_net(b_states).gather(1, b_actions).squeeze(1)
            with torch.no_grad():
                next_q = self.target_q_net(b_next_states).max(1)[0]
                # Kuhn Poker는 감쇠 계수 gamma = 1.0 (종단 보상)
                target_q = b_rewards + (1.0 - b_dones) * next_q

            rl_loss = nn.MSELoss()(curr_q, target_q)
            self.rl_optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.rl_optimizer.step()
            rl_loss_val = rl_loss.item()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        # B. Train SL (Average Policy - Reservoir Buffer Supervised Learning)
        sl_loss_val = 0.0
        if len(self.sl_buffer) >= self.batch_size:
            sl_samples = self.sl_buffer.sample(self.batch_size)
            sl_states, sl_actions = zip(*sl_samples)

            b_sl_states = torch.stack(sl_states).to(self.device)
            b_sl_actions = torch.tensor(sl_actions, dtype=torch.int64, device=self.device)

            logits = self.avg_policy_net.net(b_sl_states)
            sl_loss = nn.CrossEntropyLoss()(logits, b_sl_actions)

            self.sl_optimizer.zero_grad()
            sl_loss.backward()
            nn.utils.clip_grad_norm_(self.avg_policy_net.parameters(), 1.0)
            self.sl_optimizer.step()
            sl_loss_val = sl_loss.item()

        return rl_loss_val, sl_loss_val

    def get_average_policy_dict(self) -> Dict[str, Tuple[float, float]]:
        """12개 정보집합별 평균 정책 딕셔너리 추출"""
        policy = {}
        for c_val, c_name in [(1, 'J'), (2, 'Q'), (3, 'K')]:
            for hist in ["", "pb", "p", "b"]:
                st = encode_kuhn_state(c_val, hist, self.device)
                probs = self.avg_policy_net.get_action_probs(st)
                policy[f"{c_name}:{hist}"] = (float(probs[0]), float(probs[1]))
        return policy

    def get_best_response_policy_dict(self) -> Dict[str, Tuple[float, float]]:
        """12개 정보집합별 최신 Best Response Q-Network 소프트맥스 정책 추출"""
        policy = {}
        with torch.no_grad():
            for c_val, c_name in [(1, 'J'), (2, 'Q'), (3, 'K')]:
                for hist in ["", "pb", "p", "b"]:
                    st = encode_kuhn_state(c_val, hist, self.device)
                    q = self.q_net(st)
                    probs = torch.softmax(q / 0.2, dim=-1).cpu().numpy()
                    policy[f"{c_name}:{hist}"] = (float(probs[0]), float(probs[1]))
        return policy
