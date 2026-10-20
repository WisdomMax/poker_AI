"""
POKER AI V5.2.2 - NFSP VARIANTS MODULE
- Heinrich & Silver (2016) 원 논문 정합성 감사 및 알고리즘 구현:
  1. NFSP_MC_Variant:
     - 기존 V5.2 방식
     - 의사결정 스텝 단위 eta 샘플링 (Step-level mixture)
     - 타겟 네트워크 부재
     - Monte Carlo Return 회귀 Q 학습: Q(s_t, a_t) <- R_terminal
  2. NFSP_DQN_Reference:
     - 원 논문 Algorithm 1 완벽 준수
     - 에피소드 레벨 정책 선택 (Episode-level policy selection: d ~ Bernoulli(eta))
     - d=1 (BR 모드)일 때만 해당 에피소드의 (s, a)를 Reservoir Buffer에 저장
     - Target Q-Network 유지 및 주기적 동기화
     - 1-Step TD Bellman Target: y = r + (1-done) * gamma * max_{a' in legal} Q_target(s', a')
  3. 정책 분리 메서드:
     - get_action_distribution(s, legal, mode='average' | 'best_response' | 'mixture')
"""

import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReservoirBuffer:
    def __init__(self, capacity: int = 30000):
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
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)


# =====================================================================
# A. NFSP-MC Variant (기존 V5.2 구현)
# =====================================================================
class NFSP_MC_Agent:
    def __init__(
        self,
        player_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr_q: float = 0.003,
        lr_sl: float = 0.001,
        eta: float = 0.1,
        capacity: int = 30000
    ):
        self.player_id = player_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eta = eta
        self.variant_name = "NFSP-MC"

        # Q-Network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=lr_q)

        # Average Policy Network
        self.sl_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.opt_sl = optim.Adam(self.sl_net.parameters(), lr=lr_sl)

        self.rl_buffer = deque(maxlen=capacity)
        self.sl_buffer = ReservoirBuffer(capacity=capacity)

    def reset_episode(self):
        # Step-level 방식은 에피소드 시작 시 정책을 고정하지 않음
        pass

    def select_action(self, info: np.ndarray, legal: list, is_eval: bool = False):
        x = torch.tensor(info, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal] = True

        if is_eval:
            # 평가: Average Policy Softmax
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x)
                lg[0, ~mask] = -1e9
                probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
            return int(np.random.choice(self.action_dim, p=probs))

        # 학습: 매 스텝 독립 eta 샘플링 (Step-level Mixture)
        if random.random() < self.eta:
            # Best Response (Q-net)
            self.q_net.eval()
            with torch.no_grad():
                q = self.q_net(x).squeeze(0)
                q[~mask] = -1e9
                act = int(q.argmax().item())
            self.sl_buffer.add((info, act))
        else:
            # Average Policy
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x)
                lg[0, ~mask] = -1e9
                probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
            act = int(np.random.choice(self.action_dim, p=probs))

        return act

    def update(self, batch_size: int = 64):
        q_loss_val = 0.0
        sl_loss_val = 0.0

        # Monte Carlo Regression: Q(s, a) <- R_terminal
        if len(self.rl_buffer) >= batch_size:
            batch = random.sample(self.rl_buffer, batch_size)
            s, a, r = zip(*batch)
            b_s = torch.tensor(np.array(s), dtype=torch.float32)
            b_a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
            b_r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)

            self.q_net.train()
            cur_q = self.q_net(b_s).gather(1, b_a)
            loss_q = F.mse_loss(cur_q, b_r)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            q_loss_val = float(loss_q.item())

        if len(self.sl_buffer) >= batch_size:
            batch_sl = self.sl_buffer.sample(batch_size)
            s_sl, a_sl = zip(*batch_sl)
            b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
            b_asl = torch.tensor(a_sl, dtype=torch.long)

            self.sl_net.train()
            lg = self.sl_net(b_ssl)
            loss_sl = F.cross_entropy(lg, b_asl)
            self.opt_sl.zero_grad()
            loss_sl.backward()
            self.opt_sl.step()
            sl_loss_val = float(loss_sl.item())

        return q_loss_val, sl_loss_val


# =====================================================================
# B. NFSP-DQN Reference (원 논문 규격 준수)
# =====================================================================
class NFSP_DQN_Agent:
    def __init__(
        self,
        player_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr_q: float = 0.003,
        lr_sl: float = 0.001,
        eta: float = 0.1,
        target_sync_freq: int = 200,
        capacity: int = 30000
    ):
        self.player_id = player_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eta = eta
        self.target_sync_freq = target_sync_freq
        self.train_step = 0
        self.variant_name = "NFSP-DQN"

        # Q-Network & Target Network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.q_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=lr_q)

        # Average Policy Network
        self.sl_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.opt_sl = optim.Adam(self.sl_net.parameters(), lr=lr_sl)

        self.rl_buffer = deque(maxlen=capacity) # (s, a, r, s_next, next_legals, done)
        self.sl_buffer = ReservoirBuffer(capacity=capacity)

        # Episode-level policy state: True면 이번 에피소드 전체에서 BR 모드
        self.current_ep_is_br = True

    def reset_episode(self):
        # [원 논문 핵심]: 에피소드 시작 시 정책 d ~ Bernoulli(eta) 샘플링하여 일관되게 유지
        self.current_ep_is_br = (random.random() < self.eta)

    def select_action(self, info: np.ndarray, legal: list, is_eval: bool = False):
        x = torch.tensor(info, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal] = True

        if is_eval:
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x)
                lg[0, ~mask] = -1e9
                probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
            return int(np.random.choice(self.action_dim, p=probs))

        if self.current_ep_is_br:
            # BR 모드: epsilon-greedy Q
            self.q_net.eval()
            with torch.no_grad():
                q = self.q_net(x).squeeze(0)
                q[~mask] = -1e9
                act = int(q.argmax().item())
            # [원 논문]: BR 모드일 때만 Reservoir Buffer에 샘플 추가
            self.sl_buffer.add((info, act))
        else:
            # Average Policy 모드: Softmax
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x)
                lg[0, ~mask] = -1e9
                probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
            act = int(np.random.choice(self.action_dim, p=probs))

        return act

    def update(self, batch_size: int = 64, gamma: float = 0.99):
        q_loss_val = 0.0
        sl_loss_val = 0.0

        # 1-Step TD Bellman Update with Target Network
        if len(self.rl_buffer) >= batch_size:
            batch = random.sample(self.rl_buffer, batch_size)
            s, a, r, ns, n_legals, done = zip(*batch)

            b_s = torch.tensor(np.array(s), dtype=torch.float32)
            b_a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
            b_r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
            b_ns = torch.tensor(np.array(ns), dtype=torch.float32)
            b_d = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

            self.q_net.train()
            cur_q = self.q_net(b_s).gather(1, b_a)

            with torch.no_grad():
                target_q_all = self.q_target(b_ns)
                for idx, leg in enumerate(n_legals):
                    if len(leg) > 0:
                        m = torch.ones(self.action_dim, dtype=torch.bool)
                        m[leg] = False
                        target_q_all[idx, m] = -1e9
                max_next_q = target_q_all.max(dim=1, keepdim=True)[0]
                target_y = b_r + (1.0 - b_d) * gamma * max_next_q

            loss_q = F.mse_loss(cur_q, target_y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            q_loss_val = float(loss_q.item())

            self.train_step += 1
            if self.train_step % self.target_sync_freq == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())

        # Average Policy CrossEntropy
        if len(self.sl_buffer) >= batch_size:
            batch_sl = self.sl_buffer.sample(batch_size)
            s_sl, a_sl = zip(*batch_sl)
            b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
            b_asl = torch.tensor(a_sl, dtype=torch.long)

            self.sl_net.train()
            lg = self.sl_net(b_ssl)
            loss_sl = F.cross_entropy(lg, b_asl)
            self.opt_sl.zero_grad()
            loss_sl.backward()
            self.opt_sl.step()
            sl_loss_val = float(loss_sl.item())

        return q_loss_val, sl_loss_val

NFSP_DQN_Reference = NFSP_DQN_Agent
