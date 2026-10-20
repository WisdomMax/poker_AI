"""
POKER AI V4.2 - PHASE 2: Optimized NFSP 구현 및 8대 하이퍼파라미터 Ablation Study
- DeepMind 공식 규격 일치:
  1. action_probs 분포를 Reservoir Buffer에 저장하여 Cross-Entropy 학습
  2. Average Policy 모드 플레이 시에도 DQN Replay 경험 수징
  3. Huber Loss (SmoothL1Loss) 적용
  4. 점진적 Epsilon 감쇠 스케줄링
"""

import sys
import os
import random
import copy
import time
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.utils.reservoir_buffer import ReservoirBuffer
from v4.openspiel_adapter import DictPolicyWrapper
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from v4.v41_nfsp import QNetwork, AveragePolicyNetwork, CircularReplayBuffer, encode_kuhn_state


class OptimizedNFSPAgent:
    """
    OpenSpiel 공식 규격과 100% 일치하도록 개선된 PyTorch NFSP 에이전트
    """
    def __init__(
        self,
        player_id: int,
        state_dim: int = 7,
        num_actions: int = 2,
        hidden_dim: int = 64,
        anticipatory_param: float = 0.1,
        rl_learning_rate: float = 0.01,
        sl_learning_rate: float = 0.005,
        rl_buffer_capacity: int = 50000,
        sl_buffer_capacity: int = 200000,
        batch_size: int = 128,
        target_update_interval: int = 300,
        epsilon_start: float = 0.08,
        epsilon_end: float = 0.005,
        epsilon_decay_steps: int = 100000,
        use_action_probs_sl: bool = True,
        use_huber_loss: bool = True,
        device: Optional[torch.device] = None
    ):
        self.player_id = player_id
        self.num_actions = num_actions
        self.eta = anticipatory_param
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.use_action_probs_sl = use_action_probs_sl
        self.use_huber_loss = use_huber_loss
        self.device = device or torch.device("cpu")

        # 1. Best Response Q-Network & Target Network
        self.q_net = QNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.rl_optimizer = optim.Adam(self.q_net.parameters(), lr=rl_learning_rate)
        self.rl_buffer = CircularReplayBuffer(rl_buffer_capacity)

        # 2. Average Policy Network
        self.avg_policy_net = AveragePolicyNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.sl_optimizer = optim.Adam(self.avg_policy_net.parameters(), lr=sl_learning_rate)
        self.sl_buffer = ReservoirBuffer(sl_buffer_capacity)

        # 3. Epsilon Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0
        self.current_mode = 'best_response'

    def sample_episode_mode(self):
        self.current_mode = 'best_response' if random.random() < self.eta else 'average_policy'

    def get_epsilon(self) -> float:
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = self.total_steps / self.epsilon_decay_steps
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, card: int, history: str, is_evaluation: bool = False) -> Tuple[int, np.ndarray]:
        state_tensor = encode_kuhn_state(card, history, self.device)

        if is_evaluation or self.current_mode == 'average_policy':
            probs = self.avg_policy_net.get_action_probs(state_tensor)
            action = int(np.random.choice(ACTIONS, p=probs))
            return action, probs
        else:
            # Best Response (DQN epsilon-greedy)
            eps = 0.0 if is_evaluation else self.get_epsilon()
            with torch.no_grad():
                q_vals = self.q_net(state_tensor).cpu().numpy()
            best_act = int(np.argmax(q_vals))

            # Epsilon-greedy action probabilities
            probs = np.full(self.num_actions, eps / self.num_actions, dtype=np.float32)
            probs[best_act] += (1.0 - eps)

            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = best_act
            return action, probs

    def train_step(self):
        self.total_steps += 1
        rl_loss_val, sl_loss_val = 0.0, 0.0

        # A. RL (DQN) 업데이트
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
                target_q = b_rewards + (1.0 - b_dones) * next_q

            loss_fn = nn.SmoothL1Loss() if self.use_huber_loss else nn.MSELoss()
            rl_loss = loss_fn(curr_q, target_q)

            self.rl_optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.rl_optimizer.step()
            rl_loss_val = rl_loss.item()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        # B. SL (Average Policy) 업데이트
        if len(self.sl_buffer) >= self.batch_size:
            sl_samples = self.sl_buffer.sample(self.batch_size)
            sl_states, sl_targets = zip(*sl_samples)
            b_sl_states = torch.stack(sl_states).to(self.device)

            logits = self.avg_policy_net.net(b_sl_states)

            if self.use_action_probs_sl:
                # OpenSpiel 공식: Cross Entropy over action probability distribution
                b_probs = torch.tensor(np.array(sl_targets), dtype=torch.float32, device=self.device)
                log_probs = torch.log_softmax(logits, dim=-1)
                sl_loss = -torch.mean(torch.sum(b_probs * log_probs, dim=-1))
            else:
                b_actions = torch.tensor(sl_targets, dtype=torch.int64, device=self.device)
                sl_loss = nn.CrossEntropyLoss()(logits, b_actions)

            self.sl_optimizer.zero_grad()
            sl_loss.backward()
            nn.utils.clip_grad_norm_(self.avg_policy_net.parameters(), 1.0)
            self.sl_optimizer.step()
            sl_loss_val = sl_loss.item()

        return rl_loss_val, sl_loss_val

    def get_average_policy_dict(self) -> Dict[str, Tuple[float, float]]:
        policy = {}
        for c_val, c_name in [(1, 'J'), (2, 'Q'), (3, 'K')]:
            for hist in ["", "pb", "p", "b"]:
                st = encode_kuhn_state(c_val, hist, self.device)
                probs = self.avg_policy_net.get_action_probs(st)
                policy[f"{c_name}:{hist}"] = (float(probs[0]), float(probs[1]))
        return policy


def train_single_experiment(
    episodes: int = 25000,
    eta: float = 0.1,
    rl_lr: float = 0.01,
    sl_lr: float = 0.005,
    hidden_dim: int = 64,
    batch_size: int = 128,
    target_update: int = 300,
    eps_decay: int = 50000,
    use_probs: bool = True,
    use_huber: bool = True,
    seed: int = 42
) -> float:
    """단일 조건 학습 후 최종 NashConv 반환"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = pyspiel.load_game("kuhn_poker")
    device = torch.device("cpu")

    a0 = OptimizedNFSPAgent(0, hidden_dim=hidden_dim, anticipatory_param=eta, rl_learning_rate=rl_lr,
                            sl_learning_rate=sl_lr, batch_size=batch_size, target_update_interval=target_update,
                            epsilon_decay_steps=eps_decay, use_action_probs_sl=use_probs, use_huber_loss=use_huber, device=device)
    a1 = OptimizedNFSPAgent(1, hidden_dim=hidden_dim, anticipatory_param=eta, rl_learning_rate=rl_lr,
                            sl_learning_rate=sl_lr, batch_size=batch_size, target_update_interval=target_update,
                            epsilon_decay_steps=eps_decay, use_action_probs_sl=use_probs, use_huber_loss=use_huber, device=device)

    deck = [1, 2, 3]
    for ep in range(episodes):
        a0.sample_episode_mode()
        a1.sample_episode_mode()

        cards = np.random.permutation(deck)
        c0, c1 = cards[0], cards[1]
        traj0, traj1 = [], []

        # P0 turn
        s0_root = encode_kuhn_state(c0, "", device)
        act0, probs0 = a0.act(c0, "")
        traj0.append((s0_root, act0))
        if a0.current_mode == 'best_response':
            target_data = probs0 if use_probs else act0
            a0.sl_buffer.add((s0_root.cpu(), target_data))

        if act0 == PASS:
            # P1 turn
            s1_p = encode_kuhn_state(c1, "p", device)
            act1, probs1 = a1.act(c1, "p")
            traj1.append((s1_p, act1))
            if a1.current_mode == 'best_response':
                target_data = probs1 if use_probs else act1
                a1.sl_buffer.add((s1_p.cpu(), target_data))

            if act1 == PASS:
                fin_h = "pp"
            else:
                s0_pb = encode_kuhn_state(c0, "pb", device)
                act0_2, probs0_2 = a0.act(c0, "pb")
                traj0.append((s0_pb, act0_2))
                if a0.current_mode == 'best_response':
                    target_data = probs0_2 if use_probs else act0_2
                    a0.sl_buffer.add((s0_pb.cpu(), target_data))
                fin_h = "pbp" if act0_2 == PASS else "pbb"
        else:
            s1_b = encode_kuhn_state(c1, "b", device)
            act1, probs1 = a1.act(c1, "b")
            traj1.append((s1_b, act1))
            if a1.current_mode == 'best_response':
                target_data = probs1 if use_probs else act1
                a1.sl_buffer.add((s1_b.cpu(), target_data))
            fin_h = "bp" if act1 == PASS else "bb"

        pay0, pay1 = evaluate_payoff(fin_h, c0, c1)

        # RL 버퍼 저장 (모든 모드에서 상대 대응 학습)
        if len(traj0) == 1:
            st, ac = traj0[0]
            a0.rl_buffer.push(st.cpu(), ac, pay0, st.cpu(), True)
        elif len(traj0) == 2:
            st1, ac1 = traj0[0]; st2, ac2 = traj0[1]
            a0.rl_buffer.push(st1.cpu(), ac1, 0.0, st2.cpu(), False)
            a0.rl_buffer.push(st2.cpu(), ac2, pay0, st2.cpu(), True)

        if len(traj1) == 1:
            st, ac = traj1[0]
            a1.rl_buffer.push(st.cpu(), ac, pay1, st.cpu(), True)

        if ep > 0 and ep % 64 == 0:
            a0.train_step()
            a1.train_step()

    # 정책 결합 및 NashConv 평가
    p0_dict = a0.get_average_policy_dict()
    p1_dict = a1.get_average_policy_dict()
    comb = {}
    for c in ['J', 'Q', 'K']:
        comb[f"{c}:"] = p0_dict[f"{c}:"]; comb[f"{c}:pb"] = p0_dict[f"{c}:pb"]
        comb[f"{c}:p"] = p1_dict[f"{c}:p"]; comb[f"{c}:b"] = p1_dict[f"{c}:b"]
    
    nc = exploitability.nash_conv(game, DictPolicyWrapper(game, comb))
    return float(nc)


def run_ablation_study():
    print("=" * 75)
    print("🔬 [PHASE 2] NFSP 8대 하이퍼파라미터 Ablation Study (각 25,000판)")
    print("=" * 75)

    base_config = {
        "episodes": 25000,
        "eta": 0.1,
        "rl_lr": 0.01,
        "sl_lr": 0.005,
        "hidden_dim": 64,
        "batch_size": 128,
        "target_update": 300,
        "eps_decay": 50000,
        "use_probs": True,
        "use_huber": True
    }

    # 기준선 (Baseline)
    print("\n▶ Baseline (Optimized NFSP 기준 설정) 실행 중...")
    base_nc = train_single_experiment(**base_config, seed=42)
    print(f"  * Baseline NashConv: {base_nc:.6f}")

    variations = [
        ("A. SL Target: Discrete Action (V4.1 방식)", {"use_probs": False}),
        ("B. Loss: MSELoss (V4.1 방식)", {"use_huber": False}),
        ("C. Anticipatory Param: eta = 0.2", {"eta": 0.2}),
        ("D. Learning Rate: rl_lr = 0.005, sl_lr = 0.002", {"rl_lr": 0.005, "sl_lr": 0.002}),
        ("E. Network Dim: hidden_dim = 32", {"hidden_dim": 32}),
        ("F. Batch Size: batch_size = 64", {"batch_size": 64}),
        ("G. Target Update: interval = 100", {"target_update": 100}),
        ("H. Epsilon Decay: fast decay 15000 (V4.1 방식)", {"eps_decay": 15000})
    ]

    results = [("Baseline (Optimized)", base_nc, 0.0)]

    for name, var_dict in variations:
        cfg = copy.deepcopy(base_config)
        cfg.update(var_dict)
        print(f"\n▶ 테스트: {name} ...")
        t0 = time.time()
        nc_val = train_single_experiment(**cfg, seed=42)
        elapsed = time.time() - t0
        diff = nc_val - base_nc
        results.append((name, nc_val, diff))
        print(f"  * NashConv: {nc_val:.6f} (Baseline 대비: {diff:+.6f}, 소요: {elapsed:.1f}s)")

    print("\n" + "=" * 75)
    print("📊 [Ablation Study 종합 결과표 (NashConv 낮을수록 우수)]")
    print("=" * 75)
    print(f"{'실험 변수':<48} | {'NashConv':<12} | {'영향도 (vs Base)':<15}")
    print("-" * 80)
    for name, val, diff in results:
        status = "기준선" if diff == 0 else ("개선 (더 우수) ✨" if diff < -0.01 else ("퇴보 ❌" if diff > 0.01 else "동등"))
        print(f"{name:<48} | {val:.6f}     | {diff:>+8.4f} ({status})")
    print("=" * 75)


if __name__ == "__main__":
    run_ablation_study()
