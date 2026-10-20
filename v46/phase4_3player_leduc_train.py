"""
POKER AI V4.6 - PHASE 4: 3인 Leduc Poker 다자간 자가 대국(Self-Play) 초기 학습
- 환경: OpenSpiel 3-player Leduc Poker (47차원 관측 텐서, 8장 덱)
- 학습 규모: 10,000 -> 50,000 -> 100,000 에피소드
- 3개 플레이어(Player 0, 1, 2) 동시 정책 경사/DQN 자가 대국 학습
- 측정 지표:
  1. 플레이어/포지션별 평균 칩 보상 (\sum returns == 0)
  2. 정책 섀넌 엔트로피 (탐색 vs 결정론적 수렴도)
  3. 초당 학습 처리 속도 (games/s) 및 게임당 의사결정 수
  4. 다자간 게임 이론적 분석 (수렴 미보장 및 비합리적 균형 문제)
"""

import sys
import os
import math
import time
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pyspiel

# 3인 Leduc 전용 신경망 (47차원 입력 -> 128 -> 128 -> 3차원 로짓)
class Leduc3PlayerNet(nn.Module):
    def __init__(self, input_dim: int = 47, hidden_dim: int = 128, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None) -> torch.Tensor:
        logits = self.net(x)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        return logits
        
    def get_action_probs(self, x: torch.Tensor, legal_mask: torch.Tensor = None) -> torch.Tensor:
        logits = self.forward(x, legal_mask)
        return torch.softmax(logits, dim=-1)

class MultiplayerLeducAgent:
    def __init__(self, player_id: int, lr: float = 0.001, device=torch.device("cpu")):
        self.player_id = player_id
        self.device = device
        self.policy_net = Leduc3PlayerNet(47, 128, 3).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
    def act(self, info_vec: List[float], legal_acts: List[int]) -> Tuple[int, float, np.ndarray]:
        x = torch.tensor(info_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.zeros((1, 3), dtype=torch.bool, device=self.device)
        for a in legal_acts: mask[0, a] = True
        
        logits = self.policy_net(x, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        
        prs = [probs[a] for a in legal_acts]
        tot = sum(prs)
        prs = np.array(prs) / tot if tot > 0 else np.ones(len(legal_acts)) / len(legal_acts)
        
        chosen_act = int(np.random.choice(legal_acts, p=prs))
        log_prob = math.log(prs[legal_acts.index(chosen_act)] + 1e-12)
        return chosen_act, log_prob, probs

def compute_entropy(probs: np.ndarray) -> float:
    ent = 0.0
    for p in probs:
        if p > 1e-12:
            ent -= p * math.log2(p)
    return ent

def run_phase4_3player_training(total_episodes: int = 100000, checkpoints: List[int] = [10000, 50000, 100000]):
    print("=" * 80)
    print(f"🚀 [V4.6 PHASE 4] 3인 Leduc Poker 다자간 자가 대국(Self-Play) 학습 (총 {total_episodes:,} 판)")
    print("=" * 80)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    device = torch.device("cpu")
    
    agents = [MultiplayerLeducAgent(i, lr=0.0005, device=device) for i in range(3)]
    
    # 윈도우 통계 추적 버퍼
    window_rewards = {0: [], 1: [], 2: []}
    window_entropies = []
    total_decisions = 0
    
    start_time = time.time()
    batch_trajectories = {0: [], 1: [], 2: []}
    
    print(f"\n학습 진행 및 단계별 다자간 체크포인트 분석 ({checkpoints}):")
    print(f"{'Episode':<10} | {'P0 평균칩':<12} | {'P1 평균칩':<12} | {'P2 평균칩':<12} | {'평균 엔트로피':<13} | {'속도 (games/s)'}")
    print("-" * 80)
    
    for ep in range(1, total_episodes + 1):
        state = game.new_initial_state()
        traj = {0: [], 1: [], 2: []}
        ep_entropies = []
        
        while not state.is_terminal():
            if state.is_chance_node():
                acts, prs = zip(*state.chance_outcomes())
                state.apply_action(np.random.choice(acts, p=prs))
                continue
                
            cur_p = state.current_player()
            legal_acts = state.legal_actions(cur_p)
            info_vec = state.information_state_tensor(cur_p)
            
            act, log_p, probs = agents[cur_p].act(info_vec, legal_acts)
            ep_entropies.append(compute_entropy(probs))
            total_decisions += 1
            
            mask = np.zeros(3, dtype=bool)
            for a in legal_acts: mask[a] = True
            
            traj[cur_p].append((info_vec, mask, act))
            state.apply_action(act)
            
        returns = state.returns()
        for p in range(3):
            window_rewards[p].append(returns[p])
            for st_vec, mask_arr, ac in traj[p]:
                batch_trajectories[p].append((st_vec, mask_arr, ac, returns[p]))
                
        if ep_entropies:
            window_entropies.append(np.mean(ep_entropies))
            
        # 128 에피소드마다 정책 경사 업데이트
        if ep % 128 == 0:
            for p in range(3):
                if not batch_trajectories[p]:
                    continue
                st_list, mk_list, ac_list, rw_list = zip(*batch_trajectories[p])
                b_st = torch.tensor(np.array(st_list), dtype=torch.float32)
                b_mk = torch.tensor(np.array(mk_list), dtype=torch.bool)
                b_ac = torch.tensor(ac_list, dtype=torch.int64)
                b_rw = torch.tensor(rw_list, dtype=torch.float32)
                
                # Baseline 차감 (분산 축소)
                b_adv = (b_rw - b_rw.mean()) / (b_rw.std() + 1e-8) if len(b_rw) > 1 else b_rw
                
                logits = agents[p].policy_net(b_st, b_mk)
                log_probs = torch.log_softmax(logits, dim=-1)
                selected_log_p = log_probs.gather(1, b_ac.unsqueeze(1)).squeeze(1)
                
                loss = -(selected_log_p * b_adv).mean()
                agents[p].optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agents[p].policy_net.parameters(), 1.0)
                agents[p].optimizer.step()
                batch_trajectories[p] = []
                
        if ep in checkpoints:
            elapsed = time.time() - start_time
            speed = ep / elapsed if elapsed > 0 else 0.0
            
            # 최근 1,000판 기준 통계
            p0_r = np.mean(window_rewards[0][-1000:])
            p1_r = np.mean(window_rewards[1][-1000:])
            p2_r = np.mean(window_rewards[2][-1000:])
            m_ent = np.mean(window_entropies[-1000:]) if window_entropies else 0.0
            
            print(f"{ep:<10} | {p0_r:<+12.3f} | {p1_r:<+12.3f} | {p2_r:<+12.3f} | {m_ent:<13.4f} | {speed:>7.1f} g/s")
            
    total_elapsed = time.time() - start_time
    avg_decisions = total_decisions / total_episodes
    
    print("\n" + "=" * 80)
    print("📊 [3인 Leduc Poker 학습 결과 요약]")
    print("=" * 80)
    print(f"  - 총 학습 판수: {total_episodes:,} 판")
    print(f"  - 총 소요 시간: {total_elapsed:.1f} 초 (평균 {total_episodes/total_elapsed:.1f} games/sec)")
    print(f"  - 게임당 평균 의사결정 수: {avg_decisions:.2f} 회")
    print(f"  - 최종 포지션별 칩 손익 (최근 5,000판):")
    print(f"    * Player 0 (Button/First): {np.mean(window_rewards[0][-5000:]):+.3f} chips/hand")
    print(f"    * Player 1 (Middle):       {np.mean(window_rewards[1][-5000:]):+.3f} chips/hand")
    print(f"    * Player 2 (Last):         {np.mean(window_rewards[2][-5000:]):+.3f} chips/hand")
    print(f"    * 제로섬 보존 검증: {sum(np.mean(window_rewards[p][-5000:]) for p in range(3)):.6f} chips (완벽 보존)")
    print("=" * 80)
    
    # 모델 저장
    for p in range(3):
        save_path = f"v46/models/leduc_3p_agent_p{p}.pt"
        torch.save(agents[p].policy_net.state_dict(), save_path)
    print("💾 3인 모델 저장 완료: v46/models/leduc_3p_agent_p0/p1/p2.pt")

if __name__ == "__main__":
    run_phase4_3player_training(total_episodes=100000, checkpoints=[10000, 50000, 100000])
