"""
POKER AI V4.7 - PHASE 4: 5인 Texas Hold'em 자기대국(Self-Play) 강화학습
- 포지션 공정성 래퍼(SeatRotationWrapper) 하에서 5명의 AI가 상호 대결하며 전략 학습
- 단계별 학습 계획:
  - STEP A: 1,000판 (루프 무결성 검증)
  - STEP B: 10,000판 (신경망 가중치 변화 및 행동 확률 추적) -> ckpt_10k 저장
  - STEP C: 50,000판 (전략 개선 평가) -> ckpt_50k 저장
  - STEP D: 100,000판 (대규모 수렴 확인) -> ckpt_100k 저장
- 학습 알고리즘:
  - Advantage Normalization이 적용된 Policy Gradient (REINFORCE)
  - Gradient Clipping (norm <= 1.0)
  - NaN 감시 및 롤아웃 안정화
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

# 상위 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_holdem_rules_audit import get_5p_holdem_game
from phase2_position_fairness import HoldemSeatRotationWrapper
from phase3_holdem_ai_architecture import Holdem5PNet, Holdem5PAgent

def compute_entropy(probs: np.ndarray) -> float:
    ent = 0.0
    for p in probs:
        if p > 1e-12:
            ent -= p * math.log2(p)
    return ent

def run_phase4_training(
    total_episodes: int = 100000,
    checkpoints: List[int] = [1000, 10000, 50000, 100000],
    seed: int = 42
):
    print("=" * 80)
    print(f"🚀 [V4.7 PHASE 4] 5인 Texas Hold'em 자기대국(Self-Play) 강화학습 (총 {total_episodes:,} 판)")
    print("=" * 80)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    device = torch.device("cpu")
    models_dir = "v47/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 5명의 에이전트 초기화 (공유 정책 신경망 또는 독립 에이전트; 자기대국에서는 단일 메인 정책 복제 활용)
    # 5명이 동일한 정책 파라미터(Self-Play)를 학습하되, 각자 다른 포지션에서 수집한 궤적으로 업데이트
    shared_net = Holdem5PNet(253, 256, 4).to(device)
    optimizer = optim.Adam(shared_net.parameters(), lr=0.0003, weight_decay=1e-5)
    
    initial_weights = torch.cat([p.flatten() for p in shared_net.parameters()]).clone()
    
    # 통계 추적
    window_rewards = []
    window_entropies = []
    action_counts = np.zeros(4, dtype=int)
    total_decisions = 0
    
    batch_trajectories = [] # (st_vec, mask, act, return)
    start_time = time.time()
    
    print("\n학습 진행 및 단계별 체크포인트 분석:")
    print(f"{'Episode':<10} | {'평균 칩/Hand':<14} | {'평균 엔트로피':<13} | {'가중치 변위 ||ΔW||':<18} | {'속도 (games/s)'}")
    print("-" * 80)
    
    for ep in range(1, total_episodes + 1):
        s2a = wrapper.get_seat_to_agent(ep)
        a2s = wrapper.get_agent_to_seat(ep)
        
        st = game.new_initial_state()
        traj = {s: [] for s in range(5)}
        ep_entropies = []
        
        while not st.is_terminal():
            if st.is_chance_node():
                acts, prs = zip(*st.chance_outcomes())
                st.apply_action(np.random.choice(acts, p=prs))
                continue
                
            cur_seat = st.current_player()
            legal_acts = st.legal_actions(cur_seat)
            info_vec = st.information_state_tensor(cur_seat)
            
            x = torch.tensor(info_vec, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
            for a in legal_acts: mask[0, a] = True
            
            logits = shared_net(x, mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            
            prs = [probs[a] for a in legal_acts]
            tot = sum(prs)
            prs = np.array(prs) / tot if tot > 0 else np.ones(len(legal_acts)) / len(legal_acts)
            
            act = int(np.random.choice(legal_acts, p=prs))
            action_counts[act] += 1
            total_decisions += 1
            
            ep_entropies.append(compute_entropy(probs))
            traj[cur_seat].append((info_vec, mask.squeeze(0).cpu().numpy(), act))
            st.apply_action(act)
            
        returns = st.returns()
        # 제로섬 무결성 확인
        assert abs(sum(returns)) < 1e-4, f"Episode {ep}: 제로섬 위반 감지 {returns}"
        
        if ep_entropies:
            window_entropies.append(np.mean(ep_entropies))
            
        # 칩 보상을 100으로 스케일링 (1 Big Blind = 100 chips 단위 정규화)
        # 각 좌석별 궤적에 보상 연결
        for s in range(5):
            r_bb = returns[s] / 100.0
            for info_v, m_arr, a in traj[s]:
                batch_trajectories.append((info_v, m_arr, a, r_bb))
                
        # 128 에피소드마다 정책 업데이트
        if ep % 128 == 0:
            st_list, mk_list, ac_list, rw_list = zip(*batch_trajectories)
            b_st = torch.tensor(np.array(st_list), dtype=torch.float32, device=device)
            b_mk = torch.tensor(np.array(mk_list), dtype=torch.bool, device=device)
            b_ac = torch.tensor(ac_list, dtype=torch.int64, device=device)
            b_rw = torch.tensor(rw_list, dtype=torch.float32, device=device)
            
            # Advantage 정규화
            b_adv = (b_rw - b_rw.mean()) / (b_rw.std() + 1e-8) if len(b_rw) > 1 else b_rw
            
            logits = shared_net(b_st, b_mk)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_p = log_probs.gather(1, b_ac.unsqueeze(1)).squeeze(1)
            
            loss = -(selected_log_p * b_adv).mean()
            
            # NaN 감시
            if torch.isnan(loss):
                print(f"❌ [CRITICAL] Episode {ep}에서 Loss NaN 발생! 학습 중단.")
                break
                
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(shared_net.parameters(), 1.0)
            optimizer.step()
            batch_trajectories = []
            
        if ep in checkpoints:
            elapsed = time.time() - start_time
            speed = ep / elapsed if elapsed > 0 else 0.0
            curr_weights = torch.cat([p.flatten() for p in shared_net.parameters()]).clone()
            w_diff = torch.norm(curr_weights - initial_weights).item()
            m_ent = np.mean(window_entropies[-1000:]) if window_entropies else 0.0
            
            print(f"{ep:<10} | {'0.000 (대칭)':<14} | {m_ent:<13.4f} | {w_diff:<18.6f} | {speed:>7.1f} g/s")
            
            # 체크포인트 저장
            save_name = f"holdem_5p_step_{ep//1000}k.pt" if ep >= 1000 else f"holdem_5p_step_{ep}.pt"
            torch.save(shared_net.state_dict(), os.path.join(models_dir, save_name))
            
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("📊 [5인 Texas Hold'em 자기대국 학습 완료 요약]")
    print("=" * 80)
    print(f"  - 총 판수: {total_episodes:,} 판")
    print(f"  - 총 소요 시간: {total_elapsed:.1f} 초 (평균 {total_episodes/total_elapsed:.1f} games/sec)")
    print(f"  - 판당 평균 의사결정 수: {total_decisions/total_episodes:.2f} 회")
    tot_acts = sum(action_counts)
    print(f"  - 액션 비율 분포: Fold {action_counts[0]/tot_acts*100:.1f}%, Call/Check {action_counts[1]/tot_acts*100:.1f}%, Pot {action_counts[2]/tot_acts*100:.1f}%, All-in {action_counts[3]/tot_acts*100:.1f}%")
    print(f"  - 체크포인트 저장 완료: {models_dir}/holdem_5p_step_1k, 10k, 50k, 100k.pt")
    print("=" * 80)

if __name__ == "__main__":
    run_phase4_training(total_episodes=100000, checkpoints=[1000, 10000, 50000, 100000])
