"""
POKER AI V4.4 - PHASE 3: Leduc Poker NFSP 학습 개선 및 대규모 자가 대국
- 보상 정규화 (Reward Normalization /10.0) 및 Huber Loss 적용
- Legal Action Masking이 적용된 Target Q-Network
- 50,000 -> 100,000 -> 200,000 에피소드 단계별 수렴도 측정
- 최적 검증 체크포인트 자동 선택 및 모델 저장
"""

import sys
import os
import time
import copy
import random
from typing import List, Dict, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from v4.v43_leduc_engine import LeducNFSPAgent, LeducNFSPPolicyWrapper


def train_leduc_optimized_nfsp(
    total_episodes: int = 200000,
    checkpoints: List[int] = [10000, 50000, 100000, 150000, 200000],
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = pyspiel.load_game("leduc_poker")
    device = torch.device("cpu")
    models_dir = "v44/models"
    os.makedirs(models_dir, exist_ok=True)

    # Leduc 최적화 에이전트 초기화 (학습률 rl=0.002, sl=0.001, target_update=200)
    agent0 = LeducNFSPAgent(0, rl_lr=0.002, sl_lr=0.001, target_update=200,
                            eps_decay=total_episodes, device=device)
    agent1 = LeducNFSPAgent(1, rl_lr=0.002, sl_lr=0.001, target_update=200,
                            eps_decay=total_episodes, device=device)
    wrapper = LeducNFSPPolicyWrapper(game, agent0, agent1)

    print("=" * 80)
    print(f"🚀 [V4.4 Leduc Optimized NFSP 학습] 총 {total_episodes:,} 판 (Device: CPU)")
    print(f"체크포인트: {checkpoints} | 보상 정규화: /10.0 적용 | Legal Action Masking 활성화")
    print("=" * 80)

    start_time = time.time()
    ckpt_idx = 0
    history_log = []
    best_nc = 999.0
    best_ep = 0

    for ep in range(1, total_episodes + 1):
        agent0.sample_episode_mode()
        agent1.sample_episode_mode()

        state = game.new_initial_state()
        traj0, traj1 = [], []

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, prs = zip(*outcomes)
                ch_act = np.random.choice(acts, p=prs)
                state.apply_action(ch_act)
                continue

            cur_p = state.current_player()
            info_vec = state.information_state_tensor(cur_p)
            legal_acts = state.legal_actions(cur_p)
            agent = agent0 if cur_p == 0 else agent1

            act, probs = agent.act(info_vec, legal_acts)
            mask = np.zeros(3, dtype=np.float32)
            mask[legal_acts] = 1.0

            st_tensor = torch.tensor(info_vec, dtype=torch.float32)
            mk_tensor = torch.tensor(mask, dtype=torch.float32)

            if agent.current_mode == 'best_response':
                agent.sl_buffer.add((st_tensor, probs, mk_tensor))

            if cur_p == 0:
                traj0.append((st_tensor, act, mk_tensor))
            else:
                traj1.append((st_tensor, act, mk_tensor))

            state.apply_action(act)

        returns = state.returns()
        # 보상 정규화 (Leduc 팟 스케일 조정: r / 10.0)
        r0 = returns[0] / 10.0
        r1 = returns[1] / 10.0

        for i, (st, ac, mk) in enumerate(traj0):
            is_done = (i == len(traj0) - 1)
            nst = traj0[i+1][0] if not is_done else st
            rw = r0 if is_done else 0.0
            agent0.rl_buffer.push(st, ac, rw, nst, is_done, mk)

        for i, (st, ac, mk) in enumerate(traj1):
            is_done = (i == len(traj1) - 1)
            nst = traj1[i+1][0] if not is_done else st
            rw = r1 if is_done else 0.0
            agent1.rl_buffer.push(st, ac, rw, nst, is_done, mk)

        if ep % 64 == 0:
            agent0.train_step()
            agent1.train_step()

        # 체크포인트 평가
        if ckpt_idx < len(checkpoints) and ep == checkpoints[ckpt_idx]:
            ckpt_ep = checkpoints[ckpt_idx]
            ckpt_idx += 1
            nc = float(exploitability.nash_conv(game, wrapper))
            exp = nc / 2.0
            el = time.time() - start_time

            is_best = False
            if nc < best_nc:
                best_nc = nc
                best_ep = ckpt_ep
                is_best = True
                torch.save(agent0.avg_net.state_dict(), f"{models_dir}/v44_best_leduc_nfsp_p0.pt")
                torch.save(agent1.avg_net.state_dict(), f"{models_dir}/v44_best_leduc_nfsp_p1.pt")

            star = " 🌟 [NEW BEST!]" if is_best else ""
            print(f"  💾 [Leduc 체크포인트 {ckpt_ep:>6} 판] NashConv: {nc:.6f} (Exp: {exp*100:>5.2f} bb/100) | 소요: {el:>5.1f}s{star}")
            history_log.append({
                "episode": ckpt_ep,
                "nash_conv": nc,
                "exploitability": exp,
                "time": el,
                "is_best": is_best
            })

    total_time = time.time() - start_time
    gps = total_episodes / total_time
    print("=" * 80)
    print(f"🎯 [V4.4 Leduc 최적 모델 학습 완료]")
    print(f"  - 총 학습 판 수: {total_episodes:,} 판 ({total_time:.1f}초, {gps:.1f} games/sec)")
    print(f"  - 최고의 체크포인트: {best_ep:,} 판 (NashConv: {best_nc:.6f}, Exploit: {best_nc*50:.2f} bb/100)")
    print(f"  - V4.3(1.9996) 대비 격차 축소: {1.999607 - best_nc:+.6f}")
    print(f"  - 가중치 저장 완료: {models_dir}/v44_best_leduc_nfsp_p0.pt / p1.pt")
    print("=" * 80)

    return history_log, best_nc, best_ep


if __name__ == "__main__":
    train_leduc_optimized_nfsp(total_episodes=200000)
