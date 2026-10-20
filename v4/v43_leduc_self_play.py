"""
POKER AI V4.3 - PHASE 5 & 6: Leduc Poker 환경 구축 및 자가 대국(Self-Play) 실험
- 4대 모델 비교: Random Policy, OpenSpiel CFR, NFSP Self-Play, NFSP + Safe Bayesian
- 학습 규모: 10,000, 50,000, 100,000 에피소드
- Kuhn vs Leduc 복잡도 및 학습 속도 정량 비교
"""

import sys
import os
import time
import copy
import random
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, cfr
from v4.v43_leduc_engine import LeducNFSPAgent, LeducNFSPPolicyWrapper, ACTIONS_3


class UniformRandomPolicy(policy.Policy):
    def __init__(self, game):
        super().__init__(game, list(range(game.num_players())))

    def action_probabilities(self, state, player_id=None):
        if player_id is None: player_id = state.current_player()
        legal = state.legal_actions(player_id)
        p = 1.0 / len(legal)
        return {a: p for a in legal}


def run_leduc_cfr_benchmark(game, iterations_list=[10, 50, 100]):
    print("\n[Leduc CFR 공식 기준선 수렴도 측정]")
    solver = cfr.CFRSolver(game)
    cfr_records = []
    t0 = time.time()
    for it in range(1, max(iterations_list) + 1):
        solver.evaluate_and_update_policy()
        if it in iterations_list:
            avg_pol = solver.average_policy()
            nc = float(exploitability.nash_conv(game, avg_pol))
            el = time.time() - t0
            cfr_records.append((it, nc, el))
            print(f"  * CFR Iteration {it:>3}: NashConv = {nc:.6f} (Exp: {nc/2:.6f}, 소요: {el:.1f}s)")
    return cfr_records


def train_leduc_nfsp(
    total_episodes: int = 50000,
    checkpoints: List[int] = [10000, 30000, 50000],
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = pyspiel.load_game("leduc_poker")
    device = torch.device("cpu")
    models_dir = "v4/models"
    os.makedirs(models_dir, exist_ok=True)

    agent0 = LeducNFSPAgent(0, device=device)
    agent1 = LeducNFSPAgent(1, device=device)
    wrapper = LeducNFSPPolicyWrapper(game, agent0, agent1)

    print("=" * 80)
    print(f"🃏 [Leduc Poker NFSP 자가 대국 시작] 총 {total_episodes:,} 판")
    print(f"상태 공간: 30차원 | 액션 공간: 3개 (0:Fold, 1:Call, 2:Raise) | 장치: CPU")
    print("=" * 80)

    start_time = time.time()
    ckpt_idx = 0
    history_log = []
    total_updates = 0

    for ep in range(1, total_episodes + 1):
        agent0.sample_episode_mode()
        agent1.sample_episode_mode()

        state = game.new_initial_state()
        traj0, traj1 = [], [] # list of (info_vec, action, probs, legal_mask, player)

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

        returns = state.returns() # [R0, R1]
        r0, r1 = returns[0], returns[1]

        # RL 버퍼 저장
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
            total_updates += 2

        # 체크포인트 평가
        if ckpt_idx < len(checkpoints) and ep == checkpoints[ckpt_idx]:
            ckpt_ep = checkpoints[ckpt_idx]
            ckpt_idx += 1
            nc = float(exploitability.nash_conv(game, wrapper))
            el = time.time() - start_time
            print(f"  💾 [Leduc 체크포인트 {ckpt_ep:>6} 판] NashConv: {nc:.6f} (Exp: {nc/2:.6f}) | 소요: {el:>5.1f}s | Updates: {total_updates}")
            history_log.append((ckpt_ep, nc, el, total_updates))
            torch.save(agent0.avg_net.state_dict(), f"{models_dir}/leduc_nfsp_p0_ckpt_{ckpt_ep}.pt")
            torch.save(agent1.avg_net.state_dict(), f"{models_dir}/leduc_nfsp_p1_ckpt_{ckpt_ep}.pt")

    total_time = time.time() - start_time
    gps = total_episodes / total_time
    print("=" * 80)
    print(f"✅ Leduc NFSP 학습 완료: 총 {total_episodes:,} 판, {total_time:.1f}초 ({gps:.1f} games/sec)")
    return history_log, gps, wrapper


def run_phase5_and_6():
    print("=" * 80)
    print("🏰 [V4.3 PHASE 5 & 6] Leduc Poker 환경 구축 및 비교 연구")
    print("=" * 80)

    game = pyspiel.load_game("leduc_poker")

    # 1. Random Policy 평가
    rand_pol = UniformRandomPolicy(game)
    nc_rand = float(exploitability.nash_conv(game, rand_pol))
    print(f"\n[A. Leduc Uniform Random Policy]")
    print(f"  - NashConv:       {nc_rand:.6f}")
    print(f"  - Exploitability: {nc_rand/2:.6f}")

    # 2. CFR 기준선
    cfr_recs = run_leduc_cfr_benchmark(game, iterations_list=[10, 50, 100])

    # 3. NFSP Self-Play 학습 (50,000 에피소드 초기 검증)
    history_log, leduc_gps, wrapper = train_leduc_nfsp(total_episodes=50000, checkpoints=[10000, 30000, 50000])

    # 4. Kuhn vs Leduc 복잡도 및 성능 정량 비교
    print("\n" + "=" * 80)
    print("📊 [Kuhn Poker vs Leduc Poker 정량적 비교표]")
    print("=" * 80)
    print(f"{'비교 항목':<25} | {'Kuhn Poker':<24} | {'Leduc Poker':<24}")
    print("-" * 80)
    print(f"{'덱 크기 (총 카드 수)':<25} | {'3장 (J, Q, K)':<24} | {'6장 (J, Q, K 각 2장)':<24}")
    print(f"{'베팅 라운드':<25} | {'1 라운드 (프리플롭)':<24} | {'2 라운드 (프리플롭 + 플롭)':<24}")
    print(f"{'커뮤니티 카드':<25} | {'0장 (없음)':<24} | {'1장 (플롭 공개)':<24}")
    print(f"{'정보집합(InfoSet) 노드 수':<25} | {'12 개':<24} | {'936 개 (78배 증가)':<24}")
    print(f"{'상태 벡터 차원':<25} | {'7 차원 (OS 11차원)':<24} | {'30 차원':<24}")
    print(f"{'행동 공간 (Action Space)':<25} | {'2개 (Pass, Bet)':<24} | {'3개 (Fold, Call, Raise)':<24}")
    print(f"{'Random NashConv':<25} | {'0.916667':<24} | {nc_rand:<24.6f}")
    final_leduc_nc = history_log[-1][1]
    print(f"{'NFSP 50k NashConv':<25} | {'0.258535':<24} | {final_leduc_nc:<24.6f}")
    print(f"{'학습 처리 속도':<25} | {'약 8,050 games/sec':<24} | {f'{leduc_gps:.1f} games/sec':<24}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase5_and_6()
