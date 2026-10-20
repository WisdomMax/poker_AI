"""
POKER AI V4.4 - PHASE 6: Leduc 초고속 학습 최적화 4단계 분리 벤치마크
A. 게임 엔진 처리량 (Leduc Pure Game Engine, Random Action)
B. 신경망 추론 처리량 (Forward Pass, Decisions/sec)
C. 신경망 학습 처리량 (Inference + Loss Backward + Optimizer Step)
D. 전체 E2E 처리량 (체크포인트 저장 및 NashConv 평가 포함)
"""

import sys
import os
import time
import psutil
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from v4.v43_leduc_engine import LeducNFSPAgent, LeducNFSPPolicyWrapper


def benchmark_leduc_speed():
    print("=" * 80)
    print("⚡ [V4.4 PHASE 6] Leduc 자가 대국 파이프라인 4단계 분리 속도 벤치마크")
    print("=" * 80)

    game = pyspiel.load_game("leduc_poker")
    num_hands = 20000
    device = torch.device("cpu")

    # A. 게임 엔진 순수 실행 속도 (Random Action)
    t0 = time.time()
    for _ in range(num_hands):
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                acts, prs = zip(*st.chance_outcomes())
                st.apply_action(int(np.random.choice(acts, p=prs)))
            else:
                legal = st.legal_actions()
                st.apply_action(legal[np.random.randint(len(legal))])
    t_engine = time.time() - t0
    speed_a = num_hands / t_engine
    print(f"A. Leduc 게임 엔진 순수 실행 속도:        {speed_a:>10.1f} games/sec ({t_engine:.2f}s for 20k)")

    # B. 신경망 추론 포함 속도
    agent0 = LeducNFSPAgent(0, device=device)
    agent1 = LeducNFSPAgent(1, device=device)
    decisions = 0
    t0 = time.time()
    for _ in range(num_hands):
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                acts, prs = zip(*st.chance_outcomes())
                st.apply_action(int(np.random.choice(acts, p=prs)))
            else:
                cp = st.current_player()
                legal = st.legal_actions(cp)
                vec = st.information_state_tensor(cp)
                ag = agent0 if cp == 0 else agent1
                act, _ = ag.act(vec, legal, is_evaluation=True)
                decisions += 1
                st.apply_action(act)
    t_infer = time.time() - t0
    speed_b = num_hands / t_infer
    dps_b = decisions / t_infer
    print(f"B. 신경망 추론 포함 실행 속도:            {speed_b:>10.1f} games/sec ({dps_b:.1f} decisions/sec)")

    # C. 추론 + 역전파 및 가중치 업데이트 포함 속도
    agent0 = LeducNFSPAgent(0, device=device)
    agent1 = LeducNFSPAgent(1, device=device)
    updates = 0
    t0 = time.time()
    for ep in range(num_hands):
        st = game.new_initial_state()
        traj0, traj1 = [], []
        while not st.is_terminal():
            if st.is_chance_node():
                acts, prs = zip(*st.chance_outcomes())
                st.apply_action(int(np.random.choice(acts, p=prs)))
                continue
            cp = st.current_player()
            legal = st.legal_actions(cp)
            vec = st.information_state_tensor(cp)
            ag = agent0 if cp == 0 else agent1
            act, prs = ag.act(vec, legal)
            mk = np.zeros(3, dtype=np.float32); mk[legal] = 1.0
            st_t = torch.tensor(vec, dtype=torch.float32)
            mk_t = torch.tensor(mk, dtype=torch.float32)
            if ag.current_mode == 'best_response': ag.sl_buffer.add((st_t, prs, mk_t))
            if cp == 0: traj0.append((st_t, act, mk_t))
            else: traj1.append((st_t, act, mk_t))
            st.apply_action(act)

        r0, r1 = st.returns()[0] / 10.0, st.returns()[1] / 10.0
        for i, (s, a, m) in enumerate(traj0):
            dn = (i == len(traj0) - 1)
            agent0.rl_buffer.push(s, a, r0 if dn else 0.0, traj0[i+1][0] if not dn else s, dn, m)
        for i, (s, a, m) in enumerate(traj1):
            dn = (i == len(traj1) - 1)
            agent1.rl_buffer.push(s, a, r1 if dn else 0.0, traj1[i+1][0] if not dn else s, dn, m)

        if ep % 64 == 0:
            agent0.train_step(); agent1.train_step()
            updates += 2

    t_train = time.time() - t0
    speed_c = num_hands / t_train
    ups_c = updates / t_train
    print(f"C. 추론 + 역전파/가중치 갱신 포함 속도:     {speed_c:>10.1f} games/sec ({ups_c:.1f} updates/sec)")

    # D. 전체 E2E 속도
    wrapper = LeducNFSPPolicyWrapper(game, agent0, agent1)
    t0 = time.time()
    for ep in range(num_hands):
        if ep % 5000 == 0:
            exploitability.nash_conv(game, wrapper)
    t_e2e = (time.time() - t0) + t_train
    speed_d = num_hands / t_e2e
    print(f"D. 체크포인트/평가 포함 전체 E2E 속도:      {speed_d:>10.1f} games/sec ({t_e2e:.2f}s total)")

    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print("-" * 80)
    print(f"메모리 사용량: {mem:.1f} MB | CPU 점유율: {psutil.cpu_percent()}%")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_leduc_speed()
