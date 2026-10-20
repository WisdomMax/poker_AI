"""
POKER AI V4.2 - PHASE 4: 학습 속도 4단계 정밀 분리 벤치마크
A. 게임 엔진만 실행한 속도 (Pure Simulation, Random Action)
B. 신경망 추론(Inference/Forward) 포함 속도
C. 신경망 추론 + 역전파 및 가중치 업데이트(Training Update) 포함 속도
D. 체크포인트 저장 및 NashConv 평가 포함 전체 E2E 속도
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
from v4.openspiel_adapter import DictPolicyWrapper
from kuhn_engine_v3 import evaluate_payoff, PASS, BET, ACTIONS
from v4.v42_phase2_ablation import OptimizedNFSPAgent, encode_kuhn_state


def benchmark_speed():
    print("=" * 75)
    print("⚡ [PHASE 4] 자가 대국 파이프라인 4단계 분리 속도 벤치마크")
    print("=" * 75)

    num_hands = 50000
    deck = [1, 2, 3]
    device = torch.device("cpu")

    # -------------------------------------------------------------
    # A. 순수 게임 엔진 실행 속도 (신경망 없음, Random Action)
    # -------------------------------------------------------------
    t0 = time.time()
    for _ in range(num_hands):
        cards = np.random.permutation(deck)
        c0, c1 = cards[0], cards[1]
        act0 = np.random.choice(ACTIONS)
        if act0 == PASS:
            act1 = np.random.choice(ACTIONS)
            fin = "pp" if act1 == PASS else ("pbp" if np.random.choice(ACTIONS) == PASS else "pbb")
        else:
            act1 = np.random.choice(ACTIONS)
            fin = "bp" if act1 == PASS else "bb"
        evaluate_payoff(fin, c0, c1)
    t_engine = time.time() - t0
    speed_a = num_hands / t_engine

    print(f"A. 게임 엔진 순수 실행 속도:            {speed_a:>10.1f} games/sec ({t_engine:.2f}s for 50k)")

    # -------------------------------------------------------------
    # B. 신경망 추론(Forward Pass) 포함 속도
    # -------------------------------------------------------------
    a0 = OptimizedNFSPAgent(0, device=device)
    a1 = OptimizedNFSPAgent(1, device=device)
    decisions_count = 0

    t0 = time.time()
    for _ in range(num_hands):
        cards = np.random.permutation(deck)
        c0, c1 = cards[0], cards[1]

        act0, _ = a0.act(c0, "", is_evaluation=True)
        decisions_count += 1
        if act0 == PASS:
            act1, _ = a1.act(c1, "p", is_evaluation=True)
            decisions_count += 1
            if act1 == PASS:
                fin = "pp"
            else:
                act0_2, _ = a0.act(c0, "pb", is_evaluation=True)
                decisions_count += 1
                fin = "pbp" if act0_2 == PASS else "pbb"
        else:
            act1, _ = a1.act(c1, "b", is_evaluation=True)
            decisions_count += 1
            fin = "bp" if act1 == PASS else "bb"
        evaluate_payoff(fin, c0, c1)
    t_infer = time.time() - t0
    speed_b = num_hands / t_infer
    dps_b = decisions_count / t_infer

    print(f"B. 신경망 추론 포함 실행 속도:          {speed_b:>10.1f} games/sec ({dps_b:.1f} decisions/sec)")

    # -------------------------------------------------------------
    # C. 추론 + 역전파 및 가중치 업데이트 포함 속도
    # -------------------------------------------------------------
    a0 = OptimizedNFSPAgent(0, device=device)
    a1 = OptimizedNFSPAgent(1, device=device)
    updates_count = 0

    t0 = time.time()
    for ep in range(num_hands):
        a0.sample_episode_mode()
        a1.sample_episode_mode()
        cards = np.random.permutation(deck)
        c0, c1 = cards[0], cards[1]

        s0 = encode_kuhn_state(c0, "", device)
        act0, pr0 = a0.act(c0, "")
        if a0.current_mode == 'best_response': a0.sl_buffer.add((s0.cpu(), pr0))

        if act0 == PASS:
            s1 = encode_kuhn_state(c1, "p", device)
            act1, pr1 = a1.act(c1, "p")
            if a1.current_mode == 'best_response': a1.sl_buffer.add((s1.cpu(), pr1))
            if act1 == PASS: fin = "pp"
            else:
                s0_2 = encode_kuhn_state(c0, "pb", device)
                act0_2, pr0_2 = a0.act(c0, "pb")
                if a0.current_mode == 'best_response': a0.sl_buffer.add((s0_2.cpu(), pr0_2))
                fin = "pbp" if act0_2 == PASS else "pbb"
        else:
            s1 = encode_kuhn_state(c1, "b", device)
            act1, pr1 = a1.act(c1, "b")
            if a1.current_mode == 'best_response': a1.sl_buffer.add((s1.cpu(), pr1))
            fin = "bp" if act1 == PASS else "bb"

        pay0, pay1 = evaluate_payoff(fin, c0, c1)
        a0.rl_buffer.push(s0.cpu(), act0, pay0, s0.cpu(), True)
        a1.rl_buffer.push(s1.cpu(), act1, pay1, s1.cpu(), True)

        if ep > 0 and ep % 64 == 0:
            a0.train_step()
            a1.train_step()
            updates_count += 2

    t_train = time.time() - t0
    speed_c = num_hands / t_train
    ups_c = updates_count / t_train

    print(f"C. 추론 + 역전파/가중치 갱신 포함 속도:   {speed_c:>10.1f} games/sec ({ups_c:.1f} updates/sec)")

    # -------------------------------------------------------------
    # D. 체크포인트 저장 및 NashConv 평가 포함 전체 E2E 속도
    # -------------------------------------------------------------
    game = pyspiel.load_game("kuhn_poker")
    t0 = time.time()
    # 50,000판 중 5회 체크포인트 평가 수행
    eval_count = 0
    for ep in range(num_hands):
        if ep % 10000 == 0:
            p0_d = a0.get_average_policy_dict()
            p1_d = a1.get_average_policy_dict()
            comb = {f"{c}:": p0_d[f"{c}:"] for c in ['J', 'Q', 'K']}
            for c in ['J', 'Q', 'K']:
                comb[f"{c}:pb"] = p0_d[f"{c}:pb"]
                comb[f"{c}:p"] = p1_d[f"{c}:p"]
                comb[f"{c}:b"] = p1_d[f"{c}:b"]
            exploitability.nash_conv(game, DictPolicyWrapper(game, comb))
            eval_count += 1
    t_e2e = (time.time() - t0) + t_train # 학습 시간 + 평가 시간
    speed_d = num_hands / t_e2e

    print(f"D. 체크포인트/평가 포함 전체 E2E 속도:    {speed_d:>10.1f} games/sec ({t_e2e:.2f}s total)")

    print("-" * 75)
    print(f"메모리 사용량: {psutil.Process().memory_info().rss / (1024*1024):.1f} MB | CPU 사용률: {psutil.cpu_percent()}%")
    print("=" * 75)


if __name__ == "__main__":
    benchmark_speed()
