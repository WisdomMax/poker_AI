"""
POKER AI V4.3 - PHASE 1: NFSP 원본 논문 vs OpenSpiel 구현 교차 비교
- Heinrich & Silver (2016) 논문: SL 메모리에 (s, a) Sampled Discrete Action 저장
- OpenSpiel JAX/PyTorch: SL 메모리에 (s, action_probs) Softmax/Epsilon-greedy 확률 분포 저장
- 5개 독립 시드 (201, 202, 203, 204, 205)에서 동일한 조건으로 비교 실험
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from v4.openspiel_adapter import DictPolicyWrapper
from v4.v42_phase2_ablation import train_single_experiment


def run_phase1_experiment():
    print("=" * 80)
    print("🔬 [V4.3 PHASE 1] NFSP 원본 논문 (Sampled Action) vs OpenSpiel (Action Probs) 비교")
    print("=" * 80)

    seeds = [201, 202, 203, 204, 205]
    episodes = 25000

    results_probs = []
    results_sampled = []

    print(f"\n▶ 5개 독립 시드 {seeds}로 각 25,000판 비교 실행 중...")

    for s in seeds:
        # A. Action Probability Target (OpenSpiel 방식)
        t0 = time.time()
        nc_p = train_single_experiment(
            episodes=episodes, use_probs=True, use_huber=False,
            rl_lr=0.005, sl_lr=0.002, target_update=100, seed=s
        )
        t_p = time.time() - t0
        results_probs.append(nc_p)

        # B. Sampled Action Target (Heinrich & Silver 2016 원본 논문 방식)
        t0 = time.time()
        nc_s = train_single_experiment(
            episodes=episodes, use_probs=False, use_huber=False,
            rl_lr=0.005, sl_lr=0.002, target_update=100, seed=s
        )
        t_s = time.time() - t0
        results_sampled.append(nc_s)

        print(f"  * Seed {s:>3} | Probs Target NashConv: {nc_p:.6f} ({t_p:.1f}s) | Sampled Action NashConv: {nc_s:.6f} ({t_s:.1f}s)")

    # 통계 계산
    p_mean = np.mean(results_probs)
    p_sem = np.std(results_probs, ddof=1) / np.sqrt(len(seeds))
    p_ci = 2.776 * p_sem

    s_mean = np.mean(results_sampled)
    s_sem = np.std(results_sampled, ddof=1) / np.sqrt(len(seeds))
    s_ci = 2.776 * s_sem

    print("\n" + "=" * 80)
    print("📊 [PHASE 1 결과: 원본 논문 vs OpenSpiel 통계 비교 (5회 시드 Mean ± 95% CI)]")
    print("=" * 80)
    print(f"A. Action Probs Target (OpenSpiel)   : NashConv = {p_mean:.6f} ± {p_ci:.4f} (Exploit: {p_mean*50:.2f} bb/100)")
    print(f"B. Sampled Action Target (원본 논문): NashConv = {s_mean:.6f} ± {s_ci:.4f} (Exploit: {s_mean*50:.2f} bb/100)")
    print(f"차이 (B - A)                         : {s_mean - p_mean:+.6f}")
    print("-" * 80)

    print("\n[장단점 심층 분석]")
    print("1. Sampled Action (원본 논문):")
    print("   - 장점: 메모리 절약(액션 정수 1개 저장), 연속 행동 공간 및 대형 게임에서 구현이 매우 간단.")
    print("   - 단점: 유한한 소형 게임(Kuhn 등)에서는 샘플링 분산(Monte Carlo noise)이 그대로 평균 정책에 전파되어")
    print("     동일한 에피소드 수에서 수렴 속도가 상대적으로 느리고 분산이 큼.")
    print("2. Action Probability Target (OpenSpiel):")
    print("   - 장점: BR 정책의 전체 확률 분포(Softmax/Epsilon-greedy)를 소프트 라벨로 전달하므로,")
    print("     샘플링 분산이 제거되어 적은 에피소드(소표본)에서도 평균 정책이 훨씬 매끄럽고 빠르게 수렴함.")
    print("   - 단점: 메모리 사용량 증가(행동 수만큼 부동소수점 벡터 저장 필요).")
    print("=" * 80)


if __name__ == "__main__":
    run_phase1_experiment()
