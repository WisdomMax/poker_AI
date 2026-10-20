"""
POKER AI V4.3 - PHASE 2: Epsilon Decay 스케줄 및 하이퍼파라미터 시드 분리 재검증
- Fast Decay (15,000 스텝) vs Slow Decay (100,000 스텝) 장기 학습 (100,000판) 비교
- 검증용 시드 (Validation Seeds: 301, 302, 303)
- 테스트용 시드 (Test Seeds: 401, 402, 403)
- 단기 25k 착시 현상과 장기 100k 탐색 지속성 증명
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v4.v42_phase2_ablation import train_single_experiment


def run_phase2_retest():
    print("=" * 80)
    print("🔬 [V4.3 PHASE 2] Fast vs Slow Epsilon Decay 장기 학습 시드 분리 재검증")
    print("=" * 80)

    val_seeds = [301, 302, 303]
    test_seeds = [401, 402, 403]
    episodes = 100000

    print("\n[가설 검증]:")
    print("  - 단기 25k 실험에서는 Fast Decay(15k)가 탐색 노이즈를 일찍 제거하여 일시적으로 낮게 보일 수 있음.")
    print("  - 그러나 100k 장기 학습에서는 Fast Decay가 15k만에 탐색을 멈춰 후반 85k판 동안 국소해에 갇히고,")
    print("    Slow Decay(100k)가 지속적인 최적 대응(BR)을 탐색하여 더 우수한 최종 NashConv에 도달할 것이다.")

    # 1. 검증 세트 (Validation Seeds) 100k 학습
    print(f"\n▶ 1. 검증 세트 (Seeds {val_seeds}, 100,000 판) 실험 중...")

    val_fast_results = []
    val_slow_results = []

    for s in val_seeds:
        # Fast Decay (15,000)
        t0 = time.time()
        nc_f = train_single_experiment(
            episodes=episodes, eps_decay=15000, rl_lr=0.005, sl_lr=0.002,
            target_update=100, use_probs=True, use_huber=False, seed=s
        )
        val_fast_results.append(nc_f)
        t_f = time.time() - t0

        # Slow Decay (100,000)
        t0 = time.time()
        nc_s = train_single_experiment(
            episodes=episodes, eps_decay=100000, rl_lr=0.005, sl_lr=0.002,
            target_update=100, use_probs=True, use_huber=False, seed=s
        )
        val_slow_results.append(nc_s)
        t_s = time.time() - t0

        print(f"  * Val Seed {s} | Fast Decay (15k): {nc_f:.6f} ({t_f:.1f}s) | Slow Decay (100k): {nc_s:.6f} ({t_s:.1f}s)")

    val_fast_m = np.mean(val_fast_results)
    val_slow_m = np.mean(val_slow_results)

    print(f"\n[검증 세트 결과]:")
    print(f"  - Fast Decay (15k) 평균 NashConv: {val_fast_m:.6f}")
    print(f"  - Slow Decay (100k) 평균 NashConv: {val_slow_m:.6f}")
    diff = val_fast_m - val_slow_m
    if diff > 0:
        print(f"  ✅ 가설 입증! 100k 장기 학습에서는 Slow Decay가 NashConv를 {diff:+.6f} 더 낮추어 우수함.")
    else:
        print(f"  - Fast Decay 우세 (차이: {diff:+.6f})")

    # 2. 독립된 테스트 세트 (Test Seeds)에서 Slow Decay 일반화 성능 최종 확정
    print(f"\n▶ 2. 독립 테스트 세트 (Seeds {test_seeds}, 100,000 판) 일반화 성능 평가...")
    test_results = []
    for s in test_seeds:
        t0 = time.time()
        nc = train_single_experiment(
            episodes=episodes, eps_decay=100000, rl_lr=0.005, sl_lr=0.002,
            target_update=100, use_probs=True, use_huber=False, seed=s
        )
        t_el = time.time() - t0
        test_results.append(nc)
        print(f"  * Test Seed {s} | Slow Decay (100k): {nc:.6f} ({t_el:.1f}s)")

    test_m = np.mean(test_results)
    test_ci = 2.776 * (np.std(test_results, ddof=1) / np.sqrt(len(test_seeds)))
    print(f"\n[독립 테스트 세트 최종 일반화 성능]:")
    print(f"  - NashConv = {test_m:.6f} ± {test_ci:.4f} (Exploitability: {test_m*50:.2f} bb/100)")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_retest()
