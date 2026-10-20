"""
Poker AI V4.0 - PHASE 2: OpenSpiel 수학적 교차 검증 모듈
자체 kuhn_exact_solver vs DeepMind open_spiel.python.algorithms.exploitability
1. 이론적 Nash 전략 (NashConv = 0)
2. 균등 랜덤 정책 (NashConv = 11/12 = 0.916667)
3. CFR 평균 전략 3방향 일치 검증
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from kuhn_exact_solver import (
    get_kuhn_analytical_nash,
    compute_exact_exploitability
)
from cfr_v3 import CFRTrainerV3
from v4.openspiel_adapter import DictPolicyWrapper


def cross_validate_all():
    game = pyspiel.load_game("kuhn_poker")
    
    print("----------------------------------------------------------")
    print("1. [이론적 Kuhn Nash 전략 검증 (alpha = 1/3)]")
    print("----------------------------------------------------------")
    nash_strat = get_kuhn_analytical_nash(1.0 / 3.0)
    nash_policy_os = DictPolicyWrapper(game, nash_strat)
    
    # OpenSpiel 공식 계산
    os_nash_conv = exploitability.nash_conv(game, nash_policy_os)
    os_exploitability = exploitability.exploitability(game, nash_policy_os)
    
    # 자체 Exact Solver 계산
    exact_res = compute_exact_exploitability(nash_strat)
    
    print(f"  - OpenSpiel 공식 NashConv:       {os_nash_conv:.8f} (이론값: 0.00000000)")
    print(f"  - 자체 Solver NashConv:          {exact_res['nash_conv']:.8f} (이론값: 0.00000000)")
    print(f"  - OpenSpiel 공식 Exploitability: {os_exploitability:.8f} (이론값: 0.00000000)")
    print(f"  - 자체 Solver Exploitability:    {exact_res['overall_exploitability']:.8f} (이론값: 0.00000000)")
    
    assert abs(os_nash_conv) < 1e-6, "OpenSpiel NashConv가 0이 아닙니다!"
    assert abs(exact_res['nash_conv']) < 1e-6, "자체 Solver NashConv가 0이 아닙니다!"
    print("  ✅ 이론적 Nash 전략 교차 검증 100% 일치 (NashConv = 0.000)\n")

    print("----------------------------------------------------------")
    print("2. [균등 랜덤 정책 검증 (Uniform Random Policy)]")
    print("----------------------------------------------------------")
    # 모든 상황에서 50:50
    uniform_strat = {
        'J:': (0.5, 0.5), 'Q:': (0.5, 0.5), 'K:': (0.5, 0.5),
        'J:p': (0.5, 0.5), 'Q:p': (0.5, 0.5), 'K:p': (0.5, 0.5),
        'J:b': (0.5, 0.5), 'Q:b': (0.5, 0.5), 'K:b': (0.5, 0.5),
        'J:pb': (0.5, 0.5), 'Q:pb': (0.5, 0.5), 'K:pb': (0.5, 0.5)
    }
    uniform_policy_os = policy.UniformRandomPolicy(game)
    
    os_rand_nash_conv = exploitability.nash_conv(game, uniform_policy_os)
    os_rand_exploit = exploitability.exploitability(game, uniform_policy_os)
    exact_rand_res = compute_exact_exploitability(uniform_strat)
    
    expected_rand_nashconv = 11.0 / 12.0 # 0.91666667
    print(f"  - 이론적 기대값 (11/12):         {expected_rand_nashconv:.8f}")
    print(f"  - OpenSpiel 공식 NashConv:       {os_rand_nash_conv:.8f}")
    print(f"  - 자체 Solver NashConv:          {exact_rand_res['nash_conv']:.8f}")
    print(f"  - OpenSpiel 공식 Exploitability: {os_rand_exploit:.8f}")
    print(f"  - 자체 Solver Exploitability:    {exact_rand_res['overall_exploitability']:.8f}")
    
    assert abs(os_rand_nash_conv - expected_rand_nashconv) < 1e-5
    assert abs(exact_rand_res['nash_conv'] - expected_rand_nashconv) < 1e-5
    print("  ✅ 균등 랜덤 정책 이론값(11/12) 교차 검증 100% 일치\n")

    print("----------------------------------------------------------")
    print("3. [CFR 평균 훈련 전략 3방향 일치 검증]")
    print("----------------------------------------------------------")
    trainer = CFRTrainerV3()
    trainer.train(200000)
    cfr_strat = trainer.get_strategy_dict()
    cfr_policy_os = DictPolicyWrapper(game, cfr_strat)
    
    os_cfr_nash_conv = exploitability.nash_conv(game, cfr_policy_os)
    os_cfr_exploit = exploitability.exploitability(game, cfr_policy_os)
    exact_cfr_res = compute_exact_exploitability(cfr_strat)
    
    print(f"  - A. 자체 Exact Solver NashConv:          {exact_cfr_res['nash_conv']:.6f} (Exploitability: {exact_cfr_res['overall_exploitability']:.6f})")
    print(f"  - B. OpenSpiel 공식 Best Response NashConv: {os_cfr_nash_conv:.6f}")
    print(f"  - C. OpenSpiel 공식 Exploitability:         {os_cfr_exploit:.6f}")
    
    diff_nashconv = abs(os_cfr_nash_conv - exact_cfr_res['nash_conv'])
    diff_exploit = abs(os_cfr_exploit - exact_cfr_res['overall_exploitability'])
    
    print(f"  - 자체 vs OpenSpiel NashConv 오차:         {diff_nashconv:.8f}")
    print(f"  - 자체 vs OpenSpiel Exploitability 오차:    {diff_exploit:.8f}")
    
    assert diff_nashconv < 1e-5, f"NashConv 불일치! {diff_nashconv}"
    assert diff_exploit < 1e-5, f"Exploitability 불일치! {diff_exploit}"
    print("  ✅ CFR 모델 자체 Solver vs DeepMind OpenSpiel 3방향 100% 일치 확인!")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    print("==========================================================")
    print("🔬 [PHASE 2] DeepMind OpenSpiel 수학적 교차 검증")
    print("==========================================================")
    cross_validate_all()
