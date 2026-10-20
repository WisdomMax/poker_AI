"""
POKER AI V4.1 - STEP 5 & 6: 적대적 전략 평가 및 Safe Bayesian 결합 실험
- 5. 정확한 Best Response 및 적대적 상대 7종 평가 (Nash GTO, Rock, Maniac, Station, Random, Adversarial theta*, 이전 REINFORCE)
- 6. Safe Bayesian 결합 실험:
     A. Pure NFSP
     B. Pure Safe Bayesian (GTO 기반)
     C. NFSP + Safe Bayesian 하이브리드
     상대 급변(Rock -> Maniac) 시 적응 속도 및 누적 손실 비교
"""

import sys
import os
import copy
from typing import Dict, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from v4.openspiel_adapter import DictPolicyWrapper
from kuhn_exact_solver import (
    compute_exact_exploitability,
    compute_best_response_p0,
    compute_best_response_p1,
    get_kuhn_analytical_nash
)
from v4.phase2b_exact_oracle import (
    get_rock_strategy,
    get_maniac_strategy,
    get_station_strategy,
    get_random_strategy
)
from v4.phase3_safe_bayesian import play_match, FixedStrategyBot, SafeBayesianPlayer, SwitchStrategyBot
from v4.v41_nfsp import NFSPAgent
from v4.v41_train_nfsp import combine_policies
from v4.phase4_self_play import KuhnPolicyNet, extract_full_policy_dict


def run_adversarial_and_bayesian_eval():
    print("=" * 75)
    print("⚔️ [V4.1 STEP 5] NFSP 모델 적대적 전략 전수 평가")
    print("=" * 75)

    game = pyspiel.load_game("kuhn_poker")

    # 1. NFSP 최종 모델 로드 (25,000 에피소드 체크포인트)
    p0 = NFSPAgent(0)
    p1 = NFSPAgent(1)
    p0.avg_policy_net.load_state_dict(torch.load("v4/models/nfsp_p0_avg_ckpt_25000.pt"))
    p1.avg_policy_net.load_state_dict(torch.load("v4/models/nfsp_p1_avg_ckpt_25000.pt"))
    nfsp_policy = combine_policies(p0, p1, mode="avg")

    # REINFORCE 이전 세대 모델 로드
    rf_model = KuhnPolicyNet(32)
    rf_model.load_state_dict(torch.load("v4/models/self_play_ckpt_final.pt"))
    rf_policy = extract_full_policy_dict(rf_model)

    # 2. 정확한 Best Response 계산
    exact_metrics = compute_exact_exploitability(nfsp_policy)
    br_p0_val, best_p0_strat = compute_best_response_p0(nfsp_policy)
    br_p1_val, best_p1_strat = compute_best_response_p1(nfsp_policy)

    print(f"  [NFSP 수학적 착취 취약성 (Exact Best Response Analysis)]")
    print(f"    - P0 대상 최적 대응 (br_v0): {br_p0_val:+.6f} bb/hand")
    print(f"    - P1 대상 최적 대응 (br_v1): {br_p1_val:+.6f} bb/hand")
    print(f"    - NashConv:                {exact_metrics['nash_conv']:.6f}")
    print(f"    - Exploitability:          {exact_metrics['overall_exploitability']:.6f} bb/hand ({exact_metrics['overall_exploitability']*100:.2f} bb/100)")

    # 3. 7대 상대와의 실전 경기 (각 10,000판)
    # Adversarial theta* 전략 정의 (P0/P1에 대한 최적 대응 전략 딕셔너리)
    adv_theta_star = {}
    for k in best_p0_strat:
        act = best_p0_strat[k]
        adv_theta_star[k] = (1.0 if act == 0 else 0.0, 1.0 if act == 1 else 0.0)
    for k in best_p1_strat:
        act = best_p1_strat[k]
        adv_theta_star[k] = (1.0 if act == 0 else 0.0, 1.0 if act == 1 else 0.0)

    opponents = {
        "1. Nash GTO": get_kuhn_analytical_nash(1/3),
        "2. Rock": get_rock_strategy(),
        "3. Maniac": get_maniac_strategy(),
        "4. Calling Station": get_station_strategy(),
        "5. Uniform Random": get_random_strategy(),
        "6. Adversarial θ* (Exact BR)": adv_theta_star,
        "7. 이전 세대 (REINFORCE)": rf_policy
    }

    nfsp_bot = FixedStrategyBot("NFSP Avg", nfsp_policy)

    print("\n  [상대 7종과의 10,000판 대칭 대결 성적]")
    for opp_name, opp_strat in opponents.items():
        opp_bot = FixedStrategyBot(opp_name, opp_strat)
        p0_prof, _, _ = play_match(nfsp_bot, opp_bot, 5000, seed=10)
        _, p1_prof, _ = play_match(opp_bot, nfsp_bot, 5000, seed=20)
        tot_prof = p0_prof + p1_prof
        bb_100 = (tot_prof / 10000.0) * 100.0
        print(f"    vs {opp_name:<28}: {bb_100:>+6.2f} bb/100 (총 수익: {tot_prof:>+7.1f} bb)")

    # =====================================================================
    # STEP 6. Safe Bayesian 결합 실험
    # =====================================================================
    print("\n" + "=" * 75)
    print("🤝 [V4.1 STEP 6] Safe Bayesian 결합 실험 (3종 에이전트 비교)")
    print("=" * 75)

    gto_strat = get_kuhn_analytical_nash(1/3)

    # Agent A: Pure NFSP (고정)
    agent_a = FixedStrategyBot("Pure NFSP", nfsp_policy)

    # Agent B: Safe Bayesian (GTO 사전분포 기반)
    agent_b = SafeBayesianPlayer(gto_strategy=gto_strat, max_exploit_weight=0.75)

    # Agent C: NFSP + Safe Bayesian (NFSP 정책을 사전분포 Baseline으로 결합)
    agent_c = SafeBayesianPlayer(gto_strategy=nfsp_policy, max_exploit_weight=0.75)

    # 테스트 1: 고정 상대 장기 수익 (Rock, Maniac, Station 각 5,000판)
    print("\n[테스트 1: 고정 상대 5,000판 장기 수익률 (bb/100)]")
    for opp_name in ["Rock", "Maniac", "Calling Station"]:
        opp_s = opponents[f"2. {opp_name}" if opp_name=="Rock" else (f"3. {opp_name}" if opp_name=="Maniac" else "4. Calling Station")]
        
        # A
        p0_a, _, _ = play_match(agent_a, FixedStrategyBot(opp_name, opp_s), 2500, seed=31)
        _, p1_a, _ = play_match(FixedStrategyBot(opp_name, opp_s), agent_a, 2500, seed=32)
        bb_a = ((p0_a + p1_a) / 5000.0) * 100.0

        # B
        b0 = SafeBayesianPlayer(gto_strat, max_exploit_weight=0.75)
        b1 = SafeBayesianPlayer(gto_strat, max_exploit_weight=0.75)
        p0_b, _, _ = play_match(b0, FixedStrategyBot(opp_name, opp_s), 2500, seed=31)
        _, p1_b, _ = play_match(FixedStrategyBot(opp_name, opp_s), b1, 2500, seed=32)
        bb_b = ((p0_b + p1_b) / 5000.0) * 100.0

        # C
        c0 = SafeBayesianPlayer(nfsp_policy, max_exploit_weight=0.75)
        c1 = SafeBayesianPlayer(nfsp_policy, max_exploit_weight=0.75)
        p0_c, _, _ = play_match(c0, FixedStrategyBot(opp_name, opp_s), 2500, seed=31)
        _, p1_c, _ = play_match(FixedStrategyBot(opp_name, opp_s), c1, 2500, seed=32)
        bb_c = ((p0_c + p1_c) / 5000.0) * 100.0

        print(f"  ▶ vs {opp_name:<16}: Pure NFSP={bb_a:>+6.2f} | Safe Bayesian(GTO)={bb_b:>+6.2f} | NFSP+Safe Bayesian={bb_c:>+6.2f} bb/100")

    # 테스트 2: 상대 전략 급변 (Rock 5,000판 -> Maniac 5,000판 급변 적응)
    print("\n[테스트 2: 상대 전략 급변 (Rock 5,000판 -> Maniac 5,000판) 적응력 및 전환기 손실]")
    for p_name, make_agent in [
        ("Pure NFSP", lambda: FixedStrategyBot("Pure NFSP", nfsp_policy)),
        ("Safe Bayesian (GTO)", lambda: SafeBayesianPlayer(gto_strat, max_exploit_weight=0.75)),
        ("NFSP + Safe Bayesian", lambda: SafeBayesianPlayer(nfsp_policy, max_exploit_weight=0.75))
    ]:
        ag = make_agent()
        sw_bot = SwitchStrategyBot(switch_at_hand=5000)
        prof, _, history = play_match(ag, sw_bot, 10000, seed=999)
        p_rock = history[4999]
        p_transition = history[5499] - history[4999] # 전환 직후 500판 손익
        p_maniac = history[9999] - history[4999]
        total_p = history[-1]
        print(f"  - {p_name:<22}: 전반(Rock)={(p_rock/5000)*100:>+6.2f} | 직후500판={(p_transition/500)*100:>+6.2f} | 후반(Maniac)={(p_maniac/5000)*100:>+6.2f} | 총수익={total_p:>+6.1f} bb")

    print("=" * 75)


if __name__ == "__main__":
    run_adversarial_and_bayesian_eval()
