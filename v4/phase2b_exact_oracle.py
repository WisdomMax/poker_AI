"""
POKER AI V4.0 - PHASE 2-B: Exact Best Response Oracle 및 수학적 Regret >= 0 증명
- 몬테카를로 샘플링을 완전히 배제하고 64개 순수 전략 전수조사를 통한 해석적 최적 대응 계산
- 모든 고정 상대에 대해 Oracle_EV >= Agent_EV <=> Regret >= 0 불변식 검증
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from kuhn_exact_solver import (
    evaluate_game_value,
    compute_best_response_p0,
    compute_best_response_p1,
    get_kuhn_analytical_nash,
    P0_INFONSETS,
    P1_INFONSETS
)
from kuhn_engine_v3 import PASS, BET


def get_rock_strategy():
    """Rock: K일 때만 베팅/콜, 나머지는 무조건 체크/폴드"""
    strat = {}
    strat['J:'] = (1.0, 0.0); strat['Q:'] = (1.0, 0.0); strat['K:'] = (0.0, 1.0)
    strat['J:pb'] = (1.0, 0.0); strat['Q:pb'] = (1.0, 0.0); strat['K:pb'] = (0.0, 1.0)
    strat['J:p'] = (1.0, 0.0); strat['Q:p'] = (1.0, 0.0); strat['K:p'] = (0.0, 1.0)
    strat['J:b'] = (1.0, 0.0); strat['Q:b'] = (1.0, 0.0); strat['K:b'] = (0.0, 1.0)
    return strat


def get_maniac_strategy():
    """Maniac: 무조건 100% 베팅 및 콜"""
    strat = {}
    for info in P0_INFONSETS: strat[info] = (0.0, 1.0)
    for info in P1_INFONSETS: strat[info] = (0.0, 1.0)
    return strat


def get_station_strategy():
    """Calling Station: 먼저는 절대 베팅 안 함(체크), 베팅 오면 100% 콜"""
    strat = {}
    strat['J:'] = (1.0, 0.0); strat['Q:'] = (1.0, 0.0); strat['K:'] = (1.0, 0.0)
    strat['J:pb'] = (0.0, 1.0); strat['Q:pb'] = (0.0, 1.0); strat['K:pb'] = (0.0, 1.0)
    strat['J:p'] = (1.0, 0.0); strat['Q:p'] = (1.0, 0.0); strat['K:p'] = (1.0, 0.0)
    strat['J:b'] = (0.0, 1.0); strat['Q:b'] = (0.0, 1.0); strat['K:b'] = (0.0, 1.0)
    return strat


def get_random_strategy():
    """Uniform Random: 모든 상황 50% 확률"""
    strat = {}
    for info in P0_INFONSETS: strat[info] = (0.5, 0.5)
    for info in P1_INFONSETS: strat[info] = (0.5, 0.5)
    return strat


def run_phase2b_verification():
    print("=" * 65)
    print("🔬 [PHASE 2-B] Exact Best Response Oracle 및 Regret >= 0 증명")
    print("=" * 65)

    gto_strat = get_kuhn_analytical_nash(1.0 / 3.0)
    rand_strat = get_random_strategy()

    opponents = {
        "Rock": get_rock_strategy(),
        "Maniac": get_maniac_strategy(),
        "Calling Station": get_station_strategy(),
        "Uniform Random": rand_strat,
        "Nash GTO": gto_strat
    }

    all_regrets_non_negative = True

    for opp_name, opp_strat in opponents.items():
        print(f"\n--- [상대: {opp_name}] ---")

        # 1. Oracle Best Response 계산 (64개 순수 전략 전수조사)
        # P0 최적 대응 vs Opponent P1
        br_v0, best_p0 = compute_best_response_p0(opp_strat)
        # P1 최적 대응 vs Opponent P0
        br_v1, best_p1 = compute_best_response_p1(opp_strat)
        
        # Oracle의 평균 기대 이득 (P0와 P1을 50:50으로 플레이)
        oracle_avg_ev = 0.5 * (br_v0 + br_v1)

        print(f"  [Oracle (Exact Best Response)]")
        print(f"    - P0 출전 시 최대 EV (br_v0): {br_v0:+.6f} bb/hand ({br_v0 * 100:+.2f} bb/100)")
        print(f"    - P1 출전 시 최대 EV (br_v1): {br_v1:+.6f} bb/hand ({br_v1 * 100:+.2f} bb/100)")
        print(f"    - 평균 기대 이득 (Oracle_EV): {oracle_avg_ev:+.6f} bb/hand ({oracle_avg_ev * 100:+.2f} bb/100)")

        # 2. 고정 GTO 에이전트 성능 계산
        gto_v0 = evaluate_game_value(gto_strat, opp_strat)       # GTO가 P0, 상대가 P1
        gto_v1 = -evaluate_game_value(opp_strat, gto_strat)      # 상대가 P0, GTO가 P1
        gto_avg_ev = 0.5 * (gto_v0 + gto_v1)
        gto_regret = oracle_avg_ev - gto_avg_ev

        print(f"  [Agent: Nash GTO]")
        print(f"    - P0 EV: {gto_v0:+.6f}, P1 EV: {gto_v1:+.6f} -> 평균: {gto_avg_ev:+.6f} bb/hand ({gto_avg_ev * 100:+.2f} bb/100)")
        print(f"    - Regret (Oracle - GTO): {gto_regret:+.6f} bb/hand ({gto_regret * 100:+.2f} bb/100)")

        # 3. Random 에이전트 성능 계산
        rand_v0 = evaluate_game_value(rand_strat, opp_strat)
        rand_v1 = -evaluate_game_value(opp_strat, rand_strat)
        rand_avg_ev = 0.5 * (rand_v0 + rand_v1)
        rand_regret = oracle_avg_ev - rand_avg_ev

        print(f"  [Agent: Uniform Random]")
        print(f"    - P0 EV: {rand_v0:+.6f}, P1 EV: {rand_v1:+.6f} -> 평균: {rand_avg_ev:+.6f} bb/hand ({rand_avg_ev * 100:+.2f} bb/100)")
        print(f"    - Regret (Oracle - Rand): {rand_regret:+.6f} bb/hand ({rand_regret * 100:+.2f} bb/100)")

        # 수학적 검증: Regret >= -1e-12 이어야 함 (부동소수점 오차 허용)
        if gto_regret < -1e-9 or rand_regret < -1e-9:
            print("  ❌ 오류: 음수 후회(Negative Regret) 발생! 불변식 위반!")
            all_regrets_non_negative = False
        else:
            print(f"  ✅ Regret >= 0 수학적 불변식 완벽 성립 (오차 0)")

    print("\n" + "=" * 65)
    if all_regrets_non_negative:
        print("🎯 [Phase 2-B 결과]: 모든 고정 상대에 대해 Oracle_EV >= Agent_EV (Regret >= 0)")
        print("   V3.6의 몬테카를로 오라클 표본 오차로 인한 음수 후회 문제가 완전히 해결되었습니다.")
    else:
        print("❌ [Phase 2-B 결과]: 검증 실패")
    print("=" * 65)


if __name__ == "__main__":
    run_phase2b_verification()
