"""
Kuhn Poker V3.6 - 정확한 해석적 Best Response 및 Exploitability 계산기
몬테카를로 샘플링 없이 64개 순수 전략 전수조사를 통한 100% 엄밀한 수학 계산
"""
from itertools import permutations
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES

# Kuhn Poker의 카드 6가지 순열 (각각 발생 확률 1/6)
CARD_PERMUTATIONS = list(permutations([1, 2, 3], 2))

# 정보집합 정의
P0_INFONSETS = ['J:', 'Q:', 'K:', 'J:pb', 'Q:pb', 'K:pb'] # 6개
P1_INFONSETS = ['J:p', 'Q:p', 'K:p', 'J:b', 'Q:b', 'K:b'] # 6개


def get_kuhn_analytical_nash(alpha: float = 1.0 / 3.0) -> dict[str, tuple[float, float]]:
    """
    Kuhn (1950) 논문에 정의된 무한한 내쉬 균형 전략군 (0 <= alpha <= 1/3)
    기본값 alpha = 1/3
    이 전략을 입력하면 이론상 Exploitability가 정확히 0.000이어야 함!
    """
    strat = {}
    # P0
    strat['J:'] = (1.0 - alpha, alpha) # J로 alpha 확률 블러핑
    strat['Q:'] = (1.0, 0.0)           # Q는 항상 체크
    strat['K:'] = (1.0 - 3.0 * alpha, 3.0 * alpha) # K는 3*alpha 확률 베팅
    strat['J:pb'] = (1.0, 0.0)         # J로 체크-베팅 직면 시 100% 폴드
    strat['Q:pb'] = (2.0 / 3.0 - alpha, 1.0 / 3.0 + alpha) # Q로 블러프 캐치
    strat['K:pb'] = (0.0, 1.0)         # K로 100% 콜

    # P1
    strat['J:p'] = (2.0 / 3.0, 1.0 / 3.0) # J로 상대 체크 시 1/3 블러핑
    strat['Q:p'] = (1.0, 0.0)             # Q로 상대 체크 시 100% 체크
    strat['K:p'] = (0.0, 1.0)             # K로 상대 체크 시 100% 밸류 베팅
    strat['J:b'] = (1.0, 0.0)             # J로 상대 베팅 직면 시 100% 폴드
    strat['Q:b'] = (2.0 / 3.0, 1.0 / 3.0) # Q로 상대 베팅 직면 시 1/3 콜
    strat['K:b'] = (0.0, 1.0)             # K로 상대 베팅 직면 시 100% 콜

    return strat


def evaluate_game_value(p0_strat: dict, p1_strat: dict) -> float:
    """
    P0 전략과 P1 전략이 주어졌을 때 P0의 엄밀한 기대 이득 v_0 (P1 기대 이득은 -v_0)
    6가지 카드 순열과 모든 가능한 트리 경로 확률 가중 합산
    """
    total_p0_expected = 0.0

    for c0, c1 in CARD_PERMUTATIONS:
        c0_name = CARD_NAMES[c0]
        c1_name = CARD_NAMES[c1]

        # P0 first action prob
        p0_b_prob = p0_strat.get(f"{c0_name}:", (0.5, 0.5))[BET]
        p0_p_prob = 1.0 - p0_b_prob

        # 경로 1: P0 Pass ('p')
        if p0_p_prob > 0:
            p1_b_prob = p1_strat.get(f"{c1_name}:p", (0.5, 0.5))[BET]
            p1_p_prob = 1.0 - p1_b_prob

            # 1.1: pp (쇼다운)
            if p1_p_prob > 0:
                p0_pay, _ = evaluate_payoff("pp", c0, c1)
                total_p0_expected += (1.0 / 6.0) * p0_p_prob * p1_p_prob * p0_pay

            # 1.2: pb (P1 베팅)
            if p1_b_prob > 0:
                p0_call_prob = p0_strat.get(f"{c0_name}:pb", (0.5, 0.5))[BET]
                p0_fold_prob = 1.0 - p0_call_prob

                # 1.2.1: pbp (P0 폴드)
                if p0_fold_prob > 0:
                    p0_pay, _ = evaluate_payoff("pbp", c0, c1)
                    total_p0_expected += (1.0 / 6.0) * p0_p_prob * p1_b_prob * p0_fold_prob * p0_pay

                # 1.2.2: pbb (P0 콜, 쇼다운)
                if p0_call_prob > 0:
                    p0_pay, _ = evaluate_payoff("pbb", c0, c1)
                    total_p0_expected += (1.0 / 6.0) * p0_p_prob * p1_b_prob * p0_call_prob * p0_pay

        # 경로 2: P0 Bet ('b')
        if p0_b_prob > 0:
            p1_call_prob = p1_strat.get(f"{c1_name}:b", (0.5, 0.5))[BET]
            p1_fold_prob = 1.0 - p1_call_prob

            # 2.1: bp (P1 폴드)
            if p1_fold_prob > 0:
                p0_pay, _ = evaluate_payoff("bp", c0, c1)
                total_p0_expected += (1.0 / 6.0) * p0_b_prob * p1_fold_prob * p0_pay

            # 2.2: bb (P1 콜, 쇼다운)
            if p1_call_prob > 0:
                p0_pay, _ = evaluate_payoff("bb", c0, c1)
                total_p0_expected += (1.0 / 6.0) * p0_b_prob * p1_call_prob * p0_pay

    return float(total_p0_expected)


def compute_best_response_p1(p0_strat: dict) -> tuple[float, dict[str, int]]:
    """
    P0 전략에 대한 P1의 최적 대응(Best Response) 계산 (64개 순수 전략 전수조사)
    반환값: (P1의 최대 기대 이득 v_1 = -v_0, P1의 최적 순수 전략)
    """
    best_p1_val = -999.0
    best_p1_strat = {}

    for i in range(64):
        p1_pure = {}
        for bit_idx, info_set in enumerate(P1_INFONSETS):
            action = (i >> bit_idx) & 1
            p1_pure[info_set] = (1.0 if action == PASS else 0.0, 1.0 if action == BET else 0.0)

        # P0의 기대 이득 계산 -> P1의 이득은 -v0
        v0 = evaluate_game_value(p0_strat, p1_pure)
        v1 = -v0

        if v1 > best_p1_val:
            best_p1_val = v1
            best_p1_strat = {info_set: (i >> bit_idx) & 1 for bit_idx, info_set in enumerate(P1_INFONSETS)}

    return best_p1_val, best_p1_strat


def compute_best_response_p0(p1_strat: dict) -> tuple[float, dict[str, int]]:
    """
    P1 전략에 대한 P0의 최적 대응(Best Response) 계산 (64개 순수 전략 전수조사)
    반환값: (P0의 최대 기대 이득 v_0, P0의 최적 순수 전략)
    """
    best_p0_val = -999.0
    best_p0_strat = {}

    for i in range(64):
        p0_pure = {}
        for bit_idx, info_set in enumerate(P0_INFONSETS):
            action = (i >> bit_idx) & 1
            p0_pure[info_set] = (1.0 if action == PASS else 0.0, 1.0 if action == BET else 0.0)

        v0 = evaluate_game_value(p0_pure, p1_strat)

        if v0 > best_p0_val:
            best_p0_val = v0
            best_p0_strat = {info_set: (i >> bit_idx) & 1 for bit_idx, info_set in enumerate(P0_INFONSETS)}

    return best_p0_val, best_p0_strat


def compute_exact_exploitability(full_strategy: dict) -> dict[str, float]:
    """
    Kuhn Poker에서 전략 쌍 (sigma_0, sigma_1)의 엄밀한 Exploitability 및 NashConv 계산
    
    이론적 정의:
    - v0_actual = v(sigma_0, sigma_1)
    - v1_actual = -v0_actual
    - br_v0 = max_{sigma_0'} v(sigma_0', sigma_1)  (P0가 P1을 착취할 수 있는 최대 이득)
    - br_v1 = max_{sigma_1'} -v(sigma_0, sigma_1') (P1이 P0을 착취할 수 있는 최대 이득)
    
    - P0의 Exploitability = br_v1 - v1_actual = br_v1 + v0_actual
      (이상적 GTO에서 v0_actual = -1/18, br_v1 = 1/18 이므로 Exploitability = 0!)
    - P1의 Exploitability = br_v0 - v0_actual
      (이상적 GTO에서 v0_actual = -1/18, br_v0 = -1/18 이므로 Exploitability = 0!)
      
    - NashConv = P0 Exploitability + P1 Exploitability = (br_v0 + br_v1)
    - Overall Exploitability = NashConv / 2
    """
    v0_actual = evaluate_game_value(full_strategy, full_strategy)
    
    br_v1, best_p1_strat = compute_best_response_p1(full_strategy)
    br_v0, best_p0_strat = compute_best_response_p0(full_strategy)

    p0_exploitability = br_v1 + v0_actual
    p1_exploitability = br_v0 - v0_actual
    nash_conv = p0_exploitability + p1_exploitability
    overall_exploitability = nash_conv / 2.0

    return {
        "v0_actual": v0_actual,
        "br_v0": br_v0,
        "br_v1": br_v1,
        "p0_exploitability": p0_exploitability,
        "p1_exploitability": p1_exploitability,
        "nash_conv": nash_conv,
        "overall_exploitability": overall_exploitability,
        "best_p0_strat": best_p0_strat,
        "best_p1_strat": best_p1_strat
    }
