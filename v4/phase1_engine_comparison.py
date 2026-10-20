"""
Poker AI V4.0 - PHASE 1: 공식 연구 코드 도입 및 엔진 1:1 비교 검증
자체 kuhn_engine_v3 vs Google DeepMind OpenSpiel kuhn_poker
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
from itertools import permutations
from kuhn_engine_v3 import evaluate_payoff, CARD_NAMES

def compare_all_terminal_payoffs():
    """
    모든 6개 카드 배분 x 5개 종료 경로에 대해
    자체 evaluate_payoff와 OpenSpiel state.returns()를 1:1 전수 비교
    """
    game = pyspiel.load_game("kuhn_poker")
    
    # 5개 종료 경로와 액션 시퀀스 (0=Pass, 1=Bet)
    # pp: Pass(0), Pass(0)
    # bp: Bet(1), Pass(0)
    # bb: Bet(1), Bet(1)
    # pbp: Pass(0), Bet(1), Pass(0)
    # pbb: Pass(0), Bet(1), Bet(1)
    paths = [
        ("pp", [0, 0]),
        ("bp", [1, 0]),
        ("bb", [1, 1]),
        ("pbp", [0, 1, 0]),
        ("pbb", [0, 1, 1]),
    ]
    
    # 카드 순열 (OpenSpiel 0, 1, 2 <-> 자체 1, 2, 3)
    card_perms = list(permutations([0, 1, 2], 2))
    
    total_checks = 0
    mismatches = []
    
    for os_c0, os_c1 in card_perms:
        v3_c0 = os_c0 + 1
        v3_c1 = os_c1 + 1
        
        for path_name, actions in paths:
            total_checks += 1
            
            # OpenSpiel 시뮬레이션
            state = game.new_initial_state()
            state.apply_action(os_c0) # Deal P0
            state.apply_action(os_c1) # Deal P1
            for a in actions:
                state.apply_action(a)
                
            os_returns = state.returns()
            os_p0_pay = float(os_returns[0])
            os_p1_pay = float(os_returns[1])
            
            # 자체 V3 evaluate_payoff
            v3_p0_pay, v3_p1_pay = evaluate_payoff(path_name, v3_c0, v3_c1)
            
            if os_p0_pay != float(v3_p0_pay) or os_p1_pay != float(v3_p1_pay):
                mismatches.append({
                    "cards": (v3_c0, v3_c1),
                    "path": path_name,
                    "openspiel": (os_p0_pay, os_p1_pay),
                    "v3": (v3_p0_pay, v3_p1_pay)
                })

    print(f"총 {total_checks}개 경로 전수조사 완료.")
    if mismatches:
        print(f"❌ 불일치 발견! ({len(mismatches)}개)")
        for m in mismatches:
            print(m)
        return False
    else:
        print(f"✅ 자체 V3 엔진과 OpenSpiel 공식 엔진의 보상(Returns) 100% 일치 확인 ({total_checks}/{total_checks})")
        return True


if __name__ == "__main__":
    print("==========================================================")
    print("🔬 [PHASE 1] 자체 V3 엔진 vs DeepMind OpenSpiel 1:1 비교 검증")
    print("==========================================================")
    success = compare_all_terminal_payoffs()
    if not success:
        sys.exit(1)
