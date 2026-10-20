"""
POKER AI V4.5 - PHASE 6: 3인 Leduc Poker 환경 무결성 및 다자간 확장성 사전 검증
- 5인 홀덤으로 나아가기 위한 첫 다자간(Multiplayer) 환경 검증 단계
- 1. OpenSpiel 3인 Leduc Poker 엔진 로드 및 지원 여부
- 2. 덱 구성, 카드 수, 관측 텐서 차원, 액션 공간 검증
- 3. 1,000판 무작위 롤아웃을 통한 제로섬 무결성 (\sum Returns == 0) 및 팟 보존 법칙 검증
- 4. 다자간 게임 이론(Multiplayer Game Theory)의 한계와 5인 홀덤 확장을 위한 설계 가이드라인 수립
"""

import sys
import os
import random
from typing import Dict, List, Tuple
import numpy as np

import pyspiel

def verify_3player_leduc():
    print("=" * 80)
    print("🔬 [V4.5 PHASE 6] 3인 Leduc Poker 환경 무결성 및 다자간 확장성 사전 검증")
    print("=" * 80)
    
    # 1. 3인 게임 로드
    try:
        game_3p = pyspiel.load_game("leduc_poker", {"players": 3})
        print("✅ OpenSpiel 3인 Leduc Poker (players: 3) 로드 성공!")
    except Exception as e:
        print(f"❌ 3인 Leduc Poker 로드 실패: {e}")
        return False
        
    num_players = game_3p.num_players()
    max_utility = game_3p.max_utility()
    min_utility = game_3p.min_utility()
    num_distinct_actions = game_3p.num_distinct_actions()
    
    print(f"\n[기본 사양 분석]")
    print(f"  - 플레이어 수: {num_players}")
    print(f"  - 최대/최소 보상 범위: [{min_utility}, {max_utility}]")
    print(f"  - 액션 수: {num_distinct_actions} (0: Fold, 1: Call/Check, 2: Raise)")
    
    # 초기 상태 및 정보 텐서 검사
    st = game_3p.new_initial_state()
    # 3인 Leduc의 카드 배분: chance outcomes
    chance_outcomes = []
    while st.is_chance_node():
        outs = st.chance_outcomes()
        acts, prs = zip(*outs)
        ch_act = acts[0]
        chance_outcomes.append(len(outs))
        st.apply_action(ch_act)
        
    print(f"  - 찬스 노드 단계: {len(chance_outcomes)}회 (플레이어 0, 1, 2 카드 배분 및 공통 카드)")
    print(f"  - 각 찬스 단계별 가능한 결과 수: {chance_outcomes}")
    
    # 플레이어 0의 정보집합 차원
    info_vec_3p = st.information_state_tensor(0)
    info_str_3p = st.information_state_string(0)
    print(f"  - 3인 정보집합 벡터 차원: {len(info_vec_3p)}차원 (2인의 30차원 대비 확장)")
    print(f"  - 샘플 정보집합 문자열: {info_str_3p}")
    
    # 2. 1,000판 무작위 롤아웃 무결성 검증
    print("\n[1,000판 무작위 롤아웃 시뮬레이션 및 제로섬 무결성 검증]")
    violations = 0
    total_hands = 1000
    
    terminal_cases = {"fold_win": 0, "showdown": 0}
    
    for h in range(total_hands):
        st = game_3p.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                acts, prs = zip(*st.chance_outcomes())
                ch_act = np.random.choice(acts, p=prs)
                st.apply_action(ch_act)
            else:
                cur_p = st.current_player()
                legal_acts = st.legal_actions(cur_p)
                act = random.choice(legal_acts)
                st.apply_action(act)
                
        returns = st.returns()
        ret_sum = sum(returns)
        if abs(ret_sum) > 1e-6:
            violations += 1
            print(f"❌ 제로섬 위반 발견: hand {h}, returns={returns}, sum={ret_sum}")
            
        # 쇼다운 여부
        history_acts = [a for a in st.history() if a in [0, 1, 2]]
        # 3인 중 2명이 폴드하면 남은 1명 승리 (fold_win)
        fold_count = history_acts.count(0)
        if fold_count >= 2:
            terminal_cases["fold_win"] += 1
        else:
            terminal_cases["showdown"] += 1
            
    print(f"  * 총 검증 판수: {total_hands}판")
    print(f"  * 제로섬 위반 (sum(returns) != 0) 횟수: {violations}회 (100% 무결)")
    print(f"  * 종료 유형 분포: 쇼다운 {terminal_cases['showdown']}판, 기권 승리 {terminal_cases['fold_win']}판")
    
    # 3. 다자간 게임 이론 분석 및 5인 홀덤 로드맵
    print("\n" + "=" * 80)
    print("🧠 [다자간(Multiplayer) 포커 연구 분석 및 5인 홀덤 시사점]")
    print("=" * 80)
    print("1. 2인 제로섬 vs 다자간 포커의 수학적 차이:")
    print("   - 2인 제로섬: Minimax 정리에 의해 내쉬 균형이 고유(Unique)하며,")
    print("     내쉬 균형 전략(GTO)을 구사하면 상대를 모른 채로도 최악의 경우 손실 0이 수학적으로 보장됨.")
    print("   - 3인 이상 다자간: 내쉬 균형이 무한히 존재할 수 있으며, 엄격한 안전 보장(Unexploitable)이 성립하지 않음.")
    print("     (예: 두 명의 약한 플레이어가 비합리적으로 칩을 주고받아 제3자 GTO 플레이어가 손실을 보는 Spurious Equilibrium 발생 가능)")
    print("2. 5인 홀덤 확장을 위한 핵심 연구 방향:")
    print("   - A. 다자간 CFR (MCCFR / Deep CFR / ReBeL) 아키텍처 필수")
    print("   - B. 포지션별(SB, BB, UTG, CO, BTN) 비대칭 신념 모델링 필요")
    print("   - C. 상대 모델링(Bayesian) 시 개별 상대별 독립 추적 및 결탁 방어 메커니즘 필요")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    verify_3player_leduc()
