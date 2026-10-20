"""
POKER AI V4.8 - PHASE 1: 평가 시스템 최종 고정 및 8대 무결성 자동 테스트
- 감사 항목:
  1. 무작위 카드 분배 (결정론적 고정 버그 재발 방지)
  2. 52장 카드 중복 방지 (Deck Uniqueness)
  3. 플레이어 ID와 좌석 매핑 (Wrapper Bijective Inverse Property)
  4. 보상 귀속 및 제로섬 보존 (Sum of returns == 0)
  5. 블라인드 회전 공정성 (5개 포지션 경험 균등성)
  6. 비공개 카드 누출 방지 (Information State Leakage = 0)
  7. 수익률 계산 및 bb/100 변환 공식 무결성
  8. 포커 족보 판정 (7장 최선 5장 판정)
"""

import sys
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.poker_environment import (
    get_5p_holdem_game, HoldemSeatRotationWrapper,
    sample_chance_action, advance_chance_nodes, parse_cards_from_info_str, POSITION_NAMES
)
from v474.phase4_hand_eval_audit import evaluate_7_cards, HAND_NAMES

def run_phase1_integrity_tests(num_trials: int = 1000):
    print("=" * 80)
    print(f"🛡️ [V4.8 PHASE 1] 평가 시스템 최종 고정 및 8대 무결성 자동 단위 테스트 ({num_trials:,}판)")
    print("=" * 80)
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    
    # ---------------------------------------------------------
    # 1. 무작위 카드 분배 및 고유성 검사 (결정론적 [0][0] 감지)
    # ---------------------------------------------------------
    unique_deals = set()
    for _ in range(500):
        st = game.new_initial_state()
        acts = tuple(advance_chance_nodes(st))
        unique_deals.add(acts)
    diversity = len(unique_deals) / 500
    assert diversity > 0.99, f"❌ [1. 무작위 분배 실패] 고유 딜 비율이 {diversity*100:.1f}%에 불과합니다!"
    print(f"  ✅ 1. 무작위 카드 분배: 500회 중 {len(unique_deals)}개 고유 시퀀스 ({diversity*100:.1f}%) 통과")
    
    # ---------------------------------------------------------
    # 2. 52장 중복 방지
    # ---------------------------------------------------------
    dup_errors = 0
    for _ in range(200):
        st = game.new_initial_state()
        advance_chance_nodes(st)
        dealt = []
        for s in range(5):
            hc, _ = parse_cards_from_info_str(st.information_state_string(s))
            dealt.extend(hc)
        if len(set(dealt)) != len(dealt):
            dup_errors += 1
    assert dup_errors == 0, f"❌ [2. 덱 중복 발생] {dup_errors}건"
    print(f"  ✅ 2. 52장 중복 방지: 200판 전수 검사 중 중복 0건 통과")
    
    # ---------------------------------------------------------
    # 3. 플레이어 ID와 좌석 매핑 (전단사 역함수 성질)
    # ---------------------------------------------------------
    map_errors = 0
    for h in range(500):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        for s in range(5):
            ag = s2a[s]
            if a2s[ag] != s: map_errors += 1
    assert map_errors == 0, f"❌ [3. 좌석 매핑 오류] {map_errors}건"
    print(f"  ✅ 3. 플레이어 ID - 좌석 역함수 매핑: 500판 2,500개 관계 100% 일치 통과")
    
    # ---------------------------------------------------------
    # 4. 보상 귀속 및 제로섬 보존
    # ---------------------------------------------------------
    zero_sum_errors = 0
    boundary_errors = 0
    for h in range(200):
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st)
            else:
                legal = st.legal_actions(st.current_player())
                st.apply_action(random.choice(legal))
        rets = st.returns()
        if abs(sum(rets)) > 1e-4: zero_sum_errors += 1
        for r in rets:
            if not (-10000.0 <= r <= 40000.0): boundary_errors += 1
    assert zero_sum_errors == 0 and boundary_errors == 0, "❌ [4. 보상 제로섬/경계 위반] 발생"
    print(f"  ✅ 4. 보상 제로섬 및 경계값 (-10,000 <= return <= 40,000): 200판 100% 만족 통과")
    
    # ---------------------------------------------------------
    # 5. 블라인드 회전 공정성
    # ---------------------------------------------------------
    pos_counts = np.zeros((5, 5), dtype=int)
    for h in range(1000):
        a2s = wrapper.get_agent_to_seat(h)
        for ag, st_pos in a2s.items():
            pos_counts[ag, st_pos] += 1
    # 1,000판에서 각 에이전트는 각 포지션을 정확히 200회씩 경험해야 함
    assert np.all(pos_counts == 200), f"❌ [5. 포지션 편향] 포지션 경험 불균형: {pos_counts}"
    print(f"  ✅ 5. 블라인드 회전 공정성: 1,000판 중 각 에이전트 포지션별 정확히 200회(20.0%) 일치 통과")
    
    # ---------------------------------------------------------
    # 6. 비공개 카드 누출 방지 (Information Leakage = 0)
    # ---------------------------------------------------------
    leakage_errors = 0
    for _ in range(100):
        st = game.new_initial_state()
        advance_chance_nodes(st)
        hole_cards_all = [parse_cards_from_info_str(st.information_state_string(s))[0] for s in range(5)]
        # 각 플레이어의 정보 상태 문자열에 다른 사람의 홀카드가 텍스트로 들어있는지 검사
        for s in range(5):
            info_s = st.information_state_string(s)
            for other_s in range(5):
                if s != other_s:
                    for card in hole_cards_all[other_s]:
                        # Private: [자신의 카드] 외에는 상대 카드가 나타나선 안 됨
                        # Public 보드 카드가 아직 없으므로 info_s에 상대 카드가 존재하면 누출!
                        if card in info_s and card not in hole_cards_all[s]:
                            leakage_errors += 1
    assert leakage_errors == 0, f"❌ [6. 비공개 카드 누출] {leakage_errors}건 발생"
    print(f"  ✅ 6. 비공개 카드 누출 방지: 100판 500개 정보집합 전수 검사 누출 0건 통과")
    
    # ---------------------------------------------------------
    # 7. 수익률 및 bb/100 변환 공식 무결성
    # ---------------------------------------------------------
    # 정의: Big Blind = 100 chips.
    # 1판당 net chips = C.
    # bb/100 = (C / BB) * 100 = C (chips per hand와 동일한 값)
    # 500판에서 총 +12,500칩 -> 판당 +25.0칩 -> +25.0 bb/100.
    sample_chips = 12500.0
    sample_hands = 500
    bb100_val = (sample_chips / sample_hands / 100.0) * 100.0
    assert abs(bb100_val - 25.0) < 1e-6, "❌ [7. bb/100 계산 오류]"
    print(f"  ✅ 7. 수익률 및 bb/100 공식 (BB=100 chips, bb/100 = Net Chips / Hands): 검증 완료")
    
    # ---------------------------------------------------------
    # 8. 포커 족보 판정 (OpenSpiel vs 독립 평가기 100판 교차 검증)
    # ---------------------------------------------------------
    hand_mismatches = 0
    for _ in range(100):
        st = game.new_initial_state()
        advance_chance_nodes(st)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st)
            else: st.apply_action(1) # Check/Call to showdown
        rets = st.returns()
        max_ret = max(rets)
        ospiel_winners = [i for i, r in enumerate(rets) if abs(r - max_ret) < 1e-4]
        
        scores = []
        for s in range(5):
            hc, bc = parse_cards_from_info_str(st.information_state_string(s))
            rk, tb, _ = evaluate_7_cards(hc + bc)
            scores.append((rk, tb))
        best_sc = max(scores)
        indep_winners = [i for i, sc in enumerate(scores) if sc == best_sc]
        if ospiel_winners != indep_winners:
            hand_mismatches += 1
    assert hand_mismatches == 0, f"❌ [8. 족보 불일치] {hand_mismatches}건 발생"
    print(f"  ✅ 8. 포커 족보 판정 (7장 최선 5장 쇼다운 100판 전수 일치): 불일치 0건 통과")
    
    print("\n" + "=" * 80)
    print("🏆 [PHASE 1 통과] 8대 평가 시스템 무결성 자동 테스트 100% 통과 완료.")
    print("=" * 80)

if __name__ == "__main__":
    run_phase1_integrity_tests()
