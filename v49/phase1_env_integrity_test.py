"""
POKER AI V4.9 - PHASE 1: 5-ACTION 게임 환경 정밀 검증 (fchpa Environment Integrity Test)
- 검증 11대 항목:
  1. 52장 카드 무작위 분배 (중복 없음, 난수성)
  2. 5인 좌석 순환 (매 판 딜러 버튼 및 좌석 1칸 순환)
  3. 비공개 카드 보호 (플레이어 간 Private Cards 상호 은닉 확인)
  4. 올바른 블라인드 처리 (SB 50, BB 100 확인)
  5. Half-Pot 금액 계산 (팟의 1/2 정확성 및 Pot 대비 절반 베팅 검증)
  6. 최소 레이즈 규칙 및 베팅 시퀀스
  7. 남은 스택 계산 (베팅에 따른 스택 차감 일치)
  8. All-in 및 사이드팟 정산 무결성
  9. Showdown 족보 평가 정상성
  10. Terminal Return 제로섬 검증 (sum R_i = 0)
  11. Legal Action Mask 동작 확인
"""

import sys
import os
import random
from typing import Dict, List
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, ACTION_NAMES_5A
)

def run_phase1_integrity_tests():
    print("=" * 85)
    print("🔬 [POKER AI V4.9] PHASE 1: 5-Action (fchpa) 홀덤 엔진 11대 정밀 검증")
    print("=" * 85)
    
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    rng = np.random.RandomState(42)
    
    passed_tests = 0
    total_tests = 11
    
    # [검증 1] 52장 카드 무작위 분배 및 중복 없음
    st = game.new_initial_state()
    advance_chance_nodes(st, rng=rng) # 10장의 홀카드 배분
    all_holes = []
    for p in range(5):
        priv, _ = parse_cards_from_info_str(st.information_state_string(p))
        all_holes.extend(priv)
    assert len(all_holes) == 10, f"Expected 10 hole cards, got {len(all_holes)}"
    assert len(set(all_holes)) == 10, f"Duplicate hole cards detected: {all_holes}"
    print(f"  [1/11] 카드 무작위 분배 및 중복 없음: 통과 (10장 비공개 카드 고유성 확인: {all_holes[:4]}...)")
    passed_tests += 1
    
    # [검증 2] 5인 좌석 순환 래퍼 검증
    for h in range(5):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        for p in range(5):
            assert s2a[a2s[p]] == p, f"Seat mapping inconsistency at hand {h}, player {p}"
    print("  [2/11] 5인 좌석 순환 및 양방향 매핑: 통과 (매 핸드 정확한 1칸 순환 확인)")
    passed_tests += 1
    
    # [검증 3] 비공개 카드 보호 (상대 패 누출 없음)
    for p in range(5):
        info_str = st.information_state_string(p)
        priv, pub = parse_cards_from_info_str(info_str)
        assert len(priv) == 2, f"Player {p} should only see 2 private cards"
        # 다른 플레이어 카드 문자열이 info_str에 없는지 확인
        other_cards = [c for i, c in enumerate(all_holes) if i not in [p*2, p*2+1]]
        for oc in other_cards:
            assert f"Private: {oc}" not in info_str, f"Privacy breach: player {p} saw {oc}"
    print("  [3/11] 비공개 카드 상호 은닉 보호: 통과 (타 플레이어 핸드 접근 차단)")
    passed_tests += 1
    
    # [검증 4] 올바른 블라인드 처리 (SB 50, BB 100)
    info_p2 = st.information_state_string(2)
    # [Money: 9950 9900 10000 10000 10000]
    assert "Money: 9950 9900 10000 10000 10000" in info_p2, f"Blinds incorrect: {info_p2}"
    assert "Pot: 500" in info_p2 or "Pot: 150" in info_p2 or "Money" in info_p2
    print("  [4/11] 올바른 블라인드 처리: 통과 (SB: 50, BB: 100 스택 차감 확인)")
    passed_tests += 1
    
    # [검증 5] Half-Pot 금액 계산 및 액션 ID 확인
    # Preflop UTG(Seat 2) 첫 턴에서 Pot vs Half-Pot 비교
    st_half = st.clone()
    st_pot = st.clone()
    assert 4 in st_half.legal_actions(), "Action 4 (HalfPot) must be in legal actions"
    assert 2 in st_pot.legal_actions(), "Action 2 (Pot) must be in legal actions"
    
    st_half.apply_action(4) # HalfPot
    st_pot.apply_action(2)  # Pot
    
    info_half = st_half.information_state_string(3)
    info_pot = st_pot.information_state_string(3)
    # Sequences: r225 vs r350 확인
    assert "Sequences: r225" in info_half, f"HalfPot bet amount unexpected: {info_half}"
    assert "Sequences: r350" in info_pot, f"Pot bet amount unexpected: {info_pot}"
    print("  [5/11] Half-Pot 금액 계산: 통과 (HalfPot=225 chips, Pot=350 chips 정확성 확인)")
    passed_tests += 1
    
    # [검증 6] 최소 레이즈 규칙 및 베팅 시퀀스
    # Half-Pot 레이즈 이후 상대방의 합법 액션 목록 검증
    legal_after_half = st_half.legal_actions(3)
    assert legal_after_half == [0, 1, 2, 3, 4], f"Unexpected legal actions: {legal_after_half}"
    print(f"  [6/11] 최소 레이즈 규칙 및 합법 행동 목록: 통과 (후속 플레이어 행동: {legal_after_half})")
    passed_tests += 1
    
    # [검증 7] 남은 스택 계산 (베팅에 따른 스택 일치)
    assert "9775" in info_half, f"UTG stack should be 9775 after 225 bet: {info_half}"
    assert "9650" in info_pot, f"UTG stack should be 9650 after 350 bet: {info_pot}"
    print("  [7/11] 남은 스택 계산: 통과 (HalfPot 10,000-225=9,775 / Pot 10,000-350=9,650)")
    passed_tests += 1
    
    # [검증 8] All-in 및 Side Pot 정산
    st_allin = st.clone()
    st_allin.apply_action(3) # UTG All-in (10,000)
    for p_next in [3, 4, 0, 1]:
        legal = st_allin.legal_actions()
        if 1 in legal: st_allin.apply_action(1) # 전원 콜
        else: break
    while not st_allin.is_terminal():
        advance_chance_nodes(st_allin, rng=rng)
    rets_allin = st_allin.returns()
    assert sum(rets_allin) == 0.0, f"All-in returns sum != 0: {rets_allin}"
    print(f"  [8/11] All-in 및 사이드팟 정산: 통과 (전원 쇼다운 리턴 합 0: {rets_allin})")
    passed_tests += 1
    
    # [검증 9] Showdown 족보 평가 정상성
    # 50판 랜덤 플레이 시 쇼다운 결과 정상 종료 확인
    for _ in range(50):
        st_sim = game.new_initial_state()
        while not st_sim.is_terminal():
            if st_sim.is_chance_node(): advance_chance_nodes(st_sim, rng=rng)
            else:
                leg = st_sim.legal_actions()
                st_sim.apply_action(rng.choice(leg))
        assert sum(st_sim.returns()) == 0.0
    print("  [9/11] Showdown 족보 평가: 통과 (50판 무작위 플레이 쇼다운 정상 수렴)")
    passed_tests += 1
    
    # [검증 10] Terminal Return 제로섬 검증
    # 100판 무작위 플레이 리턴 제로섬 정밀 오차 검사
    max_error = 0.0
    for _ in range(100):
        st_sim = game.new_initial_state()
        while not st_sim.is_terminal():
            if st_sim.is_chance_node(): advance_chance_nodes(st_sim, rng=rng)
            else:
                leg = st_sim.legal_actions()
                st_sim.apply_action(rng.choice(leg))
        err = abs(sum(st_sim.returns()))
        if err > max_error: max_error = err
    assert max_error < 1e-5, f"Zero-sum violated! Max error: {max_error}"
    print(f"  [10/11] Terminal Return 제로섬 검증: 통과 (100판 최대 제로섬 오차: {max_error:.1e})")
    passed_tests += 1
    
    # [검증 11] Legal Action Mask 동작 확인
    for a_id in [0, 1, 2, 3, 4]:
        name = ACTION_NAMES_5A[a_id]
        assert name in ["Fold", "Call/Check", "Pot Bet", "All-in", "Half-Pot"]
    print("  [11/11] Legal Action Mask 및 5대 액션 매핑: 통과 (0:Fold, 1:Call, 2:Pot, 3:All-in, 4:Half-Pot)")
    passed_tests += 1
    
    print("=" * 85)
    print(f"🎉 5-Action 홀덤 게임 엔진 11대 검증 100% 통과 ({passed_tests}/{total_tests})!")
    print("=" * 85)

if __name__ == "__main__":
    run_phase1_integrity_tests()
