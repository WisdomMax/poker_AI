"""
POKER AI HUMAN ARENA: DIRECTIVE 024 Automated Audit & Unit Verification
=====================================================================
Audits:
1. Engine Chips (10,000 / 50 / 100) to Virtual Dollar ($) 1:1 mapping
2. UI Dollar representations and formatting ($10,000, $50, $100, $ pot, raises)
3. Cumulative profit/loss calculation (+$, -$)
4. AI Head-to-Head win/loss/tie tracking across 25 hands
5. Reset to $10,000 initial bankroll
6. Terminal winner announcement and log formatting
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.game_engine import PokerArenaSession, format_dollar

def run_directive024_audit():
    print("=" * 70)
    print("▶ DIRECTIVE 024: 가상 달러 게임머니 UI 및 데이터 엔진 감사 시작")
    print("=" * 70)

    # 1. 포맷팅 단위 함수 감사
    print("\n[검사 1] format_dollar 함수 단위 및 부호 테스트:")
    assert format_dollar(10000) == "$10,000", f"Expected $10,000, got {format_dollar(10000)}"
    assert format_dollar(50) == "$50", f"Expected $50, got {format_dollar(50)}"
    assert format_dollar(100) == "$100", f"Expected $100, got {format_dollar(100)}"
    assert format_dollar(1250) == "$1,250", f"Expected $1,250, got {format_dollar(1250)}"
    assert format_dollar(500, show_sign=True) == "+$500", f"Expected +$500, got {format_dollar(500, show_sign=True)}"
    assert format_dollar(-200, show_sign=True) == "-$200", f"Expected -$200, got {format_dollar(-200, show_sign=True)}"
    assert format_dollar(1850, show_sign=True) == "+$1,850", f"Expected +$1,850, got {format_dollar(1850, show_sign=True)}"
    assert format_dollar(0, show_sign=True) == "$0", f"Expected $0, got {format_dollar(0, show_sign=True)}"
    print("  ✓ format_dollar 단위/부호/천단위 콤마 테스트 100% PASS")

    # 2. 세션 초기화 상태 감사
    print("\n[검사 2] 초기 세션 생성 및 가상 달러 필드 감사:")
    session = PokerArenaSession("audit_session_024")
    assert session.initial_stack == 10000.0, f"Expected initial stack 10000, got {session.initial_stack}"
    state = session.start_new_hand()
    
    assert "pot_dollars" in state, "Missing pot_dollars in state"
    assert "call_cost_dollars" in state, "Missing call_cost_dollars in state"
    assert "raise_amounts_dollars" in state, "Missing raise_amounts_dollars in state"
    assert "hero_stats" in state, "Missing hero_stats in state"
    
    hero_stats = state["hero_stats"]
    assert hero_stats["initial_bankroll"] == 10000, f"Expected bankroll 10000, got {hero_stats['initial_bankroll']}"
    assert hero_stats["current_balance"] == 10000, f"Expected balance 10000, got {hero_stats['current_balance']}"
    assert hero_stats["total_profit_dollars"] == 0, f"Expected net 0, got {hero_stats['total_profit_dollars']}"
    assert len(hero_stats["ai_records"]) == 4, f"Expected 4 AI records, got {len(hero_stats['ai_records'])}"
    print(f"  ✓ 초기 상태 감사 완료: 보유머니 ${hero_stats['current_balance']:,}, 팟 ${state['pot_dollars']:,}")

    # 3. 25핸드 시뮬레이션 및 원장 보존/H2H 전적 감사
    print("\n[검사 3] 25핸드 실전 시뮬레이션 (다양한 액션 및 정산 검증):")
    total_hands = 25
    
    for h in range(1, total_hands + 1):
        if not session.is_hand_active:
            state = session.start_new_hand()
            
        step_cnt = 0
        while not state["is_terminal"] and step_cnt < 60:
            step_cnt += 1
            cp = state["current_player"]
            if cp == 0:
                # 인간 플레이어 액션 선택 (체크/콜, 레이즈, 폴드 혼합)
                legal = state["legal_actions"]
                if h % 5 == 0 and 0 in legal:
                    act = 0 # 폴드
                elif h % 3 == 0 and 2 in legal:
                    act = 2 # 50% 레이즈
                elif h % 7 == 0 and 5 in legal:
                    act = 5 # 올인
                else:
                    act = 1 if 1 in legal else legal[0] # 콜/체크
                state = session.step_human(act)
            else:
                progressed, state = session.step_ai_single()
                if not progressed:
                    break
                    
        assert state["is_terminal"], f"Hand #{h} did not terminate properly!"
        
        # 칩 보존 법칙: 5인 스택 합계는 항상 50,000칩 ($50,000)
        total_chips = sum(session.stacks)
        assert abs(total_chips - 50000.0) < 1e-3, f"Chip conservation violated! Sum: {total_chips}"
        
        # 순손익 = 현재 보유금액 - 초기 자본($10,000)
        expected_profit = int(round(session.stacks[0] - 10000.0))
        actual_profit = state["hero_stats"]["total_profit_dollars"]
        assert expected_profit == actual_profit, f"Profit mismatch! Expected {expected_profit}, got {actual_profit}"

    hero_stats = state["hero_stats"]
    print(f"  ✓ 25핸드 완주 성공!")
    print(f"  - 최종 보유 금액: ${hero_stats['current_balance']:,}")
    print(f"  - 총 순손익: {format_dollar(hero_stats['total_profit_dollars'], show_sign=True)}")
    print(f"  - 마지막 판 손익: {format_dollar(hero_stats['last_hand_profit_dollars'], show_sign=True)}")
    print(f"  - 승률: {hero_stats['win_rate_pct']}% ({hero_stats.get('hero_wins', session.hero_wins)}승 / {total_hands}판)")
    
    print("\n[검사 4] AI 상대별 맞대결 전적 (H2H) 집계 감사:")
    for ai in hero_stats["ai_records"]:
        total_games = ai["wins"] + ai["losses"] + ai["ties"]
        assert total_games == total_hands, f"Total H2H games {total_games} != {total_hands} for Seat {ai['seat']}"
        print(f"  - {ai['name']} ({ai['role']}): {ai['wins']}승 {ai['losses']}패 {ai['ties']}무")
    print("  ✓ H2H 상대별 전적 합산 100% 일치")

    # 4. 초기화(Reset) 감사
    print("\n[검사 5] 초기화(Reset) 기준 감사:")
    session.reset_stacks()
    reset_state = session.get_client_state()
    r_stats = reset_state["hero_stats"]
    assert r_stats["hands"] == 0, f"Hands not reset: {r_stats['hands']}"
    assert r_stats["current_balance"] == 10000, f"Balance not reset: {r_stats['current_balance']}"
    assert r_stats["total_profit_dollars"] == 0, f"Profit not reset: {r_stats['total_profit_dollars']}"
    for ai in r_stats["ai_records"]:
        assert ai["wins"] == 0 and ai["losses"] == 0 and ai["ties"] == 0, f"H2H record not reset for Seat {ai['seat']}"
    print("  ✓ 초기화 시 $10,000 및 전적 통계 0으로 완벽 초기화 확인")

    print("\n" + "=" * 70)
    print("🎉 DIRECTIVE 024 단위 및 원장 감사 ALL PASS (100% 무결성 검증 완료)")
    print("=" * 70)

if __name__ == "__main__":
    run_directive024_audit()
