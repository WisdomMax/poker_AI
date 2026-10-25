#!/usr/bin/env python3
"""
Unit & Regression Tests for DIRECTIVE 041 Integrity Poker Metrics
Verifies:
1. VPIP based on actual dollar spent (BB free check excluded, voluntary call/raise included).
2. Separation of Open/Push All-in vs Facing All-in Call.
3. Uncalled Push All-in must have zero showdowns and 100% win rate.
4. Direct Action 5 vs Effective Stack All-in distinction.
5. 3-Bet Opportunity-based calculation.
"""
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from v1024.integrity_poker_metrics import HandIntegrityTracker, STARTING_STACK

def test_bb_free_check_vpip():
    """BB 포지션에서 아무도 레이즈하지 않아 무료 체크한 경우: VPIP는 반드시 False여야 함"""
    tracker = HandIntegrityTracker(agent_seat=1, seed=100) # BB seat = 1, forced_blind = 100.0
    
    # Preflop: Action 1 (Check), to_call = 0, spent_before = 100, spent_after = 100
    tracker.record_agent_decision(
        round_idx=0, act_6=1, nat_act=1, cur_to_call=0.0,
        spent_before=100.0, spent_after=100.0,
        cur_opp_allin_active=False, opp_allin_seat=None
    )
    
    # Flop: Action 1 (Check)
    tracker.record_agent_decision(
        round_idx=1, act_6=1, nat_act=1, cur_to_call=0.0,
        spent_before=100.0, spent_after=100.0,
        cur_opp_allin_active=False, opp_allin_seat=None
    )
    
    res = tracker.finalize_hand(
        final_spent=[100.0, 100.0, 0.0, 0.0, 0.0],
        returns=[-100.0, 100.0, 0.0, 0.0, 0.0],
        priv_cards={1: ("Ah", "Kd")},
        board_cards=["Qh", "Jc", "Ts", "2d", "3s"]
    )
    
    assert res["vpip"] is False, "BB free check must NOT be counted as VPIP!"
    assert res["pfr"] is False
    assert res["pnl_bb"] == 1.0
    print("[PASS] test_bb_free_check_vpip: BB 무료 체크는 VPIP에서 완벽 제외 확인.")

def test_bb_paid_call_vpip():
    """BB 포지션에서 상대 레이즈($300)에 $200을 더 내고 콜한 경우: VPIP는 반드시 True여야 함"""
    tracker = HandIntegrityTracker(agent_seat=1, seed=101) # BB seat = 1
    
    # 상대가 $300으로 레이즈
    tracker.record_opponent_decision(cp=2, round_idx=0, act_6=2, cur_to_call=0.0, spent_before=0.0, spent_after=300.0)
    
    # Agent BB: Action 1 (Call), to_call = 200, spent_before = 100, spent_after = 300
    tracker.record_agent_decision(
        round_idx=0, act_6=1, nat_act=1, cur_to_call=200.0,
        spent_before=100.0, spent_after=300.0,
        cur_opp_allin_active=False, opp_allin_seat=None
    )
    
    res = tracker.finalize_hand(
        final_spent=[50.0, 300.0, 300.0, 0.0, 0.0],
        returns=[-50.0, -300.0, 350.0, 0.0, 0.0],
        priv_cards={1: ("Th", "9h")},
        board_cards=["8h", "7c", "2s", "3d", "4s"]
    )
    
    assert res["vpip"] is True, "BB paying extra call must be counted as VPIP!"
    assert res["pfr"] is False
    print("[PASS] test_bb_paid_call_vpip: BB 유료 콜은 VPIP에 정상 반영 확인.")

def test_uncalled_push_allin_integrity():
    """에이전트가 먼저 올인(Push)을 하고 상대가 모두 폴드한 경우: 쇼다운 0건, 반드시 양수 이익이어야 함"""
    tracker = HandIntegrityTracker(agent_seat=0, seed=102) # SB seat = 0
    
    # Agent pushes all-in (Action 5)
    tracker.record_agent_decision(
        round_idx=0, act_6=5, nat_act=5, cur_to_call=50.0,
        spent_before=50.0, spent_after=STARTING_STACK,
        cur_opp_allin_active=False, opp_allin_seat=None
    )
    
    # All opponents fold
    for opp_seat in [1, 2, 3, 4]:
        tracker.record_opponent_decision(cp=opp_seat, round_idx=0, act_6=0, cur_to_call=10000.0, spent_before=0.0, spent_after=0.0)
        
    res = tracker.finalize_hand(
        final_spent=[STARTING_STACK, 100.0, 0.0, 0.0, 0.0],
        returns=[100.0, -100.0, 0.0, 0.0, 0.0],
        priv_cards={0: ("As", "Ah")},
        board_cards=[]
    )
    
    assert res["pushed_allin"] is True
    assert res["pushed_allin_called"] is False
    assert res["called_allin"] is False
    assert res["showdown"] is False
    assert res["pnl_bb"] > 0.0
    print("[PASS] test_uncalled_push_allin_integrity: 콜 없는 푸시 올인은 쇼다운 0건 및 100% 수익성 확인.")

def test_facing_allin_call_separation():
    """상대가 먼저 올인했고 에이전트가 콜(Action 1)하여 올인된 경우: pushed_allin=False, called_allin=True여야 함"""
    tracker = HandIntegrityTracker(agent_seat=2, seed=103)
    
    # 상대(Seat 3)가 먼저 전액 올인 ($10,000)
    tracker.record_opponent_decision(cp=3, round_idx=0, act_6=5, cur_to_call=0.0, spent_before=0.0, spent_after=STARTING_STACK)
    
    # Agent(Seat 2): 상대 올인에 콜 (Action 1) -> 남은 스택 전액 투자
    tracker.record_agent_decision(
        round_idx=0, act_6=1, nat_act=1, cur_to_call=STARTING_STACK,
        spent_before=0.0, spent_after=STARTING_STACK,
        cur_opp_allin_active=True, opp_allin_seat=3
    )
    
    # Showdown 결과 발생
    res = tracker.finalize_hand(
        final_spent=[50.0, 100.0, STARTING_STACK, STARTING_STACK, 0.0],
        returns=[-50.0, -100.0, -STARTING_STACK, STARTING_STACK + 150.0, 0.0],
        priv_cards={2: ("Kd", "Qd")},
        board_cards=["Ah", "Ks", "2c", "7d", "Jc"]
    )
    
    assert res["pushed_allin"] is False, "Agent did NOT push; agent CALLED an all-in!"
    assert res["called_allin"] is True, "Must be classified as called_allin!"
    assert res["showdown"] is True
    assert res["pnl_bb"] == -100.0
    print("[PASS] test_facing_allin_call_separation: 상대 올인에 콜한 올인과 먼저 민 올인의 완전 분리 확인.")

def test_three_bet_opportunity_calculation():
    """에이전트 이전에 1회 레이즈가 발생했을 때만 3-Bet 기회로 인정"""
    # Case A: 에이전트 앞에 림프만 있는 경우 (레이즈 0회) -> 3-Bet 기회 아님 (Open Raise 기회)
    tracker_a = HandIntegrityTracker(agent_seat=3, seed=104)
    tracker_a.record_opponent_decision(cp=2, round_idx=0, act_6=1, cur_to_call=0.0, spent_before=0.0, spent_after=100.0)
    tracker_a.record_agent_decision(round_idx=0, act_6=3, nat_act=3, cur_to_call=100.0, spent_before=0.0, spent_after=400.0, cur_opp_allin_active=False, opp_allin_seat=None)
    res_a = tracker_a.finalize_hand([50, 100, 100, 400, 0], [-50, -100, -100, 250, 0], {3: ("As", "Ks")}, [])
    assert res_a["3bet_opportunity"] is False
    assert res_a["3bet"] is False
    assert res_a["pfr"] is True

    # Case B: 에이전트 앞에 1회 레이즈가 있는 경우 -> 3-Bet 기회 발생
    tracker_b = HandIntegrityTracker(agent_seat=3, seed=105)
    tracker_b.record_opponent_decision(cp=2, round_idx=0, act_6=2, cur_to_call=0.0, spent_before=0.0, spent_after=300.0)
    tracker_b.record_agent_decision(round_idx=0, act_6=3, nat_act=3, cur_to_call=300.0, spent_before=0.0, spent_after=900.0, cur_opp_allin_active=False, opp_allin_seat=None)
    res_b = tracker_b.finalize_hand([50, 100, 300, 900, 0], [-50, -100, -300, 450, 0], {3: ("As", "Ks")}, [])
    assert res_b["3bet_opportunity"] is True
    assert res_b["3bet"] is True
    print("[PASS] test_three_bet_opportunity_calculation: 3-Bet 기회 대비 3-Bet 실행 정밀 산출 확인.")

def run_all_tests():
    print("=" * 70)
    print("RUNNING DIRECTIVE 041 INTEGRITY METRICS REGRESSION TESTS")
    print("=" * 70)
    test_bb_free_check_vpip()
    test_bb_paid_call_vpip()
    test_uncalled_push_allin_integrity()
    test_facing_allin_call_separation()
    test_three_bet_opportunity_calculation()
    print("=" * 70)
    print("ALL DIRECTIVE 041 EVALUATOR TESTS PASSED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
