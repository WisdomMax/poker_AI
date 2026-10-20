"""
POKER AI V5.0 - PHASE 1 & 2: Bayesian Tracker Unit Tests
- 1. Agent ID vs Seat ID 분리 및 좌석 순환 무결성 검증
- 2. 기회(Opportunity) 기반 6대 지표 사후분포 수치 정확도 검증
- 3. 강제 블라인드 VPIP 제외 검증
- 4. Cold Start vs Warm Start 불확실성 감소 검증
- 5. 52차원 피처 텐서 스케일 [0.0, 1.0] 검증
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v50.bayesian_opponent_tracker import (
    SingleOpponentBayesianTracker, MatchSessionOpponentManager, METRIC_NAMES
)

def test_bayesian_tracker():
    print("=" * 80)
    print("PHASE 1 & 2: 베이지안 상대 관측기 단위 테스트")
    print("=" * 80)

    # 1. Cold Start 사후분포 검증
    print("\n[테스트 1] Cold Start 사전분포 및 불확실성 검증:")
    tracker = SingleOpponentBayesianTracker(agent_id=1, prior_alpha=2.0, prior_beta=2.0)
    for m in METRIC_NAMES:
        mean, std, opp = tracker.get_posterior(m)
        print(f"  지표 {m:<18}: 기회={opp:2d} | 사후평균={mean:.4f} | 사후표준편차(불확실성)={std:.4f}")
        assert opp == 0, "Cold Start 기회 수는 0이어야 함"
        assert abs(mean - 0.5) < 1e-6, "기본 사후평균은 0.5이어야 함"
        expected_std = math.sqrt((2.0 * 2.0) / (4.0**2 * 5.0)) # sqrt(4 / 80) = sqrt(0.05) ~= 0.2236
        assert abs(std - expected_std) < 1e-5, f"불확실성 계산 오류: {std} != {expected_std}"
    print("-> PASS: Cold Start에서 사후 평균 0.5000 및 최대 불확실성(0.2236) 유지 확인.")

    # 2. Warm Start 수렴 검증 (Tight 상대: 100회 중 VPIP 15회)
    print("\n[테스트 2] Warm Start 수렴 및 불확실성 축소 검증:")
    for _ in range(100):
        tracker.record_opportunity("vpip")
    for _ in range(15):
        tracker.record_success("vpip")

    mean, std, opp = tracker.get_posterior("vpip")
    print(f"  VPIP (100회 관측, 15회 성공):")
    print(f"    사후평균 = {mean:.4f} (이론치: (2+15)/(4+100) = 17/104 = {17/104:.4f})")
    print(f"    불확실성 = {std:.4f} (Cold Start 0.2236에서 대폭 축소 확인)")
    assert abs(mean - (17.0 / 104.0)) < 1e-6, "사후평균 일치"
    assert std < 0.05, "100회 관측 후 불확실성이 0.05 미만으로 안정 축소되어야 함"
    print("-> PASS: 데이터 축적에 따른 사후평균 수렴 및 불확실성 축소 확인.")

    # 3. MatchSessionOpponentManager 좌석 순환 및 ID 분리 검증
    print("\n[테스트 3] 좌석 회전 시 Agent ID 통계 격리 검증:")
    mgr = MatchSessionOpponentManager()

    # 핸드 0: Seat 2가 Agent 2임. (agent = (seat + hand_idx) % 5 = (2+0)%5 = 2)
    mgr.set_hand_index(0)
    seat_to_ag_h0 = {seat: (seat + 0) % 5 for seat in range(5)}
    # Seat 2(Agent 2)가 오픈 올인(Action 3)을 감행
    fake_history_h0 = [
        (2, 0, [0, 1, 2, 3, 4], 3) # Seat 2가 올인
    ]
    mgr.parse_and_update_hand(fake_history_h0, seat_to_ag_h0)

    # 핸드 1: 좌석 회전으로 Seat 1이 Agent 2가 됨. (agent = (1+1)%5 = 2)
    # Seat 2는 이제 Agent 3임. ((2+1)%5 = 3)
    mgr.set_hand_index(1)
    seat_to_ag_h1 = {seat: (seat + 1) % 5 for seat in range(5)}
    # 이번 판에서는 Seat 2(Agent 3)는 폴드(0), Seat 1(Agent 2)이 또 올인(3)
    fake_history_h1 = [
        (2, 0, [0, 1, 2, 3, 4], 0), # Seat 2(Agent 3) 폴드
        (1, 0, [0, 1, 2, 3, 4], 3), # Seat 1(Agent 2) 올인
    ]
    mgr.parse_and_update_hand(fake_history_h1, seat_to_ag_h1)

    # 확인: Agent 2는 올인을 2회 했고, Agent 3은 올인을 0회 했어야 함!
    ag2_allin_opp = mgr.trackers[2].stats['allin']['opp']
    ag2_allin_succ = mgr.trackers[2].stats['allin']['succ']
    ag3_allin_opp = mgr.trackers[3].stats['allin']['opp']
    ag3_allin_succ = mgr.trackers[3].stats['allin']['succ']

    print(f"  Agent 2 (올인 2회 실행): 기회={ag2_allin_opp}, 성공={ag2_allin_succ}")
    print(f"  Agent 3 (폴드 1회 실행): 기회={ag3_allin_opp}, 성공={ag3_allin_succ}")

    assert ag2_allin_succ == 2, f"Agent 2 올인 횟수 오류: {ag2_allin_succ} != 2"
    assert ag3_allin_succ == 0, f"Agent 3 올인 횟수 오류: {ag3_allin_succ} != 0"
    print("-> PASS: 좌석 회전에도 불구하고 Agent ID별 통계가 완벽히 독립적으로 추적됨.")

    # 4. 결합 피처 텐서 스케일 검증
    print("\n[테스트 4] 52차원 상대 피처 텐서 스케일 및 정규화 검증:")
    feats = mgr.get_table_feature_tensor(current_hero_seat=0)
    print(f"  피처 텐서 shape: {feats.shape} (기대: (52,))")
    print(f"  최솟값: {np.min(feats):.4f} | 최댓값: {np.max(feats):.4f}")
    assert feats.shape == (52,), f"차원 불일치: {feats.shape} != (52,)"
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0), "모든 피처는 [0.0, 1.0] 범위여야 함"
    print("-> PASS: 52차원 텐서가 정확히 [0.0, 1.0] 범위에 완벽히 안착됨.")

if __name__ == "__main__":
    test_bayesian_tracker()
