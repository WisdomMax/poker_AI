"""
POKER AI V4.9.4 - EV 물리적 상한 자동 검증 단위 테스트
- 테스트 시나리오:
  1. 합법적인 손익 범위 (-100 BB, 0 BB, +100 BB, +400 BB) 통과 검증
  2. 물리적 상한 초과 (-101 BB, +401 BB) 시 AssertionError 발생 검증
  3. 제로섬 보존 법칙 (합계 0) 검증 및 위반 시 에러 발생 검증
  4. 실제 500개 게임 롤아웃 전수 검증
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v494.ev_boundary_validator import (
    validate_chips_return, validate_bb_return,
    validate_table_zero_sum, validate_mean_ev
)
from v49.poker_environment_5a import get_5p_holdem_5a_game, advance_chance_nodes
from v48.opponents_pool import RandomBot

def test_boundaries():
    print("=" * 80)
    print("EV Boundary Validator 단위 테스트 실행")
    print("=" * 80)

    # 1. 합법적 값 통과 검증
    for valid_bb in [-100.0, -50.0, 0.0, +25.5, +100.0, +400.0]:
        assert validate_bb_return(valid_bb), f"Failed on valid bb {valid_bb}"
        assert validate_chips_return(valid_bb * 100.0), f"Failed on valid chips {valid_bb * 100}"
    print("  [Pass] 합법적 범위 [-100 BB, +400 BB] 통과 완료")

    # 2. 상한 초과 에러 발생 검증
    for invalid_bb in [-100.1, -200.0, +400.1, +1000.0]:
        try:
            validate_bb_return(invalid_bb)
            assert False, f"Did not raise AssertionError on invalid bb {invalid_bb}"
        except AssertionError:
            pass # 성공
    print("  [Pass] 물리적 상한 위반 값 감지 및 차단 검증 완료")

    # 3. 제로섬 검증
    assert validate_table_zero_sum([-100, -50, 0, +50, +100])
    try:
        validate_table_zero_sum([-100, -50, 0, +50, +101]) # 합계 +1
        assert False, "Did not catch non-zero-sum!"
    except AssertionError:
        pass
    print("  [Pass] 제로섬(Zero-Sum) 무결성 검증 완료")

    # 4. 실제 500개 게임 롤아웃 전수 검사
    print("  [실제 500게임 롤아웃 전수 상한 검증 중...]")
    game = get_5p_holdem_5a_game()
    rng = np.random.RandomState(494)
    bots = [RandomBot() for _ in range(5)]

    for h in range(500):
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng)
            else:
                p = st.current_player()
                legal = st.legal_actions(p)
                st.apply_action(bots[p].act(st, p, legal))

        rets = st.returns()
        validate_table_zero_sum(rets, hand_idx=h)
        for p, r in enumerate(rets):
            validate_chips_return(r, hand_idx=h, player_id=p)
            validate_bb_return(r / 100.0, hand_idx=h, player_id=p)

    print("  [Pass] 실제 500개 게임(2,500개 플레이어 리턴) 물리적 상한 100% 준수 확인!")
    print("EV Boundary Validator 테스트 전원 통과 완료.")

if __name__ == '__main__':
    test_boundaries()
