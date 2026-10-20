"""
POKER AI V4.9.4 - EV 물리적 상한 자동 검증 모듈 (EV Boundary Validator)
- 5인 홀덤 No-Limit 환경의 칩 및 BB 보존 법칙 및 물리적 상한 검증
- 규칙:
  * 플레이어 스택: 10,000 chips = 100 BB
  * 단일 핸드 순수익 한계: [-10,000 chips, +40,000 chips] = [-100.0 BB, +400.0 BB]
  * 전체 5인 테이블 순수익 합계(Zero-Sum): sum(returns) == 0 (부동소수점 오차 1e-4 이내)
"""

from typing import List, Union
import numpy as np

STACK_CHIPS = 10000.0
STACK_BB = 100.0
NUM_PLAYERS = 5

MIN_CHIPS = -STACK_CHIPS
MAX_CHIPS = +(NUM_PLAYERS - 1) * STACK_CHIPS # +40,000 chips

MIN_BB = -STACK_BB
MAX_BB = +(NUM_PLAYERS - 1) * STACK_BB       # +400.0 BB

def validate_chips_return(chips: float, hand_idx: int = -1, player_id: int = -1) -> bool:
    """단일 핸드 순수익 chips가 물리적 상한 [-10,000, +40,000] 내에 있는지 검증"""
    if chips < MIN_CHIPS - 1e-4 or chips > MAX_CHIPS + 1e-4:
        raise AssertionError(
            f"[FATAL CHIPS BOUNDARY ERROR] Hand {hand_idx}, Player {player_id}: "
            f"chips={chips:+.2f} is outside physical bounds [{MIN_CHIPS:+.0f}, {MAX_CHIPS:+.0f}]"
        )
    return True

def validate_bb_return(bb: float, hand_idx: int = -1, player_id: int = -1) -> bool:
    """단일 핸드 순수익 BB가 물리적 상한 [-100.0, +400.0] 내에 있는지 검증"""
    if bb < MIN_BB - 1e-4 or bb > MAX_BB + 1e-4:
        raise AssertionError(
            f"[FATAL BB BOUNDARY ERROR] Hand {hand_idx}, Player {player_id}: "
            f"bb={bb:+.4f} is outside physical bounds [{MIN_BB:+.1f}, {MAX_BB:+.1f}]"
        )
    return True

def validate_table_zero_sum(returns: Union[List[float], np.ndarray], hand_idx: int = -1) -> bool:
    """5인 테이블 모든 플레이어의 순수익 합계가 0(Zero-Sum)인지 검증"""
    total = sum(returns)
    if abs(total) > 1e-3:
        raise AssertionError(
            f"[FATAL ZERO-SUM ERROR] Hand {hand_idx}: sum of returns = {total:+.4f} != 0.0"
        )
    return True

def validate_mean_ev(mean_bb_hand: float, metric_name: str = "Mean_EV") -> bool:
    """핸드당 평균 EV가 물리적 한계 [-100.0, +400.0] BB 이내인지 검증"""
    if mean_bb_hand < MIN_BB - 1e-4 or mean_bb_hand > MAX_BB + 1e-4:
        raise AssertionError(
            f"[FATAL MEAN EV ERROR] {metric_name}: mean_bb_hand={mean_bb_hand:+.4f} "
            f"exceeds physical limits [{MIN_BB:+.1f}, {MAX_BB:+.1f}] BB/hand!"
        )
    return True
