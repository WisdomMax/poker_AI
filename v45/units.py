"""
V4.5 단위 표준화 및 보상 변환 모듈 (units.py)

Leduc Poker의 수학적 단위 규격:
- Ante: 1 chip
- Round 1 Bet: 2 chips (Small Bet / 1 Big Blind)
- Round 2 Bet: 4 chips (Big Bet / 2 Big Blinds)
- 1 Big Blind (1 bb) = 2 Chips
- Game Utility: OpenSpiel Leduc Poker의 terminal returns (칩 기준 순손익)
- Chips per Hand: 총 순손익 칩 / 총 판수
- Chips per 100 Hands: (총 순손익 칩 / 총 판수) * 100
- bb/100: Chips per 100 Hands / 2.0 (1 bb = 2 chips)
- NashConv: 정책 프로필 pi의 각 플레이어 Best Response 갭의 합 (\sum_i (u_i(BR_i, pi_{-i}) - u_i(pi)))
- Exploitability: NashConv / num_players (플레이어당 착취 허용도, 칩 단위)
- Exploitability (bb/100): (Exploitability / 2.0) * 100
"""

from typing import Dict, Any

CHIPS_PER_BB = 2.0  # Leduc Poker의 1 Big Blind = 2 Chips

def chips_to_bb(chips: float) -> float:
    """칩 단위를 Big Blind 단위로 변환 (1 bb = 2 chips)"""
    return chips / CHIPS_PER_BB

def bb_to_chips(bb: float) -> float:
    """Big Blind 단위를 칩 단위로 변환"""
    return bb * CHIPS_PER_BB

def calculate_metrics(total_chips: float, num_hands: int) -> Dict[str, float]:
    """총 칩 손익과 판수로부터 표준 지표 계산"""
    if num_hands <= 0:
        return {
            "total_chips": 0.0,
            "chips_per_hand": 0.0,
            "chips_per_100": 0.0,
            "bb_per_100": 0.0,
        }
    
    chips_per_hand = total_chips / num_hands
    chips_per_100 = chips_per_hand * 100.0
    bb_per_100 = chips_per_100 / CHIPS_PER_BB
    
    return {
        "total_chips": float(total_chips),
        "chips_per_hand": float(chips_per_hand),
        "chips_per_100": float(chips_per_100),
        "bb_per_100": float(bb_per_100),
    }

def format_metrics(total_chips: float, num_hands: int) -> str:
    """지표를 사람이 읽기 쉬운 표준 문자열로 포맷"""
    m = calculate_metrics(total_chips, num_hands)
    return (
        f"{m['total_chips']:+.1f} chips ({m['chips_per_hand']:+.4f} chips/hand, "
        f"{m['chips_per_100']:+.2f} chips/100, {m['bb_per_100']:+.2f} bb/100)"
    )

def nashconv_to_exploitability(nashconv: float, num_players: int = 2) -> Dict[str, float]:
    """NashConv를 플레이어당 Exploitability 및 bb/100으로 변환"""
    exp_chips = nashconv / float(num_players)
    exp_bb_per_100 = (exp_chips / CHIPS_PER_BB) * 100.0
    return {
        "nashconv": float(nashconv),
        "exploitability_chips": float(exp_chips),
        "exploitability_bb_per_100": float(exp_bb_per_100),
    }
