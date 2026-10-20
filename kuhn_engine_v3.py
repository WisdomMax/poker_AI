"""
Kuhn Poker V3.0 - 무결점 게임 룰 엔진 및 손익 계산 모듈
"""

# 액션 정의: 0 = PASS (Check/Fold), 1 = BET (Call/Raise)
PASS = 0
BET = 1
ACTIONS = [PASS, BET]
ACTION_CHARS = {PASS: 'p', BET: 'b'}
CHAR_TO_ACTION = {'p': PASS, 'b': BET}

# 카드 정의: 1 = Jack, 2 = Queen, 3 = King
CARD_NAMES = {1: 'J', 2: 'Q', 3: 'K'}
NAME_TO_CARD = {'J': 1, 'Q': 2, 'K': 3}


def is_terminal(history: str) -> bool:
    """게임 종료 여부 판별"""
    return history in ("pp", "bp", "bb", "pbp", "pbb")


def evaluate_payoff(history: str, c0: int, c1: int) -> tuple[int, int]:
    """
    모든 종료 상태에 대한 엄밀한 Zero-Sum 손익 계산 (P0 손익, P1 손익)
    Ante = 1, Bet = 1
    
    1. pp  : Check - Check (쇼다운)
             P0 앤티 1, P1 앤티 1 (팟 2) -> 승자 +1, 패자 -1
    2. bp  : Bet - Fold (P1 폴드)
             P0 승리! P0 +1, P1 -1
    3. bb  : Bet - Call (쇼다운)
             P0 앤티1+베팅1=2, P1 앤티1+콜1=2 (팟 4) -> 승자 +2, 패자 -2
    4. pbp : Check - Bet - Fold (P0 폴드)
             P1 승리! P0 -1, P1 +1
    5. pbb : Check - Bet - Call (쇼다운)
             P0 앤티1+콜1=2, P1 앤티1+베팅1=2 (팟 4) -> 승자 +2, 패자 -2
    """
    if history == "pp":
        if c0 > c1:
            return (1, -1)
        elif c0 < c1:
            return (-1, 1)
        else:
            return (0, 0)

    elif history == "bp":
        # P0이 베팅했고 P1이 패스(폴드) -> P0 승리
        return (1, -1)

    elif history == "bb":
        # P0 베팅, P1 콜 -> 쇼다운 (팟 4, 각자 2 지불)
        if c0 > c1:
            return (2, -2)
        elif c0 < c1:
            return (-2, 2)
        else:
            return (0, 0)

    elif history == "pbp":
        # P0 체크, P1 베팅, P0 패스(폴드) -> P1 승리
        return (-1, 1)

    elif history == "pbb":
        # P0 체크, P1 베팅, P0 콜 -> 쇼다운 (팟 4, 각자 2 지불)
        if c0 > c1:
            return (2, -2)
        elif c0 < c1:
            return (-2, 2)
        else:
            return (0, 0)

    raise ValueError(f"유효하지 않은 종료 상태: {history}")


def get_current_player(history: str) -> int:
    """현재 차례의 플레이어 번호 (0 또는 1)"""
    return len(history) % 2
