"""
Poker AI V4.0 - OpenSpiel 어댑터 모듈
자체 Python 딕셔너리 전략 <-> OpenSpiel Policy 객체 상호 변환기
"""
import pyspiel
from open_spiel.python import policy

# 카드 매핑: OpenSpiel (0=J, 1=Q, 2=K) <-> 자체 V3 (1=J, 2=Q, 3=K, 문자 'J', 'Q', 'K')
OS_CARD_TO_CHAR = {0: 'J', 1: 'Q', 2: 'K'}
CHAR_TO_OS_CARD = {'J': 0, 'Q': 1, 'K': 2}

def openspiel_infostate_to_v3_key(os_infostate_str: str) -> str:
    """
    OpenSpiel info state ('0', '1p', '0pb' 등)을
    자체 V3 정보셋 키 ('J:', 'Q:p', 'J:pb')로 변환
    """
    card_num = int(os_infostate_str[0])
    card_char = OS_CARD_TO_CHAR[card_num]
    history = os_infostate_str[1:] # 'p', 'b', 'pb' 또는 빈 문자열
    return f"{card_char}:{history}"

def v3_key_to_openspiel_infostate(v3_key: str) -> str:
    """
    자체 V3 키 ('J:', 'Q:p', 'J:pb')를
    OpenSpiel info state ('0', '1p', '0pb')로 변환
    """
    card_char, history = v3_key.split(':')
    card_num = CHAR_TO_OS_CARD[card_char]
    return f"{card_num}{history}"


class DictPolicyWrapper(policy.Policy):
    """
    자체 V3 전략 딕셔너리 { 'J:': (p_pass, p_bet), ... }를
    OpenSpiel이 요구하는 policy.Policy 객체로 래핑
    """
    def __init__(self, game, strategy_dict: dict[str, tuple[float, float]]):
        super().__init__(game, list(range(game.num_players())))
        self.strategy_dict = strategy_dict

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        
        info_state_str = state.information_state_string(player_id)
        v3_key = openspiel_infostate_to_v3_key(info_state_str)
        
        # 기본값: 50/50
        probs = self.strategy_dict.get(v3_key, (0.5, 0.5))
        # OpenSpiel 포맷: {action_id: probability}
        return {0: float(probs[0]), 1: float(probs[1])}
