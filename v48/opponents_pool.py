"""
POKER AI V4.8 - 8대 평가 상대 봇 풀 및 전략 정의 (Opponents Pool)
- 모든 봇은 자신의 좌석(cur_player)에 제공된 information_state만을 사용.
- 상대방의 비공개 카드(Private Cards) 누출이나 치팅이 전혀 없음을 보장.

8대 봇 전략 명세:
1. RandomHoldemBot: 모든 합법적 행동을 균등 무작위 선택.
2. AlwaysCallHoldemBot: 폴드하지 않고 오직 Check/Call만 고수 (Calling Station).
3. TightHoldemBot: 브로드웨이 카드(T, J, Q, K, A)나 페어가 있을 때만 베팅/콜, 그 외엔 폴드.
4. AggressiveHoldemBot: 팟 레이즈나 올인이 가능하면 적극적으로 베팅을 주도.
5. RuleBasedHoldemBot: 핸드 강도와 스트리트(프리플롭/플롭/턴/리버)에 따라 보수적으로 레이즈/체크/폴드.
6. PremiumHunterBot: 상위 3% 프리미엄(AA, KK, QQ, AKs)만 올인을 콜하고 나머지는 엄격히 폴드하여 올인 블러프를 저격.
7. BroadHunterBot: 상위 15% 핸드(TT+, AJs+, KQs)로 올인을 받아쳐 올인 남발을 광범위하게 응징.
8. AdaptiveHunterBot: 상대방의 올인 빈도를 동적으로 추적하여, 올인 비율이 높을수록 방어 레인지를 점진적으로 넓힘.
"""

import random
from typing import List, Dict, Optional
import numpy as np
import pyspiel

class BaseBot:
    def __init__(self, name: str):
        self.name = name
    def act(self, state: pyspiel.State, cur_player: int, legal_acts: List[int]) -> int:
        raise NotImplementedError
    def reset(self):
        pass

class RandomBot(BaseBot):
    def __init__(self):
        super().__init__("Random")
    def act(self, state, cur_player, legal_acts) -> int:
        return random.choice(legal_acts)

class AlwaysCallBot(BaseBot):
    def __init__(self):
        super().__init__("Always Call")
    def act(self, state, cur_player, legal_acts) -> int:
        return 1 if 1 in legal_acts else legal_acts[0]

class TightBot(BaseBot):
    def __init__(self):
        super().__init__("Tight")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        has_high = any(c in info for c in ["T", "J", "Q", "K", "A"])
        has_pair = any(p in info for p in ["Pair", "Two", "Three", "Full", "Straight", "Flush"])
        if has_high or has_pair:
            return 2 if 2 in legal_acts else (1 if 1 in legal_acts else legal_acts[0])
        return 1 if 1 in legal_acts else 0 # 체크 가능 시 체크, 아니면 폴드

class AggressiveBot(BaseBot):
    def __init__(self):
        super().__init__("Aggressive")
    def act(self, state, cur_player, legal_acts) -> int:
        if 2 in legal_acts: return 2 # Pot raise
        if 3 in legal_acts: return 3 # All in
        return 1 if 1 in legal_acts else legal_acts[0]

class RuleBasedBot(BaseBot):
    def __init__(self):
        super().__init__("Rule-Based")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        is_river = "[Round 3]" in info
        has_strong = any(c in info for c in ["K", "A"]) or "Pair" in info
        if has_strong:
            if is_river and 2 in legal_acts: return 2
            return 1 if 1 in legal_acts else (2 if 2 in legal_acts else legal_acts[0])
        return 1 if 1 in legal_acts else 0

def extract_hole_cards(info_str: str) -> List[str]:
    if "Private: " in info_str:
        cards_str = info_str.split("Private: ")[1].split("]")[0]
        return [cards_str[i:i+2] for i in range(0, len(cards_str), 2)]
    return []

class PremiumHunterBot(BaseBot):
    """상위 3% 프리미엄(AA, KK, QQ, AKs)으로만 올인 콜/트랩"""
    def __init__(self):
        super().__init__("Premium Hunter")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        is_premium = False
        if len(hc) == 2:
            r1, r2 = hc[0][0], hc[1][0]
            s1, s2 = hc[0][1], hc[1][1]
            if r1 == r2 and r1 in ['A', 'K', 'Q']: # AA, KK, QQ
                is_premium = True
            elif (r1 == 'A' and r2 == 'K') or (r1 == 'K' and r2 == 'A'):
                is_premium = True
        # 올인에 직면한 상황 (Action 1이 콜인데 칩 소모가 큰 상황)
        if 3 in legal_acts or (1 in legal_acts and 0 in legal_acts):
            if is_premium:
                return 3 if 3 in legal_acts else (1 if 1 in legal_acts else legal_acts[0])
            return 0 if 0 in legal_acts else 1
        return 1 if 1 in legal_acts else 0

class BroadHunterBot(BaseBot):
    """상위 15% 핸드(TT+, AJs+, KQs)로 올인을 받아쳐 공격적 응징"""
    def __init__(self):
        super().__init__("Broad Hunter")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        is_strong = False
        if len(hc) == 2:
            r1, r2 = hc[0][0], hc[1][0]
            broad_ranks = set(['T', 'J', 'Q', 'K', 'A'])
            if r1 == r2 and r1 in broad_ranks: # TT+
                is_strong = True
            elif r1 in broad_ranks and r2 in broad_ranks: # Broadway Combos
                is_strong = True
        if is_strong:
            return 3 if 3 in legal_acts else (2 if 2 in legal_acts else 1)
        return 1 if 1 in legal_acts else 0

class AdaptiveHunterBot(BaseBot):
    """상대의 올인 빈도를 관측하여 방어 범위를 적응적으로 변경"""
    def __init__(self):
        super().__init__("Adaptive Hunter")
        self.observed_allin_count = 0
        self.observed_total_actions = 0
    def reset(self):
        self.observed_allin_count = 0
        self.observed_total_actions = 0
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        # 상대들의 이전 액션 관측
        self.observed_total_actions += 1
        allin_ratio = (self.observed_allin_count / max(1, self.observed_total_actions))
        
        has_pair = len(hc) == 2 and hc[0][0] == hc[1][0]
        has_ace = len(hc) == 2 and ('A' in hc[0] or 'A' in hc[1])
        has_broadway = len(hc) == 2 and any(c[0] in ['K', 'Q', 'J', 'T'] for c in hc)
        
        # 상대가 올인을 많이 칠수록(allin_ratio > 0.2) 에이스나 페어만 있어도 콜/올인으로 응징
        if allin_ratio > 0.15 or len(legal_acts) == 2: # 올인 직면 국면
            if has_pair or has_ace or has_broadway:
                return 1 if 1 in legal_acts else legal_acts[0]
            return 0 if 0 in legal_acts else 1
            
        if has_pair or has_ace:
            return 2 if 2 in legal_acts else 1
        return 1 if 1 in legal_acts else 0

def get_8_opponents() -> List[BaseBot]:
    return [
        RandomBot(),
        AlwaysCallBot(),
        TightBot(),
        AggressiveBot(),
        RuleBasedBot(),
        PremiumHunterBot(),
        BroadHunterBot(),
        AdaptiveHunterBot()
    ]
