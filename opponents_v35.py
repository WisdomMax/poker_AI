"""
Poker AI V3.5 - [2. 신규 상대 전략 5종 (Opponents V3.5)]
모든 전략은 Kuhn Poker의 합법적인 정보만 활용
"""
import random
import numpy as np
from kuhn_engine_v3 import PASS, BET, ACTIONS, CARD_NAMES

class BalancedPlayer:
    """A. Balanced Player: CFR 균형 전략(GTO)을 엄격히 사용하는 상대"""
    def __init__(self, name: str, gto_strategy: dict[str, tuple[float, float]]):
        self.name = name
        self.gto_strategy = gto_strategy

    def act(self, card: int, history: str) -> int:
        info_set = f"{CARD_NAMES[card]}:{history}"
        p = self.gto_strategy.get(info_set, (0.5, 0.5))
        return int(np.random.choice(ACTIONS, p=p))


class TrickyPlayer:
    """
    B. Tricky Player:
    - 강한 카드(K)로 의도적으로 체크를 섞어 상대를 유인(Trap)
    - 약한 카드(J)로 고빈도 블러핑
    - 중간 카드(Q)로 절반은 콜, 절반은 폴드 믹싱
    """
    def __init__(self, name="Tricky"):
        self.name = name

    def act(self, card: int, history: str) -> int:
        # P0 선공일 때
        if history == "":
            if card == 1: return BET if random.random() < 0.45 else PASS # 45% 블러핑
            if card == 2: return PASS # Q는 항상 체크
            if card == 3: return PASS if random.random() < 0.50 else BET # K로 50% 슬로우플레이(체크 트랩)

        # 상대 체크 후 (history == "p")
        if history == "p":
            if card == 1: return BET if random.random() < 0.50 else PASS # 50% 스틸 블러핑
            if card == 2: return PASS # Q는 쇼다운 체크
            if card == 3: return BET # K는 밸류 베팅

        # 상대 베팅을 만났을 때 (history == "b" 또는 "pb")
        if history in ("b", "pb"):
            if card == 1: return PASS # J는 무조건 폴드
            if card == 2: return BET if random.random() < 0.50 else PASS # Q로 50% 캐치 콜
            if card == 3: return BET # K는 100% 콜

        return PASS


class AdaptiveOpponentPlayer:
    """
    C. Adaptive Player:
    상대의 폴드율을 실시간으로 관찰하고 자신의 블러핑/밸류 빈도를 동적으로 조절하는 상대
    """
    def __init__(self, name="AdaptiveOpponent"):
        self.name = name
        self.times_faced_bet = 0
        self.times_opponent_folded = 0

    def record_opponent_response(self, opponent_folded: bool):
        self.times_faced_bet += 1
        if opponent_folded:
            self.times_opponent_folded += 1

    @property
    def opp_fold_rate(self) -> float:
        if self.times_faced_bet < 5: return 0.5
        return self.times_opponent_folded / self.times_faced_bet

    def act(self, card: int, history: str) -> int:
        fold_rate = self.opp_fold_rate
        # 상대가 잘 접는다 -> 블러핑 빈도 증가
        bluff_freq = 0.70 if fold_rate > 0.55 else (0.15 if fold_rate < 0.35 else 0.33)

        if history == "":
            if card == 1: return BET if random.random() < bluff_freq else PASS
            if card == 2: return PASS
            if card == 3: return BET
        elif history == "p":
            if card == 1: return BET if random.random() < bluff_freq else PASS
            if card == 2: return PASS
            if card == 3: return BET
        elif history in ("b", "pb"):
            if card == 1: return PASS
            if card == 2: return BET if fold_rate < 0.40 else PASS
            if card == 3: return BET
        return PASS


class RandomizedPlayer:
    """
    D. Randomized Player:
    카드와 행동 이력별로 0~1 사이의 무작위 고정 확률을 부여받은 미지의 상대
    """
    def __init__(self, name="Randomized", seed: int = 42):
        self.name = name
        rng = np.random.RandomState(seed)
        # 12개 정보셋별 BET 확률
        self.probs = {}
        for c in ['J', 'Q', 'K']:
            for h in ['', 'p', 'b', 'pb']:
                self.probs[f"{c}:{h}"] = float(rng.uniform(0.0, 1.0))

    def act(self, card: int, history: str) -> int:
        key = f"{CARD_NAMES[card]}:{history}"
        p_bet = self.probs.get(key, 0.5)
        return BET if random.random() < p_bet else PASS


class CounterExploiterPlayer:
    """
    E. Counter Exploiter:
    V3의 성향(상대가 체크하면 약한 것으로 간주하고 J로 스틸하려는 성향)을 역이용하도록 설계
    - P0일 때 K나 Q로 체크를 하여 V3의 블러핑을 유도(Induce Bluff)한 뒤,
      V3가 pb를 치면 pbb로 스냅콜하여 V3를 사냥하는 트랩 봇
    """
    def __init__(self, name="CounterExploiter"):
        self.name = name

    def act(self, card: int, history: str) -> int:
        if history == "":
            # K나 Q일 때 100% 체크하여 V3에게 '약한 척' 신호 송신
            return PASS
        elif history == "p":
            # 상대가 체크했을 때: K면 밸류 베팅, J는 20% 블러핑
            if card == 3: return BET
            if card == 1: return BET if random.random() < 0.20 else PASS
            return PASS
        elif history == "pb":
            # P0 입장에서: 내가 체크했는데 V3가 찔러왔다!
            # K는 100% 트랩 콜, Q도 80% 블러프 캐치 콜! J는 폴드
            if card == 3: return BET
            if card == 2: return BET if random.random() < 0.85 else PASS
            return PASS
        elif history == "b":
            # V3가 먼저 선베팅했을 때
            if card == 3: return BET
            return PASS
        return PASS
