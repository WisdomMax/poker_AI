"""
Kuhn Poker V3.0 - CFR GTO 학습기 및 이론값 수학 검증 모듈
"""
import random
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES

class CFRNodeV3:
    def __init__(self, info_set: str):
        self.info_set = info_set
        self.regret_sum = np.zeros(2, dtype=np.float64)
        self.strategy = np.zeros(2, dtype=np.float64)
        self.strategy_sum = np.zeros(2, dtype=np.float64)

    def get_strategy(self, realization_weight: float) -> np.ndarray:
        normalizing_sum = 0.0
        for a in ACTIONS:
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0.0
            normalizing_sum += self.strategy[a]

        for a in ACTIONS:
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 0.5
            self.strategy_sum[a] += realization_weight * self.strategy[a]

        return self.strategy

    def get_average_strategy(self) -> np.ndarray:
        avg_strategy = np.zeros(2, dtype=np.float64)
        normalizing_sum = np.sum(self.strategy_sum)
        for a in ACTIONS:
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 0.5
        return avg_strategy


class CFRTrainerV3:
    def __init__(self):
        self.node_map: dict[str, CFRNodeV3] = {}

    def get_node(self, info_set: str) -> CFRNodeV3:
        if info_set not in self.node_map:
            self.node_map[info_set] = CFRNodeV3(info_set)
        return self.node_map[info_set]

    def cfr(self, cards: list[int], history: str, p0: float, p1: float) -> float:
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if is_terminal(history):
            p0_payoff, p1_payoff = evaluate_payoff(history, cards[0], cards[1])
            return p0_payoff if player == 0 else p1_payoff

        info_set = f"{CARD_NAMES[cards[player]]}:{history}"
        node = self.get_node(info_set)
        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = np.zeros(2, dtype=np.float64)
        node_util = 0.0

        for a in ACTIONS:
            next_history = history + ("p" if a == PASS else "b")
            if player == 0:
                util[a] = -self.cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        for a in ACTIONS:
            regret = util[a] - node_util
            prob = p1 if player == 0 else p0
            node.regret_sum[a] += prob * regret

        return node_util

    def train(self, iterations: int = 200000) -> float:
        cards = [1, 2, 3]
        total_util = 0.0
        for _ in range(iterations):
            random.shuffle(cards)
            total_util += self.cfr(cards, "", 1.0, 1.0)
        return total_util / iterations

    def get_strategy_dict(self) -> dict[str, tuple[float, float]]:
        res = {}
        for k, v in self.node_map.items():
            avg = v.get_average_strategy()
            res[k] = (float(avg[0]), float(avg[1]))
        return res
