"""
CFR (Counterfactual Regret Minimization) - 가상 후회 최소화 알고리즘
포커 AI(Libratus, Pluribus)의 핵심 두뇌 엔진
"""
import random
import time
import numpy as np

# 액션 정의: 0 = PASS (Check/Fold), 1 = BET (Call/Raise)
PASS = 0
BET = 1
NUM_ACTIONS = 2

class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.strategy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.strategy_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)

    def get_strategy(self, realization_weight):
        # Regret-Matching: 양수인 후회값에 비례하여 확률 배분
        normalizing_sum = 0.0
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += self.strategy[a]

        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / NUM_ACTIONS
            self.strategy_sum[a] += realization_weight * self.strategy[a]

        return self.strategy

    def get_average_strategy(self):
        # 수렴된 최종 내쉬 균형(GTO) 전략
        avg_strategy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        normalizing_sum = np.sum(self.strategy_sum)
        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / NUM_ACTIONS
        return avg_strategy

    def __str__(self):
        avg = self.get_average_strategy()
        return f"{self.info_set:<6} ➔ Pass(Check/Fold): {avg[0]:.1%}, Bet(Call/Raise): {avg[1]:.1%}"


class KuhnTrainer:
    def __init__(self):
        self.node_map = {}

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # 종료 조건 (Terminal State) 판별
        if plays >= 2:
            # 1. 둘 다 패스 (Check - Check)
            if history[-2:] == "pp":
                return 1 if cards[player] > cards[opponent] else -1
            # 2. 베팅 후 패스 (Bet - Fold)
            if history[-2:] == "bp":
                return 1
            # 3. 베팅 후 콜 (Bet - Call)
            if history[-2:] == "bb":
                return 2 if cards[player] > cards[opponent] else -2

        # 현재 플레이어의 정보 세트 (내 카드 + 액션 히스토리)
        card_str = "J" if cards[player] == 1 else ("Q" if cards[player] == 2 else "K")
        info_set = card_str + history

        if info_set not in self.node_map:
            self.node_map[info_set] = Node(info_set)
        node = self.node_map[info_set]

        # 현재 확률 전략 추출
        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = np.zeros(NUM_ACTIONS, dtype=np.float64)
        node_util = 0.0

        # 각 액션에 대한 재귀 탐색
        for a in range(NUM_ACTIONS):
            next_history = history + ("p" if a == PASS else "b")
            if player == 0:
                util[a] = -self.cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        # 가상 후회(Counterfactual Regret) 누적
        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            prob = p1 if player == 0 else p0
            node.regret_sum[a] += prob * regret

        return node_util

    def train(self, iterations=100000):
        cards = [1, 2, 3] # J=1, Q=2, K=3
        util = 0.0
        start = time.time()

        for i in range(1, iterations + 1):
            random.shuffle(cards)
            util += self.cfr(cards, "", 1.0, 1.0)

        elapsed = time.time() - start
        return elapsed, util / iterations

if __name__ == "__main__":
    print("==========================================================")
    print("🧠 CFR(Counterfactual Regret Minimization) 포커 자가대국 훈련")
    print("==========================================================")
    iterations = 200000
    print(f"총 {iterations:,} 판 무화면 초고속 자가 대국(Self-Play) 시작...")

    trainer = KuhnTrainer()
    elapsed, avg_util = trainer.train(iterations)

    print(f"⏱️ 훈련 완료 소요 시간: {elapsed:.2f}초 (초당 {iterations/elapsed:,.0f} 판 연산)")
    print(f"📊 게임 이론적 가치(Game Value, P1 기대값): {avg_util:.4f} (수학 이론값: -0.0555)")
    print("----------------------------------------------------------")
    print("🏆 [CFR이 스스로 도출해 낸 최적 GTO 전략 표]")
    print("----------------------------------------------------------")

    # 정렬하여 출력
    sorted_nodes = sorted(trainer.node_map.values(), key=lambda n: (len(n.info_set), n.info_set))
    for node in sorted_nodes:
        print(node)

    print("==========================================================")
    print("💡 수학적 검증 포인트:")
    print("1. J를 든 P1은 약 33%(1/3) 확률로 블러핑(Bet)을 칩니다.")
    print("2. K를 든 P1은 상대의 블러핑을 유도하기 위해 체크와 베팅을 섞습니다.")
    print("3. Q를 든 플레이어는 블러핑에 당하지 않는 정확한 방어 빈도를 유지합니다.")
    print("AI에게 규칙 외에는 아무것도 안 가르쳤지만, 수십만 판의 후회 계산만으로 완벽한 내쉬 균형에 도달했습니다!")
