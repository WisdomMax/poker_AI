"""
포커 AI 리그 아레나 (League Arena)
CFR AI vs 다양한 인간 스타일(Rock, Maniac, Calling Station) 맞대결 및 bb/100 수익률 측정
"""
import random
import numpy as np
from cfr_engine import KuhnTrainer, PASS, BET

class CFRPlayer:
    def __init__(self, name, node_map):
        self.name = name
        self.node_map = node_map

    def act(self, card, history):
        card_str = "J" if card == 1 else ("Q" if card == 2 else "K")
        info_set = card_str + history
        if info_set in self.node_map:
            strategy = self.node_map[info_set].get_average_strategy()
            return np.random.choice([PASS, BET], p=strategy)
        return PASS

class RockPlayer:
    """극보수 바위: K(가장 높은 패)일 때만 베팅하고 나머지는 무조건 사림"""
    def __init__(self, name="Rock(극보수 바위)"):
        self.name = name

    def act(self, card, history):
        if card == 3: # King
            return BET
        return PASS

class ManiacPlayer:
    """블러핑 광인: 패가 뭐든 무조건 100% 베팅/레이즈만 지름"""
    def __init__(self, name="Maniac(블러핑 광인)"):
        self.name = name

    def act(self, card, history):
        return BET

class CallingStationPlayer:
    """콜링 스테이션(호구): 먼저 베팅은 안 하지만 상대가 베팅하면 무조건 콜"""
    def __init__(self, name="CallingStation(호구)"):
        self.name = name

    def act(self, card, history):
        if len(history) > 0 and history[-1] == 'b':
            return BET # Call
        return PASS # Check

def play_match(p0, p1, num_hands=50000):
    cards = [1, 2, 3]
    p0_profit = 0.0

    for _ in range(num_hands):
        random.shuffle(cards)
        c0, c1 = cards[0], cards[1]
        history = ""

        # Hand Loop
        while True:
            plays = len(history)
            curr_p = p0 if plays % 2 == 0 else p1
            curr_c = c0 if plays % 2 == 0 else c1

            action = curr_p.act(curr_c, history)
            history += ("p" if action == PASS else "b")

            # 종료 판정
            if len(history) >= 2:
                if history[-2:] == "pp":
                    winner = 0 if c0 > c1 else 1
                    p0_profit += (1 if winner == 0 else -1)
                    break
                elif history[-2:] == "bp":
                    # 먼저 친 사람이 승리 (상대 폴드)
                    winner = 0 if len(history) % 2 != 0 else 1
                    p0_profit += (1 if winner == 0 else -1)
                    break
                elif history[-2:] == "bb":
                    winner = 0 if c0 > c1 else 1
                    p0_profit += (2 if winner == 0 else -2)
                    break

    # bb/100 계산 (100판당 획득 빅블라인드)
    bb_per_100 = (p0_profit / num_hands) * 100
    return bb_per_100, p0_profit

if __name__ == "__main__":
    print("==========================================================")
    print("⚔️ 포커 AI 리그 (League Tournament): CFR vs 인간 성향 봇")
    print("==========================================================")
    
    # 1. CFR AI 사전 훈련
    trainer = KuhnTrainer()
    trainer.train(200000)
    cfr_agent = CFRPlayer("CFR GTO AI", trainer.node_map)

    opponents = [
        ManiacPlayer(),
        CallingStationPlayer(),
        RockPlayer()
    ]

    print("\n각 상대와 50,000판씩 맞대결 진행 중...\n")

    for opp in opponents:
        # P0(CFR) vs Opponent
        bb100_as_p0, profit_p0 = play_match(cfr_agent, opp, num_hands=50000)
        # 포지션 공정성을 위해 포지션 바꿔서 50,000판 추가
        bb100_as_p1, profit_p1 = play_match(opp, cfr_agent, num_hands=50000)
        
        # 종합 bb/100
        net_profit = profit_p0 - profit_p1
        total_bb100 = (net_profit / 100000) * 100

        verdict = "압승 🏆" if total_bb100 > 0 else "패배"
        print(f"🥊 vs {opp.name:<25}")
        print(f"   ➔ 총 100,000판 대결 손익: {net_profit:+,.0f} chips")
        print(f"   ➔ 국제 표준 실력 지표: {total_bb100:+.2f} bb/100 ({verdict})")
        print("----------------------------------------------------------")
