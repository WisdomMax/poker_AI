"""
Kuhn Poker V3.0 - 종합 검증 및 다이내믹 벤치마크 프레임워크
"""
import random
import time
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer

# ==========================================
# 1. 벤치마크용 상대 플레이어 클래스들
# (상대 내부 로직을 Bayesian AI는 절대 보지 못함)
# ==========================================
class RandomPlayer:
    def __init__(self, name="Random"):
        self.name = name
    def act(self, card: int, history: str) -> int:
        return random.choice(ACTIONS)

class CFRGTOPlayer:
    def __init__(self, name: str, gto_strategy: dict[str, tuple[float, float]]):
        self.name = name
        self.gto_strategy = gto_strategy
    def act(self, card: int, history: str) -> int:
        info_set = f"{CARD_NAMES[card]}:{history}"
        p = self.gto_strategy.get(info_set, (0.5, 0.5))
        return int(np.random.choice(ACTIONS, p=p))

class RockPlayer:
    def __init__(self, name="Rock"):
        self.name = name
    def act(self, card: int, history: str) -> int:
        return BET if card == 3 else PASS

class ManiacPlayer:
    def __init__(self, name="Maniac"):
        self.name = name
    def act(self, card: int, history: str) -> int:
        return BET

class CallingStationPlayer:
    def __init__(self, name="CallingStation"):
        self.name = name
    def act(self, card: int, history: str) -> int:
        if len(history) > 0 and history[-1] == 'b':
            return BET # Call
        return PASS # Check

class MixedPlayer:
    """복합 전략 플레이어: 50% Maniac, 30% CallingStation, 20% Rock"""
    def __init__(self, name="Mixed"):
        self.name = name
        self.rock = RockPlayer()
        self.maniac = ManiacPlayer()
        self.station = CallingStationPlayer()
    def act(self, card: int, history: str) -> int:
        r = random.random()
        if r < 0.50:
            return self.maniac.act(card, history)
        elif r < 0.80:
            return self.station.act(card, history)
        else:
            return self.rock.act(card, history)

class LegacyHardcodedExploiterPlayer:
    """기존 V2 하드코딩 룰 착취 플레이어 (단, 수정된 V3 손익 엔진 기준 테스트용)"""
    def __init__(self, name: str, gto_strategy: dict[str, tuple[float, float]]):
        self.name = name
        self.gto_strategy = gto_strategy
        self.times_faced_bet = 0
        self.times_folded = 0
    def act(self, card: int, history: str) -> int:
        fold_rate = (self.times_folded / self.times_faced_bet) if self.times_faced_bet >= 5 else 0.5
        if fold_rate >= 0.60:
            if history == "p" and card == 1: return BET
            if history.endswith("b"): return PASS
            if history == "" and card == 3: return BET
        info_set = f"{CARD_NAMES[card]}:{history}"
        p = self.gto_strategy.get(info_set, (0.5, 0.5))
        return int(np.random.choice(ACTIONS, p=p))


# ==========================================
# 2. 단일 핸드 시뮬레이터
# ==========================================
def play_single_hand(p0, p1, cards: list[int]) -> tuple[str, int, int]:
    history = ""
    while not is_terminal(history):
        curr_p = p0 if len(history) % 2 == 0 else p1
        curr_c = cards[len(history) % 2]
        action = curr_p.act(curr_c, history)
        history += ("p" if action == PASS else "b")

    p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
    return history, p0_pay, p1_pay


# ==========================================
# 3. 다이내믹 상대 100,000판 시뮬레이션
# ==========================================
def run_dynamic_100k(ai_player, seed: int) -> tuple[float, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    
    # 5개 구간 정의 (각 20,000판)
    phases = [
        ("0~20k (Rock)", RockPlayer()),
        ("20k~40k (Maniac)", ManiacPlayer()),
        ("40k~60k (CallingStation)", CallingStationPlayer()),
        ("60k~80k (Mixed)", MixedPlayer()),
        ("80k~100k (Rock 복귀)", RockPlayer())
    ]
    
    cards = [1, 2, 3]
    total_ai_profit = 0
    phase_profits = []
    
    for phase_name, opp in phases:
        p_profit = 0
        for i in range(20000):
            random.shuffle(cards)
            ai_is_p0 = (i % 2 == 0) # 50% 포지션 공정 배정
            
            p0 = ai_player if ai_is_p0 else opp
            p1 = opp if ai_is_p0 else ai_player
            c0, c1 = cards[0], cards[1]
            
            history, p0_pay, p1_pay = play_single_hand(p0, p1, cards)
            ai_pay = p0_pay if ai_is_p0 else p1_pay
            p_profit += ai_pay
            
            # 베이지안 플레이어의 사후 업데이트 (상대 카드는 쇼다운 시에만 제공)
            if hasattr(ai_player, "model"):
                ai_pos = 0 if ai_is_p0 else 1
                opp_pos = 1 - ai_pos
                # 쇼다운 여부: pp, bb, pbb
                is_showdown = history in ("pp", "bb", "pbb")
                opp_card_shown = (c1 if ai_is_p0 else c0) if is_showdown else None
                ai_player.model.update_hand(history, opp_pos, opp_card_shown, c0 if ai_is_p0 else c1)
            
            # 레거시 하드코딩 봇 트래커 업데이트
            if hasattr(ai_player, "times_faced_bet"):
                ai_pos = 0 if ai_is_p0 else 1
                for t in range(len(history)):
                    if t % 2 != ai_pos and (t > 0 and history[t-1] == 'b'):
                        ai_player.times_faced_bet += 1
                        if history[t] == 'p':
                            ai_player.times_folded += 1

        phase_bb100 = (p_profit / 20000) * 100
        phase_profits.append(phase_bb100)
        total_ai_profit += p_profit

    overall_bb100 = (total_ai_profit / 100000) * 100
    return overall_bb100, phase_profits


if __name__ == "__main__":
    print("==========================================================")
    print("🔬 [프로젝트: Poker AI V3.0 종합 검증 및 다이내믹 벤치마크]")
    print("==========================================================")
    
    # 1. CFR GTO 사전 훈련
    print("1. CFR GTO 사전 훈련 진행 중 (200,000 iterations)...")
    cfr_trainer = CFRTrainerV3()
    cfr_trainer.train(200000)
    gto_strategy = cfr_trainer.get_strategy_dict()
    print("   ➔ CFR GTO 전략 학습 완료.")

    # 2. GTO vs GTO 이론값 검증 (100,000판)
    print("\n2. GTO vs GTO 수학적 균형 이론값 검증 (100,000판)...")
    gto_p0 = CFRGTOPlayer("GTO_P0", gto_strategy)
    gto_p1 = CFRGTOPlayer("GTO_P1", gto_strategy)
    
    # P0 기준 이론값: -1/18 (-0.05555... 즉 -5.556 bb/100)
    cards = [1, 2, 3]
    p0_fixed_profit = 0
    for _ in range(100000):
        random.shuffle(cards)
        _, p0_pay, _ = play_single_hand(gto_p0, gto_p1, cards)
        p0_fixed_profit += p0_pay
    
    p0_fixed_bb100 = (p0_fixed_profit / 100000) * 100
    print(f"   ➔ P0 고정 포지션 기대값: {p0_fixed_bb100:+.3f} bb/100 (이론값: -5.556 bb/100)")
    
    # 포지션 교대 공정 대결 시 이론값: 0.000 bb/100
    fair_profit = 0
    for i in range(100000):
        random.shuffle(cards)
        p0 = gto_p0 if i % 2 == 0 else gto_p1
        p1 = gto_p1 if i % 2 == 0 else gto_p0
        _, pay0, pay1 = play_single_hand(p0, p1, cards)
        fair_profit += (pay0 if i % 2 == 0 else pay1)
    fair_bb100 = (fair_profit / 100000) * 100
    print(f"   ➔ 포지션 교대 공정 대결: {fair_bb100:+.3f} bb/100 (이론값: 0.000 bb/100)")

    # 3. 4종 AI x 10개 독립 시드 다이내믹 벤치마크
    print("\n3. 4종 AI x 10개 독립 시드(Seed 1..10) 100,000판 다이내믹 벤치마크 실행...")
    print("   [구간 구성] 0~20k: Rock | 20k~40k: Maniac | 40k~60k: Station | 60k~80k: Mixed | 80k~100k: Rock복귀\n")

    agents_to_test = [
        ("A. Random Player", lambda: RandomPlayer()),
        ("B. CFR GTO Player", lambda: CFRGTOPlayer("CFR_GTO", gto_strategy)),
        ("C. Legacy Hardcoded AI", lambda: LegacyHardcodedExploiterPlayer("Legacy", gto_strategy)),
        ("D. Bayesian Exploiter V3", lambda: BayesianExploiterPlayer("Bayesian_V3", gto_strategy, decay_factor=0.9992))
    ]

    all_results = {}

    for agent_label, constructor in agents_to_test:
        seed_overalls = []
        seed_phases = []
        t0 = time.time()
        
        for s in range(1, 11):
            player = constructor()
            overall, phases = run_dynamic_100k(player, seed=s)
            seed_overalls.append(overall)
            seed_phases.append(phases)

        elapsed = time.time() - t0
        mean_overall = np.mean(seed_overalls)
        std_overall = np.std(seed_overalls, ddof=1)
        ci95 = 1.96 * (std_overall / np.sqrt(10))
        mean_phases = np.mean(seed_phases, axis=0)

        all_results[agent_label] = {
            "mean": mean_overall,
            "ci95": ci95,
            "phases": mean_phases,
            "elapsed": elapsed
        }
        print(f"[{agent_label}] 10개 시드 완료 ({elapsed:.1f}초) ➔ 평균: {mean_overall:+.2f} ± {ci95:.2f} bb/100")

    print("\n==========================================================")
    print("📊 [4종 AI 최종 다이내믹 벤치마크 결과 종합]")
    print("==========================================================")
    header = f"{'AI 모델':<25} | {'전체 bb/100 (95% CI)':<22} | {'0~20k(Rock)':<11} | {'20~40k(Maniac)':<14} | {'40~60k(Station)':<15} | {'60~80k(Mixed)':<13} | {'80~100k(Rock2)':<13}"
    print(header)
    print("-" * len(header))

    for label, r in all_results.items():
        ci_str = f"{r['mean']:+.2f} ± {r['ci95']:.2f}"
        p = r["phases"]
        print(f"{label:<25} | {ci_str:<22} | {p[0]:+8.2f}    | {p[1]:+11.2f}    | {p[2]:+12.2f}    | {p[3]:+10.2f}   | {p[4]:+10.2f}")
    print("==========================================================")
