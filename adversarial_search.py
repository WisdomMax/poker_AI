"""
Poker AI V3.5 - [3. 자동 적대적 전략 탐색 (Adversarial Search)]
진화 전략(Evolution Strategy)을 통해 V3를 가장 아프게 털어먹는 적대적 상대 파라미터 θ* 자동 발견
"""
import random
import time
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer

INFO_SETS = [
    'J:', 'Q:', 'K:',       # P0 첫 행동 (3개)
    'J:p', 'Q:p', 'K:p',    # P1 상대 체크 후 행동 (3개)
    'J:b', 'Q:b', 'K:b',    # P1 상대 베팅 직면 시 행동 (3개)
    'J:pb', 'Q:pb', 'K:pb'  # P0 체크 후 상대 베팅 직면 시 행동 (3개)
]

class ParametricOpponent:
    """12차원 베팅 확률 벡터 theta를 사용하는 파라메트릭 상대"""
    def __init__(self, theta: np.ndarray, name="ParametricOpponent"):
        self.theta = np.clip(theta, 0.0, 1.0)
        self.name = name
        self.prob_map = {info_set: float(self.theta[i]) for i, info_set in enumerate(INFO_SETS)}

    def act(self, card: int, history: str) -> int:
        key = f"{CARD_NAMES[card]}:{history}"
        p_bet = self.prob_map.get(key, 0.5)
        return BET if random.random() < p_bet else PASS


def evaluate_against_v3(theta: np.ndarray, gto_strategy: dict, num_hands: int = 4000, seed: int = 42) -> float:
    """
    V3를 상대로 theta 상대가 얻는 평균 수익 (bb/100) 측정
    높을수록 V3에게 큰 손실을 입히는 강력한 적대적 전략
    """
    random.seed(seed)
    np.random.seed(seed)
    
    opponent = ParametricOpponent(theta)
    v3 = BayesianExploiterPlayer("V3", gto_strategy, decay_factor=0.9992)
    
    cards = [1, 2, 3]
    opp_total_profit = 0
    
    for i in range(num_hands):
        random.shuffle(cards)
        opp_is_p0 = (i % 2 == 0)
        p0 = opponent if opp_is_p0 else v3
        p1 = v3 if opp_is_p0 else opponent
        
        history = ""
        while not is_terminal(history):
            curr_p = p0 if len(history) % 2 == 0 else p1
            curr_c = cards[len(history) % 2]
            action = curr_p.act(curr_c, history)
            history += ("p" if action == PASS else "b")
            
        p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
        opp_pay = p0_pay if opp_is_p0 else p1_pay
        opp_total_profit += opp_pay
        
        # V3 베이지안 업데이트
        v3_pos = 1 if opp_is_p0 else 0
        opp_pos = 1 - v3_pos
        is_showdown = history in ("pp", "bb", "pbb")
        opp_card_shown = (cards[0] if opp_is_p0 else cards[1]) if is_showdown else None
        v3.model.update_hand(history, opp_pos, opp_card_shown, cards[v3_pos])

    return (opp_total_profit / num_hands) * 100.0


def run_evolutionary_search(gto_strategy: dict, pop_size: int = 16, generations: int = 12) -> tuple[np.ndarray, float]:
    """
    (μ + λ) 진화 전략으로 V3 킬러 전략 탐색
    """
    dim = len(INFO_SETS)
    # 초기 개체군: GTO 근처 + 랜덤 다양성
    population = [np.random.uniform(0.0, 1.0, size=dim) for _ in range(pop_size)]
    # GTO 파라미터도 시드로 주입
    gto_theta = np.array([gto_strategy.get(k, (0.5, 0.5))[1] for k in INFO_SETS])
    population[0] = gto_theta.copy()
    
    best_theta = gto_theta.copy()
    best_score = -999.0
    
    sigma = 0.25 # 돌연변이 표준편차
    
    print(f"진화 전략 탐색 시작 (개체수: {pop_size}, 세대수: {generations})...")
    t0 = time.time()
    
    for gen in range(1, generations + 1):
        scores = []
        for i, ind in enumerate(population):
            score = evaluate_against_v3(ind, gto_strategy, num_hands=2500, seed=gen * 100 + i)
            scores.append((score, ind))
            if score > best_score:
                best_score = score
                best_theta = ind.copy()
                
        scores.sort(key=lambda x: x[0], reverse=True)
        top_half = [x[1] for x in scores[:pop_size // 2]]
        
        print(f"  [Gen {gen:02d}] 최고 적대자 성적: {scores[0][0]:+6.2f} bb/100 (누적 최고: {best_score:+6.2f} bb/100)")
        
        # 차세대 생성: 교배 + 가우시안 돌연변이
        new_pop = [best_theta.copy()] # 엘리티즘
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(top_half, 2)
            crossover = np.where(np.random.rand(dim) > 0.5, p1, p2)
            mutation = crossover + np.random.normal(0, sigma, size=dim)
            new_pop.append(np.clip(mutation, 0.0, 1.0))
            
        population = new_pop
        sigma *= 0.90 # 점진적 수렴
        
    elapsed = time.time() - t0
    print(f"탐색 완료! ({elapsed:.1f}초)")
    return best_theta, best_score


if __name__ == "__main__":
    print("==========================================================")
    print("🎯 [3. 자동 적대적 전략 탐색 (Adversarial Search)]")
    print("==========================================================")
    
    trainer = CFRTrainerV3()
    trainer.train(100000)
    gto_strat = trainer.get_strategy_dict()
    
    best_theta, train_score = run_evolutionary_search(gto_strat, pop_size=16, generations=10)
    
    print("\n----------------------------------------------------------")
    print("🔥 [발견된 최강 적대적 전략(V3 Killer) 파라미터 θ*]")
    print("----------------------------------------------------------")
    for i, info_set in enumerate(INFO_SETS):
        print(f"  {info_set:<8} ➔ Bet 확률: {best_theta[i]:.1%}")
        
    print("\n독립 테스트 데이터 및 시드로 50,000판 재검증 실행 중...")
    retest_score = evaluate_against_v3(best_theta, gto_strat, num_hands=50000, seed=99999)
    print(f"🏆 독립 시드 50,000판 재검증 결과: 적대자 {retest_score:+6.2f} bb/100 (V3 손실: {-retest_score:+6.2f} bb/100)")
