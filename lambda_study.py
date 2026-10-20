"""
Poker AI V3.5 - [5. 베이지안 학습 안정성 및 망각 계수(Lambda) 비교]
λ = 0.9900, 0.9950, 0.9990, 0.9992, 0.9999 비교
1) 정적 고정 환경 (Stationary: Rock 고정 30,000판)
2) 급변 동적 환경 (Non-Stationary: 2,000판마다 Rock <-> Maniac 교대)
"""
import random
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer
from benchmark_v3 import RockPlayer, ManiacPlayer

LAMBDAS = [0.9900, 0.9950, 0.9990, 0.9992, 0.9999]

def evaluate_lambda_stationary(lam: float, gto_strategy: dict, num_hands=20000, runs=5) -> float:
    cards = [1, 2, 3]
    run_profits = []
    rock = RockPlayer()
    
    for seed in range(runs):
        random.seed(seed * 31 + 7)
        np.random.seed(seed * 31 + 7)
        v3 = BayesianExploiterPlayer("V3", gto_strategy, decay_factor=lam)
        profit = 0
        
        for i in range(num_hands):
            random.shuffle(cards)
            ai_p0 = (i % 2 == 0)
            p0 = v3 if ai_p0 else rock
            p1 = rock if ai_p0 else v3
            
            history = ""
            while not is_terminal(history):
                curr_p = p0 if len(history) % 2 == 0 else p1
                curr_c = cards[len(history) % 2]
                history += ("p" if curr_p.act(curr_c, history) == PASS else "b")
                
            p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
            profit += (p0_pay if ai_p0 else p1_pay)
            
            ai_pos = 0 if ai_p0 else 1
            opp_pos = 1 - ai_pos
            is_showdown = history in ("pp", "bb", "pbb")
            opp_card_shown = (cards[1] if ai_p0 else cards[0]) if is_showdown else None
            v3.model.update_hand(history, opp_pos, opp_card_shown, cards[ai_pos])
            
        run_profits.append((profit / num_hands) * 100)
    return float(np.mean(run_profits))


def evaluate_lambda_dynamic(lam: float, gto_strategy: dict, cycle_len=2000, total_cycles=6, runs=5) -> float:
    cards = [1, 2, 3]
    run_profits = []
    
    for seed in range(runs):
        random.seed(seed * 43 + 11)
        np.random.seed(seed * 43 + 11)
        v3 = BayesianExploiterPlayer("V3", gto_strategy, decay_factor=lam)
        total_hands = cycle_len * total_cycles
        profit = 0
        
        for c_idx in range(total_cycles):
            # 짝수 사이클은 Rock, 홀수 사이클은 Maniac
            current_opp = RockPlayer() if c_idx % 2 == 0 else ManiacPlayer()
            for i in range(cycle_len):
                random.shuffle(cards)
                ai_p0 = (i % 2 == 0)
                p0 = v3 if ai_p0 else current_opp
                p1 = current_opp if ai_p0 else v3
                
                history = ""
                while not is_terminal(history):
                    curr_p = p0 if len(history) % 2 == 0 else p1
                    curr_c = cards[len(history) % 2]
                    history += ("p" if curr_p.act(curr_c, history) == PASS else "b")
                    
                p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
                profit += (p0_pay if ai_p0 else p1_pay)
                
                ai_pos = 0 if ai_p0 else 1
                opp_pos = 1 - ai_pos
                is_showdown = history in ("pp", "bb", "pbb")
                opp_card_shown = (cards[1] if ai_p0 else cards[0]) if is_showdown else None
                v3.model.update_hand(history, opp_pos, opp_card_shown, cards[ai_pos])
                
        run_profits.append((profit / total_hands) * 100)
    return float(np.mean(run_profits))


if __name__ == "__main__":
    print("==========================================================")
    print("🔬 [5. 베이지안 학습 안정성 및 망각 계수(Lambda) 비교]")
    print("==========================================================")
    
    trainer = CFRTrainerV3()
    trainer.train(100000)
    gto_strat = trainer.get_strategy_dict()

    print(f"{'망각 계수 (λ)':<15} | {'정적 환경 (Rock 고정 20k)':<25} | {'급변 환경 (2k마다 교대 12k)':<25} | {'특성 분석'}")
    print("-" * 90)

    for lam in LAMBDAS:
        stat_score = evaluate_lambda_stationary(lam, gto_strat, num_hands=20000, runs=4)
        dyn_score = evaluate_lambda_dynamic(lam, gto_strat, cycle_len=2000, total_cycles=6, runs=4)
        
        trait = ""
        if lam <= 0.9950:
            trait = "빠른 적응, 높은 표본 노이즈"
        elif lam >= 0.9999:
            trait = "극도의 안정성, 둔감한 적응"
        else:
            trait = "균형 잡힌 절충 (Robust)"
            
        print(f"λ = {lam:<11.4f} | {stat_score:+10.2f} bb/100           | {dyn_score:+10.2f} bb/100           | {trait}")
    print("==========================================================")
