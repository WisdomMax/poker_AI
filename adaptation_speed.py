"""
Poker AI V3.5 - [4. 적응 속도 및 구간별 성능 측정 (Adaptation Speed)]
상대 전략 급변 직후 구간별(0~100, 101~300, 301~500, 501~1k, 1k~2k, 2k~5k) 수익률 및 베이지안 파라미터 오차 추적
"""
import random
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer
from benchmark_v3 import RockPlayer, ManiacPlayer, CallingStationPlayer

BINS = [
    ("0~100판", 0, 100),
    ("101~300판", 100, 300),
    ("301~500판", 300, 500),
    ("501~1,000판", 500, 1000),
    ("1,001~2,000판", 1000, 2000),
    ("2,001~5,000판", 2000, 5000)
]

def measure_adaptation_bins(gto_strategy: dict, pre_opponent, target_opponent, num_runs: int = 10):
    """
    1) pre_opponent와 5,000판 대결하여 AI의 신념(Belief)이 pre_opponent에 젖어들게 함.
    2) T=0 시점에 갑자기 target_opponent로 변경! (AI에 알리지 않음, 데이터 리셋 없음)
    3) 이후 5,000판 동안 각 빈별 bb/100 및 추정 오차(MAE) 측정
    """
    cards = [1, 2, 3]
    bin_profits = {b[0]: [] for b in BINS}
    bin_maes = {b[0]: [] for b in BINS}

    for seed in range(num_runs):
        random.seed(seed * 77 + 1)
        np.random.seed(seed * 77 + 1)

        v3 = BayesianExploiterPlayer("V3", gto_strategy, decay_factor=0.9992)

        # 1. 이전 상대로 사전 노출 (Pre-conditioning 5,000판)
        for i in range(5000):
            random.shuffle(cards)
            ai_p0 = (i % 2 == 0)
            p0 = v3 if ai_p0 else pre_opponent
            p1 = pre_opponent if ai_p0 else v3
            
            history = ""
            while not is_terminal(history):
                curr_p = p0 if len(history) % 2 == 0 else p1
                curr_c = cards[len(history) % 2]
                history += ("p" if curr_p.act(curr_c, history) == PASS else "b")
                
            p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
            ai_pos = 0 if ai_p0 else 1
            opp_pos = 1 - ai_pos
            is_showdown = history in ("pp", "bb", "pbb")
            opp_card_shown = (cards[1] if ai_p0 else cards[0]) if is_showdown else None
            v3.model.update_hand(history, opp_pos, opp_card_shown, cards[ai_pos])

        # 2. 갑자기 target_opponent로 변경 후 5,000판 추적
        current_hand = 0
        bin_idx = 0
        current_bin_profit = 0

        for i in range(5000):
            random.shuffle(cards)
            ai_p0 = (i % 2 == 0)
            p0 = v3 if ai_p0 else target_opponent
            p1 = target_opponent if ai_p0 else v3
            
            history = ""
            while not is_terminal(history):
                curr_p = p0 if len(history) % 2 == 0 else p1
                curr_c = cards[len(history) % 2]
                history += ("p" if curr_p.act(curr_c, history) == PASS else "b")
                
            p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
            ai_pay = p0_pay if ai_p0 else p1_pay
            current_bin_profit += ai_pay
            
            ai_pos = 0 if ai_p0 else 1
            opp_pos = 1 - ai_pos
            is_showdown = history in ("pp", "bb", "pbb")
            opp_card_shown = (cards[1] if ai_p0 else cards[0]) if is_showdown else None
            v3.model.update_hand(history, opp_pos, opp_card_shown, cards[ai_pos])
            
            current_hand += 1
            
            # 구간 끝에 도달했을 때 기록
            b_name, b_start, b_end = BINS[bin_idx]
            if current_hand == b_end:
                hands_in_bin = b_end - b_start
                bb100 = (current_bin_profit / hands_in_bin) * 100
                bin_profits[b_name].append(bb100)
                
                # 추정 오차 (MAE) 계산:
                # target_opponent가 Rock일 때: J:b, Q:b, K:b 등에서 실제 Bet 확률과 AI의 추정 Bet 확률 차이
                # 예: Rock의 실제 Bet 확률: J:p = 0.0, K:p = 1.0, J: = 0.0, K: = 1.0
                est_errs = []
                for c in [1, 2, 3]:
                    for h in ["", "p"]:
                        actual_bet = 1.0 if c == 3 else 0.0
                        est_bet = v3.model.get_action_prob(c, h)[BET]
                        est_errs.append(abs(actual_bet - est_bet))
                bin_maes[b_name].append(np.mean(est_errs))
                
                current_bin_profit = 0
                bin_idx += 1
                if bin_idx >= len(BINS):
                    break

    return bin_profits, bin_maes


if __name__ == "__main__":
    print("==========================================================")
    print("⏱️ [4. 적응 속도 및 구간별 성능 측정 (Adaptation Speed)]")
    print("==========================================================")
    
    trainer = CFRTrainerV3()
    trainer.train(100000)
    gto_strat = trainer.get_strategy_dict()

    print("시나리오: Maniac(공격적 상대)에 5,000판 적응되어 있던 V3에게,")
    print("          갑자기 T=0에 극보수 Rock(바위 상대)으로 변경되었을 때의 적응 곡선 측정\n")

    maniac = ManiacPlayer()
    rock = RockPlayer()

    bin_profits, bin_maes = measure_adaptation_bins(gto_strat, maniac, rock, num_runs=10)

    print(f"{'구간 (상대 교체 후)':<16} | {'평균 bb/100 (10회 평균)':<22} | {'전략 추정 오차 (MAE)':<20} | {'적응 상태 평가':<15}")
    print("-" * 80)
    
    for b_name, _, _ in BINS:
        p_list = bin_profits[b_name]
        m_list = bin_maes[b_name]
        mean_p = np.mean(p_list)
        std_p = np.std(p_list, ddof=1)
        mean_m = np.mean(m_list)
        
        status = "적응 전환기" if mean_p < 10 else ("착취 궤도 진입 🚀" if mean_p > 20 else "점진적 개선")
        print(f"{b_name:<16} | {mean_p:+6.2f} ± {1.96*std_p/np.sqrt(10):4.2f} bb/100    | MAE: {mean_m:.3f} ({mean_m*100:.1f}%)   | {status}")
    print("==========================================================")
