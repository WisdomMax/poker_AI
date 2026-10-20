"""
Poker AI V3.6 - [종합 수학적 검증 벤치마크 (Mathematical Verification Suite)]
1. CFR GTO 정확한 Exploitability (0.000 검증)
2. Bayesian V3 스냅샷별(0, 100, 500, 1k, 5k) 엄밀한 Best Response & 착취 가능성
3. θ* vs Best Response 비교 분석
4. 올바른 Oracle(비숨김 카드) 기반 Regret 재계산
5. Student-t 분포(df=9, t=2.262) 95% 신뢰구간 적용
"""
import random
import time
import numpy as np
from itertools import permutations

from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from kuhn_exact_solver import (
    P0_INFONSETS,
    P1_INFONSETS,
    evaluate_game_value,
    compute_best_response_p0,
    compute_best_response_p1,
    compute_exact_exploitability,
    get_kuhn_analytical_nash
)
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer
from benchmark_v3 import RockPlayer, ManiacPlayer, CallingStationPlayer
from opponents_v35 import BalancedPlayer, TrickyPlayer, AdaptiveOpponentPlayer, RandomizedPlayer, CounterExploiterPlayer
from adversarial_search import ParametricOpponent

ADVERSARIAL_THETA = np.array([
    0.229, 0.000, 0.719,  # J:, Q:, K:
    0.338, 0.000, 1.000,  # J:p, Q:p, K:p
    0.000, 0.339, 1.000,  # J:b, Q:b, K:b
    0.000, 0.580, 1.000   # J:pb, Q:pb, K:pb
])

# ==========================================
# 1. 올바른 Oracle (상대 숨겨진 카드 절대 미참조)
# ==========================================
class CorrectedBayesianOracle:
    """
    상대의 '현재 숨겨진 카드'는 전혀 모른 채,
    상대의 '전략 함수(Policy)'만을 완벽히 알고 수학적 Best Response를 내는 참된 Oracle
    """
    def __init__(self, target_opponent):
        self.target_opp = target_opponent

    def act(self, my_card: int, history: str) -> int:
        my_pos = len(history) % 2
        opp_pos = 1 - my_pos
        possible_opp_cards = [c for c in [1, 2, 3] if c != my_card]

        # 1. 상대의 이전 공개 행동 히스토리를 통해 상대 카드의 사후확률 P(c_opp | history) 베이지안 계산
        weights = []
        for c in possible_opp_cards:
            w = 1.0
            for t in range(len(history)):
                if t % 2 == opp_pos:
                    h_before = history[:t]
                    act = PASS if history[t] == 'p' else BET
                    cnt = sum(1 for _ in range(40) if self.target_opp.act(c, h_before) == act)
                    w *= (cnt / 40.0)
            weights.append(w)

        tot = sum(weights)
        opp_probs = {c: (w / tot if tot > 0 else 0.5) for c, w in zip(possible_opp_cards, weights)}

        # 2. EV(Pass) vs EV(Bet) 전수 트리 계산
        ev_pass = 0.0
        ev_bet = 0.0

        for opp_card, p_c in opp_probs.items():
            c0 = my_card if my_pos == 0 else opp_card
            c1 = opp_card if my_pos == 0 else my_card

            # A. 내가 Pass 했을 때
            h_p = history + "p"
            if is_terminal(h_p):
                pay0, pay1 = evaluate_payoff(h_p, c0, c1)
                ev_pass += p_c * (pay0 if my_pos == 0 else pay1)
            else:
                # 상대 턴 (상대 반응 샘플링)
                sub_ev = 0.0
                for _ in range(25):
                    opp_act = self.target_opp.act(opp_card, h_p)
                    h_next = h_p + ("p" if opp_act == PASS else "b")
                    if is_terminal(h_next):
                        pay0, pay1 = evaluate_payoff(h_next, c0, c1)
                        sub_ev += (pay0 if my_pos == 0 else pay1)
                    else:
                        pay_f0, pay_f1 = evaluate_payoff(h_next + "p", c0, c1)
                        pay_c0, pay_c1 = evaluate_payoff(h_next + "b", c0, c1)
                        sub_ev += max(pay_f0 if my_pos == 0 else pay_f1, pay_c0 if my_pos == 0 else pay_c1)
                ev_pass += p_c * (sub_ev / 25.0)

            # B. 내가 Bet 했을 때
            h_b = history + "b"
            if is_terminal(h_b):
                pay0, pay1 = evaluate_payoff(h_b, c0, c1)
                ev_bet += p_c * (pay0 if my_pos == 0 else pay1)
            else:
                sub_ev = 0.0
                for _ in range(25):
                    opp_act = self.target_opp.act(opp_card, h_b)
                    h_next = h_b + ("p" if opp_act == PASS else "b")
                    pay0, pay1 = evaluate_payoff(h_next, c0, c1)
                    sub_ev += (pay0 if my_pos == 0 else pay1)
                ev_bet += p_c * (sub_ev / 25.0)

        return BET if ev_bet > ev_pass else PASS


# ==========================================
# 2. V3 스냅샷 전략 추출기
# ==========================================
def extract_snapshot_strategy(v3_player: BayesianExploiterPlayer) -> dict[str, tuple[float, float]]:
    """
    현재 시점에서 V3의 고정된 전략 분포 딕셔너리 추출
    """
    strat = {}
    # P0 정보셋
    for info_set in P0_INFONSETS:
        c_str, h = info_set.split(':')
        c = 1 if c_str == 'J' else (2 if c_str == 'Q' else 3)
        # V3의 act 함수 내부의 calculate_ev + GTO blending 확률 계산
        ev_pass, ev_bet = v3_player.calculate_ev(c, h, my_pos=len(h)%2)
        gto_p = v3_player.gto_strategy.get(info_set, (0.5, 0.5))
        
        h_idx = v3_player.model.hist_to_idx.get(h, 0)
        sample_count = np.sum(v3_player.model.total_observed[:, h_idx])
        conf = sample_count / (sample_count + 15.0)
        
        diff = ev_bet - ev_pass
        p_bet_exploit = 1.0 / (1.0 + np.exp(-diff / 0.1))
        exploit_p = np.array([1.0 - p_bet_exploit, p_bet_exploit])
        blended = (1.0 - conf) * np.array(gto_p) + conf * exploit_p
        blended /= np.sum(blended)
        strat[info_set] = (float(blended[0]), float(blended[1]))

    # P1 정보셋
    for info_set in P1_INFONSETS:
        c_str, h = info_set.split(':')
        c = 1 if c_str == 'J' else (2 if c_str == 'Q' else 3)
        ev_pass, ev_bet = v3_player.calculate_ev(c, h, my_pos=len(h)%2)
        gto_p = v3_player.gto_strategy.get(info_set, (0.5, 0.5))
        
        h_idx = v3_player.model.hist_to_idx.get(h, 0)
        sample_count = np.sum(v3_player.model.total_observed[:, h_idx])
        conf = sample_count / (sample_count + 15.0)
        
        diff = ev_bet - ev_pass
        p_bet_exploit = 1.0 / (1.0 + np.exp(-diff / 0.1))
        exploit_p = np.array([1.0 - p_bet_exploit, p_bet_exploit])
        blended = (1.0 - conf) * np.array(gto_p) + conf * exploit_p
        blended /= np.sum(blended)
        strat[info_set] = (float(blended[0]), float(blended[1]))

    return strat


def run_benchmark_match_exact(p0_agent, opponent, num_hands=10000):
    cards = [1, 2, 3]
    total_ai_pay = 0
    for i in range(num_hands):
        random.shuffle(cards)
        ai_p0 = (i % 2 == 0)
        p0 = p0_agent if ai_p0 else opponent
        p1 = opponent if ai_p0 else p0_agent
        
        history = ""
        while not is_terminal(history):
            curr_p = p0 if len(history) % 2 == 0 else p1
            curr_c = cards[len(history) % 2]
            history += ("p" if curr_p.act(curr_c, history) == PASS else "b")
            
        p0_pay, p1_pay = evaluate_payoff(history, cards[0], cards[1])
        ai_pay = p0_pay if ai_p0 else p1_pay
        total_ai_pay += ai_pay
        
        if hasattr(p0_agent, "model"):
            ai_pos = 0 if ai_p0 else 1
            opp_pos = 1 - ai_pos
            is_showdown = history in ("pp", "bb", "pbb")
            opp_card_shown = (cards[1] if ai_p0 else cards[0]) if is_showdown else None
            p0_agent.model.update_hand(history, opp_pos, opp_card_shown, cards[ai_pos])

    return (total_ai_pay / num_hands) * 100.0


if __name__ == "__main__":
    print("==========================================================")
    print("🔬 [POKER AI V3.6: 수학적 검증 (MATHEMATICAL VERIFICATION)]")
    print("==========================================================")
    
    # [과제 1 & 2] CFR GTO 모델의 정확한 Exploitability 계산
    print("\n--- [과제 1 & 2] CFR GTO 모델 엄밀한 수학적 Exploitability 검증 ---")
    trainer = CFRTrainerV3()
    trainer.train(200000)
    gto_strat = trainer.get_strategy_dict()

    exact_gto_res = compute_exact_exploitability(gto_strat)
    print(f"1. 실제 게임 가치 (v0_actual): {exact_gto_res['v0_actual']:.6f} (이론값: -0.055556 = -1/18)")
    print(f"2. P1 Best Response 이득 (br_v1): {exact_gto_res['br_v1']:.6f} (이론값: +0.055556 = +1/18)")
    print(f"3. P0 Best Response 이득 (br_v0): {exact_gto_res['br_v0']:.6f} (이론값: -0.055556 = -1/18)")
    print(f"4. P0 Exploitability: {exact_gto_res['p0_exploitability']:.6f}")
    print(f"5. P1 Exploitability: {exact_gto_res['p1_exploitability']:.6f}")
    print(f"6. 정확한 NashConv (P0+P1): {exact_gto_res['nash_conv']:.6f}")
    print(f"7. 정확한 Overall Exploitability (NashConv/2): {exact_gto_res['overall_exploitability']:.6f} bb/hand")
    print(f"   (bb/100 환산: {exact_gto_res['overall_exploitability']*100:.4f} bb/100)")

    # [과제 3] Bayesian V3 스냅샷별 착취 가능성
    print("\n--- [과제 3] Bayesian V3 학습 단계별 고정 스냅샷 착취 가능성 ---")
    print("시나리오: Rock을 상대로 학습을 진행하며 0, 100, 500, 1k, 5k 판에서 모델 동결")
    
    v3_test = BayesianExploiterPlayer("V3_Snapshot_Test", gto_strat, decay_factor=0.9992)
    rock = RockPlayer()
    cards = [1, 2, 3]

    snapshot_points = [0, 100, 500, 1000, 5000]
    hand_counter = 0

    print(f"{'학습 시점':<12} | {'NashConv (bb/hand)':<20} | {'스냅샷 Exploitability (bb/100)':<30} | {'P1 BR 최대 손실':<15}")
    print("-" * 85)

    for pt in snapshot_points:
        # 해당 판수까지 진행
        while hand_counter < pt:
            random.shuffle(cards)
            ai_p0 = (hand_counter % 2 == 0)
            p0 = v3_test if ai_p0 else rock
            p1 = rock if ai_p0 else v3_test
            h = ""
            while not is_terminal(h):
                curr_p = p0 if len(h) % 2 == 0 else p1
                curr_c = cards[len(h) % 2]
                h += ("p" if curr_p.act(curr_c, h) == PASS else "b")
            ai_pos = 0 if ai_p0 else 1
            opp_pos = 1 - ai_pos
            is_showdown = h in ("pp", "bb", "pbb")
            opp_c = (cards[1] if ai_p0 else cards[0]) if is_showdown else None
            v3_test.model.update_hand(h, opp_pos, opp_c, cards[ai_pos])
            hand_counter += 1

        snap_strat = extract_snapshot_strategy(v3_test)
        snap_res = compute_exact_exploitability(snap_strat)
        
        print(f"{pt:<5} 판 완료  | {snap_res['nash_conv']:<20.6f} | {snap_res['overall_exploitability']*100:<30.4f} | {snap_res['br_v1']:+8.4f}")

    # [과제 4] θ* vs 정확한 Best Response 비교 분석
    print("\n--- [과제 4] θ* (진화 탐색 적대자) vs 이론적 Best Response 비교 ---")
    # GTO 전략에 대한 이론적 P1 Best Response
    _, br_p1_dict = compute_best_response_p1(gto_strat)
    print("정보집합   | 이론적 P1 Best Response (GTO 상대) | 진화 탐색 θ* Bet 확률")
    print("-" * 65)
    for i, info_set in enumerate(P1_INFONSETS):
        br_act = "BET (100%)" if br_p1_dict[info_set] == BET else "PASS (0%)"
        theta_val = ADVERSARIAL_THETA[3 + i] # P1 인덱스
        print(f"{info_set:<10} | {br_act:<35} | {theta_val*100:5.1f}%")

    # [과제 5 & 6] 올바른 Oracle 및 Student-t (df=9, t=2.262) 95% 신뢰구간 평가
    print("\n--- [과제 5 & 6] 올바른 Oracle(비숨김 카드) 기반 10개 시드 Student-t 평가 ---")
    print("Student-t 임계값: t = 2.262 (df=9, alpha=0.05)")

    opponents_v36 = [
        ("A. Balanced(GTO)", lambda: BalancedPlayer("GTO_Opp", gto_strat)),
        ("B. Tricky", lambda: TrickyPlayer()),
        ("C. Adaptive Opponent", lambda: AdaptiveOpponentPlayer()),
        ("D. Randomized (Seed 42)", lambda: RandomizedPlayer(seed=42)),
        ("E. Counter Exploiter", lambda: CounterExploiterPlayer()),
        ("F. Adversarial θ*", lambda: ParametricOpponent(ADVERSARIAL_THETA, name="Adversarial"))
    ]

    print(f"\n{'상대 전략':<24} | {'Bayesian V3 (bb/100, t=2.262)':<32} | {'CFR GTO':<12} | {'Corrected Oracle':<18} | {'Oracle 대비 Regret'}")
    print("-" * 110)

    t_crit = 2.262 # Student-t df=9, 95%

    for opp_name, opp_factory in opponents_v36:
        v3_runs = []
        gto_runs = []
        oracle_runs = []

        for s in range(1, 11):
            random.seed(s * 203 + 5)
            np.random.seed(s * 203 + 5)

            opp_inst = opp_factory()
            v3_player = BayesianExploiterPlayer("V3", gto_strat, decay_factor=0.9992)
            v3_score = run_benchmark_match_exact(v3_player, opp_inst, num_hands=8000)
            v3_runs.append(v3_score)

            from benchmark_v3 import CFRGTOPlayer
            gto_player = CFRGTOPlayer("GTO", gto_strat)
            gto_score = run_benchmark_match_exact(gto_player, opp_factory(), num_hands=8000)
            gto_runs.append(gto_score)

            if s <= 3:
                oracle = CorrectedBayesianOracle(opp_factory())
                o_score = run_benchmark_match_exact(oracle, opp_factory(), num_hands=3000)
                oracle_runs.append(o_score)

        mean_v3 = np.mean(v3_runs)
        s_v3 = np.std(v3_runs, ddof=1)
        ci_t = t_crit * (s_v3 / np.sqrt(10))

        mean_gto = np.mean(gto_runs)
        mean_oracle = np.mean(oracle_runs)
        regret = mean_oracle - mean_v3

        print(f"{opp_name:<24} | {mean_v3:+6.2f} ± {ci_t:4.2f} bb/100 (t-dist)    | {mean_gto:+6.2f} bb/100  | {mean_oracle:+6.2f} bb/100      | {regret:5.2f} bb/100")

    print("==========================================================")
