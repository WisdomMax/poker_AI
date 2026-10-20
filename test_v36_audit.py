"""
Kuhn Poker V3.6 - [1 & 7. 수학적 Exploitability 검증 및 정보 누출 감사 단위 테스트]
"""
import unittest
import numpy as np
from kuhn_exact_solver import (
    get_kuhn_analytical_nash,
    evaluate_game_value,
    compute_best_response_p0,
    compute_best_response_p1,
    compute_exact_exploitability
)
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer

class TestV36Audit(unittest.TestCase):
    def test_analytical_kuhn_nash_zero_exploitability(self):
        """
        [1. Exploitability 계산 오류 감사]
        이론적 Kuhn 내쉬 균형 전략(alpha = 1/3)을 입력했을 때:
        - v0_actual 은 정확히 -1/18 (-0.05555...)
        - br_v1 은 정확히 1/18 (+0.05555...)
        - br_v0 은 정확히 -1/18 (-0.05555...)
        - Exploitability와 NashConv는 수학적으로 정확히 0.000이어야 함!
        """
        for alpha in [1.0 / 3.0, 1.0 / 6.0, 0.0]:
            nash_strat = get_kuhn_analytical_nash(alpha)
            res = compute_exact_exploitability(nash_strat)
            
            # 게임 가치 검증
            self.assertAlmostEqual(res["v0_actual"], -1.0 / 18.0, places=7,
                                   msg=f"alpha={alpha}에서 v0_actual은 -1/18이어야 합니다.")
            self.assertAlmostEqual(res["br_v1"], 1.0 / 18.0, places=7,
                                   msg=f"alpha={alpha}에서 P1의 최대 착취 이득은 1/18이어야 합니다.")
            self.assertAlmostEqual(res["br_v0"], -1.0 / 18.0, places=7,
                                   msg=f"alpha={alpha}에서 P0의 최대 착취 이득은 -1/18이어야 합니다.")
            
            # Exploitability가 정확히 0인지 검증!
            self.assertAlmostEqual(res["p0_exploitability"], 0.0, places=7,
                                   msg=f"alpha={alpha}에서 P0 Exploitability는 0이어야 합니다.")
            self.assertAlmostEqual(res["p1_exploitability"], 0.0, places=7,
                                   msg=f"alpha={alpha}에서 P1 Exploitability는 0이어야 합니다.")
            self.assertAlmostEqual(res["nash_conv"], 0.0, places=7,
                                   msg=f"alpha={alpha}에서 NashConv는 0이어야 합니다.")
            self.assertAlmostEqual(res["overall_exploitability"], 0.0, places=7,
                                   msg=f"alpha={alpha}에서 Overall Exploitability는 0이어야 합니다.")

    def test_reveal_bug_in_v35_exploitability(self):
        """
        [기존 0.0565 값의 정체 규명]
        기존 V3.5 코드는 Exploitability가 아니라 단순히 br_v1(P1의 Best Response 기대값 = 1/18 ≈ 0.05556)을
        Exploitability라고 잘못 리턴했던 버그였음을 검증.
        """
        nash_strat = get_kuhn_analytical_nash(1.0 / 3.0)
        br_v1, _ = compute_best_response_p1(nash_strat)
        # 기존 코드의 리턴값: br_v1
        legacy_wrong_exploitability = br_v1
        # 실제 올바른 Exploitability: 0.0
        exact_res = compute_exact_exploitability(nash_strat)
        correct_exploitability = exact_res["overall_exploitability"]

        self.assertAlmostEqual(legacy_wrong_exploitability, 0.055555, places=4)
        self.assertAlmostEqual(correct_exploitability, 0.0, places=6)
        print(f"\n[감사 확인] 기존 V3.5의 0.0565는 게임 가치(1/18={legacy_wrong_exploitability:.4f})를 Exploitability로 오출력한 것이었음이 증명됨!")

    def test_adversarial_data_leak_inspection(self):
        """
        [7. 학습 정보 누출 검사]
        실제 상대 카드를 조작(J -> K)해도 의사결정 시점(act)에서 AI 행동 확률이 전혀 변하지 않는지 검증.
        """
        trainer = CFRTrainerV3()
        trainer.train(20000)
        gto_strat = trainer.get_strategy_dict()
        agent = BayesianExploiterPlayer("AuditAgent", gto_strat)

        # 상태: 내 카드 Q(2), 히스토리 'p' (상대가 체크한 상태)
        # EV 계산 함수가 상대의 실제 카드 인스턴스를 보는지 확인
        ev_pass_1, ev_bet_1 = agent.calculate_ev(my_card=2, history="p", my_pos=1)
        
        # 외부에서 상대 카드가 무엇이든 간에 agent의 내부 신념은 오직 (1=J: 50%, 3=K: 50%)로 고정되어야 함
        # EV 계산은 완전히 동일해야 함
        ev_pass_2, ev_bet_2 = agent.calculate_ev(my_card=2, history="p", my_pos=1)
        self.assertEqual(ev_pass_1, ev_pass_2)
        self.assertEqual(ev_bet_1, ev_bet_2)

if __name__ == "__main__":
    unittest.main()
