"""
Poker AI V3.5 - [1. 기존 V3 엔진 감사 (Audit)]
8가지 핵심 무결성 체크포인트 검증
"""
import unittest
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from cfr_v3 import CFRTrainerV3
from bayesian_agent_v3 import BayesianExploiterPlayer, BayesianOpponentModel

class TestV3EngineAudit(unittest.TestCase):
    def setUp(self):
        trainer = CFRTrainerV3()
        trainer.train(50000)
        self.gto_strat = trainer.get_strategy_dict()
        self.agent = BayesianExploiterPlayer("AuditAgent", self.gto_strat)

    def test_checkpoint_1_no_hidden_card_access(self):
        """1. 상대의 숨겨진 카드가 AI 의사결정에 절대 전달되지 않는가?"""
        # act 함수의 시그니처 검사: act(card, history) -> 오직 내 카드와 공개 히스토리만 받음
        import inspect
        sig = inspect.signature(self.agent.act)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ['card', 'history'], "act 함수는 card와 history 외에 다른 인자를 받으면 안 됩니다.")

    def test_checkpoint_2_no_opponent_strategy_access(self):
        """2. 상대 전략의 실제 파라미터나 클래스 내부를 참조하지 않는가?"""
        # agent 및 model 객체 내부 속성에 상대 인스턴스에 대한 참조가 없는지 검사
        self.assertFalse(hasattr(self.agent, "opponent"), "agent에 opponent 객체 참조가 있습니다.")
        self.assertFalse(hasattr(self.agent.model, "opponent"), "model에 opponent 객체 참조가 있습니다.")

    def test_checkpoint_3_and_4_showdown_vs_fold_update(self):
        """3 & 4. 쇼다운에서만 실제 카드를 사용하고, 폴드 시에는 잠재변수로 확률적 처리하는가?"""
        # 케이스 A: 쇼다운 발생 (pp) -> opp_card_shown = 3 (K)
        # my_card = 1 (J)
        model = BayesianOpponentModel(self.gto_strat)
        initial_counts = model.counts.copy()
        
        # 쇼다운 업데이트
        model.update_hand("pp", opp_pos=1, opp_card_shown=3, my_card=1)
        # K(인덱스 2)의 pp 경로 카운트가 정확히 1.0 증가해야 함
        h_idx = model.hist_to_idx["p"] # 상대는 'p' 액션을 함
        self.assertAlmostEqual(model.counts[2, h_idx, PASS] - initial_counts[2, h_idx, PASS], 1.0, places=5)
        # J(0)이나 Q(1)의 카운트는 증가하지 않아야 함
        self.assertAlmostEqual(model.counts[0, h_idx, PASS], 0.0)
        self.assertAlmostEqual(model.counts[1, h_idx, PASS], 0.0)

        # 케이스 B: 비쇼다운 폴드 발생 (bp) -> opp_card_shown = None
        # P0(나, J) 베팅 -> P1(상대) 폴드 (p)
        model_fold = BayesianOpponentModel(self.gto_strat)
        model_fold.update_hand("bp", opp_pos=1, opp_card_shown=None, my_card=1)
        h_idx_b = model_fold.hist_to_idx["b"] # 상대가 직면한 히스토리 'b'
        
        # my_card가 1(J)이므로 상대 카드는 2(Q) 또는 3(K)만 가능
        # 두 카드에 대한 사후확률로 분배되었으므로, Q와 K의 카운트 합계가 정확히 1.0이어야 함
        q_count = model_fold.counts[1, h_idx_b, PASS]
        k_count = model_fold.counts[2, h_idx_b, PASS]
        self.assertAlmostEqual(q_count + k_count, 1.0, places=5)
        # J(인덱스 0)은 내가 들고 있으므로 카운트가 0이어야 함 (카드 제거 효과)
        self.assertAlmostEqual(model_fold.counts[0, h_idx_b, PASS], 0.0, places=5)

    def test_checkpoint_5_no_future_information(self):
        """5. 베이지안 업데이트에 미래 정보가 섞이지 않는가?"""
        # 히스토리 t 시점의 액션 업데이트 시 t 이전의 히스토리 h[:t]만 참조하는지 검증
        model = BayesianOpponentModel(self.gto_strat)
        # pbp 경로: t=0: P0 check(''), t=1: P1 bet('p'), t=2: P0 fold('pb')
        # 상대(P1)는 오직 t=1에서 'p' 상태에서 BET을 선택했음.
        # 상대에게 'pb' 이후의 정보가 들어가지 않는지 확인
        model.update_hand("pbp", opp_pos=1, opp_card_shown=None, my_card=1)
        # 히스토리 'pb'는 P0의 액션 시점이므로 상대 카운트에 'pb'가 증가하면 안 됨!
        pb_idx = model.hist_to_idx["pb"]
        self.assertEqual(np.sum(model.counts[:, pb_idx, :]), 0.0)

    def test_checkpoint_6_fair_position_symmetry(self):
        """6. 포지션 배정 및 손익 대칭성 검증"""
        # 양 플레이어가 완벽히 대칭일 때 장기 기대값이 0인지 수학적 불변식
        p0_profit, p1_profit = evaluate_payoff("pp", 2, 1) # Q vs J
        p0_rev, p1_rev = evaluate_payoff("pp", 1, 2) # J vs Q
        self.assertEqual(p0_profit + p0_rev, 0)
        self.assertEqual(p1_profit + p1_rev, 0)

    def test_checkpoint_7_and_8_ev_and_payoff_correctness(self):
        """7 & 8. 모든 합법적 행동의 EV 계산 및 손익의 정확성"""
        # 내 카드 K(3), my_pos=0(선공), history=""
        # K를 들었을 때 쇼다운을 가면 무조건 이김!
        ev_pass, ev_bet = self.agent.calculate_ev(my_card=3, history="", my_pos=0)
        # K를 들고 있을 때의 EV는 무조건 양수여야 함
        self.assertGreater(ev_pass, 0.0)
        self.assertGreater(ev_bet, 0.0)

        # 내 카드 J(1), my_pos=1(후공), history="pb" (상대가 체크 후 내가 베팅을 맞았을 때)
        # J로 콜(pbb)하면 K나 Q한테 무조건 지므로 콜 기대값은 -2. 폴드(pbp)는 -1.
        # 즉 폴드가 콜보다 항상 덜 손해여야 함
        p0_fold, p1_fold = evaluate_payoff("pbp", 3, 1) # 상대 K, 나 J -> 나(P1) 폴드 -> -1
        p0_call, p1_call = evaluate_payoff("pbb", 3, 1) # 상대 K, 나 J -> 나(P1) 콜 -> -2
        self.assertGreater(p1_fold, p1_call)

if __name__ == "__main__":
    unittest.main()
