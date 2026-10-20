"""
단위 테스트: Kuhn Poker V3.0 엔진의 룰 및 손익 계산 검증
"""
import unittest
from itertools import permutations
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET

class TestKuhnEngineV3(unittest.TestCase):
    def setUp(self):
        # 가능한 모든 카드 조합 (J=1, Q=2, K=3) 중 서로 다른 2장
        self.card_pairs = list(permutations([1, 2, 3], 2))

    def test_terminal_states(self):
        """종료 상태 판별 테스트"""
        valid_terminals = ["pp", "bp", "bb", "pbp", "pbb"]
        for s in valid_terminals:
            self.assertTrue(is_terminal(s), f"{s}는 종료 상태여야 합니다.")

        non_terminals = ["", "p", "b", "pb", "ppp", "bpp"]
        for s in non_terminals:
            self.assertFalse(is_terminal(s), f"{s}는 종료 상태가 아니어야 합니다.")

    def test_zero_sum_property(self):
        """모든 상태와 카드 조합에서 두 플레이어의 손익 합계가 항상 0인지 검증"""
        terminals = ["pp", "bp", "bb", "pbp", "pbb"]
        for h in terminals:
            for c0, c1 in self.card_pairs:
                p0, p1 = evaluate_payoff(h, c0, c1)
                self.assertEqual(p0 + p1, 0, f"Zero-sum 위반! history={h}, c0={c0}, c1={c1}, p0={p0}, p1={p1}")

    def test_exact_payoffs(self):
        """각 종료 상태별 정확한 손익 검증"""
        # 1. pp (쇼다운, 앤티 1만 승부)
        # J(1) vs Q(2) -> P1 승리
        self.assertEqual(evaluate_payoff("pp", 1, 2), (-1, 1))
        # K(3) vs J(1) -> P0 승리
        self.assertEqual(evaluate_payoff("pp", 3, 1), (1, -1))

        # 2. bp (P0 베팅, P1 폴드 -> P0 승리, 카드 무관하게 항상 +1 / -1)
        for c0, c1 in self.card_pairs:
            self.assertEqual(evaluate_payoff("bp", c0, c1), (1, -1), f"bp 실패 for {c0}, {c1}")

        # 3. bb (P0 베팅, P1 콜 -> 쇼다운, 앤티1+베팅1=2 승부)
        # J(1) vs K(3) -> P1 승리 (-2, +2)
        self.assertEqual(evaluate_payoff("bb", 1, 3), (-2, 2))
        # K(3) vs Q(2) -> P0 승리 (+2, -2)
        self.assertEqual(evaluate_payoff("bb", 3, 2), (2, -2))

        # 4. pbp (P0 체크, P1 베팅, P0 폴드 -> P1 승리, 카드 무관하게 항상 -1 / +1)
        # ⚠️ 기존 엔진에서 P0 승리로 오판정했던 치명적 버그 부분 검증!
        for c0, c1 in self.card_pairs:
            self.assertEqual(evaluate_payoff("pbp", c0, c1), (-1, 1), f"pbp 실패 for {c0}, {c1}")

        # 5. pbb (P0 체크, P1 베팅, P0 콜 -> 쇼다운, 앤티1+베팅1=2 승부)
        # Q(2) vs J(1) -> P0 승리 (+2, -2)
        self.assertEqual(evaluate_payoff("pbb", 2, 1), (2, -2))
        # J(1) vs K(3) -> P1 승리 (-2, +2)
        self.assertEqual(evaluate_payoff("pbb", 1, 3), (-2, 2))

if __name__ == '__main__':
    unittest.main()
