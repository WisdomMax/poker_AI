"""
POKER AI V4.4 - PHASE 4: Leduc Poker Bayesian 상대 모델 확장
- 6장 덱 (J, Q, K 각 2장) 카드 제거 효과 (Card Removal Effect) 수학적 적용
- 프리플롭(남은 5장) 및 플롭 커뮤니티 카드 공개(남은 4장) 반영
- 상대 베팅 기록(체크/콜/레이즈 빈도) 기반 베이지안 신념(Belief) 사후확률 추정
- 쇼다운 시 하드 업데이트, 비쇼다운 시 사후확률 가중 소프트 업데이트
- 상대 비공개 카드 누출 전무 (자신의 카드와 보드, 공개 행동만으로 추론)
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class LeducBayesianOpponentModel:
    """
    Leduc Poker 전용 베이지안 상대 모델
    상대의 숨겨진 카드를 절대 보지 않고,
    내 카드 및 공개 커뮤니티 카드의 카드 제거 효과와
    상대의 공개된 액션(체크/콜/레이즈)으로 상대의 카드 분포 P(c_opp | history, board)를 추정
    """
    def __init__(self, prior_weight: float = 10.0, decay_factor: float = 0.999):
        self.prior_weight = prior_weight
        self.decay_factor = decay_factor

        # 3개 랭크 (0: J, 1: Q, 2: K) x 2개 라운드 (0: Round 1, 1: Round 2) x 3개 액션 (0: Fold, 1: Call/Check, 2: Raise)
        # counts[rank, round, action]
        self.action_counts = np.zeros((3, 2, 3), dtype=np.float64)
        self.total_observed = np.zeros((3, 2), dtype=np.float64)
        self.hand_count = 0

        # 사전분포: 정석 플레이어 기본 성향
        # J: 체크/콜 위주, K: 레이즈 위주
        self.prior_probs = np.array([
            [[0.1, 0.7, 0.2], [0.2, 0.6, 0.2]], # J: Round 1, Round 2
            [[0.1, 0.6, 0.3], [0.1, 0.5, 0.4]], # Q: Round 1, Round 2
            [[0.0, 0.3, 0.7], [0.0, 0.2, 0.8]], # K: Round 1, Round 2
        ], dtype=np.float64)

    def get_action_prob(self, rank: int, round_idx: int) -> np.ndarray:
        """상대가 특정 카드 랭크와 라운드에서 각 액션을 취할 사후 확률 P(a | rank, round)"""
        prior = self.prior_probs[rank, round_idx] * self.prior_weight
        observed = self.action_counts[rank, round_idx]
        post = prior + observed
        tot = np.sum(post)
        return post / tot if tot > 0 else np.array([1/3, 1/3, 1/3])

    def infer_opponent_card_belief(
        self,
        my_card: int,
        board_card: Optional[int],
        opp_actions_history: List[Tuple[int, int]] # list of (round_idx, action)
    ) -> Dict[int, float]:
        """
        카드 제거 효과가 완벽히 적용된 상대의 비공개 카드 사후 신념 P(c_opp | board, history)
        my_card와 board_card는 절대 상대가 가질 수 없음!
        """
        all_cards = list(range(6))
        # 불가능한 카드 제거
        possible_cards = [c for c in all_cards if c != my_card and (board_card is None or c != board_card)]

        # 사전확률: 남은 가능한 카드들 간의 균등 분포
        card_likelihoods = []
        for c in possible_cards:
            rank = c // 2 # 0=J, 1=Q, 2=K
            lh = 1.0
            for r_idx, act in opp_actions_history:
                probs = self.get_action_prob(rank, r_idx)
                lh *= probs[act]
            card_likelihoods.append(lh)

        total_lh = sum(card_likelihoods)
        if total_lh > 0:
            belief = {c: lh / total_lh for c, lh in zip(possible_cards, card_likelihoods)}
        else:
            belief = {c: 1.0 / len(possible_cards) for c in possible_cards}
        return belief

    def update_hand(
        self,
        opp_actions_history: List[Tuple[int, int]],
        my_card: int,
        board_card: Optional[int],
        opp_card_shown: Optional[int]
    ):
        """한 판 종료 후 베이지안 업데이트"""
        self.hand_count += 1
        self.action_counts *= self.decay_factor
        self.total_observed *= self.decay_factor

        if not opp_actions_history:
            return

        if opp_card_shown is not None:
            # 1. 쇼다운으로 상대 카드가 공개된 경우 -> Hard Update
            rank = opp_card_shown // 2
            for r_idx, act in opp_actions_history:
                self.action_counts[rank, r_idx, act] += 1.0
                self.total_observed[rank, r_idx] += 1.0
        else:
            # 2. 비쇼다운(폴드)으로 상대 카드가 비공개인 경우 -> 사후확률 가중 Soft Latent Update
            belief = self.infer_opponent_card_belief(my_card, board_card, opp_actions_history)
            for c, weight in belief.items():
                rank = c // 2
                for r_idx, act in opp_actions_history:
                    self.action_counts[rank, r_idx, act] += weight
                    self.total_observed[rank, r_idx] += weight


class LeducSafeBayesianPlayer:
    """
    Leduc Poker Safe Bayesian 하이브리드 플레이어:
    - 기본 전략: V4.4 Best NFSP 정책 (단단한 방어)
    - 실시간 베이지안 신념(Belief) 모델과 관측 표본 수에 따른 신뢰도(Confidence) 가중치로 착취 액션 결합
    - 안전 상한선 (max_exploit_weight = 0.70) 내장
    """
    def __init__(self, nfsp_agent, max_exploit_weight: float = 0.70):
        self.nfsp_agent = nfsp_agent
        self.max_exploit_weight = max_exploit_weight
        self.model = LeducBayesianOpponentModel()

    def act(
        self,
        info_vec: List[float],
        legal_actions: List[int],
        my_card: int,
        board_card: Optional[int],
        opp_actions_history: List[Tuple[int, int]]
    ) -> int:
        # 1. 기본 NFSP 전략 확률
        _, nfsp_probs = self.nfsp_agent.act(info_vec, legal_actions, is_evaluation=True)

        # 2. 상대 카드 사후 신념 추정
        belief = self.model.infer_opponent_card_belief(my_card, board_card, opp_actions_history)

        # 3. 내 패의 상대적 가치 및 착취 액션 선호도 계산
        my_rank = my_card // 2
        board_rank = (board_card // 2) if board_card is not None else None

        # 핸드 강도 (Hand Strength vs Opponent Belief)
        win_prob = 0.0
        for opp_c, p_opp in belief.items():
            opp_rank = opp_c // 2
            # 페어 여부
            my_pair = (board_rank is not None and my_rank == board_rank)
            opp_pair = (board_rank is not None and opp_rank == board_rank)

            if my_pair and not opp_pair:
                win_prob += p_opp * 1.0
            elif not my_pair and opp_pair:
                win_prob += p_opp * 0.0
            elif my_rank > opp_rank:
                win_prob += p_opp * 1.0
            elif my_rank < opp_rank:
                win_prob += p_opp * 0.0
            else:
                win_prob += p_opp * 0.5 # 동점

        # 4. 휴리스틱 착취 선호 확률 (Exploit Policy)
        exploit_probs = np.zeros(3, dtype=np.float32)
        if 2 in legal_actions and win_prob > 0.65:
            # 강한 패: 레이즈 선호
            exploit_probs[2] = 0.8
            exploit_probs[1] = 0.2
        elif 0 in legal_actions and win_prob < 0.25:
            # 약한 패에 레이즈 직면: 폴드 선호
            exploit_probs[0] = 0.8
            exploit_probs[1] = 0.2
        else:
            # 중간 패: 체크/콜
            exploit_probs[1] = 0.8
            if 2 in legal_actions: exploit_probs[2] = 0.2

        # 합법 액션에 맞게 마스킹 및 정규화
        for a in range(3):
            if a not in legal_actions: exploit_probs[a] = 0.0
        s = sum(exploit_probs[a] for a in legal_actions)
        if s > 0:
            for a in legal_actions: exploit_probs[a] /= s
        else:
            for a in legal_actions: exploit_probs[a] = 1.0 / len(legal_actions)

        # 5. 관측 표본 수에 따른 신뢰도 (Confidence Blending)
        n_obs = np.sum(self.model.total_observed)
        raw_conf = n_obs / (n_obs + 40.0) # 40판 완충 계수
        safe_conf = min(raw_conf, self.max_exploit_weight)

        # 최종 혼합 정책: (1 - conf) * NFSP + conf * Exploit
        final_probs = np.zeros(3, dtype=np.float32)
        for a in legal_actions:
            final_probs[a] = (1.0 - safe_conf) * nfsp_probs[a] + safe_conf * exploit_probs[a]
        final_probs = np.array(final_probs, dtype=np.float64)
        s = np.sum(final_probs)
        if s > 0:
            final_probs /= s
        else:
            final_probs = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        return int(np.random.choice([0, 1, 2], p=final_probs))
