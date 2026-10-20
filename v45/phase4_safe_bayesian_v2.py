"""
POKER AI V4.5 - PHASE 4: Entropy-Gated Safe Bayesian 상대 적응 시스템 v2
- 문제 해결: Random 상대 등 비정보적(Uninformative) 플레이어 상대로 과적합(Over-exploitation)되어 불필요한 폴드를 범하는 현상 완벽 방어.
- 1. Shannon Entropy Gating: 상대 행동의 엔트로피 H(A)가 최대치(균등분포 log2(3)=1.585)에 근접하면,
     상대 액션이 패에 대한 정보가 없다고 판단하여 액션 우도(Likelihood) 영향도를 0으로 차단.
- 2. Card Removal Effect (카드 제거 효과)의 엄격한 보존:
     내 패와 커뮤니티 카드를 제외한 남은 덱 구성만을 바탕으로 사전 균등 신념 유지.
- 3. Student GTO Baseline: NashConv 0.008725의 Student 정책을 기본 방패로 삼아,
     가중 결합치 w = min(w_max, Entropy_Gate * Confidence)로 안전 한도 보장.
"""

import sys
import os
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from phase2_distillation import LeducStudentNet

class LeducEntropyGatedBayesianModel:
    """
    엔트로피 게이팅이 적용된 Leduc 베이지안 상대 신념 모델
    """
    def __init__(self, prior_weight: float = 10.0, decay_factor: float = 0.999):
        self.prior_weight = prior_weight
        self.decay_factor = decay_factor
        
        # 전체 관측 액션 카운트 [Fold, Call, Raise]
        self.global_action_counts = np.zeros(3, dtype=np.float64)
        
        # 핸드 랭크별 액션 카운트 [Rank 0:J, 1:Q, 2:K][Round 0, 1][Action 0, 1, 2]
        self.action_counts = np.zeros((3, 2, 3), dtype=np.float64)
        self.total_observed = np.zeros((3, 2), dtype=np.float64)
        self.hand_count = 0
        
        # 기본 사전분포 (정석 플레이어)
        self.prior_probs = np.array([
            [[0.1, 0.7, 0.2], [0.2, 0.6, 0.2]], # J
            [[0.1, 0.6, 0.3], [0.1, 0.5, 0.4]], # Q
            [[0.0, 0.3, 0.7], [0.0, 0.2, 0.8]], # K
        ], dtype=np.float64)

    def calculate_entropy_gate(self) -> float:
        """
        상대방 전체 행동의 섀넌 엔트로피를 계산하고, 편향도(Bias / Information content)를 산출.
        - 완전 무작위 (1/3씩): H(A) = 1.585 -> gate = 0.0 (우도 반영 차단)
        - 한쪽으로 쏠린 경우 (Rock, Maniac): H(A) 낮음 -> gate -> 1.0 (적극 착취)
        """
        tot = np.sum(self.global_action_counts)
        if tot < 5.0:
            return 0.0  # 표본 부족 시 차단
            
        probs = self.global_action_counts / tot
        ent = 0.0
        for p in probs:
            if p > 1e-12:
                ent -= p * math.log2(p)
                
        max_ent = math.log2(3.0)  # 약 1.58496
        # 편향도 (0.0: 완전 무작위, 1.0: 단일 행동 고정)
        bias = max(0.0, 1.0 - (ent / max_ent))
        
        # 무작위 노이즈 필터링: 편향도가 0.15 이하(거의 무작위)이면 0으로 처리
        if bias < 0.15:
            return 0.0
            
        return float(min(1.0, (bias - 0.15) / 0.85))

    def get_action_prob(self, rank: int, round_idx: int) -> np.ndarray:
        prior = self.prior_probs[rank, round_idx] * self.prior_weight
        observed = self.action_counts[rank, round_idx]
        post = prior + observed
        tot = np.sum(post)
        return post / tot if tot > 0 else np.array([1/3, 1/3, 1/3])

    def infer_opponent_card_belief(
        self,
        my_card: int,
        board_card: Optional[int],
        opp_actions_history: List[Tuple[int, int]]
    ) -> Dict[int, float]:
        """
        상대 카드 사후 신념 추정:
        - 엔트로피 게이트가 0이면(Random 상대), 액션 우도를 모두 1.0으로 처리하여 순수 카드 제거 효과만 반영!
        """
        all_cards = list(range(6))
        possible_cards = [c for c in all_cards if c != my_card and (board_card is None or c != board_card)]
        
        gate = self.calculate_entropy_gate()
        
        card_likelihoods = []
        for c in possible_cards:
            rank = c // 2
            lh = 1.0
            for r_idx, act in opp_actions_history:
                p_act = self.get_action_prob(rank, r_idx)[act]
                # 게이트 적용: gate=0이면 (1.0)^0 * (p_act)^0 = 1.0
                effective_lh = (1.0 - gate) * (1.0 / 3.0) + gate * p_act
                lh *= effective_lh
            card_likelihoods.append(lh)
            
        total_lh = sum(card_likelihoods)
        if total_lh > 0:
            return {c: lh / total_lh for c, lh in zip(possible_cards, card_likelihoods)}
        return {c: 1.0 / len(possible_cards) for c in possible_cards}

    def update_hand(
        self,
        opp_actions_history: List[Tuple[int, int]],
        my_card: int,
        board_card: Optional[int],
        opp_card_shown: Optional[int]
    ):
        self.hand_count += 1
        self.action_counts *= self.decay_factor
        self.total_observed *= self.decay_factor
        self.global_action_counts *= self.decay_factor
        
        for _, act in opp_actions_history:
            self.global_action_counts[act] += 1.0
            
        if not opp_actions_history:
            return
            
        if opp_card_shown is not None:
            rank = opp_card_shown // 2
            for r_idx, act in opp_actions_history:
                self.action_counts[rank, r_idx, act] += 1.0
                self.total_observed[rank, r_idx] += 1.0
        else:
            belief = self.infer_opponent_card_belief(my_card, board_card, opp_actions_history)
            for c, weight in belief.items():
                rank = c // 2
                for r_idx, act in opp_actions_history:
                    self.action_counts[rank, r_idx, act] += weight
                    self.total_observed[rank, r_idx] += weight

class LeducSafeBayesianPlayerV2:
    """
    V4.5 Safe Bayesian Player v2:
    - 기본 GTO 전략: V4.5 Best Distilled Student (NashConv 0.008725)
    - 엔트로피 게이트 및 표본 신뢰도 기반 동적 착취 가중치
    """
    def __init__(self, student_model: LeducStudentNet, max_exploit_weight: float = 0.65, device=torch.device("cpu")):
        self.student_model = student_model
        self.student_model.eval()
        self.max_exploit_weight = max_exploit_weight
        self.device = device
        self.model = LeducEntropyGatedBayesianModel()

    def reset_model(self):
        self.model = LeducEntropyGatedBayesianModel()

    def act(
        self,
        info_vec: List[float],
        legal_actions: List[int],
        my_card: int,
        board_card: Optional[int],
        opp_actions_history: List[Tuple[int, int]]
    ) -> int:
        # 1. Student GTO 확률 계산
        x = torch.tensor(info_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.zeros((1, 3), dtype=torch.bool, device=self.device)
        for a in legal_actions:
            mask[0, a] = True
            
        with torch.no_grad():
            gto_probs = self.student_model.get_action_probs(x, mask).squeeze(0).cpu().numpy()
            
        # 2. 상대 카드 사후 신념 추정
        belief = self.model.infer_opponent_card_belief(my_card, board_card, opp_actions_history)
        
        # 3. 승률(Win Probability) 계산
        my_rank = my_card // 2
        board_rank = (board_card // 2) if board_card is not None else None
        
        win_prob = 0.0
        for opp_c, p_opp in belief.items():
            opp_rank = opp_c // 2
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
                win_prob += p_opp * 0.5
                
        # 4. 베이지안 착취 액션 선호도
        exploit_pref = np.zeros(3, dtype=np.float32)
        if 0 in legal_actions:  # Fold
            exploit_pref[0] = max(0.0, 0.40 - win_prob) * 2.0
        if 1 in legal_actions:  # Call/Check
            exploit_pref[1] = 0.5
        if 2 in legal_actions:  # Raise
            exploit_pref[2] = max(0.0, win_prob - 0.45) * 2.5
            
        tot_exp = sum(exploit_pref[a] for a in legal_actions)
        if tot_exp > 0:
            exploit_probs = np.zeros(3, dtype=np.float32)
            for a in legal_actions:
                exploit_probs[a] = exploit_pref[a] / tot_exp
        else:
            exploit_probs = gto_probs.copy()
            
        # 5. 동적 착취 가중치 산출 (엔트로피 게이트 x 표본 신뢰도)
        gate = self.model.calculate_entropy_gate()
        sample_conf = min(1.0, self.model.hand_count / 100.0)
        exploit_weight = self.max_exploit_weight * gate * sample_conf
        
        # 6. 하이브리드 정책 결합
        final_probs = (1.0 - exploit_weight) * gto_probs + exploit_weight * exploit_probs
        sub_sum = sum(final_probs[a] for a in legal_actions)
        if sub_sum > 0:
            for a in legal_actions:
                final_probs[a] /= sub_sum
        else:
            final_probs = gto_probs
            
        # 행동 샘플링
        acts = legal_actions
        prs = [final_probs[a] for a in legal_actions]
        prs = np.array(prs) / sum(prs)
        return int(np.random.choice(acts, p=prs))
