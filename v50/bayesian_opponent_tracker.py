"""
POKER AI V5.0 - PHASE 1 & 2: Continuous Match Session & Beta-Binomial Opponent Tracker
- Match Session: 5명의 고정 에이전트가 1개 세션(예: 100~200핸드) 동안 대국
- Agent ID (0~4)와 Physical Seat ID (0~4) 완전 분리 (좌석 회전 시 통계 왜곡 원천 차단)
- 관측 가능한 공개 행동만 사용 (비공개 카드, 미래 행동, 봇 라벨 절대 배제)
- 6대 기회(Opportunity) 기반 지표 추적:
  1. VPIP (Voluntarily Put In Pot - 강제 블라인드 제외)
  2. PFR (Preflop Raise)
  3. 3-Bet Frequency (오픈 레이즈 직면 시 재레이즈)
  4. Fold to 3-Bet (자신의 오픈 후 3-Bet 직면 시 폴드)
  5. All-in Frequency
  6. Aggression Frequency (Bet/Raise / (Bet/Raise + Call + Fold))
- Beta-Binomial 사후분포 계산:
  * Prior: Beta(alpha_0, beta_0) (기본 alpha=2, beta=2)
  * Posterior: Beta(alpha + k, beta + N - k)
  * Mean: (alpha + k) / (alpha + beta + N)
  * Uncertainty (Std): sqrt( (Mean * (1 - Mean)) / (alpha + beta + N + 1) )
  * Count: min(N, 100) / 100.0
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pyspiel

METRIC_NAMES = ["vpip", "pfr", "three_bet", "fold_to_three_bet", "allin", "aggression"]

class SingleOpponentBayesianTracker:
    """개별 상대 에이전트 1인에 대한 기회 기반 Beta-Binomial 통계 추적기"""
    def __init__(self, agent_id: int, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.agent_id = agent_id
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.reset()

    def reset(self):
        """세션 시작 시 초기화"""
        self.hands_observed = 0
        # 각 지표별 {'opp': int, 'succ': int}
        self.stats = {
            m: {'opp': 0, 'succ': 0} for m in METRIC_NAMES
        }

    def record_opportunity(self, metric: str):
        if metric in self.stats:
            self.stats[metric]['opp'] += 1

    def record_success(self, metric: str):
        if metric in self.stats:
            self.stats[metric]['succ'] += 1

    def get_posterior(self, metric: str) -> Tuple[float, float, int]:
        """(사후 평균, 사후 표준편차(불확실성), 총 관측 기회 수) 반환"""
        st = self.stats[metric]
        k = st['succ']
        n = st['opp']
        a = self.prior_alpha + k
        b = self.prior_beta + (n - k)
        
        mean = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
        std = math.sqrt(max(0.0, var))
        return mean, std, n

    def get_feature_vector(self) -> np.ndarray:
        """
        상대 1인당 13차원 정규화 피처 벡터 반환:
        - 6개 지표 x (사후평균, 불확실성(std)) = 12차원
        - 총 관측 핸드 수 정규화 (min(hands, 100) / 100.0) = 1차원
        총 13차원, 모든 값은 [0.0, 1.0] 범위에 엄격히 수렴.
        """
        feats = []
        for m in METRIC_NAMES:
            mean, std, _ = self.get_posterior(m)
            feats.append(float(mean))
            feats.append(float(std))
        
        # 정규화된 관측 핸드 수
        feats.append(float(min(self.hands_observed, 100) / 100.0))
        return np.array(feats, dtype=np.float32)

class MatchSessionOpponentManager:
    """
    세션 단위 5인 테이블 상대 모델링 관리자:
    - Hero(Agent 0)를 제외한 상대 4명(Agent 1, 2, 3, 4)의 통계를 세션 동안 추적.
    - 물리적 좌석 회전 래퍼와 연동하여 seat -> agent 변환 후 기록.
    - 매 핸드 종료 시 비공개 정보 없이 공개 베팅 시퀀스만을 파싱하여 정확한 기회/성공 업데이트.
    """
    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.trackers = {
            ag_id: SingleOpponentBayesianTracker(ag_id, prior_alpha, prior_beta)
            for ag_id in range(1, 5) # 상대 4명 (1~4)
        }
        self.current_hand_idx = 0

    def reset_session(self):
        """새로운 매치 세션 시작 시 모든 상대 통계 초기화 (Cold Start)"""
        self.current_hand_idx = 0
        for t in self.trackers.values():
            t.reset()

    def set_hand_index(self, hand_idx: int):
        self.current_hand_idx = hand_idx

    def parse_and_update_hand(
        self,
        hand_history: List[Tuple[int, int, List[int], int]], # [(physical_seat, round_idx, legal_acts, action_taken), ...]
        seat_to_agent_map: Dict[int, int]
    ):
        """
        핸드가 종료된 후 공개 액션 시퀀스로부터 6대 지표 기회 및 성공 전수 업데이트.
        - hand_history: 매 스텝마다 (물리적좌석, 라운드(0:Pre, 1:Flop, 2:Turn, 3:River), legal_acts, 취한액션)
        - 비공개 카드 누출 전혀 없음!
        """
        # 이번 핸드에 참여한 상대들의 hands_observed 증가
        for ag_id in range(1, 5):
            self.trackers[ag_id].hands_observed += 1

        # 프리플롭 추적용 상태
        preflop_actions_by_agent = {ag_id: [] for ag_id in range(1, 5)}
        opened_by_agent = None # 최초 오픈 레이즈한 에이전트 ID
        three_bet_by_agent = None

        # 1. 시퀀스 전수 스캔 및 기회/성공 판정
        # fchpa 액션 매핑: 0: Fold, 1: Call/Check, 2: Pot Raise, 3: All-in, 4: Half-Pot Raise
        current_round = 0
        raises_in_preflop = 0

        for physical_seat, r_idx, legal_acts, action in hand_history:
            ag_id = seat_to_agent_map[physical_seat]
            if ag_id == 0:
                # Hero의 액션은 상대 통계에 넣지 않음. 단 게임 맥락(오픈 레이즈 여부 등)은 업데이트
                if r_idx == 0 and action in [2, 3, 4]:
                    raises_in_preflop += 1
                    if raises_in_preflop == 1 and opened_by_agent is None:
                        opened_by_agent = 0
                    elif raises_in_preflop == 2 and three_bet_by_agent is None:
                        three_bet_by_agent = 0
                continue

            tracker = self.trackers[ag_id]

            # --- All-in Frequency ---
            tracker.record_opportunity("allin")
            if action == 3:
                tracker.record_success("allin")

            # --- Aggression Frequency ---
            # 기회: 베팅/레이즈가 가능했던 상황
            if any(a in legal_acts for a in [2, 3, 4]):
                tracker.record_opportunity("aggression")
                if action in [2, 3, 4]:
                    tracker.record_success("aggression")

            # --- 프리플롭 전용 지표 ---
            if r_idx == 0:
                preflop_actions_by_agent[ag_id].append((action, legal_acts))

                # VPIP: 프리플롭에서 첫 자발적 액션 기회
                # 물리적 SB(0번 좌석), BB(1번 좌석)의 강제 투입은 자발적 투입이 아님.
                # 단 자신의 턴이 왔을 때 Call(1) 또는 Raise(2,3,4)를 하면 VPIP 성공!
                # BB에서 아무도 레이즈 안 해서 Check(1)한 경우는 추가 투입이 아니므로 제외.
                is_bb = (physical_seat == 1)
                is_first_action = (len(preflop_actions_by_agent[ag_id]) == 1)

                if is_first_action:
                    tracker.record_opportunity("vpip")
                    if action in [2, 3, 4]: # Raise
                        tracker.record_success("vpip")
                    elif action == 1: # Call
                        # BB 자리에 앉아서 칩 추가 없이 단순 체크한 경우가 아니라면 Call은 VPIP!
                        # OpenSpiel에서 BB가 옵션 체크를 할 수 있는 상황인지 체크
                        if not is_bb or raises_in_preflop > 0:
                            tracker.record_success("vpip")

                # PFR: 프리플롭 레이즈 기회
                if any(a in legal_acts for a in [2, 3, 4]):
                    tracker.record_opportunity("pfr")
                    if action in [2, 3, 4]:
                        tracker.record_success("pfr")

                # 3-Bet & Fold to 3-Bet
                if action in [2, 3, 4]:
                    raises_in_preflop += 1
                    if raises_in_preflop == 1 and opened_by_agent is None:
                        opened_by_agent = ag_id
                    elif raises_in_preflop == 2:
                        # 3-Bet 기회였음!
                        tracker.record_opportunity("three_bet")
                        tracker.record_success("three_bet")
                        three_bet_by_agent = ag_id
                elif raises_in_preflop == 1 and opened_by_agent != ag_id:
                    # 다른 누군가가 1차 오픈 레이즈를 한 상태에서 내 차례가 옴 -> 3-Bet 기회!
                    tracker.record_opportunity("three_bet")
                    # 하지만 레이즈하지 않았으므로 성공은 아님 (Call or Fold)

                # Fold to 3-Bet
                if raises_in_preflop >= 2 and opened_by_agent == ag_id:
                    # 내가 오픈했는데 상대가 3-Bet을 날린 후 다시 내 차례가 옴!
                    tracker.record_opportunity("fold_to_three_bet")
                    if action == 0:
                        tracker.record_success("fold_to_three_bet")

    def get_table_feature_tensor(self, current_hero_seat: int) -> np.ndarray:
        """
        신경망에 주입할 상대 4인의 결합 피처 텐서 (총 52차원) 반환.
        - 배치 순서: Hero의 물리적 좌석 기준으로 시계방향 상대 좌석 순서 (Seat+1, Seat+2, Seat+3, Seat+4) % 5
        - 각 좌석에 앉아 있는 실제 상대의 Agent ID 통계를 추출하여 연결.
        - 이를 통해 신경망은 '내 바로 왼쪽 플레이어, 그 다음 플레이어...'의 상대적 포지션별 통계를 일관되게 인식함!
        """
        # 좌석 회전 래퍼 공식: agent_id = (physical_seat + hand_idx) % 5
        # 상대 4개 물리 좌석:
        feats_list = []
        for offset in range(1, 5):
            rel_seat = (current_hero_seat + offset) % 5
            target_agent = (rel_seat + self.current_hand_idx) % 5
            if target_agent == 0:
                # Hero 자신은 상대 통계에 없음 (예외 방어)
                tracker_feat = np.zeros(13, dtype=np.float32)
            else:
                tracker_feat = self.trackers[target_agent].get_feature_vector()
            feats_list.append(tracker_feat)

        return np.concatenate(feats_list) # 13 * 4 = 52차원

    def inject_synthetic_profile(self, agent_id: int, profile_name: str, num_hands: int = 100):
        """Phase 5 검증용: 특정 상대에게 합성 프로필(Tight, Aggressive 등) 주입"""
        t = self.trackers[agent_id]
        t.reset()
        t.hands_observed = num_hands

        if profile_name == "Tight":
            # VPIP 15%, PFR 10%, 3-Bet 3%, Fold to 3-Bet 80%, Allin 1%, AF 15%
            t.stats['vpip'] = {'opp': num_hands, 'succ': int(num_hands * 0.15)}
            t.stats['pfr'] = {'opp': num_hands, 'succ': int(num_hands * 0.10)}
            t.stats['three_bet'] = {'opp': int(num_hands * 0.4), 'succ': int(num_hands * 0.4 * 0.03)}
            t.stats['fold_to_three_bet'] = {'opp': int(num_hands * 0.1), 'succ': int(num_hands * 0.1 * 0.80)}
            t.stats['allin'] = {'opp': int(num_hands * 1.5), 'succ': int(num_hands * 1.5 * 0.01)}
            t.stats['aggression'] = {'opp': int(num_hands * 2.0), 'succ': int(num_hands * 2.0 * 0.15)}

        elif profile_name == "LooseCalling":
            # VPIP 65%, PFR 5%, 3-Bet 2%, Fold to 3-Bet 10%, Allin 1%, AF 10%
            t.stats['vpip'] = {'opp': num_hands, 'succ': int(num_hands * 0.65)}
            t.stats['pfr'] = {'opp': num_hands, 'succ': int(num_hands * 0.05)}
            t.stats['three_bet'] = {'opp': int(num_hands * 0.4), 'succ': int(num_hands * 0.4 * 0.02)}
            t.stats['fold_to_three_bet'] = {'opp': int(num_hands * 0.1), 'succ': int(num_hands * 0.1 * 0.10)}
            t.stats['allin'] = {'opp': int(num_hands * 1.5), 'succ': int(num_hands * 1.5 * 0.01)}
            t.stats['aggression'] = {'opp': int(num_hands * 2.0), 'succ': int(num_hands * 2.0 * 0.10)}

        elif profile_name == "Aggressive":
            # VPIP 40%, PFR 35%, 3-Bet 25%, Fold to 3-Bet 30%, Allin 20%, AF 60%
            t.stats['vpip'] = {'opp': num_hands, 'succ': int(num_hands * 0.40)}
            t.stats['pfr'] = {'opp': num_hands, 'succ': int(num_hands * 0.35)}
            t.stats['three_bet'] = {'opp': int(num_hands * 0.4), 'succ': int(num_hands * 0.4 * 0.25)}
            t.stats['fold_to_three_bet'] = {'opp': int(num_hands * 0.1), 'succ': int(num_hands * 0.1 * 0.30)}
            t.stats['allin'] = {'opp': int(num_hands * 1.5), 'succ': int(num_hands * 1.5 * 0.20)}
            t.stats['aggression'] = {'opp': int(num_hands * 2.0), 'succ': int(num_hands * 2.0 * 0.60)}

        elif profile_name == "Selective3Bettor":
            # VPIP 22%, PFR 18%, 3-Bet 15% (Tight but High 3-Bet), Fold to 3-Bet 60%, Allin 5%, AF 40%
            t.stats['vpip'] = {'opp': num_hands, 'succ': int(num_hands * 0.22)}
            t.stats['pfr'] = {'opp': num_hands, 'succ': int(num_hands * 0.18)}
            t.stats['three_bet'] = {'opp': int(num_hands * 0.4), 'succ': int(num_hands * 0.4 * 0.15)}
            t.stats['fold_to_three_bet'] = {'opp': int(num_hands * 0.1), 'succ': int(num_hands * 0.1 * 0.60)}
            t.stats['allin'] = {'opp': int(num_hands * 1.5), 'succ': int(num_hands * 1.5 * 0.05)}
            t.stats['aggression'] = {'opp': int(num_hands * 2.0), 'succ': int(num_hands * 2.0 * 0.40)}
