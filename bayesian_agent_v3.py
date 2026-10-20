from typing import Optional
import numpy as np
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES

class BayesianOpponentModel:
    """
    상대의 숨겨진 카드를 직접 보지 않고,
    공개된 행동과 쇼다운 카드로만 상대의 행동 확률 P(Action | Card, History)를 추정
    """
    def __init__(self, gto_strategy: dict[str, tuple[float, float]], decay_factor: float = 0.9995, prior_weight: float = 10.0):
        self.gto_strategy = gto_strategy
        self.decay_factor = decay_factor # 망각 계수 (상대 성향 변화 적응)
        self.prior_weight = prior_weight # 사전분포 가중치 (GTO 안전장치)
        
        # 카운트 텐서: counts[card][history_idx][action]
        # card: 1=J, 2=Q, 3=K (인덱스 0, 1, 2 사용)
        self.histories = ["", "p", "b", "pb"]
        self.hist_to_idx = {h: i for i, h in enumerate(self.histories)}
        
        # counts[card_idx, hist_idx, action]
        self.counts = np.zeros((3, len(self.histories), 2), dtype=np.float64)
        self.total_observed = np.zeros((3, len(self.histories)), dtype=np.float64)
        self.hand_count = 0

    def get_action_prob(self, opp_card: int, history: str) -> np.ndarray:
        """
        상대가 특정 카드와 히스토리에서 각 액션을 취할 사후 확률 P(Action | Card, History)
        사전분포(GTO)와 관측된 데이터의 베이지안 결합
        """
        c_idx = opp_card - 1
        h_idx = self.hist_to_idx.get(history, 0)
        
        # GTO 사전 확률
        info_set = f"{CARD_NAMES[opp_card]}:{history}"
        gto_p = self.gto_strategy.get(info_set, (0.5, 0.5))
        prior_counts = np.array([gto_p[0], gto_p[1]], dtype=np.float64) * self.prior_weight
        
        # 사후 확률 (Posterior Mean)
        posterior_counts = prior_counts + self.counts[c_idx, h_idx]
        total = np.sum(posterior_counts)
        return posterior_counts / total

    def update_hand(self, history: str, opp_pos: int, opp_card_shown: Optional[int], my_card: int):
        """
        한 판이 끝난 후 베이지안 업데이트
        - 쇼다운 시: 실제 opp_card_shown으로 직접 업데이트
        - 비쇼다운(폴드) 시: my_card를 제외한 남은 2장의 사후확률로 소프트 업데이트 (Latent Variable Expectation)
        """
        self.hand_count += 1
        
        # 망각 계수 적용 (지수 감쇠로 과거 데이터 희석 -> 상대 전략 변경 적응)
        self.counts *= self.decay_factor
        self.total_observed *= self.decay_factor
        
        # 상대방이 거쳐간 히스토리와 선택한 액션 추출
        # history 문자열에서 opp_pos 순서의 액션들
        opp_actions = []
        for t in range(len(history)):
            if t % 2 == opp_pos:
                h_before = history[:t]
                act = PASS if history[t] == 'p' else BET
                if h_before in self.hist_to_idx:
                    opp_actions.append((h_before, act))

        if not opp_actions:
            return

        possible_opp_cards = [c for c in [1, 2, 3] if c != my_card]

        if opp_card_shown is not None:
            # 1. 쇼다운으로 상대 카드가 공개된 경우 -> Hard Update
            c_idx = opp_card_shown - 1
            for h_before, act in opp_actions:
                h_idx = self.hist_to_idx[h_before]
                self.counts[c_idx, h_idx, act] += 1.0
                self.total_observed[c_idx, h_idx] += 1.0
        else:
            # 2. 폴드로 카드가 비공개인 경우 -> 베이즈 정리로 상대 카드 사후확률 P(c | actions) 계산 후 Soft Update
            # P(c | h) propto P(actions | c) * P(c)
            card_likelihoods = []
            for c in possible_opp_cards:
                lh = 1.0
                for h_before, act in opp_actions:
                    probs = self.get_action_prob(c, h_before)
                    lh *= probs[act]
                card_likelihoods.append(lh)
            
            total_lh = sum(card_likelihoods)
            if total_lh > 0:
                card_posteriors = [lh / total_lh for lh in card_likelihoods]
            else:
                card_posteriors = [0.5, 0.5]

            for c, weight in zip(possible_opp_cards, card_posteriors):
                c_idx = c - 1
                for h_before, act in opp_actions:
                    h_idx = self.hist_to_idx[h_before]
                    self.counts[c_idx, h_idx, act] += weight
                    self.total_observed[c_idx, h_idx] += weight


class BayesianExploiterPlayer:
    """
    순수 기대값(EV) 극대화 및 베이지안 신념(Belief) 기반 적응형 플레이어
    하드코딩 if-else 룰 전무, 수학적 게임 트리 역추산 집행
    """
    def __init__(self, name: str, gto_strategy: dict[str, tuple[float, float]], decay_factor: float = 0.9995):
        self.name = name
        self.gto_strategy = gto_strategy
        self.model = BayesianOpponentModel(gto_strategy, decay_factor=decay_factor)

    def calculate_ev(self, my_card: int, history: str, my_pos: int) -> tuple[float, float]:
        """
        트리 역추산: PASS와 BET의 수학적 기대값 EV(Pass), EV(Bet) 계산
        """
        opp_pos = 1 - my_pos
        possible_opp_cards = [c for c in [1, 2, 3] if c != my_card]
        
        # 현재 history까지 오면서 상대가 취한 이전 액션들을 고려한 상대 카드의 사후확률 P(c_opp | history)
        card_weights = []
        for c in possible_opp_cards:
            w = 1.0
            for t in range(len(history)):
                if t % 2 == opp_pos:
                    h_before = history[:t]
                    act = PASS if history[t] == 'p' else BET
                    w *= self.model.get_action_prob(c, h_before)[act]
            card_weights.append(w)
            
        tot_w = sum(card_weights)
        if tot_w > 0:
            opp_belief = {c: w / tot_w for c, w in zip(possible_opp_cards, card_weights)}
        else:
            opp_belief = {c: 0.5 for c in possible_opp_cards}

        # 액션별 EV 계산
        ev_pass = 0.0
        ev_bet = 0.0

        for opp_card, p_card in opp_belief.items():
            c0 = my_card if my_pos == 0 else opp_card
            c1 = opp_card if my_pos == 0 else my_card

            # --- EV of PASS ---
            next_h_pass = history + "p"
            if is_terminal(next_h_pass):
                p0_pay, p1_pay = evaluate_payoff(next_h_pass, c0, c1)
                ev_pass += p_card * (p0_pay if my_pos == 0 else p1_pay)
            else:
                # 다음 차례는 상대 (P1)
                opp_probs = self.model.get_action_prob(opp_card, next_h_pass)
                # 상대 PASS -> 쇼다운 pp
                p0_p, p1_p = evaluate_payoff("pp", c0, c1)
                pay_if_opp_p = p0_p if my_pos == 0 else p1_p
                # 상대 BET -> pbp (내 차례로 돌아옴)
                # 여기서는 한 단계 더 깊은 의사결정: 상대가 pb 했을 때 내가 콜(pbb)할지 폴드(pbp)할지의 최적 선택
                p0_fold, p1_fold = evaluate_payoff("pbp", c0, c1)
                pay_fold = p0_fold if my_pos == 0 else p1_fold
                p0_call, p1_call = evaluate_payoff("pbb", c0, c1)
                pay_call = p0_call if my_pos == 0 else p1_call
                best_response_to_bet = max(pay_fold, pay_call)

                ev_pass += p_card * (opp_probs[PASS] * pay_if_opp_p + opp_probs[BET] * best_response_to_bet)

            # --- EV of BET ---
            next_h_bet = history + "b"
            if is_terminal(next_h_bet):
                p0_pay, p1_pay = evaluate_payoff(next_h_bet, c0, c1)
                ev_bet += p_card * (p0_pay if my_pos == 0 else p1_pay)
            else:
                # 다음 차례는 상대
                opp_probs = self.model.get_action_prob(opp_card, next_h_bet)
                # 상대 PASS (bp 또는 pbp) -> 폴드
                p0_fold, p1_fold = evaluate_payoff(next_h_bet + "p", c0, c1)
                pay_fold = p0_fold if my_pos == 0 else p1_fold
                # 상대 BET (bb 또는 pbb) -> 쇼다운
                p0_call, p1_call = evaluate_payoff(next_h_bet + "b", c0, c1)
                pay_call = p0_call if my_pos == 0 else p1_call

                ev_bet += p_card * (opp_probs[PASS] * pay_fold + opp_probs[BET] * pay_call)

        return ev_pass, ev_bet

    def act(self, card: int, history: str) -> int:
        my_pos = len(history) % 2
        ev_pass, ev_bet = self.calculate_ev(card, history, my_pos)

        # GTO 안전장치 (Safety Blending)
        info_set = f"{CARD_NAMES[card]}:{history}"
        gto_p = self.gto_strategy.get(info_set, (0.5, 0.5))
        
        # 상대 관측 표본 수에 따른 신뢰도 (Confidence Weight)
        # 데이터가 적을 때는 GTO를 지키며 안전하게 출발, 데이터가 쌓일수록 100% EV 최적화로 전환
        h_idx = self.model.hist_to_idx.get(history, 0)
        sample_count = np.sum(self.model.total_observed[:, h_idx])
        confidence = sample_count / (sample_count + 15.0) # 15판 수준의 완충 계수

        # 착취 액션 확률 (EV 기반 소프트맥스 / 아르그맥스)
        diff = ev_bet - ev_pass
        # temperature = 0.1의 소프트맥스로 EV가 높은 쪽에 확률 집중
        p_bet_exploit = 1.0 / (1.0 + np.exp(-diff / 0.1))
        exploit_p = np.array([1.0 - p_bet_exploit, p_bet_exploit])

        # 최종 전략 = (1 - conf) * GTO + conf * Exploit
        blended_p = (1.0 - confidence) * np.array(gto_p) + confidence * exploit_p
        blended_p /= np.sum(blended_p)

        return int(np.random.choice(ACTIONS, p=blended_p))
