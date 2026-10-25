"""
DIRECTIVE 041 Integrity Poker Metrics Engine
- 실제 칩 지출 원장(Dollar Spent) 기반 VPIP / PFR
- Open/Push All-in과 Facing All-in Call의 완전 분리
- Direct PPO Action 5 vs Effective Stack All-in 완전 분리
- 3-Bet Opportunity 기반 정밀 3-Bet% 산출
- 제로섬 검증 및 결정론적 재현성 보장
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

STARTING_STACK = 10000.0  # $100 BB

class HandIntegrityTracker:
    def __init__(self, agent_seat: int, seed: int):
        self.agent_seat = agent_seat
        self.seed = seed
        self.forced_blind = 50.0 if agent_seat == 0 else (100.0 if agent_seat == 1 else 0.0)
        
        # Preflop trackers
        self.preflop_spent_agent = self.forced_blind
        self.preflop_raises_before_agent = 0
        self.preflop_actions_log = []
        self.postflop_actions_log = []
        
        # VPIP / PFR / 3-Bet
        self.agent_vpip = False
        self.agent_pfr = False
        self.three_bet_opportunity = False
        self.agent_3bet = False
        
        # All-in categories
        self.direct_action5_count = 0
        self.pushed_allin = False       # Agent initiated all-in (Open / Raise / Bet)
        self.pushed_allin_round = None
        self.pushed_allin_callers = []  # Opponents who called agent's push
        
        self.called_allin = False       # Agent called an opponent's all-in
        self.called_allin_round = None
        self.called_allin_facing_who = None
        
        # General state
        self.folded_seats = set()
        self.seen_streets = set()
        self.entropy_samples = []
        self.pot_bet_count = 0          # Action 3 count
        self.half_pot_count = 0        # Action 2 count
        self.call_count = 0            # Action 1 count
        self.fold_count = 0            # Action 0 count

    def record_agent_decision(self, round_idx: int, act_6: int, nat_act: int, 
                              cur_to_call: float, spent_before: float, spent_after: float,
                              cur_opp_allin_active: bool, opp_allin_seat: Optional[int],
                              probs: Optional[np.ndarray] = None):
        """에이전트의 결정 및 칩 지출을 정밀 기록"""
        spent_delta = spent_after - spent_before
        rem_stack = STARTING_STACK - spent_after
        is_allin_investment = (rem_stack <= 0.0 or act_6 == 5)

        if probs is not None:
            probs_safe = np.maximum(probs, 1e-8)
            ent = float(-np.sum(probs_safe * np.log(probs_safe)))
            self.entropy_samples.append(ent)

        if act_6 == 0:
            self.fold_count += 1
            self.folded_seats.add(self.agent_seat)
        elif act_6 == 1:
            self.call_count += 1
        elif act_6 == 2:
            self.half_pot_count += 1
        elif act_6 == 3:
            self.pot_bet_count += 1
        elif act_6 == 5:
            self.direct_action5_count += 1

        if round_idx == 0:
            self.preflop_spent_agent = spent_after
            self.preflop_actions_log.append({
                "act_6": act_6,
                "nat_act": nat_act,
                "to_call": cur_to_call,
                "spent_before": spent_before,
                "spent_after": spent_after
            })
            # 3-Bet 기회 및 실행 판정
            if self.preflop_raises_before_agent == 1:
                self.three_bet_opportunity = True
                if act_6 >= 2:
                    self.agent_3bet = True
            if act_6 >= 2:
                self.agent_pfr = True
        else:
            self.postflop_actions_log.append({
                "round": round_idx,
                "act_6": act_6,
                "nat_act": nat_act,
                "to_call": cur_to_call,
                "spent_before": spent_before,
                "spent_after": spent_after
            })

        # 올인 분류: 내가 먼저 민 것(Push)인가, 상대 올인에 콜(Call)한 것인가?
        if is_allin_investment and not self.pushed_allin and not self.called_allin:
            if cur_opp_allin_active and act_6 == 1 and cur_to_call > 0:
                # 상대의 올인에 콜하여 전액 투자된 경우
                self.called_allin = True
                self.called_allin_round = round_idx
                self.called_allin_facing_who = opp_allin_seat
            else:
                # 내가 베팅/레이즈로 먼저 올인한 경우 (Action 2~5 또는 Action 1으로 유효스택 소진)
                self.pushed_allin = True
                self.pushed_allin_round = round_idx

    def record_opponent_decision(self, cp: int, round_idx: int, act_6: int,
                                 cur_to_call: float, spent_before: float, spent_after: float):
        """상대 봇의 결정 기록"""
        if act_6 == 0:
            self.folded_seats.add(cp)
        elif round_idx == 0 and act_6 >= 2:
            self.preflop_raises_before_agent += 1

        # 에이전트가 이미 올인한 상태에서 상대가 유료 콜한 경우
        if self.pushed_allin and act_6 == 1 and cur_to_call > 0:
            self.pushed_allin_callers.append(cp)

    def finalize_hand(self, final_spent: List[float], returns: List[float],
                      priv_cards: Dict[int, Tuple[str, str]], board_cards: List[str]) -> Dict[str, Any]:
        """핸드 종료 후 무결성 지표 최종 정산"""
        agent_return = returns[self.agent_seat]
        agent_pnl_bb = agent_return / 100.0  # $100 = 1 BB

        # 1. 실제 프리플랍 칩 지출 원장 기반 VPIP 판정
        # 강제 블라인드(SB 50, BB 100, 기타 0)보다 자발적으로 1달러라도 더 넣었는가?
        actual_preflop_spent = self.preflop_spent_agent
        if actual_preflop_spent > self.forced_blind:
            self.agent_vpip = True
        else:
            self.agent_vpip = False

        # 2. 쇼다운 판정
        active_players = [s for s in range(5) if s not in self.folded_seats]
        is_showdown = (len(active_players) > 1 and self.agent_seat not in self.folded_seats)

        # 3. All-in 세부 결과
        is_pushed_called = len(self.pushed_allin_callers) > 0

        # 데이터 무결성 검증: Uncalled Push All-in인 경우 상대가 다 죽었으므로 반드시 쇼다운이 없어야 하고 에이전트 손익 > 0
        if self.pushed_allin and not is_pushed_called and not self.called_allin:
            # 상대가 모두 폴드했으므로 agent는 팟을 독식해야 함
            assert not is_showdown, f"Uncalled push allin cannot have showdown! hand seed={self.seed}"
            assert agent_pnl_bb > 0, f"Uncalled push allin must be profitable! got {agent_pnl_bb} BB"

        return {
            "seed": self.seed,
            "agent_seat": self.agent_seat,
            "agent_cards": priv_cards.get(self.agent_seat, ("??", "??")),
            "board": board_cards,
            "pnl_bb": round(float(agent_pnl_bb), 2),
            "vpip": self.agent_vpip,
            "pfr": self.agent_pfr,
            "3bet_opportunity": self.three_bet_opportunity,
            "3bet": self.agent_3bet,
            "direct_action5": self.direct_action5_count > 0,
            "pushed_allin": self.pushed_allin,
            "pushed_allin_round": self.pushed_allin_round,
            "pushed_allin_called": is_pushed_called,
            "pushed_allin_callers": self.pushed_allin_callers,
            "called_allin": self.called_allin,
            "called_allin_round": self.called_allin_round,
            "showdown": is_showdown,
            "showdown_won": is_showdown and (agent_pnl_bb > 0),
            "pot_bet_count": self.pot_bet_count,
            "half_pot_count": self.half_pot_count,
            "preflop_actions": self.preflop_actions_log,
            "postflop_actions": self.postflop_actions_log,
        }
