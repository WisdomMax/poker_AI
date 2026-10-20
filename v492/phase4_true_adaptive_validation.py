"""
POKER AI V4.9.2 - PHASE 4: Adaptive Bot 신뢰성 및 적응 능력 독립 검증
- 검증 1: V4.9.1 TrueAdaptiveBot이 과거 상대 이력에 실제 반응하는지 감사 (가짜 적응 여부 규명)
- 검증 2: 진정한 에이전트별 이력 추적 및 상황 적응 봇(RigorousAdaptiveBot) 구축
- 검증 3: 동일 카드 + 동일 현재 베팅 상황에서 상대 과거 이력 (A: All-in Spammer vs B: Tight/Passive) 주입 시 행동 분화 검증
- 검증 4: 좌석 회전 시 에이전트 ID 분리, 비공개 카드 누출 금지 검증
"""

import os
import sys
import csv
from typing import Dict, List, Optional
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, HoldemSeatRotationWrapper
from v48.opponents_pool import BaseBot, extract_hole_cards
from v491.phase4_adaptive_bot_audit import TrueAdaptiveBot as V491AdaptiveBot

class RigorousAdaptiveBot(BaseBot):
    """
    진정한 상대 적응 봇 (Rigorous Adaptive Bot):
    1. 플레이어별 독립 통계 관리 (agent_id 기준, 좌석 회전과 완벽 분리)
    2. 비공개 카드 접근 절대 금지 (오직 공개 베팅 시퀀스만 추적)
    3. 과거 상대 성향(All-in 빈도, Aggression Factor)에 따라
       동일한 핸드와 동일한 베팅 직면 상황에서도 전혀 다른 대응 행동을 동적으로 선택.
    """
    def __init__(self, name: str = "RigorousAdaptiveBot", seat_wrapper: Optional[HoldemSeatRotationWrapper] = None):
        super().__init__(name)
        self.seat_wrapper = seat_wrapper
        # agent_id별 통계: {agent_id: {'allin_count': int, 'raise_count': int, 'call_count': int, 'fold_count': int, 'hands': int}}
        self.agent_stats = {i: {'allin_count': 0, 'raise_count': 0, 'call_count': 0, 'fold_count': 0, 'hands': 0} for i in range(5)}
        self.current_hand_idx = 0

    def reset_stats(self):
        self.agent_stats = {i: {'allin_count': 0, 'raise_count': 0, 'call_count': 0, 'fold_count': 0, 'hands': 0} for i in range(5)}
        self.current_hand_idx = 0

    def set_hand_index(self, hand_idx: int):
        self.current_hand_idx = hand_idx

    def observe_action(self, physical_seat: int, action: int):
        """공개 액션을 관측하여 해당 물리적 좌석에 앉은 실제 에이전트 ID에 누적"""
        if self.seat_wrapper:
            agent_id = self.seat_wrapper.get_seat_to_agent(self.current_hand_idx)[physical_seat]
        else:
            agent_id = physical_seat
            
        stats = self.agent_stats[agent_id]
        if action == 3: # All-in
            stats['allin_count'] += 1
        elif action in [2, 4]: # Pot / Half-Pot Raise
            stats['raise_count'] += 1
        elif action == 1: # Call
            stats['call_count'] += 1
        elif action == 0: # Fold
            stats['fold_count'] += 1

    def inject_prior_history(self, agent_id: int, profile: str, sample_size: int = 50):
        """특정 에이전트에 과거 이력 프로필을 합성 주입 (실험용)"""
        stats = self.agent_stats[agent_id]
        stats['hands'] += sample_size
        if profile == 'allin_spammer':
            stats['allin_count'] += int(sample_size * 0.70)
            stats['raise_count'] += int(sample_size * 0.20)
            stats['call_count'] += int(sample_size * 0.10)
        elif profile == 'tight_passive':
            stats['allin_count'] += int(sample_size * 0.02)
            stats['raise_count'] += int(sample_size * 0.04)
            stats['call_count'] += int(sample_size * 0.34)
            stats['fold_count'] += int(sample_size * 0.60)
        elif profile == 'balanced':
            stats['allin_count'] += int(sample_size * 0.10)
            stats['raise_count'] += int(sample_size * 0.25)
            stats['call_count'] += int(sample_size * 0.35)
            stats['fold_count'] += int(sample_size * 0.30)

    def get_opponent_aggression(self, target_agent_id: int) -> float:
        stats = self.agent_stats[target_agent_id]
        total = stats['allin_count'] + stats['raise_count'] + stats['call_count'] + stats['fold_count']
        if total == 0:
            return 0.15 # 기본 중간값
        return (stats['allin_count'] * 1.5 + stats['raise_count']) / total

    def act(self, state: pyspiel.State, cur_player: int, legal_acts: List[int]) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        
        # 1. 시퀀스에서 가장 최근에 베팅/올인을 주도한 어그레서 좌석 식별
        aggressor_seat = self._find_last_aggressor(info, cur_player)
        if self.seat_wrapper:
            target_agent = self.seat_wrapper.get_seat_to_agent(self.current_hand_idx)[aggressor_seat]
        else:
            target_agent = aggressor_seat
            
        opp_agg = self.get_opponent_aggression(target_agent)
        
        # 베팅 압박 직면 여부
        is_facing_allin = (len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts)
        is_facing_bet = (len(legal_acts) > 0 and legal_acts[0] == 0) # 0: Fold가 합법이면 누군가 베팅한 상태
        
        # 핸드 강도 분류
        ranks = [c[0] for c in hc] if len(hc) == 2 else []
        is_pair = (len(ranks) == 2 and ranks[0] == ranks[1])
        is_suited = (len(hc) == 2 and hc[0][1] == hc[1][1])
        
        is_monster = (is_pair and ranks[0] in ['A', 'K', 'Q']) or ('A' in ranks and 'K' in ranks)
        is_marginal = (is_pair and ranks[0] in ['J', 'T', '9', '8', '7']) or \
                      ('A' in ranks and ranks[1] in ['Q', 'J', 'T'] if len(ranks)==2 and ranks[0]=='A' else False) or \
                      ('K' in ranks and ranks[1] in ['Q', 'J'] if len(ranks)==2 and ranks[0]=='K' else False) or \
                      (is_suited and 'J' in ranks and 'T' in ranks)
        
        # 1. 몬스터 핸드: 상대 불문 강력 응수
        if is_monster:
            if is_facing_allin:
                return 1 # Call All-in
            if 3 in legal_acts:
                return 3 # All-in 3-Bet / 4-Bet
            if 2 in legal_acts:
                return 2 # Pot
            return 1

        # 2. 트래시 핸드: 상대 불문 폴드
        if not is_marginal and not is_monster:
            if 0 in legal_acts:
                return 0
            return 1 # Check

        # 3. 마지널 핸드 (JTs, TT, AQs, KQs): 상대 이력(opp_agg)에 따라 동적 적응!
        # - 상대가 All-in Spammer / Maniac (opp_agg >= 0.35):
        #   상대 범위가 넓으므로 올인 직면 시 Call(1)! (EV 극대화)
        # - 상대가 Tight / Passive (opp_agg < 0.35):
        #   상대의 올인은 순수 몬스터이므로 칼폴드(0)!
        if is_facing_allin:
            if opp_agg >= 0.35:
                return 1 # Spammer 상대로 마지널 콜!
            else:
                return 0 # Nit 상대로 칼폴드!
                
        if is_facing_bet:
            if opp_agg >= 0.35:
                # Spammer의 오픈에 적극적 리레이즈(Pot 2 또는 HalfPot 4) 또는 Call
                if 2 in legal_acts: return 2
                if 4 in legal_acts: return 4
                return 1
            else:
                return 0 # Nit의 오픈에는 폴드!

        if 4 in legal_acts:
            return 4
        return 1

    def _find_last_aggressor(self, info_str: str, cur_player: int) -> int:
        import re
        if 'Sequences: ' not in info_str:
            return (cur_player - 1) % 5
        seq = info_str.split('Sequences: ')[1].split(']')[0]
        tokens = re.findall(r'[rcf]\d*', seq)
        # Preflop player sequence (2, 3, 4, 0, 1, 2, ...)
        p_order = [2, 3, 4, 0, 1]
        actions_by_player = []
        for i, tok in enumerate(tokens):
            p = p_order[i % 5]
            actions_by_player.append((p, tok))
        for p, tok in reversed(actions_by_player):
            if tok.startswith('r') and p != cur_player:
                return p
        return (cur_player - 1) % 5

def run_phase4_audit():
    print("=" * 80)
    print("POKER AI V4.9.2 - PHASE 4: Adaptive Bot 신뢰성 및 적응 검증")
    print("=" * 80)

    game = get_5p_holdem_5a_game()
    seat_wrap = HoldemSeatRotationWrapper(game)

    # 1. V4.9.1 TrueAdaptiveBot의 과거 이력 적응 능력 감사
    print("\n[검증 1] V4.9.1 TrueAdaptiveBot 감사: 과거 상대 이력에 반응하는가?")
    bot_v491 = V491AdaptiveBot()
    
    # 두 개의 동일한 상태 (JTs를 쥐고 BTN의 3-Bet All-in 직면)
    # A 조건: 과거 올인이 매우 많았던 상대
    # B 조건: 과거 매우 타이트했던 상대
    # V4.9.1 bot_v491에 각각 이력을 주입하고 행동 측정
    
    test_hands = ['JTs', 'TT', 'AQs', '72o', 'AA']
    hand_cards = {
        'JTs': (36, 32), # Jc, Tc
        'TT': (32, 33),  # Tc, Td
        'AQs': (48, 40), # Ac, Qc
        '72o': (20, 1),  # 7c, 2d
        'AA': (48, 51)   # Ac, As
    }

    records = []

    for hname, (c1, c2) in hand_cards.items():
        # 상태 구성: P2 HalfPot 오픈 -> P3 All-in 3-Bet -> others fold -> P2 직면
        deck = [c for c in range(52) if c not in (c1, c2)]
        st = game.new_initial_state()
        st.apply_action(deck[0]); st.apply_action(deck[1]) # P0
        st.apply_action(deck[2]); st.apply_action(deck[3]) # P1
        st.apply_action(c1); st.apply_action(c2)           # P2 (Target)
        st.apply_action(deck[4]); st.apply_action(deck[5]) # P3
        st.apply_action(deck[6]); st.apply_action(deck[7]) # P4

        st.apply_action(4) # P2 HalfPot
        st.apply_action(3) # P3 All-in 3-Bet
        st.apply_action(0); st.apply_action(0); st.apply_action(0) # P4, P0, P1 fold
        legal = st.legal_actions(2)

        # V4.9.1 봇에 All-in Spammer 이력 주입
        bot_v491.history_window = [1] * 30
        act_v491_spammer = bot_v491.act(st, 2, legal)

        # V4.9.1 봇에 Tight 이력 주입
        bot_v491.history_window = [0] * 30
        act_v491_tight = bot_v491.act(st, 2, legal)

        diff_v491 = (act_v491_spammer != act_v491_tight)
        print(f"  V4.9.1 Bot | Hand: {hname:5s} | vs Spammer: {act_v491_spammer} | vs Tight: {act_v491_tight} | 행동 분화: {diff_v491}")
        
        records.append({
            'bot_version': 'V4.9.1_TrueAdaptiveBot',
            'hand': hname,
            'act_spammer': act_v491_spammer,
            'act_tight': act_v491_tight,
            'differentiated': diff_v491
        })

    # 2. RigorousAdaptiveBot의 과거 이력 적응 능력 검증
    print("\n[검증 2] RigorousAdaptiveBot: 과거 상대 이력(Spammer vs Tight)에 따른 동적 행동 분화 검증")
    bot_rigorous = RigorousAdaptiveBot(seat_wrapper=seat_wrap)

    for hname, (c1, c2) in hand_cards.items():
        deck = [c for c in range(52) if c not in (c1, c2)]
        st = game.new_initial_state()
        st.apply_action(deck[0]); st.apply_action(deck[1]) # P0
        st.apply_action(deck[2]); st.apply_action(deck[3]) # P1
        st.apply_action(c1); st.apply_action(c2)           # P2 (Target)
        st.apply_action(deck[4]); st.apply_action(deck[5]) # P3
        st.apply_action(deck[6]); st.apply_action(deck[7]) # P4

        st.apply_action(4) # P2 HalfPot
        st.apply_action(3) # P3 All-in 3-Bet
        st.apply_action(0); st.apply_action(0); st.apply_action(0) # P4, P0, P1 fold
        legal = st.legal_actions(2)

        # A: P3(직전 상대)가 All-in Spammer인 경우
        bot_rigorous.reset_stats()
        bot_rigorous.inject_prior_history(agent_id=3, profile='allin_spammer')
        act_spammer = bot_rigorous.act(st, 2, legal)

        # B: P3(직전 상대)가 Tight Passive인 경우
        bot_rigorous.reset_stats()
        bot_rigorous.inject_prior_history(agent_id=3, profile='tight_passive')
        act_tight = bot_rigorous.act(st, 2, legal)

        diff = (act_spammer != act_tight)
        act_names = {0: 'Fold', 1: 'Call', 2: 'Pot', 3: 'All-in', 4: 'Half-Pot'}
        print(f"  RigorousBot | Hand: {hname:5s} | vs Spammer: {act_names[act_spammer]:7s} | vs Tight: {act_names[act_tight]:7s} | 행동 분화: {diff}")

        records.append({
            'bot_version': 'RigorousAdaptiveBot',
            'hand': hname,
            'act_spammer': act_spammer,
            'act_tight': act_tight,
            'differentiated': diff
        })

    # 3. 좌석 회전 및 에이전트 ID 분리 정합성 검증
    print("\n[검증 3] 좌석 회전 및 에이전트 ID 분리 정합성 검증")
    bot_rigorous.reset_stats()
    # Hand 0: Seat 3 = Agent 3
    bot_rigorous.set_hand_index(0)
    bot_rigorous.observe_action(physical_seat=3, action=3) # Seat 3 all-in
    assert bot_rigorous.agent_stats[3]['allin_count'] == 1
    assert bot_rigorous.agent_stats[2]['allin_count'] == 0

    # Hand 1: Seat 3 = Agent (3 + 1) % 5 = Agent 4!
    bot_rigorous.set_hand_index(1)
    bot_rigorous.observe_action(physical_seat=3, action=3) # Seat 3 all-in
    assert bot_rigorous.agent_stats[4]['allin_count'] == 1
    assert bot_rigorous.agent_stats[3]['allin_count'] == 1
    print("  좌석 회전 매핑 검증 통과: Seat 3의 행동이 Hand 0에서는 Agent 3에, Hand 1에서는 Agent 4에 각각 정확히 분리 누적됨!")

    # 4. 결과 CSV 저장
    os.makedirs('v492/data', exist_ok=True)
    out_csv = 'v492/data/phase4_adaptive_validation.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nPhase 4 적응 검증 데이터 저장 완료: {out_csv} (총 {len(records)} 행)")

if __name__ == '__main__':
    run_phase4_audit()
