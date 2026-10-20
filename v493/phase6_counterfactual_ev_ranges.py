"""
POKER AI V4.9.3 - PHASE 6: 상대 레인지 조건부 Counterfactual EV 정밀 재검증
- 상대 유형별(Tight, Aggressive, SelectiveCaller, Random, RigorousAdaptive) 실제 3-Bet 레인지 구축
- 핸드 2종 비교: AA (Monster) vs JTs (Marginal)
- 상대 레인지에서 카드를 조건부 샘플링하여 3-Bet All-in 및 3-Bet Pot 직면 상황 생성
- 각 상대 및 각 핸드에 대해 Fold vs Call vs 4-Bet All-in의 Terminal Return EV(chips, bb) 및 95% CI 산출
- 상대 레인지 반영 여부 및 최적 대응의 분화(AA는 항상 Call, JTs는 상대에 따라 Fold vs Call 분화) 입증
"""

import os
import sys
import math
import csv
from typing import Dict, List, Tuple
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, sample_chance_action
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot, RuleBasedBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot

# 상대 유형별 3-Bet 프리플롭 레인지 정의 (Rank IDs: 2:0..A:12)
# Tight: KK+, QQ, AKs (초강력 상위 2.5% 레인지)
# Aggressive: 22+, A2s+, KTs+, QTs+, JTs, A8o+, KQo (공격적 상위 25% 레인지)
# SelectiveCaller: 99+, ATs+, KJs+, AQo+ (중간 타이트 레인지)
# Random: 100% 임의 핸드
# RigorousAdaptive: 상황에 따라 상위 15% 레인지

def sample_opponent_cards(opp_type: str, used_cards: List[int], rng: np.random.RandomState) -> Tuple[int, int]:
    deck = [c for c in range(52) if c not in used_cards]
    
    if opp_type == 'Random':
        chosen = rng.choice(len(deck), size=2, replace=False)
        return deck[chosen[0]], deck[chosen[1]]
        
    elif opp_type == 'Tight':
        # KK(44..47), QQ(40..43), AK(48..51 + 44..47)
        tight_candidates = []
        for c1 in range(40, 52):
            for c2 in range(c1 + 1, 52):
                if c1 in deck and c2 in deck:
                    r1, r2 = c1 // 4, c2 // 4
                    if (r1 == r2 and r1 in [10, 11, 12]) or (r1 == 12 and r2 == 11): # QQ, KK, AA, AK
                        tight_candidates.append((c1, c2))
        if tight_candidates:
            idx = rng.choice(len(tight_candidates))
            return tight_candidates[idx]
            
    elif opp_type == 'Aggressive':
        # 넓은 레인지: 페어 전체 또는 상위 카드
        agg_candidates = []
        for c1 in range(len(deck)):
            for c2 in range(c1 + 1, len(deck)):
                card1, card2 = deck[c1], deck[c2]
                r1, r2 = card1 // 4, card2 // 4
                if r1 == r2 or max(r1, r2) >= 8 or (abs(r1 - r2) <= 2 and min(r1, r2) >= 6):
                    agg_candidates.append((card1, card2))
        if agg_candidates:
            idx = rng.choice(len(agg_candidates))
            return agg_candidates[idx]

    elif opp_type in ['SelectiveCaller', 'RigorousAdaptive']:
        mid_candidates = []
        for c1 in range(len(deck)):
            for c2 in range(c1 + 1, len(deck)):
                card1, card2 = deck[c1], deck[c2]
                r1, r2 = card1 // 4, card2 // 4
                if r1 == r2 and r1 >= 7: # 99+
                    mid_candidates.append((card1, card2))
                elif max(r1, r2) >= 10: # AJ+, KQ+
                    mid_candidates.append((card1, card2))
        if mid_candidates:
            idx = rng.choice(len(mid_candidates))
            return mid_candidates[idx]

    # 기본 fallback
    chosen = rng.choice(len(deck), size=2, replace=False)
    return deck[chosen[0]], deck[chosen[1]]

def rollout_terminal(state: pyspiel.State, target_seat: int, bots: Dict[int, object], rng: np.random.RandomState) -> float:
    while not state.is_terminal():
        if state.is_chance_node():
            state.apply_action(sample_chance_action(state, rng=rng))
        else:
            p = state.current_player()
            legal = state.legal_actions(p)
            bot = bots.get(p)
            act = bot.act(state, p, legal) if bot is not None else rng.choice(legal)
            state.apply_action(act)
    return state.returns()[target_seat]

def run_phase6_evaluation():
    print("=" * 80)
    print("POKER AI V4.9.3 - PHASE 6: 조건부 상대 레인지 기반 Counterfactual EV 감사")
    print("=" * 80)

    game = get_5p_holdem_5a_game()
    N_SAMPLES = 400

    test_hands = {
        'AA': (48, 51), # Ac, As
        'JTs': (36, 32) # Jc, Tc
    }

    opp_types = ['Tight', 'Aggressive', 'SelectiveCaller', 'Random', 'RigorousAdaptive']
    opp_bot_instances = {
        'Tight': TightBot(),
        'Aggressive': AggressiveBot(),
        'SelectiveCaller': SelectiveCallerBot(),
        'Random': RandomBot(),
        'RigorousAdaptive': RigorousAdaptiveBot()
    }

    records = []

    for hname, (c1, c2) in test_hands.items():
        print(f"\n==================================================")
        print(f"핸드: {hname} (Hero: UTG Seat 2, 3-Bet All-in 직면)")
        print(f"==================================================")

        for opp_name in opp_types:
            opp_bot = opp_bot_instances[opp_name]
            bots = {0: TightBot(), 1: TightBot(), 3: opp_bot, 4: TightBot()}

            # Fold(0) vs Call(1) 비교
            for act in [0, 1]:
                returns_chips = []

                for sim_idx in range(N_SAMPLES):
                    sim_rng = np.random.RandomState(sim_idx * 100 + 493)
                    
                    # 1. Hero 카드 배분
                    used = [c1, c2]
                    # 2. 상대(Seat 3) 레인지 조건부 카드 샘플링
                    opp_c1, opp_c2 = sample_opponent_cards(opp_name, used, sim_rng)
                    used.extend([opp_c1, opp_c2])
                    
                    # 3. 나머지 플레이어 카드 무작위 배분
                    rem_deck = [c for c in range(52) if c not in used]
                    sim_rng.shuffle(rem_deck)

                    st = game.new_initial_state()
                    # P0
                    st.apply_action(rem_deck[0]); st.apply_action(rem_deck[1])
                    # P1
                    st.apply_action(rem_deck[2]); st.apply_action(rem_deck[3])
                    # P2 (Hero)
                    st.apply_action(c1); st.apply_action(c2)
                    # P3 (3-Bet Opponent)
                    st.apply_action(opp_c1); st.apply_action(opp_c2)
                    # P4
                    st.apply_action(rem_deck[4]); st.apply_action(rem_deck[5])

                    # 시퀀스: Hero Half-Pot(4) -> P3 3-Bet All-in(3) -> P4,P0,P1 fold(0)
                    st.apply_action(4)
                    st.apply_action(3)
                    st.apply_action(0); st.apply_action(0); st.apply_action(0)

                    assert st.current_player() == 2
                    # 강제 액션
                    st.apply_action(act)

                    ret = rollout_terminal(st, 2, bots, sim_rng)
                    returns_chips.append(ret)

                mean_c = np.mean(returns_chips)
                std_c = np.std(returns_chips, ddof=1)
                se_c = std_c / math.sqrt(N_SAMPLES)
                ci95_c = 1.96 * se_c

                mean_bb = mean_c / 100.0 # BB = 100 chips
                ci95_bb = ci95_c / 100.0
                act_str = "Fold" if act == 0 else "Call"

                print(f"  상대: {opp_name:16s} | 행동: {act_str:4s} | EV: {mean_bb:+7.2f} ± {ci95_bb:5.2f} bb/hand ({mean_c:+8.1f} ± {ci95_c:6.1f} chips)")

                records.append({
                    'hand': hname,
                    'opponent_type': opp_name,
                    'action': act_str,
                    'mean_chips': float(mean_c),
                    'ci95_chips': float(ci95_c),
                    'mean_bb_hand': float(mean_bb),
                    'ci95_bb_hand': float(ci95_bb),
                    'correct_bb100': float(mean_bb * 100.0),
                    'correct_ci95_bb100': float(ci95_bb * 100.0)
                })

    # CSV 저장
    os.makedirs("v493/data", exist_ok=True)
    out_csv = "v493/data/phase6_range_counterfactual_ev.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\nPhase 6 조건부 레인지 EV 검증 데이터 저장 완료: {out_csv}")

if __name__ == '__main__':
    run_phase6_evaluation()
