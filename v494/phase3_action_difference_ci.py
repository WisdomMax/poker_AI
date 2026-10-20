"""
POKER AI V4.9.4 - PHASE 3: 조건부 행동별 EV 차이(Paired Difference) 신뢰구간 정밀 계산
- 동일 카드 런아웃 및 동일 상대 홀카드를 공유하는 완전 페어드(Paired) 시뮬레이션
- 검증 대상:
  1. AA Facing 3-Bet All-in (Call vs Fold) across 5 opponent types
  2. AA Facing 3-Bet Pot (Call vs Fold, 4-Bet All-in vs Fold, 4-Bet All-in vs Call)
  3. JTs Facing 3-Bet All-in (Call vs Fold) across 5 opponent types
- 산출 통계:
  * Delta EV = Return(Action A) - Return(Action B)
  * Mean Delta (bb/hand and bb/100)
  * Std of Delta, SE of Delta
  * df, t-statistic, exact two-tailed p-value
  * 95% Confidence Interval of Difference
  * 물리적 상한 자동 검증 (Assertion) 내장
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
from v493.phase6_counterfactual_ev_ranges import sample_opponent_cards
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return

def t_distribution_cdf(t: float, df: int) -> float:
    """t-분포 CDF(t, df) 수치 적분"""
    if t == 0: return 0.5
    # Simpson rule
    n_steps = 1000
    x_vals = np.linspace(0, abs(t), n_steps + 1)
    # 정규화 상수
    log_coeff = math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    coeff = math.exp(log_coeff)
    y_vals = coeff * (1.0 + (x_vals**2) / float(df)) ** (-(df + 1) / 2.0)
    integral = (abs(t) / (3.0 * n_steps)) * (y_vals[0] + y_vals[-1] + 4.0 * np.sum(y_vals[1:-1:2]) + 2.0 * np.sum(y_vals[2:-1:2]))
    cdf = 0.5 + integral if t > 0 else 0.5 - integral
    return float(np.clip(cdf, 0.0, 1.0))

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
    ret = state.returns()[target_seat]
    validate_chips_return(ret)
    validate_bb_return(ret / 100.0)
    return ret

def run_phase3_analysis():
    print("=" * 80)
    print("POKER AI V4.9.4 - PHASE 3: 행동 간 EV 차이(Paired Difference) 신뢰구간 산출")
    print("=" * 80)

    game = get_5p_holdem_5a_game()
    N_SAMPLES = 400

    opp_types = ['Tight', 'Aggressive', 'SelectiveCaller', 'Random', 'RigorousAdaptive']
    opp_bots = {
        'Tight': TightBot(),
        'Aggressive': AggressiveBot(),
        'SelectiveCaller': SelectiveCallerBot(),
        'Random': RandomBot(),
        'RigorousAdaptive': RigorousAdaptiveBot()
    }

    test_scenarios = [
        # (Hand, Situation, ActionA, ActionB, LabelA, LabelB)
        ('AA', 'SitB_Facing_3Bet_AllIn', 1, 0, 'Call', 'Fold'),
        ('JTs', 'SitB_Facing_3Bet_AllIn', 1, 0, 'Call', 'Fold'),
        ('AA', 'SitB_Facing_3Bet_Pot', 1, 0, 'Call', 'Fold'),
        ('AA', 'SitB_Facing_3Bet_Pot', 3, 0, '4Bet_Allin', 'Fold'),
        ('AA', 'SitB_Facing_3Bet_Pot', 3, 1, '4Bet_Allin', 'Call'),
    ]

    hand_cards = {
        'AA': (48, 51),
        'JTs': (36, 32)
    }

    records = []

    for hname, sit, actA, actB, labelA, labelB in test_scenarios:
        c1, c2 = hand_cards[hname]
        print(f"\n[비교 시나리오: {hname} | {sit} | {labelA} vs {labelB}]")
        print("-" * 80)

        for opp_name in opp_types:
            opp_bot = opp_bots[opp_name]
            bots = {0: TightBot(), 1: TightBot(), 3: opp_bot, 4: TightBot()}

            deltas_bb = []
            returns_A = []
            returns_B = []

            for sim_idx in range(N_SAMPLES):
                sim_seed = 494000 + sim_idx * 10
                sim_rng = np.random.RandomState(sim_seed)

                # 1. 공통 카드 배분
                used = [c1, c2]
                opp_c1, opp_c2 = sample_opponent_cards(opp_name, used, sim_rng)
                used.extend([opp_c1, opp_c2])
                rem_deck = [c for c in range(52) if c not in used]
                sim_rng.shuffle(rem_deck)

                # 게임 상태 A 생성
                st_A = game.new_initial_state()
                st_A.apply_action(rem_deck[0]); st_A.apply_action(rem_deck[1])
                st_A.apply_action(rem_deck[2]); st_A.apply_action(rem_deck[3])
                st_A.apply_action(c1); st_A.apply_action(c2)
                st_A.apply_action(opp_c1); st_A.apply_action(opp_c2)
                st_A.apply_action(rem_deck[4]); st_A.apply_action(rem_deck[5])

                # 게임 상태 B 생성 (동일 카드 100% 복제)
                st_B = game.new_initial_state()
                st_B.apply_action(rem_deck[0]); st_B.apply_action(rem_deck[1])
                st_B.apply_action(rem_deck[2]); st_B.apply_action(rem_deck[3])
                st_B.apply_action(c1); st_B.apply_action(c2)
                st_B.apply_action(opp_c1); st_B.apply_action(opp_c2)
                st_B.apply_action(rem_deck[4]); st_B.apply_action(rem_deck[5])

                # 베팅 진행
                if 'AllIn' in sit:
                    st_A.apply_action(4); st_A.apply_action(3); st_A.apply_action(0); st_A.apply_action(0); st_A.apply_action(0)
                    st_B.apply_action(4); st_B.apply_action(3); st_B.apply_action(0); st_B.apply_action(0); st_B.apply_action(0)
                else: # 3-Bet Pot
                    st_A.apply_action(4); st_A.apply_action(2); st_A.apply_action(0); st_A.apply_action(0); st_A.apply_action(0)
                    st_B.apply_action(4); st_B.apply_action(2); st_B.apply_action(0); st_B.apply_action(0); st_B.apply_action(0)

                assert st_A.current_player() == 2
                assert st_B.current_player() == 2

                # 강제 액션 적용
                st_A.apply_action(actA)
                st_B.apply_action(actB)

                # 동일 런아웃 롤아웃을 위한 동일 시드 서브 난수기
                rng_rollout_A = np.random.RandomState(sim_seed + 1)
                rng_rollout_B = np.random.RandomState(sim_seed + 1)

                ret_A = rollout_terminal(st_A, 2, bots, rng_rollout_A) / 100.0 # bb
                ret_B = rollout_terminal(st_B, 2, bots, rng_rollout_B) / 100.0 # bb

                delta = ret_A - ret_B
                deltas_bb.append(delta)
                returns_A.append(ret_A)
                returns_B.append(ret_B)

            deltas = np.array(deltas_bb)
            mean_d = float(np.mean(deltas))
            std_d = float(np.std(deltas, ddof=1))
            se_d = std_d / math.sqrt(N_SAMPLES)
            df = N_SAMPLES - 1
            t_stat = mean_d / max(1e-9, se_d)
            p_val = 2.0 * (1.0 - t_distribution_cdf(abs(t_stat), df=df))
            t_crit = 1.966 # for df=399, alpha=0.05
            ci95_margin = t_crit * se_d

            # bb/100 환산
            mean_d_bb100 = mean_d * 100.0
            ci95_bb100 = ci95_margin * 100.0

            sig_str = "유의미함 (p < 0.001)" if p_val < 0.001 else ("유의미함 (p < 0.05)" if p_val < 0.05 else "유의미하지 않음 (p >= 0.05)")

            print(f"  상대: {opp_name:16s} | ΔEV({labelA} - {labelB}): {mean_d:+7.2f} ± {ci95_margin:5.2f} BB/hand ({mean_d_bb100:+8.1f} ± {ci95_bb100:6.1f} bb/100) | t={t_stat:+7.2f} | {sig_str}")

            records.append({
                'hand': hname,
                'situation': sit,
                'action_A': labelA,
                'action_B': labelB,
                'opponent': opp_name,
                'mean_A_bb_hand': float(np.mean(returns_A)),
                'mean_B_bb_hand': float(np.mean(returns_B)),
                'mean_delta_bb_hand': mean_d,
                'ci95_delta_bb_hand': ci95_margin,
                'mean_delta_bb100': mean_d_bb100,
                'ci95_delta_bb100': ci95_bb100,
                'std_delta_bb_hand': std_d,
                'se_delta_bb_hand': se_d,
                'df': df,
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': sig_str
            })

    os.makedirs("v494/data", exist_ok=True)
    out_csv = "v494/data/phase3_action_difference_ci.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\nPhase 3 행동 간 EV 차이 신뢰구간 분석 완료: {out_csv} (총 {len(records)}개 시나리오)")

if __name__ == '__main__':
    run_phase3_analysis()
