"""
POKER AI V4.9.2 - PHASE 3: 대응 행동별 Counterfactual EV 정밀 평가
- 동일한 게임 상태를 Clone하여 합법 행동(Fold vs Call vs Bet 등)을 각각 1회 강제 실행
- 나머지 게임은 정상 정책(상대 봇 및 AI 모델)으로 종료 시점까지 진행
- Terminal Return 기준 순수익(Chips, bb)의 Monte Carlo 평균 EV 및 95% 신뢰구간 산출
- 핵심 분석: AA 3-Bet 직면 상태에서 Fold vs Call vs All-in의 상대별 EV 정밀 비교
"""

import os
import sys
import csv
import math
from typing import Dict, List, Tuple
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, sample_chance_action
from v48.opponents_pool import (
    AlwaysCallBot, AggressiveBot, RandomBot, TightBot, 
    RuleBasedBot, extract_hole_cards
)
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot

def rollout_subgame(state: pyspiel.State, target_player: int, bots: Dict[int, object], rng: np.random.RandomState) -> float:
    """
    현재 state에서부터 게임이 끝날 때까지 롤아웃 수행 후 target_player의 return 반환
    """
    while not state.is_terminal():
        if state.is_chance_node():
            state.apply_action(sample_chance_action(state, rng=rng))
        else:
            p = state.current_player()
            legal = state.legal_actions(p)
            bot = bots.get(p)
            if bot is not None:
                a = bot.act(state, p, legal)
            else:
                a = rng.choice(legal)
            state.apply_action(a)
    return state.returns()[target_player]

def run_phase3_eval():
    print("=" * 80)
    print("POKER AI V4.9.2 - PHASE 3: Counterfactual EV 정밀 평가 시작")
    print("=" * 80)

    game = get_5p_holdem_5a_game()
    rng_master = np.random.RandomState(492)
    N_SAMPLES = 300

    # 검사 대상 상대 봇 풀
    opponent_pool = {
        'Aggressive': AggressiveBot(),
        'Random': RandomBot(),
        'Tight': TightBot(),
        'RuleBased': RuleBasedBot(),
        'RigorousAdaptive': RigorousAdaptiveBot()
    }

    records = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 실험 1: AA Half-Pot 오픈 후 3-Bet All-in 직면 상태 (C-1)
    # Target Player: 2 (UTG), Cards: AA (48, 51)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[실험 1] AA Half-Pot 오픈 후 3-Bet All-in 직면 상태 (C-1)")
    print(f"  몬테카를로 샘플 수: N = {N_SAMPLES}")
    print("-" * 75)

    for opp_name, opp_bot in opponent_pool.items():
        # 테이블 봇 배치: P3(직전 3-bet 상대) = opp_bot, 나머지 = TightBot
        bots = {
            0: TightBot(),
            1: TightBot(),
            3: opp_bot,
            4: TightBot()
        }

        # 비교 대상 행동: Fold(0) vs Call(1)
        actions_to_test = [0, 1]

        for act in actions_to_test:
            returns = []
            
            for sim_idx in range(N_SAMPLES):
                # 새로운 독립 핸드 생성 (동일 조건)
                sim_seed = 10000 + sim_idx
                rng = np.random.RandomState(sim_seed)
                
                # 덱 생성: P2에게 AA 배분
                deck = [c for c in range(52) if c not in (48, 51)]
                rng.shuffle(deck)
                
                st = game.new_initial_state()
                deck_i = 0
                for p in range(5):
                    if p == 2:
                        st.apply_action(48); st.apply_action(51)
                    else:
                        st.apply_action(deck[deck_i]); deck_i += 1
                        st.apply_action(deck[deck_i]); deck_i += 1

                # 프리플롭 진행: P2 Half-Pot(4) -> P3 3-Bet All-in(3) -> P4,P0,P1 fold(0)
                st.apply_action(4)
                st.apply_action(3)
                st.apply_action(0); st.apply_action(0); st.apply_action(0)

                assert st.current_player() == 2
                
                # 강제 액션 수행
                st.apply_action(act)
                
                # 나머지 게임 정상 롤아웃
                ret = rollout_subgame(st, target_player=2, bots=bots, rng=rng)
                returns.append(ret)

            # 통계 산출 (chips 및 bb: 1 bb = 100 chips)
            mean_chips = np.mean(returns)
            std_chips = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
            sem_chips = std_chips / math.sqrt(N_SAMPLES)
            ci95_chips = 1.96 * sem_chips

            mean_bb = mean_chips / 100.0
            ci95_bb = ci95_chips / 100.0

            act_label = "Fold" if act == 0 else "Call"
            print(f"  상대: {opp_name:17s} | 행동: {act_label:4s} | EV: {mean_bb:+7.2f} ± {ci95_bb:5.2f} bb ({mean_chips:+8.1f} ± {ci95_chips:6.1f} chips)")

            records.append({
                'situation': 'SitC_AA_Facing_3Bet_AllIn',
                'hand': 'AA',
                'opponent': opp_name,
                'action_id': act,
                'action_name': act_label,
                'n_samples': N_SAMPLES,
                'mean_chips': float(mean_chips),
                'ci95_chips': float(ci95_chips),
                'mean_bb': float(mean_bb),
                'ci95_bb': float(ci95_bb)
            })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 실험 2: AA Half-Pot 오픈 후 3-Bet Pot 직면 상태 (C-2)
    # 합법 행동: Fold(0), Call(1), 4-Bet Pot(2), 4-Bet All-in(3), 4-Bet Half-Pot(4)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[실험 2] AA Half-Pot 오픈 후 3-Bet Pot 직면 상태 (C-2)")
    print(f"  몬테카를로 샘플 수: N = {N_SAMPLES}")
    print("-" * 75)

    for opp_name in ['Aggressive', 'RuleBased']:
        opp_bot = opponent_pool[opp_name]
        bots = {0: TightBot(), 1: TightBot(), 3: opp_bot, 4: TightBot()}

        for act in [0, 1, 2, 3, 4]:
            returns = []
            for sim_idx in range(N_SAMPLES):
                sim_seed = 20000 + sim_idx
                rng = np.random.RandomState(sim_seed)
                deck = [c for c in range(52) if c not in (48, 51)]
                rng.shuffle(deck)

                st = game.new_initial_state()
                deck_i = 0
                for p in range(5):
                    if p == 2:
                        st.apply_action(48); st.apply_action(51)
                    else:
                        st.apply_action(deck[deck_i]); deck_i += 1
                        st.apply_action(deck[deck_i]); deck_i += 1

                # P2 Half-Pot(4) -> P3 3-Bet Pot(2) -> others fold(0)
                st.apply_action(4)
                st.apply_action(2)
                st.apply_action(0); st.apply_action(0); st.apply_action(0)

                assert st.current_player() == 2
                st.apply_action(act)
                ret = rollout_subgame(st, target_player=2, bots=bots, rng=rng)
                returns.append(ret)

            mean_chips = np.mean(returns)
            std_chips = np.std(returns, ddof=1)
            sem_chips = std_chips / math.sqrt(N_SAMPLES)
            ci95_chips = 1.96 * sem_chips
            mean_bb = mean_chips / 100.0
            ci95_bb = ci95_chips / 100.0

            act_names = {0: 'Fold', 1: 'Call', 2: 'Pot', 3: 'All-in', 4: 'Half-Pot'}
            act_label = act_names[act]
            print(f"  상대: {opp_name:10s} | 행동: {act_label:8s} | EV: {mean_bb:+7.2f} ± {ci95_bb:5.2f} bb ({mean_chips:+8.1f} ± {ci95_chips:6.1f} chips)")

            records.append({
                'situation': 'SitC_AA_Facing_3Bet_Pot',
                'hand': 'AA',
                'opponent': opp_name,
                'action_id': act,
                'action_name': act_label,
                'n_samples': N_SAMPLES,
                'mean_chips': float(mean_chips),
                'ci95_chips': float(ci95_chips),
                'mean_bb': float(mean_bb),
                'ci95_bb': float(ci95_bb)
            })

    # CSV 저장
    os.makedirs('v492/data', exist_ok=True)
    out_csv = 'v492/data/phase3_counterfactual_ev.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nPhase 3 Counterfactual EV 데이터 저장 완료: {out_csv} (총 {len(records)} 행)")

if __name__ == '__main__':
    run_phase3_eval()
