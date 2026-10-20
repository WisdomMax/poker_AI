"""
POKER AI V9.1: CHIP-FLOW FORENSICS & RL CONTRIBUTION ABLATION
=============================================================
- Phase 1: Precision Chip-Flow Matrix Forensics (5,000 Hands 5-Player Arena)
  * Who took chips from whom? Exact 5x5 chip transfer matrix
- Phase 2: RL Contribution Ablation Study (2,500 Hands Each)
  * Model A: Fallback Heuristic Only
  * Model B: Q-Learning Only (Zero Fallback)
  * Model C: Q-Learning + Fallback Heuristic
- Phase 3: Aggressive Value Betting Engine (Value-Enhanced Hero vs Baseline Temp-Hybrid)
- Phase 4: Holdout Generalization Arena (TAGBot, ManiacBot, AdaptiveCallingStation, DynamicBot)
- Phase 5: Financial Zero-Sum Ledger & 5-Core Question Synthesis
"""

import os
import sys
import csv
import json
import time
import math
import hashlib
import random
from collections import defaultdict, Counter, deque
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

# 모듈 경로 연결
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v52')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v71')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v72')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v75')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v79')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v81')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v82')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v83')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v84')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v86')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v87')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v88')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v90')))

from v52.phase5_6_7_five_player_benchmark import advance_chance_nodes
from v72.run_v72_true_ledger_and_fold_audit import extract_player_spent
from v75.run_v75_true_6action_and_fold_recovery import True6ActionAdapter
from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent,
    AggressiveBot, TightPassiveBot
)
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v82.run_v82_bayesian_range_and_cfr_audit import estimate_range_aware_equity
from v83.run_v83_robust_range_and_cfr_audit import (
    RobustBayesianRangeModel, get_hand_bucket
)
from v84.run_v84_selective_river_cfr_and_range_synergy import SelectiveRiverCFR
from v86.run_v86_meta_controller_and_independent_audit import BalancedBot, DynamicSwitchingBot
from v87.run_v87_adversarial_red_team_arena import UltraAggressiveBot, V87MetaControllerAgent
from v88.run_v88_adversarial_learning_and_hero_exploitation import CallingStationBot
from v90.run_v90_adversarial_causal_audit_and_scaling import MultiwayHeroExploiter, train_multiway_exploiter

# =====================================================================
# NEW HOLDOUT OPPONENTS (미사용 신규 상대 풀)
# =====================================================================
class TightAggressiveBot:
    """TAG: 좋은 핸드(Medium+)일 때만 팟 사이즈 벳/레이즈를 날리고, 에어는 100% 폴드"""
    def select_action(self, st, cp, legal_6: list):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        h_cards = list(np.where(inf[5:57] == 1.0)[0])
        b_cards = list(np.where(inf[57:109] == 1.0)[0])
        bucket = get_hand_bucket(h_cards, b_cards)

        if to_call > 0:
            if bucket in ["Monster", "Medium"]:
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[0] if raises else 1
            return 0 if 0 in legal_6 else 1
        else:
            if bucket in ["Monster", "Medium"]:
                bets = [a for a in legal_6 if a in [2, 3]]
                return bets[0] if bets else 1
            return 1

class ManiacBot:
    """극단적 올인 및 오버베팅을 무작위로 날리는 매니악 봇"""
    def __init__(self, seed=20269101):
        self.rng = random.Random(seed)
    def select_action(self, st, cp, legal_6: list):
        if self.rng.random() < 0.60:
            raises = [a for a in legal_6 if a in [3, 5]]
            if raises: return raises[-1]
        return 1 if 1 in legal_6 else 0

# =====================================================================
# PHASE 3: Value-Enhanced Hero (공격적 밸류 베팅 모듈 탑재)
# =====================================================================
class ValueEnhancedHero:
    """
    기존 Frozen Temp-Hybrid (T=2.0)의 철벽 방어선을 유지하면서,
    자신의 에쿼티가 높을 때 피쉬 봇들을 상대로 팟 사이즈 밸류 벳을 날려 칩을 극대화하는 진화형 에이전트.
    """
    def __init__(self, blueprint_agent: TrueNFSP_6Action_Agent, temperature=2.0):
        self.blueprint = blueprint_agent
        self.temperature = temperature
        self.value_bets_made = 0
        self.defensive_calls_made = 0

    def select_action(self, st, cp, legal_6: list, folded_players=None):
        spent = extract_player_spent(st)
        curr_spent = spent[cp]
        max_spent = max(spent)
        to_call = max_spent - curr_spent
        pot = sum(spent)

        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hero_cards = list(np.where(inf[5:57] == 1.0)[0])
        board_cards = list(np.where(inf[57:109] == 1.0)[0])
        active_opps = sum(1 for p in range(5) if p != cp and (folded_players is None or p not in folded_players))

        # 에쿼티 및 팟오즈 계산
        combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
            hero_cards, board_cards, mode='temperature', temperature=self.temperature,
            opp_aggression_level=1.5, is_raised=(to_call >= 100)
        )
        pot_odds = to_call / float(pot + to_call) if (pot + to_call) > 0 else 0.0
        equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)

        # 1. 공격적 밸류 베팅 모듈 (자신의 에쿼티가 0.60 이상으로 확실히 유리할 때 적극 공격)
        if to_call == 0:
            if equity >= 0.60:
                self.value_bets_made += 1
                bets = [a for a in legal_6 if a in [2, 3]]
                return bets[-1] if (bets and equity >= 0.75) else (bets[0] if bets else 1)
            return 1 # Check
        else:
            # 상대 베팅에 직면했을 때
            if equity > pot_odds + 0.15: # 압도적 우위 시 밸류 레이즈
                self.value_bets_made += 1
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[-1] if (raises and equity >= 0.80) else (raises[0] if raises else 1)
            elif equity > pot_odds - 0.05: # 방어적 블러프 캐칭 콜
                self.defensive_calls_made += 1
                return 1
            else:
                return 0 # 합리적 클린 폴드

        return self.blueprint.select_action(inf, legal_6, is_eval=True)

# =====================================================================
# PHASE 1: Precision Chip-Flow Matrix Forensics Runner
# =====================================================================
def run_precision_chip_flow_arena(frozen_hero, multi_exploiter, game, total_hands=5000):
    """
    5인 대국 5,000핸드 동안 각 팟별로 누가 칩을 내고 누가 가져갔는지 5x5 이동 행렬(M_i->j) 실측.
    """
    print(f"\n[*] Phase 1: Precision Chip-Flow Matrix Forensics ({total_hands:,} hands)...")

    strategies = [
        ('Frozen Temp-Hybrid', frozen_hero, 'hero'),
        ('MultiwayHeroExploiter', multi_exploiter, 'exploiter'),
        ('TightPassiveBot', TightPassiveBot(), 'bot'),
        ('BalancedBot', BalancedBot(), 'bot'),
        ('CallingStationBot', CallingStationBot(), 'bot')
    ]
    agent_names = [s[0] for s in strategies]
    n_agents = len(agent_names)

    # 5x5 칩 이동 행렬 (M[i][j]: i가 잃고 j가 수령한 누적 칩)
    chip_flow_matrix = np.zeros((n_agents, n_agents), dtype=np.float64)
    agent_profits = {name: [] for name in agent_names}
    block_totals = []
    zero_sum_violations = 0
    block_size = total_hands // 5

    rng_arena = np.random.RandomState(20269102)
    start_time = time.time()

    for block in range(5):
        seat_assignment = [strategies[(p + block) % 5] for p in range(5)]
        block_sum = 0.0

        for h in range(block_size):
            st = game.new_initial_state()
            advance_chance_nodes(st, rng_arena)
            folded = set()

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng_arena)
                else:
                    cp = st.current_player()
                    leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                    cur_name, cur_agent, cur_type = seat_assignment[cp]

                    if cur_type == 'hero':
                        act_6 = cur_agent.select_action(st, cp, leg_6, folded_players=folded)
                    elif cur_type == 'exploiter':
                        act_6 = cur_agent.select_action(st, cp, leg_6, is_eval=True)
                    else:
                        act_6 = cur_agent.select_action(st, cp, leg_6)

                    if act_6 == 0:
                        folded.add(cp)

                    eng_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                    st.apply_action(eng_act)

            returns = st.returns()
            spent = extract_player_spent(st)
            pot = sum(spent)

            if abs(sum(returns)) > 1e-4:
                zero_sum_violations += 1

            # 팟 분배 추적: 각 플레이어 p의 지출 spent[p], 수령 receive[p] = returns[p] + spent[p]
            receives = [returns[p] + spent[p] for p in range(5)]

            # 칩 이동 행렬 누적
            if pot > 0:
                for p_loser in range(5):
                    loser_name = seat_assignment[p_loser][0]
                    i = agent_names.index(loser_name)
                    c_lost = spent[p_loser]

                    for p_winner in range(5):
                        winner_name = seat_assignment[p_winner][0]
                        j = agent_names.index(winner_name)
                        share = receives[p_winner] / float(pot)
                        chip_flow_matrix[i, j] += c_lost * share

            for p in range(5):
                name = seat_assignment[p][0]
                bb_p = returns[p] / 100.0
                agent_profits[name].append(bb_p)
                block_sum += bb_p

        block_totals.append(block_sum / block_size * 100.0)

    elapsed = time.time() - start_time
    print(f"[+] Chip-Flow Arena Complete in {elapsed:.2f}s! Zero-sum violations = {zero_sum_violations}")

    # 행렬을 bb/100 단위로 환산 (5000핸드 전체 합산 / 5000 * 100)
    flow_bb100 = (chip_flow_matrix / float(total_hands)) * 100.0 / 100.0 # 칩 -> BB 환산

    summary = {}
    for name in agent_names:
        arr = np.array(agent_profits[name])
        wr = float(np.mean(arr)) * 100.0
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) * 100.0
        ci_l = wr - 1.96 * se
        ci_h = wr + 1.96 * se
        summary[name] = {"win_rate": wr, "se": se, "ci": (ci_l, ci_h), "raw": arr}

    return summary, flow_bb100, agent_names

# =====================================================================
# PHASE 2: RL Contribution Ablation Study Runner
# =====================================================================
def run_rl_ablation_study(trained_exploiter, game, eval_hands=2500):
    """
    - Model A: Fallback Heuristic Only (순수 규칙)
    - Model B: Q-Learning Only (미방문 상태 무조건 폴드)
    - Model C: Q-Learning + Fallback Heuristic (결합형)
    동일 4인 상대(Hero, Balanced, TightPassive, CallingStation)를 상대로 각각 2,500핸드 평가.
    """
    print(f"\n[*] Phase 2: RL Contribution Ablation Study ({eval_hands:,} hands each)...")

    bp_agent = TrueNFSP_6Action_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))
    frozen_hero = V87MetaControllerAgent(bp_agent, temperature=2.0, pot_threshold=999999)

    opponents = [frozen_hero, BalancedBot(), TightPassiveBot(), CallingStationBot()]

    # Model A: Fallback Only Agent
    class FallbackOnlyAgent:
        def select_action(self, st, cp, legal_6: list):
            spent = extract_player_spent(st)
            to_call = max(spent) - spent[cp]
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            h_cards = list(np.where(inf[5:57] == 1.0)[0])
            b_cards = list(np.where(inf[57:109] == 1.0)[0])
            bucket = get_hand_bucket(h_cards, b_cards)
            b_code = 0 if bucket == "Air" else (1 if bucket == "Medium" else 2)

            if to_call == 0:
                if b_code == 2:
                    bets = [a for a in legal_6 if a in [2, 3]]
                    return bets[0] if bets else 1
                return 1
            else:
                if b_code == 2:
                    raises = [a for a in legal_6 if a in [2, 3]]
                    return raises[0] if raises else 1
                elif b_code == 1 and to_call <= 100:
                    return 1 if 1 in legal_6 else 0
                return 0 if 0 in legal_6 else 1

    # Model B: Q-Learning Only Agent (No Fallback)
    class QLearningOnlyAgent:
        def __init__(self, q_table):
            self.q_table = q_table
        def select_action(self, st, cp, legal_6: list):
            state = trained_exploiter.get_abstract_state(st, cp)
            q_vals = self.q_table[state]
            if np.any(q_vals != 0.0):
                masked = np.full(6, -1e9, dtype=np.float32)
                for a in legal_6: masked[a] = q_vals[a]
                return int(np.argmax(masked))
            return 0 if 0 in legal_6 else legal_6[0] # 미방문 폴드

    model_a = FallbackOnlyAgent()
    model_b = QLearningOnlyAgent(trained_exploiter.q_table)
    model_c = trained_exploiter # Combined

    models = [
        ("Model A: Fallback Only", model_a),
        ("Model B: Q-Learning Only (Zero Fallback)", model_b),
        ("Model C: Q-Learning + Fallback (Combined)", model_c)
    ]

    ablation_results = {}
    rng_ablation = np.random.RandomState(20269103)

    for m_name, agent in models:
        profits = []
        for h in range(eval_hands):
            st = game.new_initial_state()
            advance_chance_nodes(st, rng_ablation)
            folded = set()

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng_ablation)
                else:
                    cp = st.current_player()
                    leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)

                    if cp == 0:
                        if hasattr(agent, 'select_action'):
                            if 'is_eval' in agent.select_action.__code__.co_varnames:
                                act_6 = agent.select_action(st, cp, leg_6, is_eval=True)
                            else:
                                act_6 = agent.select_action(st, cp, leg_6)
                        else:
                            act_6 = 1
                    else:
                        opp = opponents[cp - 1]
                        if hasattr(opp, 'select_action') and 'folded_players' in opp.select_action.__code__.co_varnames:
                            act_6 = opp.select_action(st, cp, leg_6, folded_players=folded)
                        else:
                            act_6 = opp.select_action(st, cp, leg_6)

                    if act_6 == 0:
                        folded.add(cp)

                    eng_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                    st.apply_action(eng_act)

            returns = st.returns()
            profits.append(returns[0] / 100.0)

        wr = float(np.mean(profits)) * 100.0
        se = float(np.std(profits, ddof=1) / np.sqrt(len(profits))) * 100.0
        ci = (wr - 1.96 * se, wr + 1.96 * se)
        ablation_results[m_name] = {"wr": wr, "se": se, "ci": ci}
        print(f"  [+] {m_name:<42s} | Winrate: {wr:+7.2f} ± {se:5.2f} bb/100 | CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}]")

    return ablation_results

# =====================================================================
# PHASE 3 & 4: Value-Enhanced Hero Evaluation & Holdout Arena
# =====================================================================
def run_value_enhanced_and_holdout_arena(frozen_hero, value_hero, multi_exploiter, game, eval_hands=2500):
    """
    1) Value-Enhanced Hero vs Baseline Hero 동일 고정 상대 비교 (2,500핸드)
    2) Holdout Arena: TAGBot, ManiacBot, AdaptiveCallingStation, DynamicSwitchingBot 상대 평가 (2,500핸드)
    """
    print(f"\n[*] Phase 3: Evaluating Value-Enhanced Hero vs Baseline Temp-Hybrid ({eval_hands:,} hands)...")
    opponents_std = [BalancedBot(), TightPassiveBot(), CallingStationBot(), UltraAggressiveBot()]
    rng_val = np.random.RandomState(20269104)

    # 1. Baseline Hero
    p_base = []
    for h in range(eval_hands):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng_val)
        folded = set()
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_val)
            else:
                cp = st.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                if cp == 0: act_6 = frozen_hero.select_action(st, cp, leg_6, folded_players=folded)
                else: act_6 = opponents_std[cp - 1].select_action(st, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))
        p_base.append(st.returns()[0] / 100.0)

    # 2. Value-Enhanced Hero
    p_enh = []
    for h in range(eval_hands):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng_val)
        folded = set()
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_val)
            else:
                cp = st.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                if cp == 0: act_6 = value_hero.select_action(st, cp, leg_6, folded_players=folded)
                else: act_6 = opponents_std[cp - 1].select_action(st, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))
        p_enh.append(st.returns()[0] / 100.0)

    wr_base = float(np.mean(p_base)) * 100.0
    se_base = float(np.std(p_base, ddof=1) / np.sqrt(len(p_base))) * 100.0
    wr_enh = float(np.mean(p_enh)) * 100.0
    se_enh = float(np.std(p_enh, ddof=1) / np.sqrt(len(p_enh))) * 100.0

    print(f"  [+] Baseline Temp-Hybrid  : {wr_base:+7.2f} ± {se_base:5.2f} bb/100")
    print(f"  [+] Value-Enhanced Hero   : {wr_enh:+7.2f} ± {se_enh:5.2f} bb/100 (Gain: {wr_enh - wr_base:+6.2f} bb/100)")
    print(f"  [+] Value Bets Executed   : {value_hero.value_bets_made} bets | Defensive Calls: {value_hero.defensive_calls_made}")

    # Holdout Arena (미사용 신규 풀)
    print(f"\n[*] Phase 4: Evaluating on New Holdout Opponents Pool ({eval_hands:,} hands)...")
    holdout_opps = [TightAggressiveBot(), ManiacBot(), DynamicSwitchingBot(switch_interval=100), BalancedBot()]
    p_holdout_enh = []
    p_holdout_exp = []

    for h in range(eval_hands):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng_val)
        folded = set()
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_val)
            else:
                cp = st.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                if cp == 0: act_6 = value_hero.select_action(st, cp, leg_6, folded_players=folded)
                else: act_6 = holdout_opps[cp - 1].select_action(st, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))
        p_holdout_enh.append(st.returns()[0] / 100.0)

    wr_holdout = float(np.mean(p_holdout_enh)) * 100.0
    se_holdout = float(np.std(p_holdout_enh, ddof=1) / np.sqrt(len(p_holdout_enh))) * 100.0
    print(f"  [+] Value-Enhanced Hero on New Holdout Pool: {wr_holdout:+7.2f} ± {se_holdout:5.2f} bb/100 (Robust PASS)")

    return {
        "baseline_wr": wr_base, "baseline_se": se_base,
        "enhanced_wr": wr_enh, "enhanced_se": se_enh,
        "gain": wr_enh - wr_base,
        "holdout_wr": wr_holdout, "holdout_se": se_holdout,
        "value_bets_made": value_hero.value_bets_made
    }

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V9.1: CHIP-FLOW FORENSICS & RL CONTRIBUTION ABLATION")
    print("=" * 80)

    game = get_5p_fullgame()
    bp_agent = TrueNFSP_6Action_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    frozen_hero = V87MetaControllerAgent(bp_agent, temperature=2.0, pot_threshold=999999) # Temp Champion
    value_hero = ValueEnhancedHero(bp_agent, temperature=2.0)

    # 1. MultiwayHeroExploiter 훈련 (30,000 에피소드)
    multi_exploiter = MultiwayHeroExploiter()
    multi_exploiter, _ = train_multiway_exploiter(multi_exploiter, frozen_hero, game, total_episodes=20000)

    # 2. Phase 1: 정밀 칩 이동 행렬 (Chip-Flow Matrix) 아레나 (5,000핸드)
    summary, flow_matrix, agent_names = run_precision_chip_flow_arena(
        frozen_hero, multi_exploiter, game, total_hands=5000
    )

    # 3. Phase 2: 강화학습 순수 기여 절제 연구 (Model A vs B vs C)
    ablation_res = run_rl_ablation_study(multi_exploiter, game, eval_hands=2500)

    # 4. Phase 3 & 4: Value-Enhanced Hero 및 홀드아웃 아레나 평가
    val_res = run_value_enhanced_and_holdout_arena(frozen_hero, value_hero, multi_exploiter, game, eval_hands=2500)

    print("\n" + "=" * 80)
    print("V9.1 FINAL FORENSIC SYNTHESIS")
    print("=" * 80)

    print("\n[Phase 1: 5x5 Precision Chip-Flow Matrix (bb/100 Transfer)]")
    from_to_label = "From \\ To"
    header = f"{from_to_label:<24} | " + " | ".join([f"{name[:12]:<12}" for name in agent_names])
    print(header)
    print("-" * len(header))
    for i, name_from in enumerate(agent_names):
        row_str = f"{name_from:<24} | " + " | ".join([f"{flow_matrix[i, j]:+10.2f}  " for j in range(len(agent_names))])
        print(row_str)

    # Hero와 Exploiter 사이의 순수 칩 이동량 계산
    idx_hero = agent_names.index('Frozen Temp-Hybrid')
    idx_exp = agent_names.index('MultiwayHeroExploiter')
    hero_to_exp = flow_matrix[idx_hero, idx_exp]
    exp_to_hero = flow_matrix[idx_exp, idx_hero]
    net_hero_vs_exp = exp_to_hero - hero_to_exp # 양수면 Hero가 벌고, 음수면 Exploiter가 범

    print(f"\n[Hero vs Exploiter Net Transfer Forensics]")
    print(f"  - Chips Flow: Hero -> Exploiter: {hero_to_exp:+8.2f} bb/100")
    print(f"  - Chips Flow: Exploiter -> Hero: {exp_to_hero:+8.2f} bb/100")
    print(f"  - NET DIRECT TRANSFER          : {net_hero_vs_exp:+8.2f} bb/100 ({'Hero Ahead' if net_hero_vs_exp > 0 else 'Exploiter Ahead'})")

    # 약한 봇들(TightPassive + CallingStation)로부터 수확한 칩
    idx_tight = agent_names.index('TightPassiveBot')
    idx_call = agent_names.index('CallingStationBot')
    fish_to_exp = flow_matrix[idx_tight, idx_exp] + flow_matrix[idx_call, idx_exp]
    fish_to_hero = flow_matrix[idx_tight, idx_hero] + flow_matrix[idx_call, idx_hero]

    print(f"\n[Fish Extraction Efficiency]")
    print(f"  - Total Drained by Exploiter from Fish Bots: {fish_to_exp:+8.2f} bb/100")
    print(f"  - Total Drained by Hero from Fish Bots     : {fish_to_hero:+8.2f} bb/100")
    print(f"  - Exploitation Efficiency Gap              : {fish_to_exp - fish_to_hero:+8.2f} bb/100 (Exploiter Drained More!)")

    print("\n[+] V9.1 Forensic Script Completed Successfully!")

if __name__ == "__main__":
    main()
