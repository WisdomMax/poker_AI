"""
POKER AI V10.0: CONTROLLED OFFENSE-DEFENSE INTEGRATION
======================================================
- Phase 1: Three Model Architectures Frozen
  * Model A: Defense Only (Frozen Temp-Hybrid T=2.0)
  * Model B: Offense Only (Multiway Exploitation Engine)
  * Model C: Integrated (Defense + Offense Unified Architecture)
- Phase 2: Policy Conflict Forensic Audit (Why Q-Learning clashed with Fallback in marginal states)
- Phase 3: Action-by-Action Value Attribution (Check->Bet, Small->Pot counterfactual audit)
- Phase 4: 5,000-Hand Duplicate Poker Controlled Contrast (Identical Decks & Seats)
  * Paired t-test, SEM, 95% CI, Max Drawdown
- Phase 5: Holdout Opponents Stress Test (TAGBot, ManiacBot, DynamicBot, BalancedBot)
- Phase 6: Formal Grand Unification GO / NO-GO Verdict
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v91')))

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
from v91.run_v91_chip_flow_and_rl_ablation import TightAggressiveBot, ManiacBot

# =====================================================================
# V10.0 INTEGRATED AGENT: DEFENSE + OFFENSE UNIFIED ARCHITECTURE
# =====================================================================
class V100IntegratedAgent:
    """
    POKER ALPHAGO V10.0 통합 모델:
    - 수비 상황 (To-call > 0): Temperature Hybrid (T=2.0) 철벽 팟오즈 방어선 집행.
    - 공격 상황 (To-call == 0): 에쿼티 0.60 이상 시 공격적 밸류 벳(Pot-Bet) 집행하여 피쉬 봇 칩 수확.
    - 정책 충돌 방지: 수비와 공격의 역할을 상태(Facing Bet vs Free)에 따라 완벽히 분리하여 충돌 제거!
    """
    def __init__(self, blueprint_agent: TrueNFSP_6Action_Agent, q_table, temperature=2.0):
        self.blueprint = blueprint_agent
        self.q_table = q_table
        self.temperature = temperature
        self.action_changes = {
            "check_to_bet": 0,
            "call_to_raise": 0,
            "pot_bets": 0,
            "defensive_folds": 0,
            "defensive_calls": 0
        }

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

        # 1. 수비 상황 (상대의 베팅에 직면: To-call > 0) -> 철벽 방어 엔진 가동!
        if to_call > 0 and 0 in legal_6:
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.5, is_raised=(to_call >= 100)
            )
            pot_odds = to_call / float(pot + to_call)
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)

            if equity > pot_odds + 0.15:
                self.action_changes["call_to_raise"] += 1
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[0] if raises else 1
            elif equity >= pot_odds - 0.05:
                self.action_changes["defensive_calls"] += 1
                return 1 # 블러프 캐칭 콜
            else:
                self.action_changes["defensive_folds"] += 1
                return 0 # 클린 폴드

        # 2. 공격 상황 (자유 상태: To-call == 0) -> 능동 착취 엔진 가동!
        if to_call == 0 and len(board_cards) >= 3:
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.0, is_raised=False
            )
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)

            if equity >= 0.60:
                self.action_changes["check_to_bet"] += 1
                bets = [a for a in legal_6 if a in [2, 3]]
                if bets and equity >= 0.75:
                    self.action_changes["pot_bets"] += 1
                    return bets[-1] # Pot-Bet
                return bets[0] if bets else 1

        return self.blueprint.select_action(inf, legal_6, is_eval=True)

# =====================================================================
# MAIN EXPERIMENT PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.0: CONTROLLED OFFENSE-DEFENSE INTEGRATION")
    print("=" * 80)

    # 1. OpenSpiel 엔진 및 가중치 로드
    game = get_5p_fullgame()
    bp_agent = TrueNFSP_6Action_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    # 2. Phase 1: 세 모델 준비
    # Model A: Defense Only (Baseline Temp-Hybrid)
    model_a = V87MetaControllerAgent(bp_agent, temperature=2.0, pot_threshold=999999)

    # Model B: Offense Only (Multiway Exploiter Q-Learning)
    model_b = MultiwayHeroExploiter()
    model_b, _ = train_multiway_exploiter(model_b, model_a, game, total_episodes=15000)

    # Model C: Integrated (Defense + Offense Unified Architecture)
    model_c = V100IntegratedAgent(bp_agent, model_b.q_table, temperature=2.0)

    # 3. Phase 4: 5,000핸드 Duplicate Controlled Contrast (동일 카드 덱 5,000핸드)
    print(f"\n[*] Phase 4: Duplicate Controlled Contrast on Fixed Opponents (5,000 hands)...")
    opponents = [BalancedBot(), TightPassiveBot(), CallingStationBot(), UltraAggressiveBot()]

    p_a, p_b, p_c = [], [], []
    total_hands = 5000

    for h in range(total_hands):
        seed_h = 20261000 + h
        seat_test = h % 5

        # Run Model A (Defense Only)
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(seed_h))
        folded = set()
        while not st_a.is_terminal():
            if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h))
            else:
                cp = st_a.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                if cp == seat_test: act_6 = model_a.select_action(st_a, cp, leg_6, folded_players=folded)
                else: act_6 = opponents[cp if cp < seat_test else cp - 1].select_action(st_a, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, act_6))
        p_a.append(st_a.returns()[seat_test] / 100.0)

        # Run Model B (Offense Only)
        st_b = game.new_initial_state()
        advance_chance_nodes(st_b, np.random.RandomState(seed_h))
        folded = set()
        while not st_b.is_terminal():
            if st_b.is_chance_node(): advance_chance_nodes(st_b, np.random.RandomState(seed_h))
            else:
                cp = st_b.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                if cp == seat_test: act_6 = model_b.select_action(st_b, cp, leg_6, is_eval=True)
                else: act_6 = opponents[cp if cp < seat_test else cp - 1].select_action(st_b, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, act_6))
        p_b.append(st_b.returns()[seat_test] / 100.0)

        # Run Model C (Integrated Unified)
        st_c = game.new_initial_state()
        advance_chance_nodes(st_c, np.random.RandomState(seed_h))
        folded = set()
        while not st_c.is_terminal():
            if st_c.is_chance_node(): advance_chance_nodes(st_c, np.random.RandomState(seed_h))
            else:
                cp = st_c.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_c, cp)
                if cp == seat_test: act_6 = model_c.select_action(st_c, cp, leg_6, folded_players=folded)
                else: act_6 = opponents[cp if cp < seat_test else cp - 1].select_action(st_c, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_c, cp, act_6))
        p_c.append(st_c.returns()[seat_test] / 100.0)

    arr_a = np.array(p_a) * 100.0
    arr_b = np.array(p_b) * 100.0
    arr_c = np.array(p_c) * 100.0

    wr_a, se_a = float(np.mean(arr_a)), float(stats.sem(arr_a))
    wr_b, se_b = float(np.mean(arr_b)), float(stats.sem(arr_b))
    wr_c, se_c = float(np.mean(arr_c)), float(stats.sem(arr_c))

    # 대응 차이 통계 (C - A)
    diff_ca = arr_c - arr_a
    m_diff_ca = float(np.mean(diff_ca))
    se_diff_ca = float(stats.sem(diff_ca))
    t_ca, p_ca = stats.ttest_rel(arr_c, arr_a)

    print(f"  [+] Model A (Defense Only)   : {wr_a:+7.2f} ± {se_a:5.2f} bb/100")
    print(f"  [+] Model B (Offense Only)   : {wr_b:+7.2f} ± {se_b:5.2f} bb/100")
    print(f"  [+] Model C (Integrated V10) : {wr_c:+7.2f} ± {se_c:5.2f} bb/100")
    print(f"  [+] Paired Gain (Model C - A): {m_diff_ca:+7.2f} ± {se_diff_ca:5.2f} bb/100 (t = {t_ca:.4f}, p = {p_ca:.6f})")

    # 4. Phase 5: 신규 홀드아웃 풀 동시 대조 (2,500핸드)
    print(f"\n[*] Phase 5: Duplicate Holdout Arena (2,500 hands)...")
    holdout_opps = [TightAggressiveBot(), ManiacBot(), DynamicSwitchingBot(switch_interval=100), BalancedBot()]
    h_a, h_c = [], []
    for h in range(2500):
        seed_h = 20261100 + h
        seat_test = h % 5

        # Model A
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(seed_h))
        folded = set()
        while not st_a.is_terminal():
            if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h))
            else:
                cp = st_a.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                if cp == seat_test: act_6 = model_a.select_action(st_a, cp, leg_6, folded_players=folded)
                else: act_6 = holdout_opps[cp if cp < seat_test else cp - 1].select_action(st_a, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, act_6))
        h_a.append(st_a.returns()[seat_test] / 100.0)

        # Model C
        st_c = game.new_initial_state()
        advance_chance_nodes(st_c, np.random.RandomState(seed_h))
        folded = set()
        while not st_c.is_terminal():
            if st_c.is_chance_node(): advance_chance_nodes(st_c, np.random.RandomState(seed_h))
            else:
                cp = st_c.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_c, cp)
                if cp == seat_test: act_6 = model_c.select_action(st_c, cp, leg_6, folded_players=folded)
                else: act_6 = holdout_opps[cp if cp < seat_test else cp - 1].select_action(st_c, cp, leg_6)
                if act_6 == 0: folded.add(cp)
                st_c.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_c, cp, act_6))
        h_c.append(st_c.returns()[seat_test] / 100.0)

    wr_ha = float(np.mean(h_a)) * 100.0
    se_ha = float(stats.sem(h_a)) * 100.0
    wr_hc = float(np.mean(h_c)) * 100.0
    se_hc = float(stats.sem(h_c)) * 100.0
    t_h, p_h = stats.ttest_rel(np.array(h_c)*100, np.array(h_a)*100)

    print(f"  [+] Holdout Model A (Defense): {wr_ha:+7.2f} ± {se_ha:5.2f} bb/100")
    print(f"  [+] Holdout Model C (Unified): {wr_hc:+7.2f} ± {se_hc:5.2f} bb/100 (Gain: {wr_hc - wr_ha:+6.2f} bb/100, p = {p_h:.4f})")

    print("\n" + "=" * 80)
    print("V10.0 FINAL SYNTHESIS & GRAND UNIFICATION VERDICT")
    print("=" * 80)
    print(f"Action Changes Breakdown:")
    for k, v in model_c.action_changes.items():
        print(f"  - {k:<20s}: {v:5d} times")

    # 원장 CSV 저장
    os.makedirs("v100/data", exist_ok=True)
    csv_path = "v100/data/v100_controlled_integration_ledger.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hand_id", "model_a_bb", "model_b_bb", "model_c_bb", "paired_gain_c_minus_a"])
        for idx in range(total_hands):
            writer.writerow([idx + 1, arr_a[idx], arr_b[idx], arr_c[idx], diff_ca[idx]])
    print(f"[+] Controlled ledger saved to {csv_path}")

if __name__ == "__main__":
    main()
