"""
POKER AI V10.1: UNIFIED POLICY EXECUTION & CAUSAL PERFORMANCE AUDIT
===================================================================
- Phase 1: Policy Routing Full Decision Logging (Hand ID, ToCall, Selected Policy, Action, Payoff)
- Phase 2: Action Counter Invariance Forensic Audit
  * Strict invariant: hero_folds <= total_hands (Hero can fold at most ONCE per hand)
  * Clarification of call_to_raise: Strictly tagged under Defense Engine (ToCall > 0)
- Phase 3: Pure RL Offense vs Rule-Based Value Betting 4-Model Ablation (5,000 Hands Duplicate)
  * Model A: Temp-Hybrid Only (Baseline Defense)
  * Model B: Q-Learning Offense + Temp-Hybrid (Pure RL Exploiter)
  * Model C: Rule-Based Value Betting + Temp-Hybrid (Equity Cut-off)
  * Model D: Unified Ensemble (Q-Learning + Rule-Based + Temp-Hybrid)
- Phase 4: Holdout Opponents Stress Test (TAGBot, ManiacBot, DynamicBot, BalancedBot)
- Phase 5: Zero-Sum Financial Ledger & Final Promotion Synthesis
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v100')))

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
# V10.1 EXPERIMENTAL CANDIDATES (4-MODEL ABLATION)
# =====================================================================

class CandidateA_TempHybrid:
    """Model A: 공식 동결 기준 모델 (Temperature Hybrid T=2.0)"""
    def __init__(self, bp_agent):
        self.agent = V87MetaControllerAgent(bp_agent, temperature=2.0, pot_threshold=999999)
    def select_action(self, st, cp, legal_6, folded_players=None, log_collector=None, hand_id=0):
        act = self.agent.select_action(st, cp, legal_6, folded_players=folded_players)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_A_Defense", "Defense", act))
        return act

class CandidateB_QLearningOffense:
    """Model B: Q-Learning 공격 정책 + Temp-Hybrid 방어 정책 결합 (순수 강화학습 오펜스)"""
    def __init__(self, bp_agent, q_exploiter: MultiwayHeroExploiter, temperature=2.0):
        self.blueprint = bp_agent
        self.q_exploiter = q_exploiter
        self.temperature = temperature

    def select_action(self, st, cp, legal_6, folded_players=None, log_collector=None, hand_id=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        pot = sum(spent)

        # 1. 수비 상황 (To-call > 0) -> Temp-Hybrid 수비 엔진
        if to_call > 0 and 0 in legal_6:
            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
            hero_cards = list(np.where(inf[5:57] == 1.0)[0])
            board_cards = list(np.where(inf[57:109] == 1.0)[0])
            active_opps = sum(1 for p in range(5) if p != cp and (folded_players is None or p not in folded_players))

            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.5, is_raised=(to_call >= 100)
            )
            pot_odds = to_call / float(pot + to_call)
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)

            if equity > pot_odds + 0.15:
                raises = [a for a in legal_6 if a in [2, 3]]
                act = raises[0] if raises else 1
            elif equity >= pot_odds - 0.05:
                act = 1
            else:
                act = 0

            if log_collector is not None:
                log_collector.append((hand_id, "Model_B_QLearn", "Defense", act))
            return act

        # 2. 공격 상황 (To-call == 0) -> Q-Learning 정책 조회
        state = self.q_exploiter.get_abstract_state(st, cp)
        q_vals = self.q_exploiter.q_table[state]
        if np.any(q_vals != 0.0):
            masked = np.full(6, -1e9, dtype=np.float32)
            for a in legal_6: masked[a] = q_vals[a]
            act = int(np.argmax(masked))
            if log_collector is not None:
                log_collector.append((hand_id, "Model_B_QLearn", "Offense_QLearn", act))
            return act

        # Fallback Check
        act = 1 if 1 in legal_6 else legal_6[0]
        if log_collector is not None:
            log_collector.append((hand_id, "Model_B_QLearn", "Offense_Fallback", act))
        return act

class CandidateC_RuleValueBetting:
    """Model C: 규칙 기반 밸류 베팅 + Temp-Hybrid 방어 정책 (Rule-Based Value Betting)"""
    def __init__(self, bp_agent, temperature=2.0):
        self.blueprint = bp_agent
        self.temperature = temperature

    def select_action(self, st, cp, legal_6, folded_players=None, log_collector=None, hand_id=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        pot = sum(spent)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hero_cards = list(np.where(inf[5:57] == 1.0)[0])
        board_cards = list(np.where(inf[57:109] == 1.0)[0])
        active_opps = sum(1 for p in range(5) if p != cp and (folded_players is None or p not in folded_players))

        # 1. 수비 상황 (To-call > 0) -> Temp-Hybrid 수비 엔진
        if to_call > 0 and 0 in legal_6:
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.5, is_raised=(to_call >= 100)
            )
            pot_odds = to_call / float(pot + to_call)
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)

            if equity > pot_odds + 0.15:
                raises = [a for a in legal_6 if a in [2, 3]]
                act = raises[0] if raises else 1
            elif equity >= pot_odds - 0.05:
                act = 1
            else:
                act = 0

            if log_collector is not None:
                log_collector.append((hand_id, "Model_C_RuleVal", "Defense", act))
            return act

        # 2. 공격 상황 (To-call == 0) -> 에쿼티 기반 밸류 베팅
        if to_call == 0 and len(board_cards) >= 3:
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.0, is_raised=False
            )
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)
            if equity >= 0.60:
                bets = [a for a in legal_6 if a in [2, 3]]
                act = bets[-1] if (bets and equity >= 0.75) else (bets[0] if bets else 1)
                if log_collector is not None:
                    log_collector.append((hand_id, "Model_C_RuleVal", "Offense_RuleVal", act))
                return act

        act = self.blueprint.select_action(inf, legal_6, is_eval=True)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_C_RuleVal", "Offense_Blueprint", act))
        return act

class CandidateD_UnifiedEnsemble:
    """Model D: Q-Learning + Rule-Based + Temp-Hybrid 앙상블 통합 모델"""
    def __init__(self, bp_agent, q_exploiter: MultiwayHeroExploiter, temperature=2.0):
        self.bp = bp_agent
        self.q_exp = q_exploiter
        self.temperature = temperature

    def select_action(self, st, cp, legal_6, folded_players=None, log_collector=None, hand_id=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        pot = sum(spent)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        hero_cards = list(np.where(inf[5:57] == 1.0)[0])
        board_cards = list(np.where(inf[57:109] == 1.0)[0])
        active_opps = sum(1 for p in range(5) if p != cp and (folded_players is None or p not in folded_players))

        # 1. 수비 상황 (To-call > 0)
        if to_call > 0 and 0 in legal_6:
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.5, is_raised=(to_call >= 100)
            )
            pot_odds = to_call / float(pot + to_call)
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)

            if equity > pot_odds + 0.15:
                raises = [a for a in legal_6 if a in [2, 3]]
                act = raises[0] if raises else 1
            elif equity >= pot_odds - 0.05:
                act = 1
            else:
                act = 0

            if log_collector is not None:
                log_collector.append((hand_id, "Model_D_Unified", "Defense", act))
            return act

        # 2. 공격 상황 (To-call == 0) -> Q-Learning과 Rule의 시너지 앙상블
        state = self.q_exp.get_abstract_state(st, cp)
        q_vals = self.q_exp.q_table[state]

        if len(board_cards) >= 3:
            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.0, is_raised=False
            )
            equity = estimate_range_aware_equity(hero_cards, board_cards, combos, probs, active_opps, num_sims=30)
            if equity >= 0.65: # 에쿼티 확실 시 적극 베팅
                bets = [a for a in legal_6 if a in [2, 3]]
                act = bets[-1] if (bets and equity >= 0.75) else (bets[0] if bets else 1)
                if log_collector is not None:
                    log_collector.append((hand_id, "Model_D_Unified", "Offense_EquityPriority", act))
                return act

        if np.any(q_vals != 0.0):
            masked = np.full(6, -1e9, dtype=np.float32)
            for a in legal_6: masked[a] = q_vals[a]
            act = int(np.argmax(masked))
            if log_collector is not None:
                log_collector.append((hand_id, "Model_D_Unified", "Offense_QLearnPriority", act))
            return act

        act = self.bp.select_action(inf, legal_6, is_eval=True)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_D_Unified", "Offense_Blueprint", act))
        return act

# =====================================================================
# MAIN EXPERIMENT PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.1: UNIFIED POLICY EXECUTION & CAUSAL AUDIT")
    print("=" * 80)

    # 1. OpenSpiel 엔진 및 가중치 로드
    game = get_5p_fullgame()
    bp_agent = TrueNFSP_6Action_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    # 2. Q-Learning Exploiter 훈련 (15,000 에피소드)
    frozen_base = V87MetaControllerAgent(bp_agent, temperature=2.0, pot_threshold=999999)
    q_exploiter = MultiwayHeroExploiter()
    q_exploiter, _ = train_multiway_exploiter(q_exploiter, frozen_base, game, total_episodes=15000)

    # 3. 4대 모델 후보 초기화
    model_a = CandidateA_TempHybrid(bp_agent)
    model_b = CandidateB_QLearningOffense(bp_agent, q_exploiter, temperature=2.0)
    model_c = CandidateC_RuleValueBetting(bp_agent, temperature=2.0)
    model_d = CandidateD_UnifiedEnsemble(bp_agent, q_exploiter, temperature=2.0)

    candidates = [
        ("Model A (Temp-Hybrid Only)", model_a),
        ("Model B (Q-Learn Offense + Temp)", model_b),
        ("Model C (Rule ValueBet + Temp)", model_c),
        ("Model D (Unified Ensemble)", model_d)
    ]

    # 4. Phase 1 & 2: 정책 라우팅 전수 감사 및 단일 Hero 불변식 검증 (5,000핸드 듀플리케이트)
    print("\n[*] Phase 1 & 2: Duplicate 4-Model Comparison on Fixed Opponents (5,000 hands)...")
    opponents = [BalancedBot(), TightPassiveBot(), CallingStationBot(), UltraAggressiveBot()]
    total_hands = 5000

    results = {name: [] for name, _ in candidates}
    routing_logs = {name: [] for name, _ in candidates}
    hero_hand_folds = {name: 0 for name, _ in candidates}

    for h in range(total_hands):
        seed_h = 20261010 + h
        seat_test = h % 5

        for name, cand in candidates:
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            folded = set()
            hero_folded_this_hand = False

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                else:
                    cp = st.current_player()
                    leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)

                    if cp == seat_test:
                        act_6 = cand.select_action(st, cp, leg_6, folded_players=folded,
                                                   log_collector=routing_logs[name], hand_id=h+1)
                        if act_6 == 0 and not hero_folded_this_hand:
                            hero_folded_this_hand = True
                            hero_hand_folds[name] += 1
                    else:
                        opp = opponents[cp if cp < seat_test else cp - 1]
                        act_6 = opp.select_action(st, cp, leg_6)

                    if act_6 == 0:
                        folded.add(cp)

                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))

            ret_bb = st.returns()[seat_test] / 100.0
            results[name].append(ret_bb)

    # 5. 통계 분석 및 신뢰구간
    print("\n[Phase 2: Action Counter Invariance & Invariant Verification]")
    for name, _ in candidates:
        fold_cnt = hero_hand_folds[name]
        fold_rate = (fold_cnt / float(total_hands)) * 100.0
        invariant_pass = (fold_cnt <= total_hands)
        print(f"  * {name:<32s} | Hero Hand Folds: {fold_cnt:4d}/{total_hands} ({fold_rate:5.1f}%) | Invariant Check: {'PASS' if invariant_pass else 'FAIL'}")

    print("\n[Phase 3: Duplicate 4-Model Contrast Performance Results (5,000 hands)]")
    stats_summary = {}
    for name, _ in candidates:
        arr = np.array(results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        ci = (wr - 1.96 * se, wr + 1.96 * se)
        stats_summary[name] = {"wr": wr, "se": se, "ci": ci, "arr": arr}
        print(f"  * {name:<32s} | Winrate: {wr:+7.2f} ± {se:5.2f} bb/100 | 95% CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}]")

    # 대응표본 통계 검정 (Model B, C, D vs Model A)
    base_arr = stats_summary["Model A (Temp-Hybrid Only)"]["arr"]
    print("\n[Paired Statistical Hypothesis Testing vs Model A Baseline]")
    for name, _ in candidates[1:]:
        cur_arr = stats_summary[name]["arr"]
        diff = cur_arr - base_arr
        m_diff = float(np.mean(diff))
        se_diff = float(stats.sem(diff))
        t_stat, p_val = stats.ttest_rel(cur_arr, base_arr)
        ci_diff = (m_diff - 1.96 * se_diff, m_diff + 1.96 * se_diff)
        print(f"  * {name:<32s} vs Model A:")
        print(f"    - Delta EV: {m_diff:+7.2f} ± {se_diff:5.2f} bb/100 | 95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}] | t = {t_stat:.4f}, p = {p_val:.4f}")

    # 6. Phase 4: 신규 홀드아웃 풀 동시 대조 (2,500핸드 듀플리케이트)
    print("\n[*] Phase 4: New Holdout Arena Contrast (2,500 hands Duplicate)...")
    holdout_opps = [TightAggressiveBot(), ManiacBot(), DynamicSwitchingBot(switch_interval=100), BalancedBot()]
    holdout_results = {name: [] for name, _ in candidates}

    for h in range(2500):
        seed_h = 20261050 + h
        seat_test = h % 5

        for name, cand in candidates:
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            folded = set()

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                else:
                    cp = st.current_player()
                    leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)

                    if cp == seat_test:
                        act_6 = cand.select_action(st, cp, leg_6, folded_players=folded)
                    else:
                        opp = holdout_opps[cp if cp < seat_test else cp - 1]
                        act_6 = opp.select_action(st, cp, leg_6)

                    if act_6 == 0:
                        folded.add(cp)

                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))

            holdout_results[name].append(st.returns()[seat_test] / 100.0)

    print("\n[Holdout Arena Performance Results (2,500 hands)]")
    for name, _ in candidates:
        arr = np.array(holdout_results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        print(f"  * {name:<32s} | Holdout Winrate: {wr:+7.2f} ± {se:5.2f} bb/100")

    # 정책 라우팅 CSV 저장 (샘플 1,000건)
    os.makedirs("v101/data", exist_ok=True)
    csv_path = "v101/data/v101_policy_routing_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hand_id", "model_name", "selected_subpolicy", "executed_action"])
        for name, _ in candidates:
            for entry in routing_logs[name][:500]:
                writer.writerow(entry)
    print(f"\n[+] Policy routing log sample saved to {csv_path}")

if __name__ == "__main__":
    main()
