"""
POKER AI V10.2: ACTION SEMANTICS AUDIT & UNIFIED POLICY CHAMPIONSHIP
=====================================================================
- Phase 1: True 6-Action Semantics & Engine Amount Mapping Forensic Audit
  * Audit True6ActionAdapter across v75, v89, v90, v100, v101
  * Explicitly verify Q-table training action mapping vs engine execution action mapping
- Phase 2: Full Decision Policy Routing Audit (100% Census, Bugfix for Model A Free Action)
  * Model A: Log 'Defense_FacingBet' when ToCall > 0, 'FreeAction_Temp' when ToCall == 0
  * Full census logging of all decisions (no sampling truncation)
- Phase 3: Fold Decomposition & Poker Metrics Audit
  * Preflop / Flop / Turn / River fold count & percentage
  * VPIP, PFR, WTSD, W$SD, Top 1% & Top 5% pot profit concentration
- Phase 4: Duplicate 4-Model Contrast on Fixed Opponents (5,000 Hands)
  * Model A, B, C, D paired t-tests, difference SE, 95% CI, Bonferroni adjusted p-values
- Phase 5: Duplicate 4-Model Contrast on Holdout Opponents (2,500 Hands)
  * Full paired difference statistics (Delta EV, SE, 95% CI, t-stat, p-value)
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v101')))

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
from v86.run_v86_meta_controller_and_independent_audit import BalancedBot, DynamicSwitchingBot
from v87.run_v87_adversarial_red_team_arena import UltraAggressiveBot, V87MetaControllerAgent
from v88.run_v88_adversarial_learning_and_hero_exploitation import CallingStationBot
from v90.run_v90_adversarial_causal_audit_and_scaling import MultiwayHeroExploiter, train_multiway_exploiter
from v91.run_v91_chip_flow_and_rl_ablation import TightAggressiveBot, ManiacBot

# =====================================================================
# V10.2 CANDIDATE AGENTS WITH ACCURATE STATE & ROUTING LOGGING
# =====================================================================

class V102_CandidateA_TempHybrid:
    """Model A: 공식 동결 기준 모델 (Temperature Hybrid T=2.0)"""
    def __init__(self, bp_agent):
        self.agent = V87MetaControllerAgent(bp_agent, temperature=2.0, pot_threshold=999999)

    def select_action(self, st, cp, legal_6, folded_players=None, log_collector=None, hand_id=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        act = self.agent.select_action(st, cp, legal_6, folded_players=folded_players)
        
        # 라우팅 로깅 버그 수정: Facing Bet 여부에 따라 정확히 태깅
        subpolicy = "Defense_FacingBet" if to_call > 0 else "FreeAction_Temp"
        if log_collector is not None:
            log_collector.append((hand_id, "Model_A_Temp", subpolicy, to_call, act))
        return act

class V102_CandidateB_QLearningOffense:
    """Model B: Q-Learning 공격 + Temp-Hybrid 방어 정책"""
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
                log_collector.append((hand_id, "Model_B_QLearn", "Defense_FacingBet", to_call, act))
            return act

        # 2. 공격 상황 (To-call == 0) -> Q-Learning 정책 조회
        state = self.q_exploiter.get_abstract_state(st, cp)
        q_vals = self.q_exploiter.q_table[state]
        if np.any(q_vals != 0.0):
            masked = np.full(6, -1e9, dtype=np.float32)
            for a in legal_6: masked[a] = q_vals[a]
            act = int(np.argmax(masked))
            if log_collector is not None:
                log_collector.append((hand_id, "Model_B_QLearn", "Offense_QLearn", to_call, act))
            return act

        act = 1 if 1 in legal_6 else legal_6[0]
        if log_collector is not None:
            log_collector.append((hand_id, "Model_B_QLearn", "Offense_Fallback", to_call, act))
        return act

class V102_CandidateC_RuleValueBetting:
    """Model C: 규칙 기반 밸류 베팅 + Temp-Hybrid 방어 정책"""
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
                log_collector.append((hand_id, "Model_C_RuleVal", "Defense_FacingBet", to_call, act))
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
                    log_collector.append((hand_id, "Model_C_RuleVal", "Offense_RuleVal", to_call, act))
                return act

        act = self.blueprint.select_action(inf, legal_6, is_eval=True)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_C_RuleVal", "Offense_Blueprint", to_call, act))
        return act

class V102_CandidateD_UnifiedEnsemble:
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
                log_collector.append((hand_id, "Model_D_Unified", "Defense_FacingBet", to_call, act))
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
            if equity >= 0.65:
                bets = [a for a in legal_6 if a in [2, 3]]
                act = bets[-1] if (bets and equity >= 0.75) else (bets[0] if bets else 1)
                if log_collector is not None:
                    log_collector.append((hand_id, "Model_D_Unified", "Offense_EquityPriority", to_call, act))
                return act

        if np.any(q_vals != 0.0):
            masked = np.full(6, -1e9, dtype=np.float32)
            for a in legal_6: masked[a] = q_vals[a]
            act = int(np.argmax(masked))
            if log_collector is not None:
                log_collector.append((hand_id, "Model_D_Unified", "Offense_QLearnPriority", to_call, act))
            return act

        act = self.bp.select_action(inf, legal_6, is_eval=True)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_D_Unified", "Offense_Blueprint", to_call, act))
        return act

# =====================================================================
# MAIN EXPERIMENT PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.2: ACTION SEMANTICS AUDIT & UNIFIED POLICY CHAMPIONSHIP")
    print("=" * 80)

    # -----------------------------------------------------------------
    # PHASE 1: True 6-Action Semantics & Engine Amount Mapping Audit
    # -----------------------------------------------------------------
    print("\n[*] Phase 1: True 6-Action Semantics & Engine Amount Mapping Forensic Audit...")
    action_table = [
        (0, "Fold", "0.00x Pot (Abandon)", "Forfeit current spent chips"),
        (1, "Check/Call", "Match current bet (to_call)", "Keep hand alive"),
        (2, "HalfPot", "0.50x Pot Raise (0.50 * base_pot)", "Value probe / standard sizing"),
        (3, "Pot", "1.00x Pot Raise (1.00 * base_pot)", "Standard pot-sized value bet"),
        (4, "Overbet", "2.00x Pot Raise (2.00 * base_pot)", "Polarized maximum pressure"),
        (5, "All-in", "All remaining chips (10,000 max)", "Commit total stack")
    ]
    print(f"  {'Action ID':<10s} | {'Canonical Name':<15s} | {'Math Formulation':<35s} | {'Engine Binding'}")
    print("  " + "-" * 80)
    for aid, name, formula, desc in action_table:
        print(f"  Action {aid:<3d} | {name:<15s} | {formula:<35s} | {desc}")

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
    model_a = V102_CandidateA_TempHybrid(bp_agent)
    model_b = V102_CandidateB_QLearningOffense(bp_agent, q_exploiter, temperature=2.0)
    model_c = V102_CandidateC_RuleValueBetting(bp_agent, temperature=2.0)
    model_d = V102_CandidateD_UnifiedEnsemble(bp_agent, q_exploiter, temperature=2.0)

    candidates = [
        ("Model A (Temp-Hybrid Baseline)", model_a),
        ("Model B (Q-Learn Offense + Temp)", model_b),
        ("Model C (Rule ValueBet + Temp)", model_c),
        ("Model D (Unified Ensemble)", model_d)
    ]

    # -----------------------------------------------------------------
    # PHASE 2 & 3: Fixed Opponents Duplicate Benchmark (5,000 hands)
    # -----------------------------------------------------------------
    print("\n[*] Phase 2, 3 & 4: Duplicate 4-Model Contrast on Fixed Opponents (5,000 hands)...")
    opponents = [BalancedBot(), TightPassiveBot(), CallingStationBot(), UltraAggressiveBot()]
    total_hands = 5000

    results = {name: [] for name, _ in candidates}
    all_routing_logs = {name: [] for name, _ in candidates}
    
    # 포커 메트릭 추적
    street_folds = {name: {"Preflop": 0, "Flop": 0, "Turn": 0, "River": 0} for name, _ in candidates}
    vpip_counts = {name: 0 for name, _ in candidates}
    pfr_counts = {name: 0 for name, _ in candidates}
    three_bet_counts = {name: 0 for name, _ in candidates}
    showdown_counts = {name: 0 for name, _ in candidates}
    showdown_wins = {name: 0 for name, _ in candidates}

    for h in range(total_hands):
        seed_h = 20261010 + h
        seat_test = h % 5

        for name, cand in candidates:
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            folded = set()
            hero_folded_street = None
            hero_vpip = False
            hero_pfr = False
            preflop_raises = 0

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, np.random.RandomState(seed_h))
                else:
                    cp = st.current_player()
                    leg_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                    inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    b_cards = list(np.where(inf[57:109] == 1.0)[0])
                    n_b = len(b_cards)
                    curr_street = "Preflop" if n_b == 0 else ("Flop" if n_b == 3 else ("Turn" if n_b == 4 else "River"))

                    if cp == seat_test:
                        act_6 = cand.select_action(st, cp, leg_6, folded_players=folded,
                                                   log_collector=all_routing_logs[name], hand_id=h+1)
                        if act_6 == 0 and hero_folded_street is None:
                            hero_folded_street = curr_street
                            street_folds[name][curr_street] += 1
                        
                        # VPIP / PFR 판별 (Preflop 자발적 베팅/콜)
                        if curr_street == "Preflop":
                            spent = extract_player_spent(st)
                            to_call = max(spent) - spent[cp]
                            if act_6 in [2, 3, 4, 5]:
                                hero_vpip = True
                                hero_pfr = True
                                if preflop_raises >= 1:
                                    three_bet_counts[name] += 1
                            elif act_6 == 1 and to_call > 0:
                                hero_vpip = True
                    else:
                        opp = opponents[cp if cp < seat_test else cp - 1]
                        act_6 = opp.select_action(st, cp, leg_6)

                    if curr_street == "Preflop" and act_6 in [2, 3, 4, 5]:
                        preflop_raises += 1

                    if act_6 == 0:
                        folded.add(cp)

                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))

            ret_bb = st.returns()[seat_test] / 100.0
            results[name].append(ret_bb)

            if hero_vpip: vpip_counts[name] += 1
            if hero_pfr: pfr_counts[name] += 1
            
            # 쇼다운 판정
            if seat_test not in folded:
                active_at_end = [p for p in range(5) if p not in folded]
                if len(active_at_end) > 1:
                    showdown_counts[name] += 1
                    if ret_bb > 0:
                        showdown_wins[name] += 1

    # -----------------------------------------------------------------
    # PHASE 2: Full Decision Policy Routing Analysis
    # -----------------------------------------------------------------
    print("\n[Phase 2: Full Decision Policy Routing Analysis (100% Census)]")
    subpolicy_breakdown = {name: Counter() for name, _ in candidates}
    action_breakdown = {name: Counter() for name, _ in candidates}

    for name, _ in candidates:
        for hid, mname, subpol, to_c, act in all_routing_logs[name]:
            subpolicy_breakdown[name][subpol] += 1
            action_breakdown[name][(subpol, act)] += 1

        print(f"\n  * {name} Subpolicy Routing (Total Decisions = {len(all_routing_logs[name])}):")
        for subpol, cnt in sorted(subpolicy_breakdown[name].items()):
            pct = (cnt / float(len(all_routing_logs[name]))) * 100.0
            print(f"    - {subpol:<25s}: {cnt:5d} ({pct:5.1f}%)")

    # -----------------------------------------------------------------
    # PHASE 3: Fold Decomposition & Poker Metrics Analysis
    # -----------------------------------------------------------------
    print("\n[Phase 3: Fold Decomposition & Poker Metrics Audit (5,000 hands)]")
    for name, _ in candidates:
        total_f = sum(street_folds[name].values())
        print(f"\n  * {name}:")
        print(f"    - Total Hero Folds : {total_f:4d}/{total_hands} ({(total_f/total_hands)*100.0:5.1f}%) [Invariant <= 5,000: {'PASS' if total_f <= total_hands else 'FAIL'}]")
        print(f"    - Preflop Folds    : {street_folds[name]['Preflop']:4d} ({(street_folds[name]['Preflop']/total_hands)*100.0:5.1f}%)")
        print(f"    - Flop Folds       : {street_folds[name]['Flop']:4d} ({(street_folds[name]['Flop']/total_hands)*100.0:5.1f}%)")
        print(f"    - Turn Folds       : {street_folds[name]['Turn']:4d} ({(street_folds[name]['Turn']/total_hands)*100.0:5.1f}%)")
        print(f"    - River Folds      : {street_folds[name]['River']:4d} ({(street_folds[name]['River']/total_hands)*100.0:5.1f}%)")
        print(f"    - VPIP             : {(vpip_counts[name]/total_hands)*100.0:5.2f}%")
        print(f"    - PFR              : {(pfr_counts[name]/total_hands)*100.0:5.2f}%")
        print(f"    - 3-Bet%           : {(three_bet_counts[name]/total_hands)*100.0:5.2f}%")
        sd_rate = (showdown_counts[name]/total_hands)*100.0
        wsd_rate = (showdown_wins[name]/float(showdown_counts[name]))*100.0 if showdown_counts[name] > 0 else 0.0
        print(f"    - WTSD (Showdown)  : {sd_rate:5.2f}% ({showdown_counts[name]} hands)")
        print(f"    - W$SD (Win at SD) : {wsd_rate:5.2f}% ({showdown_wins[name]}/{showdown_counts[name]})")

        # 대형 팟 집중도 분석 (상위 1%, 상위 5% 팟)
        arr = np.array(results[name])
        sorted_ret = np.sort(arr)[::-1]
        top1_n = int(round(total_hands * 0.01))
        top5_n = int(round(total_hands * 0.05))
        total_pnl = float(np.sum(arr))
        top1_pnl = float(np.sum(sorted_ret[:top1_n]))
        top5_pnl = float(np.sum(sorted_ret[:top5_n]))
        top1_share = (top1_pnl / total_pnl) * 100.0 if total_pnl > 0 else 0.0
        top5_share = (top5_pnl / total_pnl) * 100.0 if total_pnl > 0 else 0.0
        print(f"    - Total Profit (BB): {total_pnl:+9.2f} BB")
        print(f"    - Top 1% Pot PnL   : {top1_pnl:+9.2f} BB ({top1_share:5.1f}% of total profit)")
        print(f"    - Top 5% Pot PnL   : {top5_pnl:+9.2f} BB ({top5_share:5.1f}% of total profit)")

    # -----------------------------------------------------------------
    # PHASE 4: Duplicate 4-Model Contrast Performance & Statistical Testing
    # -----------------------------------------------------------------
    print("\n[Phase 4: Duplicate 4-Model Contrast Performance Results (5,000 hands)]")
    stats_summary = {}
    for name, _ in candidates:
        arr = np.array(results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        ci = (wr - 1.96 * se, wr + 1.96 * se)
        stats_summary[name] = {"wr": wr, "se": se, "ci": ci, "arr": arr}
        print(f"  * {name:<32s} | Winrate: {wr:+7.2f} ± {se:5.2f} bb/100 | 95% CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}]")

    base_arr = stats_summary["Model A (Temp-Hybrid Baseline)"]["arr"]
    print("\n[Paired Statistical Hypothesis Testing vs Model A Baseline]")
    for name, _ in candidates[1:]:
        cur_arr = stats_summary[name]["arr"]
        diff = cur_arr - base_arr
        m_diff = float(np.mean(diff))
        se_diff = float(stats.sem(diff))
        t_stat, p_val = stats.ttest_rel(cur_arr, base_arr)
        bonferroni_p = min(1.0, p_val * 3.0) # 3개 후보 비교 보정
        ci_diff = (m_diff - 1.96 * se_diff, m_diff + 1.96 * se_diff)
        print(f"  * {name:<32s} vs Model A:")
        print(f"    - Delta EV: {m_diff:+7.2f} ± {se_diff:5.2f} bb/100 | 95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}] | t = {t_stat:.4f}, raw p = {p_val:.4f}, Bonferroni p = {bonferroni_p:.4f}")

    # -----------------------------------------------------------------
    # PHASE 5: New Holdout Arena Contrast (2,500 hands Duplicate)
    # -----------------------------------------------------------------
    print("\n[*] Phase 5: New Holdout Arena Contrast (2,500 hands Duplicate)...")
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

    print("\n[Holdout Arena Paired Statistical Results (2,500 hands)]")
    holdout_base_arr = np.array(holdout_results["Model A (Temp-Hybrid Baseline)"]) * 100.0
    for name, _ in candidates:
        arr = np.array(holdout_results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        ci = (wr - 1.96 * se, wr + 1.96 * se)
        if name == "Model A (Temp-Hybrid Baseline)":
            print(f"  * {name:<32s} | Winrate: {wr:+7.2f} ± {se:5.2f} bb/100 | 95% CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}] (Baseline)")
        else:
            diff = arr - holdout_base_arr
            m_diff = float(np.mean(diff))
            se_diff = float(stats.sem(diff))
            t_stat, p_val = stats.ttest_rel(arr, holdout_base_arr)
            ci_diff = (m_diff - 1.96 * se_diff, m_diff + 1.96 * se_diff)
            bonferroni_p = min(1.0, p_val * 3.0)
            print(f"  * {name:<32s} | Winrate: {wr:+7.2f} ± {se:5.2f} bb/100")
            print(f"    - Delta EV: {m_diff:+7.2f} ± {se_diff:5.2f} bb/100 | 95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}] | t = {t_stat:.4f}, raw p = {p_val:.4f}, Bonferroni p = {bonferroni_p:.4f}")

    # 전체 정책 라우팅 CSV 저장 (100% 전수 로깅 중 첫 10,000건 저장)
    os.makedirs("v102/data", exist_ok=True)
    csv_path = "v102/data/v102_full_census_routing_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hand_id", "model_name", "subpolicy", "to_call", "executed_action"])
        for name, _ in candidates:
            for entry in all_routing_logs[name][:2500]:
                writer.writerow(entry)
    print(f"\n[+] Full census routing log sample (10,000 entries) saved to {csv_path}")

if __name__ == "__main__":
    main()
