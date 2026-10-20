"""
POKER AI V10.3: SHOWDOWN INTEGRITY & RL OFFENSE INDEPENDENT VALIDATION
=======================================================================
- Phase 1: Showdown Accounting Bug Forensic Audit & HUD Metric Recalibration
  * Strict separation: Hero Fold vs Won Without Showdown vs Reached Real Showdown
  * WTSD% = (Real Showdowns / Flop Seen Hands) * 100
  * W$SD% = (Showdown Wins / Real Showdowns) * 100
  * Won Without Showdown (Uncontested Pot Steal/Bluff Take-down)
- Phase 2: Preflop Blind Defense & Hyper-Aggressive Stress Test
  * Introduce BlindStealerBot (Aggressive Blind Stealer on Cutoff/Button)
  * Introduce HyperAggressiveBot (High frequency 3-Bet & Overbet pressure)
  * Audit positional EV (SB, BB, Early, Late) and blind defense winrate
- Phase 3: Q-Learning Offense Causal Attribution Audit
  * Compare Q-Learning choice vs Baseline Blueprint choice on every free action
  * Measure action divergence rate, street of change, sizing shifts (Check -> Bet)
- Phase 4: Direct Pairwise Hypothesis Testing (B vs C, B vs D, C vs D)
  * Full paired difference stats (Mean Diff, SE, 95% CI, t-stat, Bonferroni-corrected p)
- Phase 5: Long-term Duplicate Championship on Diverse Field
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v102')))

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
from v102.run_v102_action_semantics_championship import (
    V102_CandidateA_TempHybrid, V102_CandidateB_QLearningOffense,
    V102_CandidateC_RuleValueBetting, V102_CandidateD_UnifiedEnsemble
)

# =====================================================================
# SPECIALIZED ADVERSARIES FOR PREFLOP STRESS TEST
# =====================================================================
class BlindStealerBot:
    """블라인드 스틸 특화 봇: CO, BTN, SB 포지션에서 프리플랍 100% 레이즈 구사"""
    def select_action(self, st, cp, legal_6):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
        if n_b == 0: # Preflop
            if cp in [2, 3, 4] and to_call <= 100: # Late position / SB
                raises = [a for a in legal_6 if a in [2, 3]]
                return raises[0] if raises else 1
            if to_call == 0:
                return 1
            return 1 if to_call <= 100 else 0
        # Postflop
        if to_call == 0:
            return 2 if 2 in legal_6 and random.random() < 0.40 else 1
        return 1 if to_call <= 200 else 0

class HyperAggressiveBot:
    """초공격형 봇: 3-Bet 빈도 50%, 팟 오버베팅을 밥먹듯이 시도하여 상대 폴드를 강요"""
    def select_action(self, st, cp, legal_6):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
        n_b = len(list(np.where(inf[57:109] == 1.0)[0]))
        if n_b == 0: # Preflop
            if to_call > 0 and 3 in legal_6 and random.random() < 0.60:
                return 3 # 3-Bet Full Pot
            raises = [a for a in legal_6 if a in [2, 3, 4]]
            return raises[-1] if raises and random.random() < 0.70 else 1
        # Postflop
        if to_call == 0:
            bets = [a for a in legal_6 if a in [2, 3, 4]]
            return bets[-1] if bets and random.random() < 0.60 else 1
        raises = [a for a in legal_6 if a in [2, 3, 4]]
        return raises[-1] if raises and random.random() < 0.50 else (1 if to_call <= 300 else 0)

# =====================================================================
# MAIN PIPELINE FOR V10.3
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.3: SHOWDOWN INTEGRITY & RL OFFENSE INDEPENDENT VALIDATION")
    print("=" * 80)

    # 1. OpenSpiel 엔진 및 가중치 로드
    game = get_5p_fullgame()
    bp_agent = TrueNFSP_6Action_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    # 2. Q-Learning Exploiter 로드/훈련
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

    # =================================================================
    # PHASE 1 & 4: 5,000 Hands Duplicate with Strict Showdown Accounting
    # =================================================================
    print("\n[*] Phase 1 & 4: 5,000 Hands Duplicate with Strict Showdown Accounting...")
    opponents = [BalancedBot(), TightPassiveBot(), CallingStationBot(), UltraAggressiveBot()]
    total_hands = 5000

    results = {name: [] for name, _ in candidates}
    
    # 쇼다운 정밀 지표
    hand_outcome = {name: {
        "Hero_Fold": 0,
        "Won_Without_Showdown": 0,
        "Reached_Real_Showdown": 0,
        "Won_at_Showdown": 0,
        "Lost_at_Showdown": 0,
        "Tied_at_Showdown": 0,
        "Flop_Seen_Count": 0
    } for name, _ in candidates}

    # Q-Learning 오펜스 행동 변경 추적 (Phase 3)
    rl_attribution = {
        "total_free_actions": 0,
        "qlearn_called": 0,
        "action_diverged_from_bp": 0,
        "diverged_streets": Counter(),
        "diverged_actions": Counter()
    }

    for h in range(total_hands):
        seed_h = 20261010 + h
        seat_test = h % 5

        for name, cand in candidates:
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            folded = set()
            hero_saw_flop = False

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

                    if cp == seat_test and n_b >= 3:
                        hero_saw_flop = True

                    if cp == seat_test:
                        # Phase 3: Model B 행동 분기 추적
                        if name == "Model B (Q-Learn Offense + Temp)":
                            spent = extract_player_spent(st)
                            to_call = max(spent) - spent[cp]
                            if to_call == 0:
                                rl_attribution["total_free_actions"] += 1
                                # 기본 Blueprint 추천 행동 확인
                                bp_act = bp_agent.select_action(inf, leg_6, is_eval=True)
                                act_6 = cand.select_action(st, cp, leg_6, folded_players=folded)
                                if act_6 != bp_act:
                                    rl_attribution["action_diverged_from_bp"] += 1
                                    rl_attribution["diverged_streets"][curr_street] += 1
                                    rl_attribution["diverged_actions"][(bp_act, act_6)] += 1
                            else:
                                act_6 = cand.select_action(st, cp, leg_6, folded_players=folded)
                        else:
                            act_6 = cand.select_action(st, cp, leg_6, folded_players=folded)

                        if act_6 == 0:
                            folded.add(cp)
                    else:
                        opp = opponents[cp if cp < seat_test else cp - 1]
                        act_6 = opp.select_action(st, cp, leg_6)
                        if act_6 == 0:
                            folded.add(cp)

                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))

            ret_bb = st.returns()[seat_test] / 100.0
            results[name].append(ret_bb)

            # 플랍 관측 여부
            if hero_saw_flop:
                hand_outcome[name]["Flop_Seen_Count"] += 1

            # 쇼다운 엄밀 분리 감사 (버그 완전 해결)
            alive = [p for p in range(5) if p not in folded]
            if seat_test in folded:
                hand_outcome[name]["Hero_Fold"] += 1
            else:
                if len(alive) == 1:
                    # 다른 모든 상대가 폴드하여 쇼다운 없이 승리
                    hand_outcome[name]["Won_Without_Showdown"] += 1
                else:
                    # 2인 이상 끝까지 살아남아 카드 오픈 (Real Showdown)
                    hand_outcome[name]["Reached_Real_Showdown"] += 1
                    if ret_bb > 0:
                        hand_outcome[name]["Won_at_Showdown"] += 1
                    elif ret_bb < 0:
                        hand_outcome[name]["Lost_at_Showdown"] += 1
                    else:
                        hand_outcome[name]["Tied_at_Showdown"] += 1

    # -----------------------------------------------------------------
    # PHASE 1: Showdown Accounting Bug Forensic Audit Report
    # -----------------------------------------------------------------
    print("\n[Phase 1: Showdown Accounting Forensic Audit (5,000 hands)]")
    for name, _ in candidates:
        ho = hand_outcome[name]
        tot_survived = ho["Won_Without_Showdown"] + ho["Reached_Real_Showdown"]
        flop_seen = ho["Flop_Seen_Count"]
        real_sd = ho["Reached_Real_Showdown"]
        sd_wins = ho["Won_at_Showdown"]
        wtsd_pct = (real_sd / float(flop_seen)) * 100.0 if flop_seen > 0 else 0.0
        wsd_pct = (sd_wins / float(real_sd)) * 100.0 if real_sd > 0 else 0.0

        print(f"\n  * {name}:")
        print(f"    - Total Hands              : {total_hands}")
        print(f"    - Hero Folds               : {ho['Hero_Fold']:4d} ({(ho['Hero_Fold']/total_hands)*100.0:5.1f}%)")
        print(f"    - Hero Survived Total      : {tot_survived:4d} ({(tot_survived/total_hands)*100.0:5.1f}%)")
        print(f"      + Won Without Showdown   : {ho['Won_Without_Showdown']:4d} ({(ho['Won_Without_Showdown']/total_hands)*100.0:5.1f}%) [Uncontested Pot Steals!]")
        print(f"      + Reached Real Showdown  : {ho['Reached_Real_Showdown']:4d} ({(ho['Reached_Real_Showdown']/total_hands)*100.0:5.1f}%) [Cards Actually Opened]")
        print(f"        * Won at Showdown      : {ho['Won_at_Showdown']:4d} ({(ho['Won_at_Showdown']/real_sd)*100.0 if real_sd else 0:5.1f}%)")
        print(f"        * Lost at Showdown     : {ho['Lost_at_Showdown']:4d} ({(ho['Lost_at_Showdown']/real_sd)*100.0 if real_sd else 0:5.1f}%)")
        print(f"        * Tied at Showdown     : {ho['Tied_at_Showdown']:4d}")
        print(f"    - Flop Seen Hands          : {flop_seen:4d} ({(flop_seen/total_hands)*100.0:5.1f}%)")
        print(f"    - Standard WTSD% (SD/Flop) : {wtsd_pct:5.2f}%")
        print(f"    - Standard W$SD% (Win/SD)  : {wsd_pct:5.2f}%")

    # -----------------------------------------------------------------
    # PHASE 3: Q-Learning Offense Causal Attribution Audit
    # -----------------------------------------------------------------
    print("\n[Phase 3: Q-Learning Offense Causal Attribution Audit (Model B)]")
    tot_free = rl_attribution["total_free_actions"]
    div_cnt = rl_attribution["action_diverged_from_bp"]
    print(f"  * Total Free Actions (ToCall == 0) : {tot_free}")
    print(f"  * Action Diverged from Blueprint    : {div_cnt} ({(div_cnt/tot_free)*100.0 if tot_free else 0:5.1f}%)")
    print("  * Diverged Action Street Breakdown :")
    for st_name, c in rl_attribution["diverged_streets"].most_common():
        print(f"    - {st_name:<10s}: {c:4d} ({(c/div_cnt)*100.0:5.1f}%)")
    print("  * Top 5 Policy Action Shifts (BP Action -> Q-Learn Action):")
    for (bp_a, q_a), c in rl_attribution["diverged_actions"].most_common(5):
        act_map = {0: "Fold", 1: "Check", 2: "HalfPot", 3: "Pot", 4: "Overbet", 5: "Allin"}
        print(f"    - {act_map[bp_a]} -> {act_map[q_a]}: {c:4d} times")

    # -----------------------------------------------------------------
    # PHASE 4: Full Pairwise Statistical Testing (B vs A, B vs C, B vs D, C vs D)
    # -----------------------------------------------------------------
    print("\n[Phase 4: Full Pairwise Hypothesis Testing (5,000 hands Duplicate)]")
    stats_summary = {}
    for name, _ in candidates:
        arr = np.array(results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        stats_summary[name] = {"wr": wr, "se": se, "arr": arr}
        print(f"  * {name:<32s} | Winrate: {wr:+7.2f} ± {se:5.2f} bb/100")

    def paired_compare(n1, n2, num_tests=6):
        arr1 = stats_summary[n1]["arr"]
        arr2 = stats_summary[n2]["arr"]
        diff = arr1 - arr2
        m_diff = float(np.mean(diff))
        se_diff = float(stats.sem(diff))
        t_stat, p_val = stats.ttest_rel(arr1, arr2)
        bonferroni_p = min(1.0, p_val * num_tests)
        ci_diff = (m_diff - 1.96 * se_diff, m_diff + 1.96 * se_diff)
        print(f"  * {n1} vs {n2}:")
        print(f"    - Delta EV: {m_diff:+7.2f} ± {se_diff:5.2f} bb/100 | 95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}] | t = {t_stat:.4f}, raw p = {p_val:.4f}, Bonferroni p = {bonferroni_p:.4f}")

    print("\n  [Pairwise Contrasts on Fixed Opponents]:")
    paired_compare("Model B (Q-Learn Offense + Temp)", "Model A (Temp-Hybrid Baseline)")
    paired_compare("Model B (Q-Learn Offense + Temp)", "Model C (Rule ValueBet + Temp)")
    paired_compare("Model B (Q-Learn Offense + Temp)", "Model D (Unified Ensemble)")
    paired_compare("Model C (Rule ValueBet + Temp)", "Model A (Temp-Hybrid Baseline)")
    paired_compare("Model D (Unified Ensemble)", "Model A (Temp-Hybrid Baseline)")
    paired_compare("Model C (Rule ValueBet + Temp)", "Model D (Unified Ensemble)")

    # -----------------------------------------------------------------
    # PHASE 2: Preflop Blind Defense & Hyper-Aggressive Stress Test (2,500 hands)
    # -----------------------------------------------------------------
    print("\n[*] Phase 2: Preflop Blind Defense & Hyper-Aggressive Stress Test (2,500 hands)...")
    stress_opponents = [BlindStealerBot(), HyperAggressiveBot(), BalancedBot(), CallingStationBot()]
    stress_results = {name: [] for name, _ in candidates}
    position_results = {name: {"SB": [], "BB": [], "Early": [], "Late": []} for name, _ in candidates}

    for h in range(2500):
        seed_h = 20261100 + h
        seat_test = h % 5
        pos_name = "SB" if seat_test == 3 else ("BB" if seat_test == 4 else ("Early" if seat_test in [0, 1] else "Late"))

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
                        opp = stress_opponents[cp if cp < seat_test else cp - 1]
                        act_6 = opp.select_action(st, cp, leg_6)

                    if act_6 == 0:
                        folded.add(cp)

                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))

            ret_bb = st.returns()[seat_test] / 100.0
            stress_results[name].append(ret_bb)
            position_results[name][pos_name].append(ret_bb)

    print("\n[Preflop Stress Test Performance Results (2,500 hands)]")
    stress_summary = {}
    for name, _ in candidates:
        arr = np.array(stress_results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        ci = (wr - 1.96 * se, wr + 1.96 * se)
        stress_summary[name] = {"wr": wr, "se": se, "arr": arr}
        print(f"  * {name:<32s} | Stress Winrate: {wr:+7.2f} ± {se:5.2f} bb/100 | 95% CI: [{ci[0]:+7.2f}, {ci[1]:+7.2f}]")

    print("\n  [Positional Breakdown under Aggressive Steals (bb/100)]:")
    print(f"  {'Model Name':<32s} | {'SB (Seat 3)':<15s} | {'BB (Seat 4)':<15s} | {'Early (0, 1)':<15s} | {'Late (Seat 2)':<15s}")
    print("  " + "-" * 95)
    for name, _ in candidates:
        sb_wr = float(np.mean(position_results[name]["SB"])) * 100.0
        bb_wr = float(np.mean(position_results[name]["BB"])) * 100.0
        ea_wr = float(np.mean(position_results[name]["Early"])) * 100.0
        la_wr = float(np.mean(position_results[name]["Late"])) * 100.0
        print(f"  {name:<32s} | {sb_wr:+9.2f} bb/100 | {bb_wr:+9.2f} bb/100 | {ea_wr:+9.2f} bb/100 | {la_wr:+9.2f} bb/100")

    # Pairwise contrast under stress test (Model B vs Model A, C, D)
    print("\n  [Stress Test Pairwise Hypothesis Testing (B vs A, B vs C, B vs D)]:")
    base_stress = stress_summary["Model A (Temp-Hybrid Baseline)"]["arr"]
    for comp_name in ["Model B (Q-Learn Offense + Temp)", "Model C (Rule ValueBet + Temp)", "Model D (Unified Ensemble)"]:
        comp_arr = stress_summary[comp_name]["arr"]
        diff = comp_arr - base_stress
        m_diff = float(np.mean(diff))
        se_diff = float(stats.sem(diff))
        t_stat, p_val = stats.ttest_rel(comp_arr, base_stress)
        bonferroni_p = min(1.0, p_val * 3.0)
        ci_diff = (m_diff - 1.96 * se_diff, m_diff + 1.96 * se_diff)
        print(f"  * {comp_name} vs Model A:")
        print(f"    - Delta EV: {m_diff:+7.2f} ± {se_diff:5.2f} bb/100 | 95% CI: [{ci_diff[0]:+7.2f}, {ci_diff[1]:+7.2f}] | t = {t_stat:.4f}, raw p = {p_val:.4f}, Bonf p = {bonferroni_p:.4f}")

    # B vs C, B vs D in Stress Test
    arr_b = stress_summary["Model B (Q-Learn Offense + Temp)"]["arr"]
    arr_c = stress_summary["Model C (Rule ValueBet + Temp)"]["arr"]
    arr_d = stress_summary["Model D (Unified Ensemble)"]["arr"]
    
    diff_bc = arr_b - arr_c
    t_bc, p_bc = stats.ttest_rel(arr_b, arr_c)
    print(f"  * Model B vs Model C (Stress): Delta EV: {float(np.mean(diff_bc)):+7.2f} ± {float(stats.sem(diff_bc)):5.2f} bb/100 | t = {t_bc:.4f}, p = {p_bc:.4f}")

    diff_bd = arr_b - arr_d
    t_bd, p_bd = stats.ttest_rel(arr_b, arr_d)
    print(f"  * Model B vs Model D (Stress): Delta EV: {float(np.mean(diff_bd)):+7.2f} ± {float(stats.sem(diff_bd)):5.2f} bb/100 | t = {t_bd:.4f}, p = {p_bd:.4f}")

if __name__ == "__main__":
    main()
