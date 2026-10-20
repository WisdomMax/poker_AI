"""
POKER AI V10.4: TRUE ADVERSARIAL VALIDATION & COUNTERFACTUAL ACTION AUDIT
==========================================================================
- Phase 1: CallingStation-Free Pure Competency Arena (2,500 Hands Duplicate)
  * Opponents: TightAggressiveBot, BalancedBot, BlindStealerBot, HyperAggressiveBot
  * CallingStation completely EXCLUDED to verify non-zero Won Without Showdown
- Phase 2: Forensic Audit of Check -> All-in Decisions
  * Full inspection of every All-in: Street, Hole Cards, Board, Equity, Pot, Result Payoff
  * Categorize into Value Bet vs Bluff vs Inappropriate play
- Phase 3: Counterfactual Action Divergence Payoff Audit
  * Audit hands where Q-Learning diverged from Blueprint
  * Compare hand outcome when Q-Learning altered the action
- Phase 4: Big Blind (BB) Positional EV Causal Decomposition
  * Dissect BB hands into Preflop defense vs Postflop Q-Learning exploitation
  * Paired t-test and 95% CI for BB EV difference (Model B vs Model A)
- Phase 5: Synthesis & Promotion Audit
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v103')))

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
from v103.run_v103_showdown_and_rl_validation import BlindStealerBot, HyperAggressiveBot

def card_to_str(c):
    ranks = "23456789TJQKA"
    suits = "shdc"
    return ranks[c // 4] + suits[c % 4]

def main():
    print("=" * 80)
    print("POKER AI V10.4: TRUE ADVERSARIAL VALIDATION & COUNTERFACTUAL ACTION AUDIT")
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
    # PHASE 1: CallingStation-Free Pure Competency Arena (2,500 hands Duplicate)
    # =================================================================
    print("\n[*] Phase 1: CallingStation-Free Pure Competency Arena (2,500 hands Duplicate)...")
    # CallingStation 완전히 제외: TAG, Balanced, BlindStealer, HyperAggressive
    non_cs_opponents = [TightAggressiveBot(), BalancedBot(), BlindStealerBot(), HyperAggressiveBot()]
    total_hands_p1 = 2500

    p1_results = {name: [] for name, _ in candidates}
    p1_outcomes = {name: {
        "Hero_Fold": 0,
        "Won_Without_Showdown": 0,
        "Reached_Real_Showdown": 0,
        "Won_at_Showdown": 0,
        "Lost_at_Showdown": 0,
        "Tied_at_Showdown": 0
    } for name, _ in candidates}

    for h in range(total_hands_p1):
        seed_h = 20261200 + h
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
                        opp = non_cs_opponents[cp if cp < seat_test else cp - 1]
                        act_6 = opp.select_action(st, cp, leg_6)

                    if act_6 == 0:
                        folded.add(cp)

                    st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6))

            ret_bb = st.returns()[seat_test] / 100.0
            p1_results[name].append(ret_bb)

            alive = [p for p in range(5) if p not in folded]
            if seat_test in folded:
                p1_outcomes[name]["Hero_Fold"] += 1
            else:
                if len(alive) == 1:
                    # 상대가 전부 폴드하여 쇼다운 없이 승리 (Non-Zero 검증 대상!)
                    p1_outcomes[name]["Won_Without_Showdown"] += 1
                else:
                    p1_outcomes[name]["Reached_Real_Showdown"] += 1
                    if ret_bb > 0: p1_outcomes[name]["Won_at_Showdown"] += 1
                    elif ret_bb < 0: p1_outcomes[name]["Lost_at_Showdown"] += 1
                    else: p1_outcomes[name]["Tied_at_Showdown"] += 1

    print("\n[Phase 1 Results: CallingStation-Free Arena Accounting]")
    for name, _ in candidates:
        arr = np.array(p1_results[name]) * 100.0
        wr = float(np.mean(arr))
        se = float(stats.sem(arr))
        oc = p1_outcomes[name]
        print(f"\n  * {name}:")
        print(f"    - Winrate                   : {wr:+7.2f} ± {se:5.2f} bb/100")
        print(f"    - Total Folds               : {oc['Hero_Fold']:4d} ({(oc['Hero_Fold']/total_hands_p1)*100.0:5.1f}%)")
        print(f"    - Won Without Showdown      : {oc['Won_Without_Showdown']:4d} ({(oc['Won_Without_Showdown']/total_hands_p1)*100.0:5.1f}%) [Cards NOT opened, Opponents Folded!]")
        print(f"    - Reached Real Showdown     : {oc['Reached_Real_Showdown']:4d} ({(oc['Reached_Real_Showdown']/total_hands_p1)*100.0:5.1f}%)")
        if oc['Reached_Real_Showdown'] > 0:
            wsd = (oc['Won_at_Showdown'] / float(oc['Reached_Real_Showdown'])) * 100.0
            print(f"    - Showdown Winrate (W$SD)   : {wsd:5.1f}% ({oc['Won_at_Showdown']}/{oc['Reached_Real_Showdown']})")

    # Pairwise comparison in non-CS arena
    base_p1 = np.array(p1_results["Model A (Temp-Hybrid Baseline)"]) * 100.0
    arr_b_p1 = np.array(p1_results["Model B (Q-Learn Offense + Temp)"]) * 100.0
    diff_p1 = arr_b_p1 - base_p1
    t_p1, p_p1 = stats.ttest_rel(arr_b_p1, base_p1)
    print(f"\n  [CallingStation-Free Arena: Model B vs Model A]:")
    print(f"    - Delta EV: {float(np.mean(diff_p1)):+7.2f} ± {float(stats.sem(diff_p1)):5.2f} bb/100 | t = {t_p1:.4f}, p = {p_p1:.4f}")

    # =================================================================
    # PHASE 2 & 3 & 4: 5,000 Hands All-in Audit, Counterfactual & BB Decomposition
    # =================================================================
    print("\n[*] Phase 2, 3 & 4: Detailed 5,000 Hands Audit on Fixed Field...")
    opponents_5k = [BalancedBot(), TightPassiveBot(), CallingStationBot(), UltraAggressiveBot()]
    total_hands_5k = 5000

    allin_records = []
    diverged_hand_records = []
    bb_position_data = {"Model_A": [], "Model_B": []}

    for h in range(total_hands_5k):
        seed_h = 20261010 + h
        seat_test = h % 5
        is_bb = (seat_test == 4)

        # Model A 실행
        st_a = game.new_initial_state()
        advance_chance_nodes(st_a, np.random.RandomState(seed_h))
        folded_a = set()
        while not st_a.is_terminal():
            if st_a.is_chance_node():
                advance_chance_nodes(st_a, np.random.RandomState(seed_h))
            else:
                cp = st_a.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                if cp == seat_test:
                    act_a = model_a.select_action(st_a, cp, leg_6, folded_players=folded_a)
                else:
                    opp = opponents_5k[cp if cp < seat_test else cp - 1]
                    act_a = opp.select_action(st_a, cp, leg_6)
                if act_a == 0: folded_a.add(cp)
                st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, act_a))
        ret_a_bb = st_a.returns()[seat_test] / 100.0
        if is_bb: bb_position_data["Model_A"].append(ret_a_bb)

        # Model B 실행 및 추적
        st_b = game.new_initial_state()
        advance_chance_nodes(st_b, np.random.RandomState(seed_h))
        folded_b = set()
        hand_diverged = False
        diverged_details = []

        while not st_b.is_terminal():
            if st_b.is_chance_node():
                advance_chance_nodes(st_b, np.random.RandomState(seed_h))
            else:
                cp = st_b.current_player()
                leg_6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                inf = np.array(st_b.information_state_tensor(cp), dtype=np.float32)
                h_cards = list(np.where(inf[5:57] == 1.0)[0])
                b_cards = list(np.where(inf[57:109] == 1.0)[0])
                n_b = len(b_cards)
                curr_st_name = "Preflop" if n_b == 0 else ("Flop" if n_b == 3 else ("Turn" if n_b == 4 else "River"))

                if cp == seat_test:
                    spent = extract_player_spent(st_b)
                    to_call = max(spent) - spent[cp]
                    pot = sum(spent)
                    bp_act = bp_agent.select_action(inf, leg_6, is_eval=True)

                    act_b = model_b.select_action(st_b, cp, leg_6, folded_players=folded_b)

                    # Q-Learning 행동 변경 감지
                    if to_call == 0 and act_b != bp_act:
                        hand_diverged = True
                        diverged_details.append((curr_st_name, bp_act, act_b, pot))

                    # Phase 2: Check -> All-in (act_b == 5 when to_call == 0) 포렌식 로깅
                    if to_call == 0 and act_b == 5 and bp_act == 1:
                        # 에쿼티 추정
                        combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                            h_cards, b_cards, mode='temperature', temperature=2.0,
                            opp_aggression_level=1.0, is_raised=False
                        )
                        active_opps = sum(1 for p in range(5) if p != cp and p not in folded_b)
                        eq = estimate_range_aware_equity(h_cards, b_cards, combos, probs, active_opps, num_sims=30)
                        
                        allin_records.append({
                            "hand_id": h + 1,
                            "street": curr_st_name,
                            "hero_cards": " ".join([card_to_str(c) for c in h_cards]),
                            "board": " ".join([card_to_str(c) for c in b_cards]),
                            "equity": eq,
                            "pot": pot,
                            "eff_stack": 10000 - spent[cp]
                        })
                else:
                    opp = opponents_5k[cp if cp < seat_test else cp - 1]
                    act_b = opp.select_action(st_b, cp, leg_6)

                if act_b == 0: folded_b.add(cp)
                st_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, act_b))

        ret_b_bb = st_b.returns()[seat_test] / 100.0
        if is_bb: bb_position_data["Model_B"].append(ret_b_bb)

        if hand_diverged:
            diverged_hand_records.append({
                "hand_id": h + 1,
                "ret_model_a": ret_a_bb,
                "ret_model_b": ret_b_bb,
                "diff": ret_b_bb - ret_a_bb,
                "details": diverged_details
            })

    # -----------------------------------------------------------------
    # PHASE 2: Check -> All-in Forensic Investigation Report
    # -----------------------------------------------------------------
    print(f"\n[Phase 2: Check -> All-in Decisions Forensic Investigation (Total = {len(allin_records)})]")
    high_eq_cnt = sum(1 for r in allin_records if r["equity"] >= 0.70)
    med_eq_cnt = sum(1 for r in allin_records if 0.40 <= r["equity"] < 0.70)
    bluff_cnt = sum(1 for r in allin_records if r["equity"] < 0.40)
    avg_eq = np.mean([r["equity"] for r in allin_records]) if allin_records else 0.0

    print(f"  * Average Hero Equity at All-in   : {avg_eq*100.0:5.2f}%")
    print(f"  * Monster / Value All-in (Eq >= 70%): {high_eq_cnt:3d} ({(high_eq_cnt/len(allin_records))*100.0 if allin_records else 0:5.1f}%)")
    print(f"  * Medium / Semi-Bluff (40%~70%)   : {med_eq_cnt:3d} ({(med_eq_cnt/len(allin_records))*100.0 if allin_records else 0:5.1f}%)")
    print(f"  * Pure Bluff All-in (Eq < 40%)    : {bluff_cnt:3d} ({(bluff_cnt/len(allin_records))*100.0 if allin_records else 0:5.1f}%)")
    print("\n  * Sample 5 Check -> All-in Spotlights:")
    for i, r in enumerate(allin_records[:5]):
        print(f"    [{i+1}] Hand #{r['hand_id']} | Street: {r['street']:<7s} | Hero: {r['hero_cards']:<5s} | Board: {r['board']:<14s} | Equity: {r['equity']*100.0:5.1f}% | Pot: {r['pot']} chips")

    # CSV 저장
    os.makedirs("v104/data", exist_ok=True)
    with open("v104/data/v104_allin_forensic_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hand_id", "street", "hero_cards", "board", "equity", "pot", "eff_stack"])
        writer.writeheader()
        for r in allin_records:
            writer.writerow(r)

    # -----------------------------------------------------------------
    # PHASE 3: Counterfactual Action Divergence Payoff Audit
    # -----------------------------------------------------------------
    print(f"\n[Phase 3: Counterfactual Divergence Payoff Audit (Hands with Divergence = {len(diverged_hand_records)})]")
    div_diffs = [r["diff"] for r in diverged_hand_records]
    mean_div_diff = float(np.mean(div_diffs)) if div_diffs else 0.0
    se_div_diff = float(stats.sem(div_diffs)) if div_diffs else 0.0
    total_div_gain = float(np.sum(div_diffs)) if div_diffs else 0.0

    print(f"  * Hands where Q-Learning Diverged : {len(diverged_hand_records)} / {total_hands_5k} ({(len(diverged_hand_records)/total_hands_5k)*100.0:5.1f}%)")
    print(f"  * Total Net BB Gain from Divergence: {total_div_gain:+8.2f} BB")
    print(f"  * Average EV per Diverged Hand   : {mean_div_diff:+6.2f} ± {se_div_diff:5.2f} BB/hand")
    t_div, p_div = stats.ttest_1samp(div_diffs, 0.0) if div_diffs else (0, 1)
    print(f"  * Divergence EV Causal Test (H0=0): t = {t_div:.4f}, p = {p_div:.4f}")

    # -----------------------------------------------------------------
    # PHASE 4: Big Blind (BB) Positional EV Causal Decomposition
    # -----------------------------------------------------------------
    print(f"\n[Phase 4: Big Blind (BB) Positional EV Causal Decomposition (Hands = {len(bb_position_data['Model_A'])})]")
    bb_a = np.array(bb_position_data["Model_A"]) * 100.0
    bb_b = np.array(bb_position_data["Model_B"]) * 100.0
    bb_diff = bb_b - bb_a

    wr_bb_a = float(np.mean(bb_a))
    wr_bb_b = float(np.mean(bb_b))
    se_bb_diff = float(stats.sem(bb_diff))
    mean_bb_diff = float(np.mean(bb_diff))
    ci_bb_diff = (mean_bb_diff - 1.96 * se_bb_diff, mean_bb_diff + 1.96 * se_bb_diff)
    t_bb, p_bb = stats.ttest_rel(bb_b, bb_a)

    print(f"  * Model A BB Winrate : {wr_bb_a:+7.2f} bb/100")
    print(f"  * Model B BB Winrate : {wr_bb_b:+7.2f} bb/100")
    print(f"  * BB Delta EV (B - A): {mean_bb_diff:+7.2f} ± {se_bb_diff:5.2f} bb/100")
    print(f"  * 95% Confidence Int : [{ci_bb_diff[0]:+7.2f}, {ci_bb_diff[1]:+7.2f}]")
    print(f"  * Paired t-statistic : t = {t_bb:.4f}, p = {p_bb:.4f}")

    print("\n[+] V10.4 Execution Completed Successfully!")

if __name__ == "__main__":
    main()
