import os
import sys
import json
import time
import numpy as np
import torch
import scipy.stats as stats

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent,
    AggressiveBot, TightPassiveBot
)
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v83.run_v83_robust_range_and_cfr_audit import (
    RobustBayesianRangeModel, get_hand_bucket
)
from v86.run_v86_meta_controller_and_independent_audit import BalancedBot
from v87.run_v87_adversarial_red_team_arena import UltraAggressiveBot, V87MetaControllerAgent
from v88.run_v88_adversarial_learning_and_hero_exploitation import CallingStationBot
from v90.run_v90_adversarial_causal_audit_and_scaling import MultiwayHeroExploiter, train_multiway_exploiter
from v91.run_v91_chip_flow_and_rl_ablation import TightAggressiveBot, ManiacBot
from v103.run_v103_showdown_and_rl_validation import BlindStealerBot, HyperAggressiveBot
from v106.run_v106_deterministic_replay import (
    DeterministicBalancedBot, get_deterministic_fixed_opponents, get_deterministic_non_cs_opponents
)

# =====================================================================
# PHASE 2: 전역 RNG 완전 격리 몬테카를로 에쿼티 추정기
# =====================================================================
def estimate_range_aware_equity_deterministic(
    hero_cards, board_cards, opp_combos, opp_probs, num_active_opponents,
    num_sims=100, rng_stream=None
):
    """
    전역 np.random을 전혀 사용하지 않는 순수 결정론적 에쿼티 추정기.
    rng_stream: 독립된 np.random.RandomState 인스턴스 (필수)
    """
    known_cards = set(hero_cards) | set(board_cards)
    unknown_deck = list(set(range(52)) - known_cards)
    cards_needed_board = 5 - len(board_cards)
    
    if len(unknown_deck) < (2 * num_active_opponents + cards_needed_board):
        return 0.50

    if rng_stream is None:
        raise ValueError("CRITICAL: rng_stream (RandomState) must be explicitly provided to prevent global RNG pollution!")

    wins = 0
    ties = 0
    num_combos = len(opp_combos)
    if num_combos == 0:
        return 0.50

    for _ in range(num_sims):
        # 1. Bayesian Range 가중 샘플링
        opp1_idx = rng_stream.choice(num_combos, p=opp_probs)
        opp1_hole = list(opp_combos[opp1_idx])

        # 2. 미확인 덱 차단 구성
        current_blocked = known_cards | set(opp1_hole)
        rem_deck = [c for c in unknown_deck if c not in current_blocked]

        # 3. 추가 상대 및 보드 샘플링
        extra_opps = num_active_opponents - 1
        needed_cards = cards_needed_board + 2 * extra_opps
        if len(rem_deck) < needed_cards:
            continue
        
        sim_sampled = rng_stream.choice(rem_deck, size=needed_cards, replace=False)
        sim_board = list(board_cards) + list(sim_sampled[:cards_needed_board])
        
        # Hero 점수
        hero_full = list(hero_cards) + sim_board
        hero_score = eval_7cards(hero_full)

        hero_won = True
        hero_tied = False

        # Opp1 점수
        opp1_full = opp1_hole + sim_board
        opp1_score = eval_7cards(opp1_full)
        if opp1_score > hero_score:
            hero_won = False
        elif opp1_score == hero_score:
            hero_tied = True

        # 추가 활성 상대들 점수
        if hero_won or hero_tied:
            for i in range(extra_opps):
                o_hole = list(sim_sampled[cards_needed_board + 2*i : cards_needed_board + 2*i + 2])
                o_score = eval_7cards(o_hole + sim_board)
                if o_score > hero_score:
                    hero_won = False
                    hero_tied = False
                    break
                elif o_score == hero_score:
                    hero_tied = True

        if hero_won and not hero_tied:
            wins += 1
        elif hero_tied:
            ties += 1

    return float(wins + 0.5 * ties) / float(num_sims)


# =====================================================================
# V10.8 결정론적 에이전트 클래스 (RNG 격리)
# =====================================================================
class V108_Deterministic_TempHybrid:
    """Model A: 결정론적 Temperature Hybrid T=2.0 기준 모델"""
    def __init__(self, bp_agent, num_sims=100):
        self.blueprint = bp_agent
        self.temperature = 2.0
        self.num_sims = num_sims

    def select_action(self, st, cp, legal_6, folded_players=None, hand_seed=0, decision_idx=0, log_collector=None, hand_id=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        pot = sum(spent)
        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)

        # 1. 수비 상황 (To-call > 0) -> Temp-Hybrid 수비 엔진
        if to_call > 0 and 0 in legal_6:
            hero_cards = list(np.where(inf[5:57] == 1.0)[0])
            board_cards = list(np.where(inf[57:109] == 1.0)[0])
            active_opps = sum(1 for p in range(5) if p != cp and (folded_players is None or p not in folded_players))

            combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
                hero_cards, board_cards, mode='temperature', temperature=self.temperature,
                opp_aggression_level=1.5, is_raised=(to_call >= 100)
            )
            pot_odds = to_call / float(pot + to_call)
            
            # 독립된 에쿼티 난수 스트림 생성 (전역 RNG 격리)
            eq_seed = (hand_seed * 10007 + cp * 101 + decision_idx * 17) & 0xFFFFFFFF
            eq_rng = np.random.RandomState(eq_seed)
            equity = estimate_range_aware_equity_deterministic(
                hero_cards, board_cards, combos, probs, active_opps,
                num_sims=self.num_sims, rng_stream=eq_rng
            )

            if equity > pot_odds + 0.05:
                raises = [a for a in legal_6 if a in [2, 3]]
                act = raises[0] if (raises and equity > 0.65) else (1 if 1 in legal_6 else legal_6[0])
            elif equity < pot_odds - 0.05:
                act = 0 # Clean Fold
            else:
                act = 1 if 1 in legal_6 else legal_6[0]

            if log_collector is not None:
                log_collector.append((hand_id, "Model_A_Temp", "Defense_FacingBet", to_call, act, equity, pot_odds))
            return act

        # 2. 자유 행동 상황 (To-call == 0) -> Blueprint 신경망
        act = self.blueprint.select_action(inf, legal_6, is_eval=True)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_A_Temp", "FreeAction_Blueprint", to_call, act, None, None))
        return act


class V108_Deterministic_QLearningOffense:
    """Model B: Q-Learning 공격 + Temp-Hybrid 방어 정책 (RNG 완전 격리)"""
    def __init__(self, bp_agent, q_exploiter: MultiwayHeroExploiter, num_sims=100):
        self.blueprint = bp_agent
        self.q_exploiter = q_exploiter
        self.temperature = 2.0
        self.num_sims = num_sims

    def select_action(self, st, cp, legal_6, folded_players=None, hand_seed=0, decision_idx=0, log_collector=None, hand_id=0):
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
            
            # 독립된 에쿼티 난수 스트림 생성 (전역 RNG 격리)
            eq_seed = (hand_seed * 10007 + cp * 101 + decision_idx * 17) & 0xFFFFFFFF
            eq_rng = np.random.RandomState(eq_seed)
            equity = estimate_range_aware_equity_deterministic(
                hero_cards, board_cards, combos, probs, active_opps,
                num_sims=self.num_sims, rng_stream=eq_rng
            )

            if equity > pot_odds + 0.15:
                raises = [a for a in legal_6 if a in [2, 3]]
                act = raises[0] if raises else 1
            elif equity >= pot_odds - 0.05:
                act = 1
            else:
                act = 0

            if log_collector is not None:
                log_collector.append((hand_id, "Model_B_QLearn", "Defense_FacingBet", to_call, act, equity, pot_odds))
            return act

        # 2. 공격 상황 (To-call == 0) -> Q-Learning 정책 조회
        state = self.q_exploiter.get_abstract_state(st, cp)
        q_vals = self.q_exploiter.q_table[state]
        if np.any(q_vals != 0.0):
            masked = np.full(6, -1e9, dtype=np.float32)
            for a in legal_6: masked[a] = q_vals[a]
            act = int(np.argmax(masked))
            if log_collector is not None:
                log_collector.append((hand_id, "Model_B_QLearn", "Offense_QLearn", to_call, act, None, None))
            return act

        act = 1 if 1 in legal_6 else legal_6[0]
        if log_collector is not None:
            log_collector.append((hand_id, "Model_B_QLearn", "Offense_Fallback", to_call, act, None, None))
        return act


def get_deterministic_fixed_opponents_v108(seed_h):
    """핸드별 완벽한 결정론적 상대 봇 풀 생성 (모든 확률적 봇에 명시적 시드 주입)"""
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 100 + 1),
        TightPassiveBot(),
        CallingStationBot(),
        UltraAggressiveBot(rng_seed=seed_h * 100 + 4)
    ]

def main():
    print("=" * 80)
    print("POKER AI V10.8: DETERMINISTIC BENCHMARK & REPRODUCIBILITY AUDIT")
    print("=" * 80)

    # 1. OpenSpiel 엔진 및 가중치 로드
    game = get_5p_fullgame()
    bp_agent = TrueNFSP_6Action_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    q_exploiter = MultiwayHeroExploiter()
    q_table_cache = 'v106/data/q_exploiter_15k_cache.pt'
    if os.path.exists(q_table_cache):
        print(f"[*] Loading pre-trained Q-table from {q_table_cache}...")
        loaded_dict = torch.load(q_table_cache, weights_only=False)
        for k, v in loaded_dict.items():
            q_exploiter.q_table[k] = v
        print(f"[+] Q-table successfully restored! States = {len(q_exploiter.q_table)}")
    else:
        q_exploiter, _ = train_multiway_exploiter(q_exploiter, V87MetaControllerAgent(bp_agent, 2.0, 999999), game, 15000)

    # =================================================================
    # PHASE 2: 전역 RNG 완전 분리 검증 (Self-Regression 100 Hands Test)
    # =================================================================
    print("\n[*] Phase 2: Testing Full RNG Isolation Self-Regression (100 Hands)...")
    model_a_test = V108_Deterministic_TempHybrid(bp_agent, num_sims=100)

    regression_identical = True
    max_diff = 0.0
    action_seq_mismatches = 0
    chip_flow_mismatches = 0

    for h in range(100):
        seed_h = 20261100 + h
        seat_test = h % 5

        # Run 1
        opps_1 = get_deterministic_fixed_opponents_v108(seed_h)
        st_1 = game.new_initial_state()
        deck_rng_1 = np.random.RandomState(seed_h)
        advance_chance_nodes(st_1, deck_rng_1)
        f_1 = set()
        dec_1 = 0
        acts_1 = []
        bets_1 = []

        while not st_1.is_terminal():
            if st_1.is_chance_node():
                advance_chance_nodes(st_1, deck_rng_1)
            else:
                cp = st_1.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_1, cp)
                if cp == seat_test:
                    a = model_a_test.select_action(st_1, cp, l6, folded_players=f_1, hand_seed=seed_h, decision_idx=dec_1)
                    dec_1 += 1
                else:
                    opp = opps_1[cp if cp < seat_test else cp - 1]
                    a = opp.select_action(st_1, cp, l6)
                if a == 0: f_1.add(cp)
                eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_1, cp, a)
                acts_1.append((cp, a))
                bets_1.append((cp, eng_amt))
                st_1.apply_action(eng_amt)
        ret_1 = st_1.returns()[seat_test] / 100.0

        # Run 2 (동일 조건 독립 재실행)
        opps_2 = get_deterministic_fixed_opponents_v108(seed_h)
        st_2 = game.new_initial_state()
        deck_rng_2 = np.random.RandomState(seed_h)
        advance_chance_nodes(st_2, deck_rng_2)
        f_2 = set()
        dec_2 = 0
        acts_2 = []
        bets_2 = []

        while not st_2.is_terminal():
            if st_2.is_chance_node():
                advance_chance_nodes(st_2, deck_rng_2)
            else:
                cp = st_2.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_2, cp)
                if cp == seat_test:
                    a = model_a_test.select_action(st_2, cp, l6, folded_players=f_2, hand_seed=seed_h, decision_idx=dec_2)
                    dec_2 += 1
                else:
                    opp = opps_2[cp if cp < seat_test else cp - 1]
                    a = opp.select_action(st_2, cp, l6)
                if a == 0: f_2.add(cp)
                eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_2, cp, a)
                acts_2.append((cp, a))
                bets_2.append((cp, eng_amt))
                st_2.apply_action(eng_amt)
        ret_2 = st_2.returns()[seat_test] / 100.0

        diff = abs(ret_1 - ret_2)
        if diff > max_diff:
            max_diff = diff
        if acts_1 != acts_2:
            action_seq_mismatches += 1
        if bets_1 != bets_2:
            chip_flow_mismatches += 1
        if diff > 1e-9:
            regression_identical = False

    print(f"  * Self-Regression 100 Hands Test : {'PASSED (Bit-Exact 0.000000 BB)' if regression_identical else 'FAILED'}")
    print(f"  * Action Sequence Mismatches     : {action_seq_mismatches} / 100")
    print(f"  * Chip Flow Amount Mismatches    : {chip_flow_mismatches} / 100")
    print(f"  * Max Payoff Difference Detected : {max_diff:.8f} BB")

    # =================================================================
    # PHASE 3: 동일 행동 원장 불일치 완전 해부 & Facing-Bet 감사 (5,000 Hands)
    # =================================================================
    print("\n[*] Phase 3: 5,000 Hands Ledger Audit & Exact Path Reconciliation...")
    total_hands = 5000
    model_a = V108_Deterministic_TempHybrid(bp_agent, num_sims=100)
    model_b = V108_Deterministic_QLearningOffense(bp_agent, q_exploiter, num_sims=100)

    results_a = []
    results_b = []
    
    all_path_identical_hands = []
    hero_only_identical_hands = []
    diverged_hands = []

    facing_bet_qlearn_calls = 0
    facing_bet_temp_calls = 0
    free_action_qlearn_calls = 0

    causal_rollouts = []

    t_p3_start = time.time()
    for h in range(total_hands):
        seed_h = 20261010 + h
        seat_test = h % 5

        # 1. Model A 실행
        opps_a = get_deterministic_fixed_opponents_v108(seed_h)
        st_a = game.new_initial_state()
        deck_rng_a = np.random.RandomState(seed_h)
        advance_chance_nodes(st_a, deck_rng_a)
        f_a = set()
        dec_a = 0
        path_a = []
        hero_acts_a = []

        while not st_a.is_terminal():
            if st_a.is_chance_node():
                advance_chance_nodes(st_a, deck_rng_a)
            else:
                cp = st_a.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                if cp == seat_test:
                    a = model_a.select_action(st_a, cp, l6, folded_players=f_a, hand_seed=seed_h, decision_idx=dec_a)
                    dec_a += 1
                    hero_acts_a.append(a)
                else:
                    opp = opps_a[cp if cp < seat_test else cp - 1]
                    a = opp.select_action(st_a, cp, l6)
                if a == 0: f_a.add(cp)
                eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a)
                path_a.append((cp, a, eng_amt))
                st_a.apply_action(eng_amt)
        ret_a = st_a.returns()[seat_test] / 100.0
        results_a.append(ret_a)

        # 2. Model B 실행
        opps_b = get_deterministic_fixed_opponents_v108(seed_h)
        st_b = game.new_initial_state()
        deck_rng_b = np.random.RandomState(seed_h)
        advance_chance_nodes(st_b, deck_rng_b)
        f_b = set()
        dec_b = 0
        path_b = []
        hero_acts_b = []
        hand_diverged = False

        while not st_b.is_terminal():
            if st_b.is_chance_node():
                advance_chance_nodes(st_b, deck_rng_b)
            else:
                cp = st_b.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                if cp == seat_test:
                    spent = extract_player_spent(st_b)
                    to_call = max(spent) - spent[cp]

                    # 가상으로 Model A의 행동 산출
                    act_a_hypo = model_a.select_action(st_b, cp, l6, folded_players=f_b, hand_seed=seed_h, decision_idx=dec_b)
                    act_b = model_b.select_action(st_b, cp, l6, folded_players=f_b, hand_seed=seed_h, decision_idx=dec_b)
                    dec_b += 1
                    hero_acts_b.append(act_b)

                    # 정책 라우팅 통계 수집
                    if to_call > 0 and 0 in l6:
                        facing_bet_temp_calls += 1
                    else:
                        free_action_qlearn_calls += 1

                    if act_b != act_a_hypo and not hand_diverged:
                        hand_diverged = True
                        diverged_hands.append(h)

                        # [Phase 5 반사실적 상태 복제 롤아웃 수행]
                        # Branch A: act_a_hypo 주입
                        st_cl_a = st_b.clone()
                        f_ca = set(f_b)
                        opps_cl_a = get_deterministic_fixed_opponents_v108(seed_h + 500000)
                        eng_a = True6ActionAdapter.map_6action_to_engine_amount(st_cl_a, cp, act_a_hypo)
                        st_cl_a.apply_action(eng_a)
                        if act_a_hypo == 0: f_ca.add(cp)
                        rng_ca = np.random.RandomState(seed_h + 900000)
                        dec_ca = dec_b
                        while not st_cl_a.is_terminal():
                            if st_cl_a.is_chance_node():
                                advance_chance_nodes(st_cl_a, rng_ca)
                            else:
                                p_ca = st_cl_a.current_player()
                                l_ca = True6ActionAdapter.get_legal_6actions(st_cl_a, p_ca)
                                if p_ca == seat_test:
                                    a_ca = model_a.select_action(st_cl_a, p_ca, l_ca, folded_players=f_ca, hand_seed=seed_h, decision_idx=dec_ca)
                                    dec_ca += 1
                                else:
                                    opp_ca = opps_cl_a[p_ca if p_ca < seat_test else p_ca - 1]
                                    a_ca = opp_ca.select_action(st_cl_a, p_ca, l_ca)
                                if a_ca == 0: f_ca.add(p_ca)
                                st_cl_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_cl_a, p_ca, a_ca))
                        pay_a = st_cl_a.returns()[seat_test] / 100.0

                        # Branch B: act_b 주입
                        st_cl_b = st_b.clone()
                        f_cb = set(f_b)
                        opps_cl_b = get_deterministic_fixed_opponents_v108(seed_h + 500000)
                        eng_b = True6ActionAdapter.map_6action_to_engine_amount(st_cl_b, cp, act_b)
                        st_cl_b.apply_action(eng_b)
                        if act_b == 0: f_cb.add(cp)
                        rng_cb = np.random.RandomState(seed_h + 900000)
                        dec_cb = dec_b
                        while not st_cl_b.is_terminal():
                            if st_cl_b.is_chance_node():
                                advance_chance_nodes(st_cl_b, rng_cb)
                            else:
                                p_cb = st_cl_b.current_player()
                                l_cb = True6ActionAdapter.get_legal_6actions(st_cl_b, p_cb)
                                if p_cb == seat_test:
                                    a_cb = model_b.select_action(st_cl_b, p_cb, l_cb, folded_players=f_cb, hand_seed=seed_h, decision_idx=dec_cb)
                                    dec_cb += 1
                                else:
                                    opp_cb = opps_cl_b[p_cb if p_cb < seat_test else p_cb - 1]
                                    a_cb = opp_cb.select_action(st_cl_b, p_cb, l_cb)
                                if a_cb == 0: f_cb.add(p_cb)
                                st_cl_b.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_cl_b, p_cb, a_cb))
                        pay_b = st_cl_b.returns()[seat_test] / 100.0

                        causal_rollouts.append({
                            "hand_id": h,
                            "cp": cp,
                            "to_call": to_call,
                            "act_a": act_a_hypo,
                            "act_b": act_b,
                            "pay_a": pay_a,
                            "pay_b": pay_b,
                            "causal_diff": pay_b - pay_a
                        })
                    a = act_b
                else:
                    opp = opps_b[cp if cp < seat_test else cp - 1]
                    a = opp.select_action(st_b, cp, l6)
                if a == 0: f_b.add(cp)
                eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a)
                path_b.append((cp, a, eng_amt))
                st_b.apply_action(eng_amt)
        ret_b = st_b.returns()[seat_test] / 100.0
        results_b.append(ret_b)

        # 경로 및 손익 일치성 분류
        if path_a == path_b:
            all_path_identical_hands.append((h, ret_a, ret_b))
        elif hero_acts_a == hero_acts_b:
            hero_only_identical_hands.append((h, ret_a, ret_b))

    t_p3_elapsed = time.time() - t_p3_start

    # 결과 분석
    sum_a = sum(results_a)
    sum_b = sum(results_b)
    net_diff = sum_b - sum_a

    # All-Path Identical Hands 손익 차이
    path_id_diffs = [r_b - r_a for _, r_a, r_b in all_path_identical_hands]
    max_path_id_err = max([abs(d) for d in path_id_diffs]) if path_id_diffs else 0.0
    sum_path_id_diff = sum(path_id_diffs)

    # Hero-Only Identical Hands 손익 차이
    hero_id_diffs = [r_b - r_a for _, r_a, r_b in hero_only_identical_hands]
    sum_hero_id_diff = sum(hero_id_diffs)

    print(f"\n[Phase 3 & 5 결과 요약 - 총 {total_hands}핸드, 소요시간: {t_p3_elapsed:.1f}s]:")
    print(f"  * Model A 총 손익: {sum_a:+.2f} BB ({sum_a/total_hands*100:+.2f} bb/100)")
    print(f"  * Model B 총 손익: {sum_b:+.2f} BB ({sum_b/total_hands*100:+.2f} bb/100)")
    print(f"  * 전체 손익 차이: {net_diff:+.2f} BB ({net_diff/total_hands*100:+.2f} bb/100)")
    print(f"  * 1) 전체 경로 100% 일치 핸드: {len(all_path_identical_hands)}개")
    print(f"       -> 손익 차이 합계: {sum_path_id_diff:.8f} BB (최대 오차: {max_path_id_err:.8f} BB) ==> {'PERFECT 0.000000 BB' if max_path_id_err < 1e-6 else 'MISMATCH'}")
    print(f"  * 2) Hero 행동만 일치한 핸드: {len(hero_only_identical_hands)}개 (손익 차이: {sum_hero_id_diff:+.2f} BB)")
    print(f"  * 3) 행동 분기 핸드: {len(diverged_hands)}개")
    print(f"  * 정책 라우팅 감사:")
    print(f"       - Facing-Bet 수비 상황 Q-Learning 호출: {facing_bet_qlearn_calls}회 (0회 규격 일치)")
    print(f"       - Facing-Bet 수비 상황 Temp-Hybrid 호출: {facing_bet_temp_calls}회")
    print(f"       - Free-Action 공격 상황 Q-Learning 호출: {free_action_qlearn_calls}회")

    # =================================================================
    # PHASE 5: 결정론적 상태 복제 반사실적 인과 효과 분석
    # =================================================================
    causal_diffs = [cr["causal_diff"] for cr in causal_rollouts]
    n_causal = len(causal_diffs)
    sum_causal = sum(causal_diffs)
    mean_causal = float(np.mean(causal_diffs)) if n_causal > 0 else 0.0
    se_causal = float(stats.sem(causal_diffs)) if n_causal > 1 else 0.0
    t_causal = mean_causal / se_causal if se_causal > 0 else 0.0
    p_causal = float(stats.t.sf(t_causal, df=n_causal - 1)) if n_causal > 1 else 1.0
    ci_95 = (mean_causal - 1.96 * se_causal, mean_causal + 1.96 * se_causal)

    print(f"\n[Phase 5: 반사실적 상태 복제 인과 효과 (N={n_causal})]:")
    print(f"  * 누적 인과 손익: {sum_causal:+.2f} BB")
    print(f"  * 결정당 평균 인과 효과: {mean_causal:+.4f} ± {se_causal:.4f} BB/decision")
    print(f"  * 95% 신뢰구간: [{ci_95[0]:+.4f}, {ci_95[1]:+.4f}] BB")
    print(f"  * 검정 통계량: t = {t_causal:.4f}, p = {p_causal:.4f} ({'p < 0.05 유의' if p_causal < 0.05 else '경향성 p < 0.10' if p_causal < 0.10 else '미달'})")

    # 데이터 저장
    p3_p5_summary = {
        "total_hands": total_hands,
        "model_a_bb100": sum_a / total_hands * 100,
        "model_b_bb100": sum_b / total_hands * 100,
        "net_diff_bb": net_diff,
        "all_path_identical_count": len(all_path_identical_hands),
        "all_path_identical_diff": sum_path_id_diff,
        "max_path_identical_error": max_path_id_err,
        "hero_only_identical_count": len(hero_only_identical_hands),
        "hero_only_identical_diff": sum_hero_id_diff,
        "diverged_hands_count": len(diverged_hands),
        "facing_bet_qlearn_calls": facing_bet_qlearn_calls,
        "facing_bet_temp_calls": facing_bet_temp_calls,
        "free_action_qlearn_calls": free_action_qlearn_calls,
        "causal_rollouts_n": n_causal,
        "causal_sum_bb": sum_causal,
        "causal_mean_bb": mean_causal,
        "causal_se_bb": se_causal,
        "causal_t_stat": t_causal,
        "causal_p_value": p_causal,
        "causal_ci_95": ci_95
    }

    out_file = "v108/data/phase2_3_5_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(p3_p5_summary, f, indent=2)
    print(f"[+] Phase 2, 3, 5 분석 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
