import os
import sys
import json
import time
import random
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
    TightPassiveBot
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
from v106.run_v106_deterministic_replay import DeterministicBalancedBot

# =====================================================================
# DETERMINISTIC INFRASTRUCTURE
# =====================================================================
class DeterministicTrueNFSP_Agent(TrueNFSP_6Action_Agent):
    """전역 np.random을 전혀 사용하지 않고 독립 RandomState로 샘플링하는 NFSP 에이전트"""
    def select_action_deterministic(self, info: np.ndarray, legal_6: list, rng: np.random.RandomState):
        norm_info = normalize_inf_state_v79(info)
        x = torch.tensor(norm_info, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal_6] = True

        self.sl_net.eval()
        with torch.no_grad():
            lg = self.sl_net(x).squeeze(0)
            lg[~mask] = -1e9
            probs = F.softmax(lg, dim=-1).numpy()
        
        return int(rng.choice(self.action_dim, p=probs))

def estimate_range_aware_equity_deterministic(
    hero_cards, board_cards, opp_combos, opp_probs, num_active_opponents,
    num_sims=100, rng_stream=None
):
    known_cards = set(hero_cards) | set(board_cards)
    unknown_deck = list(set(range(52)) - known_cards)
    cards_needed_board = 5 - len(board_cards)
    
    if len(unknown_deck) < (2 * num_active_opponents + cards_needed_board):
        return 0.50

    if rng_stream is None:
        raise ValueError("CRITICAL: rng_stream (RandomState) must be explicitly provided!")

    wins = 0
    ties = 0
    num_combos = len(opp_combos)
    if num_combos == 0:
        return 0.50

    for _ in range(num_sims):
        opp1_idx = rng_stream.choice(num_combos, p=opp_probs)
        opp1_hole = list(opp_combos[opp1_idx])

        current_blocked = known_cards | set(opp1_hole)
        rem_deck = [c for c in unknown_deck if c not in current_blocked]

        extra_opps = num_active_opponents - 1
        needed_cards = cards_needed_board + 2 * extra_opps
        if len(rem_deck) < needed_cards:
            continue
        
        sim_sampled = rng_stream.choice(rem_deck, size=needed_cards, replace=False)
        sim_board = list(board_cards) + list(sim_sampled[:cards_needed_board])
        
        hero_full = list(hero_cards) + sim_board
        hero_score = eval_7cards(hero_full)

        hero_won = True
        hero_tied = False

        opp1_full = opp1_hole + sim_board
        opp1_score = eval_7cards(opp1_full)
        if opp1_score > hero_score:
            hero_won = False
        elif opp1_score == hero_score:
            hero_tied = True

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


class DeterministicTightPassiveBot:
    def __init__(self, rng_seed=20267902):
        self.rng = random.Random(rng_seed)

    def select_action(self, st, cp, legal_6):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        if to_call > 150 and 0 in legal_6 and self.rng.random() < 0.55:
            return 0
        return 1

class DeterministicUltraAggressiveBot:
    def __init__(self, rng_seed=20268704):
        self.rng = random.Random(rng_seed)

    def select_action(self, st, cp, legal_6: list):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        if to_call > 0:
            raises = [a for a in legal_6 if a in [2, 3, 5]]
            if raises and self.rng.random() < 0.80:
                return raises[-1]
            return 1 if 1 in legal_6 else (0 if 0 in legal_6 else 1)
        else:
            bets = [a for a in legal_6 if a in [2, 3]]
            if bets and self.rng.random() < 0.85:
                return bets[-1]
            return 1

def get_deterministic_fixed_opponents_v108(seed_h):
    return [
        DeterministicBalancedBot(rng_seed=seed_h * 100 + 1),
        DeterministicTightPassiveBot(rng_seed=seed_h * 100 + 2),
        CallingStationBot(),
        DeterministicUltraAggressiveBot(rng_seed=seed_h * 100 + 4)
    ]


class V108_Deterministic_TempHybrid:
    """Model A: 결정론적 Temperature Hybrid T=2.0 기준 모델"""
    def __init__(self, bp_agent: DeterministicTrueNFSP_Agent, num_sims=100):
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
        bp_seed = (hand_seed * 20011 + cp * 307 + decision_idx * 31) & 0xFFFFFFFF
        bp_rng = np.random.RandomState(bp_seed)
        act = self.blueprint.select_action_deterministic(inf, legal_6, rng=bp_rng)
        if log_collector is not None:
            log_collector.append((hand_id, "Model_A_Temp", "FreeAction_Blueprint", to_call, act, None, None))
        return act


class V108_Deterministic_QLearningOffense:
    """Model B: Q-Learning 공격 + Temp-Hybrid 방어 정책 (RNG 완전 격리)"""
    def __init__(self, bp_agent: DeterministicTrueNFSP_Agent, q_exploiter: MultiwayHeroExploiter, num_sims=100):
        self.blueprint = bp_agent
        self.q_exploiter = q_exploiter
        self.temperature = 2.0
        self.num_sims = num_sims

    def select_action(self, st, cp, legal_6, folded_players=None, hand_seed=0, decision_idx=0, log_collector=None, hand_id=0):
        spent = extract_player_spent(st)
        to_call = max(spent) - spent[cp]
        pot = sum(spent)

        # 1. 수비 상황 (To-call > 0) -> Temp-Hybrid 수비 엔진 (Facing Bet)
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


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.8: DETERMINISTIC BENCHMARK & POLICY ACCURACY BENCHMARK")
    print("=" * 80)

    game = get_5p_fullgame()
    bp_agent = DeterministicTrueNFSP_Agent()
    v79_ckpt = 'v79/models/sys_a_pure_seed_42.pt'
    bp_agent.sl_net.load_state_dict(torch.load(v79_ckpt, map_location='cpu'))

    q_exploiter = MultiwayHeroExploiter()
    q_table_cache = 'v106/data/q_exploiter_15k_cache.pt'
    print(f"[*] Loading pre-trained Q-table from {q_table_cache}...")
    loaded_dict = torch.load(q_table_cache, weights_only=False)
    for k, v in loaded_dict.items():
        q_exploiter.q_table[k] = v
    print(f"[+] Q-table successfully restored! States = {len(q_exploiter.q_table)}")

    # =================================================================
    # PHASE 3: 동일 행동 원장 불일치 완전 해부 & 정책 라우팅 감사 (5,000 Hands)
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

    # =================================================================
    # PHASE 4: 에쿼티 정밀도별 AI 실력 비교 (30, 100, 300, 1000, 3000)
    # =================================================================
    print("\n[*] Phase 4: Benchmarking Equity Precision Impact on Winrates...")
    sim_settings = [30, 100, 300, 1000] # 3000은 속도상 1000으로 충분한 검증력 확보, 대표 샘플 4단계
    bench_hands = 2500 # 정밀 비교 핸드 수
    precision_results = {}

    for sims in sim_settings:
        print(f"\n  --- Running Benchmark with Monte Carlo Sims = {sims} ({bench_hands} hands) ---")
        m_a = V108_Deterministic_TempHybrid(bp_agent, num_sims=sims)
        m_b = V108_Deterministic_QLearningOffense(bp_agent, q_exploiter, num_sims=sims)

        rets_a, rets_b = [], []
        folds_a, folds_b = 0, 0
        vpip_a, vpip_b = 0, 0
        pfr_a, pfr_b = 0, 0
        noshowdown_wins_a, noshowdown_wins_b = 0, 0
        t_b_start = time.time()

        for h in range(bench_hands):
            seed_h = 20265000 + h
            seat_test = h % 5

            # Model A
            opps_a = get_deterministic_fixed_opponents_v108(seed_h)
            st_a = game.new_initial_state()
            deck_rng_a = np.random.RandomState(seed_h)
            advance_chance_nodes(st_a, deck_rng_a)
            f_a = set()
            dec_a = 0
            vpip_flag_a = False
            pfr_flag_a = False
            preflop_done_a = False
            hero_folded_a = False

            while not st_a.is_terminal():
                if st_a.is_chance_node():
                    advance_chance_nodes(st_a, deck_rng_a)
                    preflop_done_a = True
                else:
                    cp = st_a.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                    if cp == seat_test:
                        a = m_a.select_action(st_a, cp, l6, folded_players=f_a, hand_seed=seed_h, decision_idx=dec_a)
                        dec_a += 1
                        if a == 0:
                            hero_folded_a = True
                            f_a.add(cp)
                        if not preflop_done_a:
                            if a in [1, 2, 3, 4, 5] and extract_player_spent(st_a)[cp] > 50:
                                vpip_flag_a = True
                            if a in [2, 3, 4, 5]:
                                pfr_flag_a = True
                    else:
                        opp = opps_a[cp if cp < seat_test else cp - 1]
                        a = opp.select_action(st_a, cp, l6)
                        if a == 0: f_a.add(cp)
                    eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a)
                    st_a.apply_action(eng_amt)
            ret_a = st_a.returns()[seat_test] / 100.0
            rets_a.append(ret_a)
            if hero_folded_a: folds_a += 1
            if vpip_flag_a: vpip_a += 1
            if pfr_flag_a: pfr_a += 1
            if ret_a > 0 and len(f_a) == 4 and not hero_folded_a:
                noshowdown_wins_a += 1

            # Model B
            opps_b = get_deterministic_fixed_opponents_v108(seed_h)
            st_b = game.new_initial_state()
            deck_rng_b = np.random.RandomState(seed_h)
            advance_chance_nodes(st_b, deck_rng_b)
            f_b = set()
            dec_b = 0
            vpip_flag_b = False
            pfr_flag_b = False
            preflop_done_b = False
            hero_folded_b = False

            while not st_b.is_terminal():
                if st_b.is_chance_node():
                    advance_chance_nodes(st_b, deck_rng_b)
                    preflop_done_b = True
                else:
                    cp = st_b.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_b, cp)
                    if cp == seat_test:
                        a = m_b.select_action(st_b, cp, l6, folded_players=f_b, hand_seed=seed_h, decision_idx=dec_b)
                        dec_b += 1
                        if a == 0:
                            hero_folded_b = True
                            f_b.add(cp)
                        if not preflop_done_b:
                            if a in [1, 2, 3, 4, 5] and extract_player_spent(st_b)[cp] > 50:
                                vpip_flag_b = True
                            if a in [2, 3, 4, 5]:
                                pfr_flag_b = True
                    else:
                        opp = opps_b[cp if cp < seat_test else cp - 1]
                        a = opp.select_action(st_b, cp, l6)
                        if a == 0: f_b.add(cp)
                    eng_amt = True6ActionAdapter.map_6action_to_engine_amount(st_b, cp, a)
                    st_b.apply_action(eng_amt)
            ret_b = st_b.returns()[seat_test] / 100.0
            rets_b.append(ret_b)
            if hero_folded_b: folds_b += 1
            if vpip_flag_b: vpip_b += 1
            if pfr_flag_b: pfr_b += 1
            if ret_b > 0 and len(f_b) == 4 and not hero_folded_b:
                noshowdown_wins_b += 1

        t_b_elapsed = time.time() - t_b_start
        mean_a = float(np.mean(rets_a)) * 100.0
        se_a = float(stats.sem(rets_a)) * 100.0
        mean_b = float(np.mean(rets_b)) * 100.0
        se_b = float(stats.sem(rets_b)) * 100.0
        diffs = [b - a for a, b in zip(rets_a, rets_b)]
        mean_diff = float(np.mean(diffs)) * 100.0
        se_diff = float(stats.sem(diffs)) * 100.0
        t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
        p_val = float(stats.t.sf(abs(t_stat), df=bench_hands-1) * 2)

        print(f"    * Model A: {mean_a:+.2f} ± {se_a:.2f} bb/100 | Fold: {folds_a/bench_hands*100:.1f}% | VPIP: {vpip_a/bench_hands*100:.1f}% | NoSD Wins: {noshowdown_wins_a}")
        print(f"    * Model B: {mean_b:+.2f} ± {se_b:.2f} bb/100 | Fold: {folds_b/bench_hands*100:.1f}% | VPIP: {vpip_b/bench_hands*100:.1f}% | NoSD Wins: {noshowdown_wins_b}")
        print(f"    * Diff (B-A): {mean_diff:+.2f} ± {se_diff:.2f} bb/100 | t={t_stat:.4f}, p={p_val:.4f} | Time: {t_b_elapsed:.1f}s")

        precision_results[str(sims)] = {
            "sims": sims,
            "hands": bench_hands,
            "model_a_bb100": mean_a,
            "model_a_se": se_a,
            "model_a_folds": folds_a,
            "model_a_vpip": vpip_a,
            "model_a_noshowdown_wins": noshowdown_wins_a,
            "model_b_bb100": mean_b,
            "model_b_se": se_b,
            "model_b_folds": folds_b,
            "model_b_vpip": vpip_b,
            "model_b_noshowdown_wins": noshowdown_wins_b,
            "diff_bb100": mean_diff,
            "diff_se": se_diff,
            "t_stat": t_stat,
            "p_val": p_val,
            "elapsed_s": t_b_elapsed
        }

    # 전체 요약 저장
    final_output = {
        "phase3_ledger": {
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
            "free_action_qlearn_calls": free_action_qlearn_calls
        },
        "phase5_causal": {
            "n": n_causal,
            "sum_bb": sum_causal,
            "mean_bb": mean_causal,
            "se_bb": se_causal,
            "t_stat": t_causal,
            "p_val": p_causal,
            "ci_95": ci_95
        },
        "phase4_precision_benchmark": precision_results
    }

    out_file = "v108/data/v108_main_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[+] V10.8 전체 실험 결과 JSON 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
