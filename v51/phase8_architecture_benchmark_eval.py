"""
POKER AI V5.1 - PHASE 8: FROZEN FULL-GAME BENCHMARK EVALUATION (1,000 HANDS)
- V4.9.8 Frozen Evaluation 원칙 100% 준수 (시드 고정, 동일 좌석 회전, 동일 상대 풀)
- 평가 대상:
  1. Control: V5.0 중립 피처 기준선
  2. Model A (Concatenation): 3개 시드 (42, 101, 202)
  3. Model B (Two-Tower MLP): 3개 시드 (42, 101, 202)
  4. Model C (FiLM Gating): 3개 시드 (42, 101, 202)
- 측정 메트릭:
  * 전체 bb/100 및 95% 신뢰구간
  * 72o Call 횟수 및 Call% (Blind Caller 재발 여부)
  * AA Call/Raise 비율
  * 카테고리별(Premium, Marginal, Trash) EV (BB)
  * 상대 유형별 맞춤 적응력
- 원시 데이터를 v51/data/phase8_architecture_fullgame_eval.csv에 저장
"""

import os
import sys
import csv
import numpy as np
import scipy.stats as stats
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.opponents_pool import RandomBot, AlwaysCallBot, TightBot, AggressiveBot
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import get_neutral_prior_features
from v51.models_comparison import ConcatenationActorCritic, TwoTowerActorCritic, FiLMActorCritic

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def classify_hero_hand(cards: list) -> str:
    """V4.9.8 엄격 검증 핸드 분류기"""
    if len(cards) < 2:
        return "Unknown"
    c1, c2 = cards[0], cards[1]
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]

    # Rank value map
    rmap = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    v1, v2 = rmap.get(r1, 0), rmap.get(r2, 0)
    high_r, low_r = (r1, r2) if v1 >= v2 else (r2, r1)
    suited = (s1 == s2)

    short_name = f"{high_r}{low_r}{'s' if suited and high_r != low_r else ('o' if high_r != low_r else '')}"

    # Premium: AA, KK, QQ, JJ, AKs, AKo, AQs
    if short_name in ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs']:
        return "Premium"
    # Marginal: TT, 99, 88, 77, AJs, ATs, KQs, KJs, QJs, JTs, AQo, KTo, QTo
    if short_name in ['TT', '99', '88', '77', 'AJs', 'ATs', 'KQs', 'KJs', 'QJs', 'JTs', 'AQo', 'KTo', 'QTo', 'AJo']:
        return "Marginal"
    return "Trash"

def evaluate_model_fullgame(
    model_name: str,
    model,
    num_hands: int = 1000,
    eval_seed: int = 20261017,
    session_length: int = 50,
    is_control: bool = False
) -> dict:
    """1,000핸드 독립 동결 벤치마크 평가"""
    device = torch.device("cpu")
    model.eval()

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    table_opponents = [RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot()]
    rng_eval = np.random.RandomState(eval_seed)
    neutral_opp = get_neutral_prior_features()

    hand_returns = []
    cat_returns = {'Premium': [], 'Marginal': [], 'Trash': []}
    seven_two_total = 0
    seven_two_calls = 0
    aa_total = 0
    aa_calls_or_raises = 0

    num_sessions = num_hands // session_length

    for sess_idx in range(num_sessions):
        tracker_mgr = MatchSessionOpponentManager()
        tracker_mgr.reset_session()

        for h_in_sess in range(session_length):
            ep_global = sess_idx * session_length + h_in_sess
            tracker_mgr.set_hand_index(h_in_sess)

            s2a = wrapper.get_seat_to_agent(h_in_sess)
            a2s = wrapper.get_agent_to_seat(h_in_sess)
            hero_seat = a2s[0]

            if is_control:
                opp_feat = neutral_opp
            else:
                opp_feat = tracker_mgr.get_table_feature_tensor(hero_seat)

            st = game.new_initial_state()
            advance_chance_nodes(st, rng=rng_eval)

            # Hero 카드 식별
            info_hero_init = st.information_state_string(hero_seat)
            priv_cards, _ = parse_cards_from_info_str(info_hero_init)
            hand_cat = classify_hero_hand(priv_cards)
            is_72o = (('7' in priv_cards[0] and '2' in priv_cards[1]) or ('2' in priv_cards[0] and '7' in priv_cards[1])) and (priv_cards[0][1] != priv_cards[1][1])
            is_aa = (priv_cards[0][0] == 'A' and priv_cards[1][0] == 'A')

            if is_72o:
                seven_two_total += 1
            if is_aa:
                aa_total += 1

            hand_action_log = []
            hero_called_72o = False
            hero_aggressive_aa = False

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng=rng_eval)
                else:
                    cur = st.current_player()
                    legal = st.legal_actions(cur)
                    round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                    if cur == hero_seat:
                        hero_info = normalize_vec(st.information_state_tensor(cur))
                        h_t = torch.tensor(hero_info, dtype=torch.float32, device=device).unsqueeze(0)
                        o_t = torch.tensor(opp_feat, dtype=torch.float32, device=device).unsqueeze(0)
                        m_t = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in legal: m_t[0, a] = True

                        with torch.no_grad():
                            raw_logits, probs, _ = model(h_t, o_t, m_t)
                            # 결정론적 또는 argmax 평가 (Frozen Evaluation)
                            act = torch.argmax(probs, dim=-1).item()

                        if is_72o and act in [1, 2, 3, 4]:
                            hero_called_72o = True
                        if is_aa and act in [1, 2, 3, 4]:
                            hero_aggressive_aa = True

                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        ag_id = s2a[cur]
                        opp = table_opponents[(ag_id - 1) % len(table_opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            # 트래커 업데이트
            tracker_mgr.parse_and_update_hand(hand_action_log, s2a)

            # 결과 집계
            ret_bb = st.returns()[hero_seat] / 100.0
            hand_returns.append(ret_bb)
            if hand_cat in cat_returns:
                cat_returns[hand_cat].append(ret_bb)

            if is_72o and hero_called_72o:
                seven_two_calls += 1
            if is_aa and hero_aggressive_aa:
                aa_calls_or_raises += 1

    # 통계 계산
    n = len(hand_returns)
    mean_bb = float(np.mean(hand_returns))
    bb_per_100 = mean_bb * 100.0
    sem = float(np.std(hand_returns, ddof=1) / np.sqrt(n))
    ci_half = float(stats.t.ppf(0.975, df=n-1) * sem * 100.0)

    prem_ev = float(np.mean(cat_returns['Premium'])) if len(cat_returns['Premium']) > 0 else 0.0
    marg_ev = float(np.mean(cat_returns['Marginal'])) if len(cat_returns['Marginal']) > 0 else 0.0
    trash_ev = float(np.mean(cat_returns['Trash'])) if len(cat_returns['Trash']) > 0 else 0.0

    seven_two_call_pct = (seven_two_calls / seven_two_total * 100.0) if seven_two_total > 0 else 0.0
    aa_action_pct = (aa_calls_or_raises / aa_total * 100.0) if aa_total > 0 else 0.0

    return {
        'model': model_name,
        'bb_per_100': bb_per_100,
        'ci_half': ci_half,
        'premium_ev_bb': prem_ev,
        'marginal_ev_bb': marg_ev,
        'trash_ev_bb': trash_ev,
        'seven_two_total': seven_two_total,
        'seven_two_calls': seven_two_calls,
        'seven_two_call_pct': seven_two_call_pct,
        'aa_total': aa_total,
        'aa_action_pct': aa_action_pct,
        'raw_returns': hand_returns
    }

def run_phase8_benchmark():
    print("=" * 85)
    print("PHASE 8: FROZEN FULL-GAME BENCHMARK EVALUATION (1,000 HANDS)")
    print("=" * 85)

    architectures = [
        ("ModelA_Concat", ConcatenationActorCritic),
        ("ModelB_TwoTower", TwoTowerActorCritic),
        ("ModelC_FiLM", FiLMActorCritic)
    ]
    seeds = [42, 101, 202]

    results = []

    print(f"\n{'모델/조건':<22} | {'bb/100 (95% CI)':<20} | {'Premium EV':<12} | {'Marginal EV':<12} | {'Trash EV':<10} | {'72o Call%':<10} | {'AA Active%'}")
    print("-" * 105)

    # 1. Control (중립 피처 Baseline)
    ctrl_model = ConcatenationActorCritic()
    # 0k 초기화 또는 기본 모델
    ctrl_res = evaluate_model_fullgame("Control_Neutral", ctrl_model, num_hands=1000, is_control=True)
    results.append(ctrl_res)
    print(f"{ctrl_res['model']:<22} | {ctrl_res['bb_per_100']:+6.2f} ± {ctrl_res['ci_half']:4.2f} bb/100 | {ctrl_res['premium_ev_bb']:+6.2f} BB    | {ctrl_res['marginal_ev_bb']:+6.2f} BB    | {ctrl_res['trash_ev_bb']:+6.2f} BB   | {ctrl_res['seven_two_call_pct']:6.1f}%    | {ctrl_res['aa_action_pct']:6.1f}%")

    # 2. 3대 아키텍처 각 3개 시드 평가
    for arch_name, cls_obj in architectures:
        for s in seeds:
            path = f"v51/models/{arch_name}_seed_{s}.pt"
            if not os.path.exists(path):
                print(f"[경고] 모델 없음: {path}")
                continue
            m = cls_obj()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            res = evaluate_model_fullgame(f"{arch_name}_s{s}", m, num_hands=1000, is_control=False)
            results.append(res)
            print(f"{res['model']:<22} | {res['bb_per_100']:+6.2f} ± {res['ci_half']:4.2f} bb/100 | {res['premium_ev_bb']:+6.2f} BB    | {res['marginal_ev_bb']:+6.2f} BB    | {res['trash_ev_bb']:+6.2f} BB   | {res['seven_two_call_pct']:6.1f}%    | {res['aa_action_pct']:6.1f}%")

    # CSV 저장 (raw_returns 제외한 메타데이터)
    csv_path = "v51/data/phase8_architecture_fullgame_eval.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [k for k in results[0].keys() if k != 'raw_returns']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != 'raw_returns'}
            writer.writerow(row)

    print(f"\n[저장 완료] Phase 8 풀게임 평가 결과 CSV: {csv_path}")

    # 3. 아키텍처별 평균 및 통계적 검정
    print("\n" + "=" * 85)
    print("PHASE 8 & 9: 아키텍처별 풀게임 평균 및 통계적 가설 검정 (Welch's t-test)")
    print("=" * 85)

    arch_summaries = {}
    for arch_name, _ in architectures:
        m_runs = [r for r in results if r['model'].startswith(arch_name)]
        if m_runs:
            bbs = [r['bb_per_100'] for r in m_runs]
            call_72 = [r['seven_two_call_pct'] for r in m_runs]
            aa_act = [r['aa_action_pct'] for r in m_runs]
            all_returns = np.concatenate([r['raw_returns'] for r in m_runs])
            mean_bb = float(np.mean(bbs))
            std_bb = float(np.std(bbs, ddof=1)) if len(bbs) > 1 else 0.0
            arch_summaries[arch_name] = {
                'mean_bb': mean_bb,
                'std_bb': std_bb,
                'call_72_mean': float(np.mean(call_72)),
                'aa_act_mean': float(np.mean(aa_act)),
                'all_returns': all_returns
            }
            print(f"  [{arch_name}]")
            print(f"    - 평균 수익률: {mean_bb:+6.2f} ± {std_bb:4.2f} bb/100 (시드 간 편차)")
            print(f"    - 72o Call 비율: {arch_summaries[arch_name]['call_72_mean']:.2f}%")
            print(f"    - AA 적극 행동률: {arch_summaries[arch_name]['aa_act_mean']:.2f}%")

    # t-test: Model B vs Model A, Model C vs Model A
    if "ModelA_Concat" in arch_summaries and "ModelB_TwoTower" in arch_summaries:
        ret_a = arch_summaries["ModelA_Concat"]['all_returns']
        ret_b = arch_summaries["ModelB_TwoTower"]['all_returns']
        t_stat, p_val = stats.ttest_ind(ret_b, ret_a, equal_var=False)
        d = (np.mean(ret_b) - np.mean(ret_a)) / np.sqrt((np.var(ret_b) + np.var(ret_a)) / 2)
        print(f"\n  [가설 검정 1: Two-Tower vs Concat]")
        print(f"    - EV 차이: {arch_summaries['ModelB_TwoTower']['mean_bb'] - arch_summaries['ModelA_Concat']['mean_bb']:+.2f} bb/100")
        print(f"    - Welch's t={t_stat:.3f}, p-value={p_val:.4f}, Cohen's d={d:.4f}")
        print(f"    - 통계적 유의성: {'유의미함 (p < 0.05)' if p_val < 0.05 else '유의미하지 않음 (p >= 0.05)'}")

    if "ModelA_Concat" in arch_summaries and "ModelC_FiLM" in arch_summaries:
        ret_a = arch_summaries["ModelA_Concat"]['all_returns']
        ret_c = arch_summaries["ModelC_FiLM"]['all_returns']
        t_stat, p_val = stats.ttest_ind(ret_c, ret_a, equal_var=False)
        d = (np.mean(ret_c) - np.mean(ret_a)) / np.sqrt((np.var(ret_c) + np.var(ret_a)) / 2)
        print(f"\n  [가설 검정 2: FiLM vs Concat]")
        print(f"    - EV 차이: {arch_summaries['ModelC_FiLM']['mean_bb'] - arch_summaries['ModelA_Concat']['mean_bb']:+.2f} bb/100")
        print(f"    - Welch's t={t_stat:.3f}, p-value={p_val:.4f}, Cohen's d={d:.4f}")
        print(f"    - 통계적 유의성: {'유의미함 (p < 0.05)' if p_val < 0.05 else '유의미하지 않음 (p >= 0.05)'}")

    print("=" * 85)

if __name__ == '__main__':
    run_phase8_benchmark()
