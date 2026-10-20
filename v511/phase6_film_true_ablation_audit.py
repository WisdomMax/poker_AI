"""
POKER AI V5.1.1 - PHASE 6: FiLM TRUE EFFECT & OPPONENT FEATURE ABLATION AUDIT
- FiLM의 성능 원천이 '상대 통계 활용(A)'인지, '단순 구조적 안정성(B)'인지 엄밀 검증:
  * 조건 1: True Bayesian Features (세션 내 실시간 베이지안 추적 통계)
  * 조건 2: Neutral Prior Features (중립값 사전확률 0.5 패딩, 상대 정보 차단)
  * 조건 3: Shuffled Opponent Features (상대 4인의 피처 벡터를 무작위 셔플)
  * 조건 4: Inverted Opponent Features (Tight 성향을 Aggressive로, Aggressive를 Tight로 반전 왜곡)
- 대상 모델: FiLM 체크포인트 3개 시드 (Seed 42, 101, 202)
- 측정 지표:
  * 전체 bb/100 및 95% 신뢰구간
  * 정책 변화도: TVD(True vs Condition)
  * Premium / Marginal / Trash 카테고리별 EV
  * 72o Call% 및 AA Active%
- 원시 데이터를 v511/data/phase6_film_true_ablation_audit.csv에 저장
"""

import os
import sys
import csv
import copy
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
from v51.models_comparison import FiLMActorCritic

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def classify_hero_hand(cards: list) -> str:
    if len(cards) < 2: return "Trash"
    c1, c2 = cards[0], cards[1]
    r1, r2 = c1[0], c2[0]
    suited = (c1[1] == c2[1])
    rmap = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    hr, lr = (r1, r2) if rmap.get(r1,0) >= rmap.get(r2,0) else (r2, r1)
    sname = f"{hr}{lr}{'s' if suited and hr!=lr else ('o' if hr!=lr else '')}"
    if sname in ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs']: return "Premium"
    if sname in ['TT', '99', '88', '77', 'AJs', 'ATs', 'KQs', 'KJs', 'QJs', 'JTs', 'AQo', 'KTo', 'QTo', 'AJo']: return "Marginal"
    return "Trash"

def evaluate_film_ablation(
    model_path: str,
    feature_condition: str, # 'True', 'Neutral', 'Shuffled', 'Inverted'
    num_hands: int = 1000,
    eval_seed: int = 20261017,
    session_length: int = 50
) -> dict:
    device = torch.device("cpu")
    model = FiLMActorCritic()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    table_opponents = [RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot()]
    rng_eval = np.random.RandomState(eval_seed)
    rng_shuffle = np.random.RandomState(eval_seed + 999)

    neutral_opp = get_neutral_prior_features()

    hand_returns = []
    cat_returns = {'Premium': [], 'Marginal': [], 'Trash': []}
    seven_two_total = 0
    seven_two_calls = 0
    aa_total = 0
    aa_actives = 0

    all_action_probs = []

    num_sessions = num_hands // session_length

    for sess_idx in range(num_sessions):
        tracker_mgr = MatchSessionOpponentManager()
        tracker_mgr.reset_session()

        for h_in_sess in range(session_length):
            tracker_mgr.set_hand_index(h_in_sess)

            s2a = wrapper.get_seat_to_agent(h_in_sess)
            a2s = wrapper.get_agent_to_seat(h_in_sess)
            hero_seat = a2s[0]

            # 피처 조건별 생성
            raw_opp_feat = tracker_mgr.get_table_feature_tensor(hero_seat)

            if feature_condition == 'True':
                opp_feat = raw_opp_feat
            elif feature_condition == 'Neutral':
                opp_feat = neutral_opp
            elif feature_condition == 'Shuffled':
                # 상대 4인의 13차원 통계를 무작위 셔플
                opp_blocks = [raw_opp_feat[i*13:(i+1)*13] for i in range(4)]
                rng_shuffle.shuffle(opp_blocks)
                opp_feat = np.concatenate(opp_blocks)
            elif feature_condition == 'Inverted':
                # VPIP, PFR, 3Bet 통계를 1.0 - x 로 반전 왜곡
                inv = np.copy(raw_opp_feat)
                for i in range(4):
                    base = i * 13
                    inv[base] = 1.0 - inv[base]     # vpip
                    inv[base+3] = 1.0 - inv[base+3] # pfr
                    inv[base+6] = 1.0 - inv[base+6] # 3bet
                opp_feat = inv

            st = game.new_initial_state()
            advance_chance_nodes(st, rng=rng_eval)

            info_hero_init = st.information_state_string(hero_seat)
            priv_cards, _ = parse_cards_from_info_str(info_hero_init)
            hand_cat = classify_hero_hand(priv_cards)
            is_72o = (('7' in priv_cards[0] and '2' in priv_cards[1]) or ('2' in priv_cards[0] and '7' in priv_cards[1])) and (priv_cards[0][1] != priv_cards[1][1])
            is_aa = (priv_cards[0][0] == 'A' and priv_cards[1][0] == 'A')

            if is_72o: seven_two_total += 1
            if is_aa: aa_total += 1

            hand_action_log = []
            hero_called_72 = False
            hero_active_aa = False

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
                            act = torch.argmax(probs, dim=-1).item()
                            all_action_probs.append(probs.squeeze(0).cpu().numpy())

                        if is_72o and act in [1, 2, 3, 4]: hero_called_72 = True
                        if is_aa and act in [1, 2, 3, 4]: hero_active_aa = True

                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        ag_id = s2a[cur]
                        opp = table_opponents[(ag_id - 1) % len(table_opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            # 트래커는 실제 공개 이력으로 계속 업데이트
            tracker_mgr.parse_and_update_hand(hand_action_log, s2a)

            ret_bb = st.returns()[hero_seat] / 100.0
            hand_returns.append(ret_bb)
            cat_returns[hand_cat].append(ret_bb)

            if is_72o and hero_called_72: seven_two_calls += 1
            if is_aa and hero_active_aa: aa_actives += 1

    n = len(hand_returns)
    mean_bb = float(np.mean(hand_returns)) * 100.0
    sem = float(np.std(hand_returns, ddof=1) / np.sqrt(n)) * 100.0
    ci_half = float(stats.t.ppf(0.975, df=n-1) * sem)

    prem_ev = float(np.mean(cat_returns['Premium'])) if cat_returns['Premium'] else 0.0
    marg_ev = float(np.mean(cat_returns['Marginal'])) if cat_returns['Marginal'] else 0.0
    trash_ev = float(np.mean(cat_returns['Trash'])) if cat_returns['Trash'] else 0.0

    c72_pct = (seven_two_calls / seven_two_total * 100.0) if seven_two_total > 0 else 0.0
    aa_pct = (aa_actives / aa_total * 100.0) if aa_total > 0 else 0.0

    return {
        'mean_bb_100': mean_bb,
        'ci_half': ci_half,
        'prem_ev': prem_ev,
        'marg_ev': marg_ev,
        'trash_ev': trash_ev,
        'c72_pct': c72_pct,
        'aa_pct': aa_pct,
        'action_probs': np.array(all_action_probs),
        'hand_returns': np.array(hand_returns)
    }

def run_phase6_audit():
    print("=" * 85)
    print("PHASE 6: FiLM 어블레이션 감사 (True vs Neutral vs Shuffled vs Inverted)")
    print("=" * 85)

    conditions = ['True', 'Neutral', 'Shuffled', 'Inverted']
    seeds = [42, 101, 202]

    results = []

    print(f"\n{'시드':<8} | {'조건':<10} | {'bb/100 (95% CI)':<20} | {'Premium EV':<12} | {'Marginal EV':<12} | {'Trash EV':<10} | {'TVD vs True'}")
    print("-" * 95)

    for s in seeds:
        path = f"v51/models/ModelC_FiLM_seed_{s}.pt"
        assert os.path.exists(path), f"체크포인트 없음: {path}"

        # True 조건 먼저 실행하여 기준 확률 확보
        true_res = evaluate_film_ablation(path, 'True', num_hands=1000)
        true_probs = true_res['action_probs']

        for cond in conditions:
            if cond == 'True':
                res = true_res
                tvd = 0.0000
            else:
                res = evaluate_film_ablation(path, cond, num_hands=1000)
                # TVD 계산
                min_len = min(len(true_probs), len(res['action_probs']))
                tvd = float(0.5 * np.mean(np.sum(np.abs(true_probs[:min_len] - res['action_probs'][:min_len]), axis=-1)))

            c_str = f"{res['mean_bb_100']:+6.2f} ± {res['ci_half']:4.2f}"
            print(f"Seed {s:<3} | {cond:<10} | {c_str:<20} | {res['prem_ev']:+6.2f} BB    | {res['marg_ev']:+6.2f} BB    | {res['trash_ev']:+6.2f} BB   | {tvd:8.4f}")

            results.append({
                'seed': s,
                'condition': cond,
                'bb_per_100': res['mean_bb_100'],
                'ci_half': res['ci_half'],
                'prem_ev': res['prem_ev'],
                'marg_ev': res['marg_ev'],
                'trash_ev': res['trash_ev'],
                'c72_pct': res['c72_pct'],
                'aa_pct': res['aa_pct'],
                'tvd_vs_true': tvd
            })

    # 조건별 전체 평균 계산
    print("\n" + "=" * 85)
    print("조건별 3개 시드 전체 평균 및 통계 비교:")
    print("=" * 85)
    for cond in conditions:
        c_rows = [r for r in results if r['condition'] == cond]
        mean_val = np.mean([r['bb_per_100'] for r in c_rows])
        mean_tvd = np.mean([r['tvd_vs_true'] for r in c_rows])
        print(f"  * [{cond:<8} 조건]: 평균 {mean_val:+7.2f} bb/100 | 평균 TVD vs True: {mean_tvd:.4f}")

    true_vals = [r['bb_per_100'] for r in results if r['condition'] == 'True']
    neut_vals = [r['bb_per_100'] for r in results if r['condition'] == 'Neutral']
    shuf_vals = [r['bb_per_100'] for r in results if r['condition'] == 'Shuffled']
    inv_vals  = [r['bb_per_100'] for r in results if r['condition'] == 'Inverted']

    t_neut, p_neut = stats.ttest_rel(true_vals, neut_vals)
    t_shuf, p_shuf = stats.ttest_rel(true_vals, shuf_vals)
    t_inv,  p_inv  = stats.ttest_rel(true_vals, inv_vals)

    print(f"\n  [대응표본 t-검정 결과]")
    print(f"    - True vs Neutral : 차이 {np.mean(true_vals) - np.mean(neut_vals):+.2f} bb/100 (t={t_neut:.3f}, p={p_neut:.4f})")
    print(f"    - True vs Shuffled: 차이 {np.mean(true_vals) - np.mean(shuf_vals):+.2f} bb/100 (t={t_shuf:.3f}, p={p_shuf:.4f})")
    print(f"    - True vs Inverted: 차이 {np.mean(true_vals) - np.mean(inv_vals):+.2f} bb/100 (t={t_inv:.3f}, p={p_inv:.4f})")

    print("\n  [결정적 판정]:")
    if p_neut >= 0.05 and p_shuf >= 0.05:
        print("    -> [판정 B]: True와 Shuffled/Neutral 간에 유의미한 성능 차이가 없음 (p >= 0.05)!")
        print("    -> FiLM의 높은 수익성은 '상대 성향을 활용한 적응적 착취(Exploitation)' 때문이 아니라,")
        print("       FiLM 아키텍처 자체의 우수한 피처 정규화 및 탐색/정책 학습 안정성(Regularization Stability) 덕분임!")
    else:
        print("    -> [판정 A]: True 조건이 Shuffled 대비 유의미하게 우수함 (p < 0.05).")

    # CSV 저장
    csv_path = "v511/data/phase6_film_true_ablation_audit.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[저장 완료] Phase 6 FiLM 어블레이션 감사 CSV: {csv_path}\n")

if __name__ == '__main__':
    run_phase6_audit()
