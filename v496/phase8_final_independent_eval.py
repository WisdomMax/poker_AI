"""
POKER AI V4.9.6 - PHASE 8: Final Independent Generalization Evaluation & Statistical Verification
- 5인 No-Limit Texas Hold'em (100BB 스택, 5-Action)
- 비학습 상대 풀 및 독립 평가 시드 (Seed 99999, 1,000판 대국)
- 측정 메트릭:
  * 전체 bb/100 (공식: 100 * Total Net Chips / (100 * Hands))
  * 포지션별 bb/100 (BTN, CO, UTG, SB, BB)
  * 상대 봇별 bb/100 (Random, Tight, Aggressive, Adaptive, Selective, BluffCatcher 등)
  * 프리미엄 / 마지널 / 트래시 핸드별 기대수익 (bb/hand)
  * 물리적 손익 상한 [-100, +400] BB 및 테이블 Zero-Sum 자동 Assertion
  * 모델 간 Paired Difference 및 95% 신뢰구간, p-value 산출
"""

import os
import sys
import math
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
from scipy import stats
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return, validate_mean_ev
from v496.stable_ppo import StableHoldemActorCritic5A

FINAL_HANDS = 1000
FINAL_EVAL_SEED = 99999

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def classify_hand_tier(card_str: str) -> str:
    """홀카드 문자열(예: 'AsAh')을 프리미엄/마지널/트래시로 분류"""
    if not card_str or len(card_str) < 4:
        return 'trash'
    r1, s1 = card_str[0], card_str[1]
    r2, s2 = card_str[2], card_str[3]
    rank_order = "23456789TJQKA"
    try:
        idx1, idx2 = rank_order.index(r1), rank_order.index(r2)
    except:
        return 'trash'
    suited = (s1 == s2)
    high_idx = max(idx1, idx2)
    low_idx = min(idx1, idx2)
    is_pair = (high_idx == low_idx)

    # 프리미엄: AA, KK, QQ, JJ, AK
    if is_pair and high_idx >= 9: # JJ, QQ, KK, AA
        return 'premium'
    if high_idx == 12 and low_idx == 11: # AK
        return 'premium'

    # 마지널: TT, 99, 88, AQ, AJ, AT, KQ, KJ, QJ, JTs, T9s, 98s
    if is_pair and high_idx >= 6: # 88, 99, TT
        return 'marginal'
    if high_idx >= 10 and low_idx >= 8:
        return 'marginal'
    if suited and high_idx - low_idx == 1 and high_idx >= 7:
        return 'marginal'

    return 'trash'

def evaluate_model_independent(
    model: StableHoldemActorCritic5A,
    game: pyspiel.Game,
    normalize_input: bool,
    device: torch.device,
    model_name: str
) -> Dict:
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    model.eval()

    total_chips = 0.0
    pos_returns = defaultdict(list)
    tier_returns = defaultdict(list)
    action_counts = defaultdict(int)
    total_actions = 0

    rng_eval = np.random.RandomState(FINAL_EVAL_SEED)
    rng_opp = random.Random(FINAL_EVAL_SEED + 101)

    for h in range(FINAL_HANDS):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        pos_name = ['BTN', 'SB', 'BB', 'UTG', 'CO'][hero_seat]
        table_opps = rng_opp.sample(opponents, 4)

        st = game.new_initial_state()
        hero_hand_str = ""

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_eval)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    if not hero_hand_str:
                        # 홀카드 파싱
                        info_s = st.information_state_string(cur)
                        hero_hand_str = parse_cards_from_info_str(info_s)

                    vec = st.information_state_tensor(cur)
                    if normalize_input: vec = normalize_vec(vec)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True

                    act, _, _, _ = model.sample_action(x, m, deterministic=False)
                    action_counts[act] += 1
                    total_actions += 1
                else:
                    ag_id = s2a[cur]
                    opp = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp.act(st, cur, legal)
                st.apply_action(act)

        returns_all = st.returns()
        # 제로섬 검증
        assert abs(sum(returns_all)) < 1e-3, f"Zero-Sum 위반: {sum(returns_all)}"

        ret_chips = returns_all[hero_seat]
        validate_chips_return(ret_chips, hand_idx=h)
        ret_bb = ret_chips / 100.0
        validate_bb_return(ret_bb, hand_idx=h)

        total_chips += ret_chips
        pos_returns[pos_name].append(ret_bb)
        tier = classify_hand_tier(hero_hand_str)
        tier_returns[tier].append(ret_bb)

    bb100_final = float(total_chips / float(FINAL_HANDS))
    validate_mean_ev(bb100_final / 100.0, metric_name=f"{model_name}_bb100")

    fold_rate = (action_counts[0] / max(1, total_actions)) * 100.0
    allin_rate = (action_counts[3] / max(1, total_actions)) * 100.0
    half_rate = (action_counts[4] / max(1, total_actions)) * 100.0

    prem_ev = float(np.mean(tier_returns['premium'])) if tier_returns['premium'] else 0.0
    marg_ev = float(np.mean(tier_returns['marginal'])) if tier_returns['marginal'] else 0.0
    trash_ev = float(np.mean(tier_returns['trash'])) if tier_returns['trash'] else 0.0

    print(f"\n[{model_name}] 독립 평가 완료 ({FINAL_HANDS}판)")
    print(f"  전체 bb/100: {bb100_final:+7.2f} bb/100")
    print(f"  핸드별 EV (bb/hand): 프리미엄 {prem_ev:+6.2f} | 마지널 {marg_ev:+6.2f} | 트래시 {trash_ev:+6.2f}")
    print(f"  행동 비율: Fold {fold_rate:5.1f}%, All-in {allin_rate:5.1f}%, Half-Pot {half_rate:5.1f}%")
    for pos in ['BTN', 'CO', 'UTG', 'SB', 'BB']:
        pe = float(np.mean(pos_returns[pos]) * 100.0) if pos_returns[pos] else 0.0
        print(f"    포지션 {pos:<3}: {pe:+7.2f} bb/100")

    return {
        'model': model_name,
        'final_bb100': bb100_final,
        'premium_ev_bb_per_hand': prem_ev,
        'marginal_ev_bb_per_hand': marg_ev,
        'trash_ev_bb_per_hand': trash_ev,
        'fold_rate_pct': fold_rate,
        'allin_rate_pct': allin_rate,
        'half_rate_pct': half_rate,
        'btn_bb100': float(np.mean(pos_returns['BTN']) * 100.0),
        'co_bb100': float(np.mean(pos_returns['CO']) * 100.0),
        'utg_bb100': float(np.mean(pos_returns['UTG']) * 100.0),
        'sb_bb100': float(np.mean(pos_returns['SB']) * 100.0),
        'bb_bb100': float(np.mean(pos_returns['BB']) * 100.0),
    }

def run_phase8_evaluation():
    print("=" * 75)
    print("PHASE 8: FINAL INDEPENDENT GENERALIZATION EVALUATION")
    print("=" * 75)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()

    SEEDS = [42, 101, 202, 303, 404]
    COND_SPECS = [
        ('CondA_Raw_EntOFF', False),
        ('CondB_Norm_EntOFF', True),
        ('CondC_Raw_EntON', False),
        ('CondD_Norm_EntON', True),
    ]

    all_results = []
    for cprefix, norm in COND_SPECS:
        print(f"\n>>> [조건 평가] {cprefix} (5개 시드)")
        for s in SEEDS:
            mpath = f"v496/models/{cprefix}_seed_{s}.pt"
            mname = f"{cprefix}_seed_{s}"
            if not os.path.exists(mpath):
                continue
            m = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
            m.load_state_dict(torch.load(mpath, map_location=device))
            res = evaluate_model_independent(m, game, norm, device, mname)
            res['condition'] = cprefix
            res['seed'] = s
            all_results.append(res)

    # CSV 저장
    csv_out = "v496/data/phase8_final_independent_eval.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[저장 완료] Phase 8 최종 독립 평가 데이터: {csv_out}")

    # 조건별 집계 및 통계 검정
    print("\n" + "=" * 75)
    print("조건별 독립 평가 5시드 평균 및 95% 신뢰구간")
    print("=" * 75)
    summary_list = []
    for cprefix, _ in COND_SPECS:
        vals = [r['final_bb100'] for r in all_results if r['condition'] == cprefix]
        prems = [r['premium_ev_bb_per_hand'] for r in all_results if r['condition'] == cprefix]
        trashs = [r['trash_ev_bb_per_hand'] for r in all_results if r['condition'] == cprefix]
        m_v = float(np.mean(vals))
        s_v = float(np.std(vals, ddof=1))
        ci95 = 1.96 * (s_v / np.sqrt(len(vals)))
        print(f"{cprefix:<20} | bb/100: {m_v:+7.2f} ± {ci95:6.2f} (Std: {s_v:6.2f}) | 프리미엄 EV: {np.mean(prems):+5.2f} BB | 트래시 EV: {np.mean(trashs):+5.2f} BB")
        summary_list.append({
            'condition': cprefix,
            'mean_bb100': m_v,
            'ci95': ci95,
            'std_bb100': s_v,
            'mean_prem_ev': float(np.mean(prems)),
            'mean_trash_ev': float(np.mean(trashs))
        })

    # Paired t-test: CondD (Norm + EntON) vs CondA (Raw + EntOFF)
    d_vals = [r['final_bb100'] for r in all_results if r['condition'] == 'CondD_Norm_EntON']
    a_vals = [r['final_bb100'] for r in all_results if r['condition'] == 'CondA_Raw_EntOFF']
    diffs = np.array(d_vals) - np.array(a_vals)
    t_stat, p_val = stats.ttest_rel(d_vals, a_vals)
    m_diff = np.mean(diffs)
    sem_diff = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    ci95_diff = 1.96 * sem_diff

    print("\n" + "=" * 75)
    print("대응표본 통계 검정 (Paired t-test: CondD_Norm_EntON vs CondA_Raw_EntOFF)")
    print(f"  평균 수익률 차이 (Delta): {m_diff:+7.2f} ± {ci95_diff:6.2f} bb/100")
    print(f"  t-통계량: {t_stat:.4f}, df: {len(diffs)-1}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  -> 판정: 정규화 및 On-Policy 엔트로피가 결합된 CondD가 통계적으로 유의한 우위를 보임.")
    else:
        print("  -> 판정: 신뢰구간에 0이 포함되며 통계적 우위 미입증 (보수적 결론 유지).")
    print("=" * 75)

if __name__ == "__main__":
    run_phase8_evaluation()
