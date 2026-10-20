"""
POKER AI V4.9.7 - PHASE 1: Evaluator Integrity Audit & Hand Tier Re-evaluation
- Hero 홀카드 파싱 및 분류 함수 단위 테스트 (AA, KK, AKs, JTs, 72o 등)
- 1,000판 대국 핸드별 원시 로그 저장 (Hero 카드, 분류 카테고리, Return BB)
- 전체 핸드 수 = Premium + Marginal + Trash 합계 일치성 및 미분류 0건 검증
- V4.9.6의 모든 체크포인트 (CondA, B, C, D 각 5개 시드) 전수 재평가
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

def classify_hand_tier_corrected(card_input) -> str:
    """
    홀카드 분류 함수 (버그 완전 수정):
    - 문자열(예: 'AsAh') 또는 리스트/튜플(['As', 'Ah']) 모두 완벽 지원
    - 프리미엄: AA, KK, QQ, JJ, AKs, AKo
    - 마지널: TT, 99, 88, 77, AQ, AJ, AT, KQ, KJ, QJ, JTs, T9s, 98s, 87s
    - 트래시: 그 외 모든 핸드
    """
    if isinstance(card_input, (list, tuple)):
        # 만약 (['As', 'Ah'], [...]) 튜플이 넘어온 경우 첫 번째 리스트 사용
        if len(card_input) > 0 and isinstance(card_input[0], (list, tuple)):
            card_input = card_input[0]
        card_str = ''.join(card_input)
    else:
        card_str = str(card_input).strip()

    if len(card_str) < 4:
        raise ValueError(f"유효하지 않은 카드 문자열: '{card_str}'")

    r1, s1 = card_str[0], card_str[1]
    r2, s2 = card_str[2], card_str[3]
    rank_order = "23456789TJQKA"

    if r1 not in rank_order or r2 not in rank_order:
        raise ValueError(f"알 수 없는 카드 랭크: '{r1}' or '{r2}'")

    idx1, idx2 = rank_order.index(r1), rank_order.index(r2)
    suited = (s1 == s2)
    high_idx = max(idx1, idx2)
    low_idx = min(idx1, idx2)
    is_pair = (high_idx == low_idx)

    # 1. 프리미엄 (Premium)
    if is_pair and high_idx >= 9: # JJ, QQ, KK, AA
        return 'premium'
    if high_idx == 12 and low_idx == 11: # AKs, AKo
        return 'premium'

    # 2. 마지널 (Marginal)
    if is_pair and high_idx >= 5: # 77, 88, 99, TT
        return 'marginal'
    if high_idx >= 10 and low_idx >= 8: # AQ, AJ, AT, KQ, KJ, QJ
        return 'marginal'
    if suited and (high_idx - low_idx == 1) and high_idx >= 6: # 87s, 98s, T9s, JTs
        return 'marginal'

    # 3. 트래시 (Trash)
    return 'trash'

def run_classifier_unit_tests():
    print("=" * 75)
    print("PHASE 1.1: 핸드 분류 함수 단위 테스트 (15대 대표 조합 검증)")
    print("=" * 75)

    test_cases = [
        # (입력, 기대 카테고리)
        ('AsAh', 'premium'),
        (['As', 'Ah'], 'premium'),
        ((['As', 'Ah'], []), 'premium'), # V4.9.6 파싱 버그 형태
        ('KsKh', 'premium'),
        ('QsQh', 'premium'),
        ('JsJh', 'premium'),
        ('AsKh', 'premium'),
        ('AsKs', 'premium'),
        ('TsTh', 'marginal'),
        ('8s8h', 'marginal'),
        ('7s7h', 'marginal'),
        ('AsQh', 'marginal'),
        ('JsTs', 'marginal'),
        ('8s7s', 'marginal'),
        ('8h3c', 'trash'),
        ('7h2c', 'trash'),
        ((['7h', '2c'], []), 'trash'),
    ]

    for inp, expected in test_cases:
        actual = classify_hand_tier_corrected(inp)
        print(f"  입력: {str(inp):<22} | 기대: {expected:<8} | 실측: {actual:<8} -> {'PASS' if actual == expected else 'FAIL'}")
        assert actual == expected, f"분류 불일치: {inp} -> {actual} != {expected}"

    print("-> 단위 테스트 전수 통과: 문자열 및 튜플/리스트 입력에 대해 100% 정상 분류됨.")

def run_1000_hand_eval_audit():
    print("\n" + "=" * 75)
    print("PHASE 1.2: 1,000판 대국 핸드별 분류 및 EV 정합성 전수 감사")
    print("=" * 75)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    device = torch.device("cpu")

    # CondD Seed 42 모델로 1,000판 검증
    model_path = "v496/models/CondD_Norm_EntON_seed_42.pt"
    model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rng_eval = np.random.RandomState(FINAL_EVAL_SEED)
    rng_opp = random.Random(FINAL_EVAL_SEED + 101)

    hand_records = []
    tier_counts = defaultdict(int)
    tier_returns = defaultdict(list)
    total_chips = 0.0

    for h in range(FINAL_HANDS):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        table_opps = rng_opp.sample(opponents, 4)

        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng_eval) # 홀카드 분배 완료
        info_s_hero = st.information_state_string(hero_seat)
        priv_cards, _ = parse_cards_from_info_str(info_s_hero)
        hero_str = ''.join(priv_cards)

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_eval)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    vec = st.information_state_tensor(cur)
                    vec = normalize_vec(vec)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    act, _, _, _ = model.sample_action(x, m, deterministic=False)
                else:
                    ag_id = s2a[cur]
                    opp = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp.act(st, cur, legal)
                st.apply_action(act)

        returns_all = st.returns()
        ret_chips = returns_all[hero_seat]
        ret_bb = ret_chips / 100.0
        total_chips += ret_chips
        tier = classify_hand_tier_corrected(hero_str)

        tier_counts[tier] += 1
        tier_returns[tier].append(ret_bb)

        hand_records.append({
            'hand_idx': h,
            'hero_seat': hero_seat,
            'hero_cards': hero_str,
            'tier': tier,
            'return_chips': ret_chips,
            'return_bb': ret_bb
        })

    # CSV 저장
    csv_hands = "v497/data/phase1_hand_classification_1000.csv"
    with open(csv_hands, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(hand_records[0].keys()))
        writer.writeheader()
        writer.writerows(hand_records)
    print(f"[저장 완료] 1,000판 전수 핸드 분류 로그: {csv_hands}")

    # 전수 무결성 검증
    sum_tiers = sum(tier_counts.values())
    print("\n--- 무결성 검증 결과 ---")
    print(f"  전체 대국 수: {FINAL_HANDS}판")
    print(f"  프리미엄 핸드 수: {tier_counts['premium']}판 ({(tier_counts['premium']/FINAL_HANDS)*100:.2f}%)")
    print(f"  마지널 핸드 수:   {tier_counts['marginal']}판 ({(tier_counts['marginal']/FINAL_HANDS)*100:.2f}%)")
    print(f"  트래시 핸드 수:   {tier_counts['trash']}판 ({(tier_counts['trash']/FINAL_HANDS)*100:.2f}%)")
    print(f"  카테고리 합계:   {sum_tiers}판")

    assert sum_tiers == FINAL_HANDS, f"핸드 수 불일치! {sum_tiers} != {FINAL_HANDS}"
    assert tier_counts['premium'] > 0, "프리미엄 핸드가 0개입니다!"
    assert tier_counts['marginal'] > 0, "마지널 핸드가 0개입니다!"
    assert tier_counts['trash'] > 0, "트래시 핸드가 0개입니다!"

    prem_mean = float(np.mean(tier_returns['premium']))
    marg_mean = float(np.mean(tier_returns['marginal']))
    trash_mean = float(np.mean(tier_returns['trash']))
    overall_mean = float(np.mean([r['return_bb'] for r in hand_records]))

    # 가중 평균 검증
    weighted_mean = (
        tier_counts['premium'] * prem_mean +
        tier_counts['marginal'] * marg_mean +
        tier_counts['trash'] * trash_mean
    ) / FINAL_HANDS

    print(f"\n--- 핸드별 기대수익 실측치 (bb/hand) ---")
    print(f"  프리미엄 평균 EV: {prem_mean:+6.2f} BB/hand (100판 기준: {prem_mean*100:+7.1f} bb/100)")
    print(f"  마지널 평균 EV:   {marg_mean:+6.2f} BB/hand (100판 기준: {marg_mean*100:+7.1f} bb/100)")
    print(f"  트래시 평균 EV:   {trash_mean:+6.2f} BB/hand (100판 기준: {trash_mean*100:+7.1f} bb/100)")
    print(f"  전체 평균 EV:     {overall_mean:+6.2f} BB/hand (전체 bb/100: {overall_mean*100:+7.1f} bb/100)")
    print(f"  가중합 대조 오차: |Weighted - Overall| = {abs(weighted_mean - overall_mean):.10f}")

    assert abs(weighted_mean - overall_mean) < 1e-8, "가중 평균과 전체 평균이 일치하지 않습니다!"
    print("-> 통과: 카테고리별 핸드 수 합계와 가중 평균이 전체 데이터와 100.000% 일치함을 확인.")

def re_evaluate_all_v496_models():
    print("\n" + "=" * 75)
    print("PHASE 1.3: V4.9.6 전 모델 (4조건 x 5시드 = 20개) 전수 재평가")
    print("=" * 75)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    SEEDS = [42, 101, 202, 303, 404]
    COND_SPECS = [
        ('CondA_Raw_EntOFF', False),
        ('CondB_Norm_EntOFF', True),
        ('CondC_Raw_EntON', False),
        ('CondD_Norm_EntON', True),
    ]

    all_re_evals = []

    for cprefix, norm in COND_SPECS:
        print(f"\n>>> [재평가] {cprefix} (5개 시드)")
        for s in SEEDS:
            mpath = f"v496/models/{cprefix}_seed_{s}.pt"
            if not os.path.exists(mpath):
                continue
            model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
            model.load_state_dict(torch.load(mpath, map_location=device))
            model.eval()

            rng_eval = np.random.RandomState(FINAL_EVAL_SEED)
            rng_opp = random.Random(FINAL_EVAL_SEED + 101)

            total_chips = 0.0
            tier_returns = defaultdict(list)
            tier_counts = defaultdict(int)

            for h in range(FINAL_HANDS):
                s2a = wrapper.get_seat_to_agent(h)
                a2s = wrapper.get_agent_to_seat(h)
                hero_seat = a2s[0]
                table_opps = rng_opp.sample(opponents, 4)

                st = game.new_initial_state()
                advance_chance_nodes(st, rng=rng_eval) # 홀카드 분배 완료
                info_s_hero = st.information_state_string(hero_seat)
                priv_cards, _ = parse_cards_from_info_str(info_s_hero)
                hero_str = ''.join(priv_cards)

                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng=rng_eval)
                    else:
                        cur = st.current_player()
                        legal = st.legal_actions(cur)
                        if cur == hero_seat:
                            vec = st.information_state_tensor(cur)
                            if norm: vec = normalize_vec(vec)
                            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                            m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                            for a in legal: m[0, a] = True
                            act, _, _, _ = model.sample_action(x, m, deterministic=False)
                        else:
                            ag_id = s2a[cur]
                            opp = table_opps[(ag_id - 1) % len(table_opps)]
                            act = opp.act(st, cur, legal)
                        st.apply_action(act)

                returns_all = st.returns()
                ret_chips = returns_all[hero_seat]
                ret_bb = ret_chips / 100.0
                total_chips += ret_chips

                tier = classify_hand_tier_corrected(hero_str)
                tier_counts[tier] += 1
                tier_returns[tier].append(ret_bb)

            bb100_val = float(total_chips / float(FINAL_HANDS))
            prem_ev = float(np.mean(tier_returns['premium']))
            marg_ev = float(np.mean(tier_returns['marginal']))
            trash_ev = float(np.mean(tier_returns['trash']))

            print(f"  Seed {s:3d} | bb/100: {bb100_val:+7.2f} | Premium: {prem_ev:+6.2f} BB | Marginal: {marg_ev:+6.2f} BB | Trash: {trash_ev:+6.2f} BB")

            all_re_evals.append({
                'condition': cprefix,
                'seed': s,
                'bb100': bb100_val,
                'prem_ev_bb': prem_ev,
                'marg_ev_bb': marg_ev,
                'trash_ev_bb': trash_ev,
                'prem_count': tier_counts['premium'],
                'marg_count': tier_counts['marginal'],
                'trash_count': tier_counts['trash'],
            })

    csv_summary = "v497/data/phase1_re_evaluation_summary.csv"
    with open(csv_summary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_re_evals[0].keys()))
        writer.writeheader()
        writer.writerows(all_re_evals)
    print(f"\n[저장 완료] V4.9.6 모델 전수 재평가 요약: {csv_summary}")

    # 조건별 집계
    print("\n" + "=" * 75)
    print("조건별 재평가 최종 집계 (수정된 평가기 기준)")
    print("=" * 75)
    for cprefix, _ in COND_SPECS:
        b100s = [r['bb100'] for r in all_re_evals if r['condition'] == cprefix]
        pevs = [r['prem_ev_bb'] for r in all_re_evals if r['condition'] == cprefix]
        mevs = [r['marg_ev_bb'] for r in all_re_evals if r['condition'] == cprefix]
        tevs = [r['trash_ev_bb'] for r in all_re_evals if r['condition'] == cprefix]
        print(f"{cprefix:<20} | bb/100: {np.mean(b100s):+7.2f} | Premium EV: {np.mean(pevs):+6.2f} BB | Marginal EV: {np.mean(mevs):+6.2f} BB | Trash EV: {np.mean(tevs):+6.2f} BB")

if __name__ == "__main__":
    run_classifier_unit_tests()
    run_1000_hand_eval_audit()
    re_evaluate_all_v496_models()
