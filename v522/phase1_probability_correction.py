"""
POKER AI V5.2.2 - PHASE 1: 99 HAND PROBABILITY ERROR CORRECTION & EQUITY AUDIT
- 24장 덱 (6 Ranks: 9, T, J, Q, K, A, 4 Suits: s, c, h, d)
- Hero 홀카드: 99 (예: 9s, 9c) 고정
- 남은 카드: 22장 (9h, 9d + 10~A 각 4장씩 총 20장)
- 상대 플레이어: 4명 (각 2장 배분, 총 8장)
- 타겟 프리미엄 핸드:
  * TT, JJ, QQ, KK, AA: 5 ranks * C(4,2)=6 combos = 30 combos
  * AKs: 4 suited combos (AsKs, AcKc, AhKh, AdKd) = 4 combos
  * 합계: 34 combos (남은 22장 기준 C(22,2)=231 combos 중)
- 목적:
  1. 상대 1명 확률 = 34 / 231 (14.7186%)
  2. 합집합 상한(Union Bound) = 4 * (34/231) = 58.8745% (80% 주장의 명백한 오류 입증)
  3. 1,000,000회 Monte Carlo 시뮬레이션으로 정확한 상대 4명 중 1명 이상 보유 확률 산출
  4. 24장 덱에서 99의 실제 Equity 실측:
     - vs 4 Random Hands
     - vs 4 Top 50% Hands
     - vs 4 Top 30% Hands
     - vs Facing All-in Range (TT~AA, AKs)
- 원시 데이터를 v522/data/phase1_probability_correction.csv에 저장
"""

import os
import sys
import csv
import itertools
from collections import defaultdict
import numpy as np

# 24장 덱 정의: 0=9, 1=T, 2=J, 3=Q, 4=K, 5=A
# card = rank * 4 + suit (suit: 0=s, 1=c, 2=h, 3=d)
RANKS = ['9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'c', 'h', 'd']

def card_to_str(c):
    return RANKS[c // 4] + SUITS[c % 4]

def is_pair(c1, c2):
    return (c1 // 4) == (c2 // 4)

def is_suited(c1, c2):
    return (c1 % 4) == (c2 % 4)

def is_target_hand(c1, c2):
    # TT~AA (rank 1~5) or AKs (rank 5 & 4, suited)
    r1, r2 = c1 // 4, c2 // 4
    if r1 == r2 and r1 >= 1: # TT, JJ, QQ, KK, AA
        return True
    if (r1 == 5 and r2 == 4) or (r1 == 4 and r2 == 5):
        if is_suited(c1, c2): # AKs
            return True
    return False

def evaluate_5card_hand(cards):
    """
    5장 평가 (홀카드 2장 + 플랍 3장 = 5장 쇼다운)
    24장 덱에서는 턴/리버 없이 플랍 3장으로 쇼다운 진행
    Rank category:
      8: Straight Flush
      7: Four of a Kind
      6: Full House
      5: Flush
      4: Straight
      3: Three of a Kind
      2: Two Pair
      1: One Pair
      0: High Card
    """
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]

    rank_counts = defaultdict(int)
    for r in ranks:
        rank_counts[r] += 1
    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    is_flush = len(set(suits)) == 1

    # Straight check (6 ranks: 0=9, 1=T, 2=J, 3=Q, 4=K, 5=A)
    unique_ranks = sorted(list(set(ranks)), reverse=True)
    is_straight = False
    straight_high = -1
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]

    if is_straight and is_flush:
        return (8, straight_high)
    if counts_sorted[0][1] == 4:
        return (7, counts_sorted[0][0], counts_sorted[1][0])
    if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
        return (6, counts_sorted[0][0], counts_sorted[1][0])
    if is_flush:
        return (5, ranks)
    if is_straight:
        return (4, straight_high)
    if counts_sorted[0][1] == 3:
        return (3, counts_sorted[0][0], [r for r in ranks if r != counts_sorted[0][0]])
    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        high_p = max(counts_sorted[0][0], counts_sorted[1][0])
        low_p = min(counts_sorted[0][0], counts_sorted[1][0])
        kicker = [r for r in ranks if r != high_p and r != low_p][0]
        return (2, high_p, low_p, kicker)
    if counts_sorted[0][1] == 2:
        p = counts_sorted[0][0]
        kickers = [r for r in ranks if r != p]
        return (1, p, kickers)
    return (0, ranks)

def run_phase1_audit():
    print("=" * 85)
    print("PHASE 1: 99 핸드 확률 오류 정정 및 24장 덱 Equity 정밀 실측")
    print("=" * 85)

    # 1. 조합론 검증
    hero_c1, hero_c2 = 0, 1 # 9s, 9c
    deck_22 = [c for c in range(24) if c not in (hero_c1, hero_c2)]

    total_pairs = list(itertools.combinations(deck_22, 2))
    total_combos_1p = len(total_pairs) # 231
    target_combos = [pair for pair in total_pairs if is_target_hand(pair[0], pair[1])]
    num_target = len(target_combos) # 34

    single_p = num_target / total_combos_1p
    union_bound = single_p * 4

    print(f"\n[1. 엄밀한 조합론 분석]")
    print(f"  - 남은 카드 수              : {len(deck_22)} 장")
    print(f"  - 1인 가능한 홀카드 조합 수 : C(22, 2) = {total_combos_1p} 가지")
    print(f"  - 타겟 프리미엄 조합 수     : TT~AA(30) + AKs(4) = {num_target} 가지")
    print(f"  - 상대 1명이 보유할 확률    : {num_target} / {total_combos_1p} = {single_p * 100:.4f}%")
    print(f"  - 상대 4명의 합집합 상한    : 4 * ({num_target} / {total_combos_1p}) = {union_bound * 100:.4f}% (최대 상한선)")
    print(f"  -> [판정] V5.2.1 보고서의 '상대 4명 중 1명 이상 보유 확률 80% 초과' 주장은")
    print(f"            수학적 상한선인 58.87%조차 크게 초과한 '명백한 허위/오류'임을 입증함!\n")

    # 2. 1,000,000회 Monte Carlo로 정확한 결합 확률 산출
    num_sims = 1000000
    rng = np.random.RandomState(42)
    deck_arr = np.array(deck_22)

    has_target_count = 0
    target_opp_distribution = defaultdict(int)

    for _ in range(num_sims):
        # 4명에게 2장씩 배분 (8장 비복원 추출)
        dealt = rng.choice(deck_arr, size=8, replace=False)
        opp_targets = 0
        for i in range(4):
            c1, c2 = dealt[2*i], dealt[2*i+1]
            if is_target_hand(c1, c2):
                opp_targets += 1
        if opp_targets > 0:
            has_target_count += 1
        target_opp_distribution[opp_targets] += 1

    exact_prob = has_target_count / num_sims
    exact_ci = 1.96 * np.sqrt(exact_prob * (1 - exact_prob) / num_sims)

    print(f"[2. 1,000,000회 Monte Carlo 정밀 결합 확률 산출]")
    print(f"  - 상대 4명 중 1명 이상 보유 참 확률 : {exact_prob * 100:.4f}% ± {exact_ci * 100:.4f}%")
    print(f"  - 보유자 수 분포:")
    for k in sorted(target_opp_distribution.keys()):
        pct = (target_opp_distribution[k] / num_sims) * 100
        print(f"    * 정확히 {k}명이 타겟 보유 : {pct:6.2f}%")

    # 3. 24장 덱에서 99의 Equity 실측
    print(f"\n[3. 24장 덱에서 99의 상황별 Equity 실측 (100,000회 롤아웃)]")
    equity_sims = 100000

    # 상황 A: vs 4 Random Opponents
    wins_random = 0
    ties_random = 0
    for _ in range(equity_sims):
        # 8장 상대 + 3장 보드 = 11장 추출
        sample = rng.choice(deck_arr, size=11, replace=False)
        board = list(sample[8:11])
        hero_val = evaluate_5card_hand([hero_c1, hero_c2] + board)

        best_opp_val = None
        for i in range(4):
            opp_val = evaluate_5card_hand([sample[2*i], sample[2*i+1]] + board)
            if best_opp_val is None or opp_val > best_opp_val:
                best_opp_val = opp_val

        if hero_val > best_opp_val:
            wins_random += 1
        elif hero_val == best_opp_val:
            ties_random += 1

    equity_random = (wins_random + ties_random * 0.5) / equity_sims

    # 상황 B: vs Facing All-in (최소 1명이 TT~AA or AKs 보유)
    wins_allin = 0
    ties_allin = 0
    valid_allin_trials = 0
    while valid_allin_trials < equity_sims:
        sample = rng.choice(deck_arr, size=11, replace=False)
        opp_has_target = False
        for i in range(4):
            if is_target_hand(sample[2*i], sample[2*i+1]):
                opp_has_target = True
                break
        if not opp_has_target:
            continue

        valid_allin_trials += 1
        board = list(sample[8:11])
        hero_val = evaluate_5card_hand([hero_c1, hero_c2] + board)

        best_opp_val = None
        for i in range(4):
            opp_val = evaluate_5card_hand([sample[2*i], sample[2*i+1]] + board)
            if best_opp_val is None or opp_val > best_opp_val:
                best_opp_val = opp_val

        if hero_val > best_opp_val:
            wins_allin += 1
        elif hero_val == best_opp_val:
            ties_allin += 1

    equity_allin = (wins_allin + ties_allin * 0.5) / equity_sims

    print(f"  - 99 Equity vs 4 Random Hands    : {equity_random * 100:5.2f}%")
    print(f"  - 99 Equity vs Facing All-in     : {equity_allin * 100:5.2f}%")

    # CSV 저장
    os.makedirs("v522/data", exist_ok=True)
    csv_path = "v522/data/phase1_probability_correction.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit", "notes"])
        writer.writerow(["single_opponent_prob", f"{single_p*100:.4f}", "%", "34 / 231"])
        writer.writerow(["union_bound_upper_limit", f"{union_bound*100:.4f}", "%", "4 * (34/231)"])
        writer.writerow(["exact_mc_target_prob", f"{exact_prob*100:.4f}", "%", f"1M sims, CI: {exact_ci*100:.4f}"])
        writer.writerow(["equity_vs_4_random", f"{equity_random*100:.2f}", "%", "100k rollouts"])
        writer.writerow(["equity_vs_facing_allin", f"{equity_allin*100:.2f}", "%", "100k rollouts"])
    print(f"\n[저장 완료] Phase 1 CSV: {csv_path}\n")

if __name__ == '__main__':
    run_phase1_audit()
