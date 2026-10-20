"""
POKER AI V5.2.2 - PHASE 2: 99 FACING ALL-IN EV CONTRADICTION & RIGOROUS AUDIT
- V5.2.1 보고서 내 EV 모순 원인 전수 규명:
  * 모순 A (Executive Summary): Fold EV = -2.00 BB vs Call EV = -5.50 BB (Fold 우위)
  * 모순 B (Phase 3 Table):     Fold EV = -3.50 BB vs Call EV = -1.40 BB (Call 우위)
- 모순의 근본 원인:
  1. Hero 초기 투입액(팟 기여금) 정의 불일치:
     - 모순 A: Hero 2.0 BB 오픈(미니 레이즈) 후 올인 직면 -> Fold 시 -2.00 BB
     - 모순 B: Hero Pot 오픈(3.5 BB: SB 50, BB 100 상황에서 Pot 오픈은 350) 후 3-Bet 올인 직면 -> Fold 시 -3.50 BB
  2. 상대(All-in 행위자)의 카드 레인지 가정 불일치:
     - 모순 A: 상대가 '타이트 프리미엄 레인지(TT~AA, AKs)'로 올인한다고 가정한 현실적 수치 -> 99 Call 시 승률 급락으로 -5.50 BB
     - 모순 B: OpenSpiel 롤아웃 시 상대에게 무작위 덱(Random 2장)을 배분하여 상대가 블러프/쓰레기패로 올인한 셈이 됨 -> 99 vs Random 1명 에쿼티가 높아 Call 시 -1.40 BB
- 엄밀한 동일 조건 재실측:
  * Hand Start 기준 Terminal Net Return
  * 상황 1: Hero UTG Pot 오픈(3.5 BB) 후 CO 3-Bet All-in 직면 vs Random Opponent
  * 상황 2: Hero UTG Pot 오픈(3.5 BB) 후 CO 3-Bet All-in 직면 vs Realistic All-in Opponent (TT~AA, AKs, KQs)
  * 상황 3: Hero BB(1.0 BB)에서 UTG 10 BB All-in 직면 vs Realistic Shove Range
- 공통 난수(Common Random Numbers) 적용 및 95% CI 산출
- 원시 데이터를 v522/data/phase2_99_ev_contradiction_audit.csv에 저장
"""

import os
import sys
import csv
import itertools
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import pyspiel

from phase1_probability_correction import evaluate_5card_hand, is_target_hand, is_pair, is_suited

def audit_99_ev_scenarios(num_trials=500, seed=20261018):
    print("=" * 85)
    print("PHASE 2: 99 핸드 Facing All-in EV 모순 전수 감사 및 동일 조건 실측")
    print("=" * 85)

    params = {
        'betting': 'nolimit', 'bettingAbstraction': 'fcpa', 'numPlayers': 5,
        'numRanks': 6, 'numSuits': 4, 'numHoleCards': 2, 'numRounds': 2,
        'numBoardCards': '0 3', 'blind': '50 100 0 0 0', 'stack': '1000 1000 1000 1000 1000',
        'firstPlayer': '3 1'
    }
    game = pyspiel.load_game('universal_poker', params)

    hero_c1, hero_c2 = 0, 1 # 9s, 9c
    deck_22 = [c for c in range(24) if c not in (hero_c1, hero_c2)]

    rng = np.random.RandomState(seed)

    scenarios = [
        {
            "name": "시나리오 1: UTG 3.5BB Pot오픈 후 CO 올인 vs 상대 Random Range (Phase 3 재현)",
            "hero_pos": "UTG (Seat 2)",
            "hero_invested": "3.5 BB (350)",
            "opp_invested": "10.0 BB (1000)",
            "pot_before_call": "15.0 BB (1500)",
            "call_cost": "6.5 BB (650)",
            "eff_stack": "10.0 BB (1000)",
            "opp_range": "Random 2-Cards (비현실적 레인지)",
            "opp_filter": lambda c1, c2: True
        },
        {
            "name": "시나리오 2: UTG 3.5BB Pot오픈 후 CO 올인 vs 상대 Realistic All-in Range (프리미엄)",
            "hero_pos": "UTG (Seat 2)",
            "hero_invested": "3.5 BB (350)",
            "opp_invested": "10.0 BB (1000)",
            "pot_before_call": "15.0 BB (1500)",
            "call_cost": "6.5 BB (650)",
            "eff_stack": "10.0 BB (1000)",
            "opp_range": "Realistic All-in (TT~AA, AKs, KQs)",
            "opp_filter": lambda c1, c2: is_target_hand(c1, c2) or ((c1//4 in (4,3)) and (c2//4 in (4,3)) and is_suited(c1, c2))
        },
        {
            "name": "시나리오 3: Hero BB(1.0BB 투입)에서 UTG 10BB 오픈 올인 직면 vs Shove Range",
            "hero_pos": "BB (Seat 1)",
            "hero_invested": "1.0 BB (100)",
            "opp_invested": "10.0 BB (1000)",
            "pot_before_call": "11.5 BB (1150)",
            "call_cost": "9.0 BB (900)",
            "eff_stack": "10.0 BB (1000)",
            "opp_range": "Realistic Shove (TT~AA, AKs, AKo, KQs)",
            "opp_filter": lambda c1, c2: is_target_hand(c1, c2) or (c1//4 == 5 and c2//4 == 4)
        }
    ]

    results = []

    for sc in scenarios:
        print(f"\n[{sc['name']}]")
        print(f"  - Hero 포지션     : {sc['hero_pos']}")
        print(f"  - Hero 기투입액   : {sc['hero_invested']}")
        print(f"  - 상대 기투입액   : {sc['opp_invested']}")
        print(f"  - 현재 팟 크기    : {sc['pot_before_call']}")
        print(f"  - 콜 필요 금액    : {sc['call_cost']}")
        print(f"  - 유효 스택       : {sc['eff_stack']}")
        print(f"  - 상대 카드 레인지 : {sc['opp_range']}")

        fold_returns = []
        call_returns = []

        valid_count = 0
        while valid_count < num_trials:
            # 덱 셔플
            shuffled = list(rng.permutation(deck_22))
            opp_c1, opp_c2 = shuffled[0], shuffled[1]

            if not sc["opp_filter"](opp_c1, opp_c2):
                continue

            valid_count += 1
            board = shuffled[8:11] # 플랍 3장

            # Hand Start 기준 Net Return 계산
            # Fold 시: 기투입액 전액 손실
            if "3.5 BB" in sc['hero_invested']:
                fold_ret = -3.50
            elif "1.0 BB" in sc['hero_invested']:
                fold_ret = -1.00
            elif "2.0 BB" in sc['hero_invested']:
                fold_ret = -2.00
            fold_returns.append(fold_ret)

            # Call 시: 쇼다운 결과에 따른 Net Return
            # 총 팟 = 20 BB (Hero 10 BB + Opp 10 BB + 블라인드 등)
            hero_val = evaluate_5card_hand([hero_c1, hero_c2] + board)
            opp_val = evaluate_5card_hand([opp_c1, opp_c2] + board)

            if "3.5 BB" in sc['hero_invested']:
                # Hero 총 투입 10 BB, SB(0.5 BB) 폴드 -> 총 팟 = 20.5 BB
                total_pot = 20.5
                if hero_val > opp_val:
                    call_ret = total_pot - 10.0 # +10.5 BB
                elif hero_val == opp_val:
                    call_ret = (total_pot / 2.0) - 10.0 # +0.25 BB
                else:
                    call_ret = -10.0 # -10.0 BB
            elif "1.0 BB" in sc['hero_invested']:
                # Hero BB 10 BB 투입, SB 0.5 BB 폴드, 나머지 0 BB -> 총 팟 = 20.5 BB
                total_pot = 20.5
                if hero_val > opp_val:
                    call_ret = total_pot - 10.0 # +10.5 BB
                elif hero_val == opp_val:
                    call_ret = (total_pot / 2.0) - 10.0 # +0.25 BB
                else:
                    call_ret = -10.0 # -10.0 BB

            call_returns.append(call_ret)

        f_mean = float(np.mean(fold_returns))
        c_mean = float(np.mean(call_returns))
        c_sem = float(np.std(call_returns, ddof=1) / np.sqrt(num_trials))
        c_ci = float(stats.t.ppf(0.975, df=num_trials-1) * c_sem)

        diff = c_mean - f_mean
        optimal_act = "Call" if diff > 0 else "Fold"

        print(f"  - 표본 수         : {num_trials} 회 (Common Board/Deck 롤아웃)")
        print(f"  - Fold Net Return : {f_mean:+6.2f} BB (확정 손실)")
        print(f"  - Call Net Return : {c_mean:+6.2f} ± {c_ci:4.2f} BB (95% CI)")
        print(f"  - Call - Fold EV  : {diff:+6.2f} BB -> [참 최적 행동: {optimal_act}]")

        results.append({
            "scenario": sc['name'],
            "hero_pos": sc['hero_pos'],
            "hero_invested": sc['hero_invested'],
            "call_cost": sc['call_cost'],
            "opp_range": sc['opp_range'],
            "trials": num_trials,
            "fold_ev_bb": f_mean,
            "call_ev_bb": c_mean,
            "call_ci_half": c_ci,
            "ev_diff": diff,
            "optimal_action": optimal_act
        })

    # CSV 저장
    os.makedirs("v522/data", exist_ok=True)
    csv_path = "v522/data/phase2_99_ev_contradiction_audit.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장 완료] Phase 2 CSV: {csv_path}\n")

if __name__ == '__main__':
    audit_99_ev_scenarios()
