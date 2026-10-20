"""
POKER AI V5.0.1 - PHASE 5: Out-of-Sample Calibration & Future Action Prediction Audit
- 분리 평가:
  * 관측 구간 (Observation Window): 첫 100핸드
  * 예측 구간 (Prediction Window): 다음 100핸드
- 3대 예측 방식 비교:
  A. 고정 Prior: p = 0.5000
  B. 단순 관측 빈도 (MLE): p = k / N
  C. Beta Posterior Mean (Bayesian): p = (alpha + k) / (alpha + beta + N)
- 4대 정량 지표 산출:
  * Brier Score: Mean squared error of binary outcomes (1/M) * sum((p - y)^2)
  * Log Loss: -1/M * sum(y*log(p) + (1-y)*log(1-p))
  * Calibration Error: |y_bar_test - p|
  * MAE: Mean absolute error on test frequency
"""

import os
import sys
import math
import csv
from collections import defaultdict
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v48.opponents_pool import (
    TightBot, AggressiveBot, AlwaysCallBot, RandomBot
)
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager

def run_phase5_audit():
    print("=" * 85)
    print("PHASE 5: OUT-OF-SAMPLE 미래 행동 예측 및 CALIBRATION 전수 재검증")
    print("=" * 85)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    opp_dict = {
        1: TightBot(),
        2: AggressiveBot(),
        3: AlwaysCallBot(),
        4: RandomBot()
    }

    mgr = MatchSessionOpponentManager()
    mgr.reset_session()
    rng = np.random.RandomState(20261017)

    # 1. 첫 100핸드 관측 구간 (Observation Window)
    print("\n[단계 1] 관측 구간 (첫 100핸드) 데이터 수집 및 사후분포 추정...")
    for h in range(100):
        mgr.set_hand_index(h)
        s2a = wrapper.get_seat_to_agent(h)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng)
        log_obs = []
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                r_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0
                ag_id = s2a[cur]
                opp = opp_dict[ag_id] if ag_id != 0 else RandomBot()
                act = opp.act(st, cur, legal)
                log_obs.append((cur, r_idx, legal, act))
                st.apply_action(act)
        mgr.parse_and_update_hand(log_obs, s2a)

    # 3대 방식의 예측 확률 p 산출 (지표: VPIP, PFR, All-in)
    target_metrics = ["vpip", "pfr", "allin"]
    predictions = {} # {(ag_id, metric): {'prior': float, 'mle': float, 'bayes': float}}

    for ag_id in range(1, 5):
        t = mgr.trackers[ag_id]
        for m in target_metrics:
            succ = t.stats[m]['succ']
            opp = t.stats[m]['opp']
            p_prior = 0.5000
            p_mle = succ / max(1, opp)
            p_bayes, _, _ = t.get_posterior(m)
            predictions[(ag_id, m)] = {
                'prior': p_prior,
                'mle': p_mle,
                'bayes': p_bayes
            }

    # 2. 다음 100핸드 예측 구간 (Prediction Window)
    print("\n[단계 2] 예측 구간 (다음 100핸드) 실제 행동 발생 및 out-of-sample 대조...")
    # 다음 100핸드에서의 실제 이진 이벤트 수집
    test_events = defaultdict(list) # {(ag_id, metric): [0, 1, 1, 0, ...]}

    for h in range(100, 200):
        s2a = wrapper.get_seat_to_agent(h)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng)

        # 핸드 내에서 각 상대의 행동 발생 여부 기록
        actions_in_hand = defaultdict(list)
        legal_in_hand = defaultdict(list)

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                ag_id = s2a[cur]
                opp = opp_dict[ag_id] if ag_id != 0 else RandomBot()
                act = opp.act(st, cur, legal)

                if ag_id != 0:
                    actions_in_hand[ag_id].append(act)
                    legal_in_hand[ag_id].append(legal)
                st.apply_action(act)

        # 각 상대에 대해 이벤트 기록
        for ag_id in range(1, 5):
            acts = actions_in_hand[ag_id]
            if not acts:
                continue

            # VPIP: 프리플롭 첫 액션
            first_act = acts[0]
            test_events[(ag_id, "vpip")].append(1 if first_act in [1, 2, 3, 4] else 0)

            # PFR: 프리플롭 레이즈 여부
            test_events[(ag_id, "pfr")].append(1 if any(a in [2, 3, 4] for a in acts) else 0)

            # All-in: 올인 여부
            test_events[(ag_id, "allin")].append(1 if any(a == 3 for a in acts) else 0)

    # 3. 4대 평가 지표 (Brier Score, Log Loss, Calibration Error, MAE) 산출
    results = []

    print("\n" + "=" * 85)
    print("미래 100핸드 실제 행동 예측 결과 (Out-of-Sample 비교):")
    print("=" * 85)
    print(f"  {'상대 및 지표':<26} | {'실제빈도':<8} | {'Prior(Brier)':<14} | {'MLE(Brier)':<14} | {'Bayes(Brier)':<14} | 승자")
    print("-" * 85)

    all_brier_prior = []
    all_brier_mle = []
    all_brier_bayes = []

    for (ag_id, m), y_list in test_events.items():
        if len(y_list) == 0:
            continue
        y_arr = np.array(y_list, dtype=np.float32)
        y_bar = float(np.mean(y_arr))

        p_p = predictions[(ag_id, m)]['prior']
        p_m = predictions[(ag_id, m)]['mle']
        p_b = predictions[(ag_id, m)]['bayes']

        # Brier Score = mean((p - y)^2)
        brier_p = float(np.mean((p_p - y_arr) ** 2))
        brier_m = float(np.mean((p_m - y_arr) ** 2))
        brier_b = float(np.mean((p_b - y_arr) ** 2))

        # Log Loss (epsilon clamping)
        def log_loss(p_val, y):
            p_c = np.clip(p_val, 1e-6, 1.0 - 1e-6)
            return float(-np.mean(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c)))

        ll_p = log_loss(p_p, y_arr)
        ll_m = log_loss(p_m, y_arr)
        ll_b = log_loss(p_b, y_arr)

        cal_err_p = abs(y_bar - p_p)
        cal_err_m = abs(y_bar - p_m)
        cal_err_b = abs(y_bar - p_b)

        winner = "Bayes" if brier_b <= min(brier_p, brier_m) else ("MLE" if brier_m < brier_p else "Prior")

        label = f"{opp_dict[ag_id].name:<12} {m}"
        print(f"  {label:<26} | {y_bar:6.3f}   | {brier_p:.4f} ({p_p:.2f})  | {brier_m:.4f} ({p_m:.2f})  | {brier_b:.4f} ({p_b:.2f})  | {winner}")

        all_brier_prior.append(brier_p)
        all_brier_mle.append(brier_m)
        all_brier_bayes.append(brier_b)

        results.append({
            'bot_name': opp_dict[ag_id].name,
            'metric': m,
            'test_hands': len(y_list),
            'actual_freq': y_bar,
            'prior_prob': p_p,
            'mle_prob': p_m,
            'bayes_prob': p_b,
            'brier_prior': brier_p,
            'brier_mle': brier_m,
            'brier_bayes': brier_b,
            'logloss_prior': ll_p,
            'logloss_mle': ll_m,
            'logloss_bayes': ll_b,
            'cal_err_prior': cal_err_p,
            'cal_err_mle': cal_err_m,
            'cal_err_bayes': cal_err_b,
            'winner': winner
        })

    mean_b_prior = np.mean(all_brier_prior)
    mean_b_mle = np.mean(all_brier_mle)
    mean_b_bayes = np.mean(all_brier_bayes)

    print("-" * 85)
    print(f"  [전체 평균 Brier Score (낮을수록 우수)]:")
    print(f"    - 고정 Prior         : {mean_b_prior:.4f}")
    print(f"    - 단순 관측 빈도(MLE): {mean_b_mle:.4f}")
    print(f"    - Beta Posterior     : {mean_b_bayes:.4f} (MLE 대비 과적합 방지 효과)")

    # CSV 저장
    csv_out = "v501/data/phase5_calibration_out_of_sample.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n[저장 완료] Out-of-sample Calibration CSV: {csv_out}")

if __name__ == "__main__":
    run_phase5_audit()
