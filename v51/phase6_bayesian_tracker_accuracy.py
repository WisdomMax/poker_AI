"""
POKER AI V5.1 - PHASE 6: BAYESIAN OPPONENT TRACKER ACCURACY & CALIBRATION AUDIT
- 베이지안 상대 추적기의 행동 예측 정확도 검증:
  1. 대상 상대: TightBot, AggressiveBot, RandomBot, SelectiveCallerBot
  2. 관측 핸드 수: 5, 10, 20, 50, 100 핸드
  3. 지표:
     - Brier Score (VPIP, PFR)
     - Log Loss (Binary Cross-Entropy)
     - Calibration Error (ECE: Expected Calibration Error)
  4. 원시 데이터를 v51/data/phase6_bayesian_tracker_accuracy.csv에 저장
"""

import os
import sys
import csv
import numpy as np
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, advance_chance_nodes
from v48.opponents_pool import TightBot, AggressiveBot, RandomBot
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot
from v50.bayesian_opponent_tracker import SingleOpponentBayesianTracker

def compute_brier_score(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((preds - targets) ** 2))

def compute_log_loss(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> float:
    preds = np.clip(preds, eps, 1.0 - eps)
    return float(-np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds)))

def compute_ece(preds: np.ndarray, targets: np.ndarray, num_bins: int = 5) -> float:
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(preds)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (preds >= bin_lower) & (preds < bin_upper if i < num_bins - 1 else preds <= bin_upper)
        bin_size = np.sum(mask)
        if bin_size > 0:
            avg_pred = np.mean(preds[mask])
            avg_target = np.mean(targets[mask])
            ece += (bin_size / n) * np.abs(avg_pred - avg_target)
    return float(ece)

def run_phase6_audit():
    print("=" * 85)
    print("PHASE 6: BAYESIAN OPPONENT TRACKER 예측 정확도 및 캘리브레이션 감사")
    print("=" * 85)

    game = get_5p_holdem_5a_game()
    opponents = [
        ("TightBot", TightBot()),
        ("AggressiveBot", AggressiveBot()),
        ("RandomBot", RandomBot()),
        ("SelectiveCallerBot", SelectiveCallerBot())
    ]

    hand_checkpoints = [5, 10, 20, 50, 100]
    num_sessions = 5
    rng = np.random.RandomState(42)

    results = []

    print(f"\n{'상대 봇':<20} | {'학습 핸드':<9} | {'VPIP Brier':<11} | {'VPIP LogLoss':<13} | {'VPIP ECE':<10} | {'PFR Brier':<10}")
    print("-" * 85)

    for opp_name, bot in opponents:
        for num_train_hands in hand_checkpoints:
            vpip_preds = []
            vpip_actuals = []
            pfr_preds = []
            pfr_actuals = []

            for sess in range(num_sessions):
                tracker = SingleOpponentBayesianTracker(agent_id=1)

                # 1. Training Phase: 봇의 행동 관측 및 트래커 업데이트
                for h in range(num_train_hands):
                    st = game.new_initial_state()
                    advance_chance_nodes(st, rng=rng)

                    saw_action = False
                    vpip = False
                    pfr = False

                    while not st.is_terminal():
                        if st.is_chance_node():
                            advance_chance_nodes(st, rng=rng)
                        else:
                            cp = st.current_player()
                            leg = st.legal_actions(cp)
                            act = bot.act(st, cp, leg)
                            if cp == 1:
                                saw_action = True
                                if act in [1, 2, 3, 4]:
                                    vpip = True
                                if act in [2, 3, 4]:
                                    pfr = True
                            st.apply_action(act)

                    if saw_action:
                        tracker.record_opportunity('vpip')
                        if vpip:
                            tracker.record_success('vpip')
                        tracker.record_opportunity('pfr')
                        if pfr:
                            tracker.record_success('pfr')

                # 2. Testing Phase (Out-of-sample 25 hands): 현재 사후분포로 다음 행동 예측 평가
                est_vpip, _, _ = tracker.get_posterior('vpip')
                est_pfr, _, _ = tracker.get_posterior('pfr')

                for h in range(25):
                    st = game.new_initial_state()
                    advance_chance_nodes(st, rng=rng)

                    test_vpip = False
                    test_pfr = False
                    saw_action = False

                    while not st.is_terminal():
                        if st.is_chance_node():
                            advance_chance_nodes(st, rng=rng)
                        else:
                            cp = st.current_player()
                            leg = st.legal_actions(cp)
                            act = bot.act(st, cp, leg)
                            if cp == 1:
                                saw_action = True
                                if act in [1, 2, 3, 4]:
                                    test_vpip = True
                                if act in [2, 3, 4]:
                                    test_pfr = True
                            st.apply_action(act)

                    if saw_action:
                        vpip_preds.append(est_vpip)
                        vpip_actuals.append(1.0 if test_vpip else 0.0)
                        pfr_preds.append(est_pfr)
                        pfr_actuals.append(1.0 if test_pfr else 0.0)

            vp_p = np.array(vpip_preds)
            vp_a = np.array(vpip_actuals)
            pf_p = np.array(pfr_preds)
            pf_a = np.array(pfr_actuals)

            vpip_bs = compute_brier_score(vp_p, vp_a)
            vpip_ll = compute_log_loss(vp_p, vp_a)
            vpip_ece = compute_ece(vp_p, vp_a)
            pfr_bs = compute_brier_score(pf_p, pf_a)

            print(f"{opp_name:<20} | {num_train_hands:<9} | {vpip_bs:11.4f} | {vpip_ll:13.4f} | {vpip_ece:10.4f} | {pfr_bs:10.4f}")

            results.append({
                'opponent': opp_name,
                'num_hands': num_train_hands,
                'vpip_brier': vpip_bs,
                'vpip_logloss': vpip_ll,
                'vpip_ece': vpip_ece,
                'pfr_brier': pfr_bs
            })

    # CSV 저장
    csv_path = "v51/data/phase6_bayesian_tracker_accuracy.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[저장 완료] Phase 6 트래커 정확도 CSV: {csv_path}")

if __name__ == '__main__':
    run_phase6_audit()
