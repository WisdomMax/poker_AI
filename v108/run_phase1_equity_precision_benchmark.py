import time
import json
import numpy as np
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 필요한 모듈 임포트
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v82.run_v82_bayesian_range_and_cfr_audit import estimate_range_aware_equity
from v83.run_v83_robust_range_and_cfr_audit import RobustBayesianRangeModel

def run_phase1_benchmark():
    print("=" * 80)
    print("PHASE 1: MONTE CARLO EQUITY ENGINE PRECISION BENCHMARK")
    print("=" * 80)

    # 1. 테스트 케이스 설계
    # 4개 대표적 보드 상황 (Flop 2개, Turn 1개, River 1개)
    # 특히 Equity가 45%~55% 부근에 위치하여 Pot Odds(40~60%) 경계선에 걸리는 케이스 포함
    test_cases = [
        {
            "name": "Case 1: Flop Flush Draw & Overcards (Ah Kh on Jh 8h 2c) - Coin Flip",
            "hero": [48, 44], # Ah, Kh (A-suit=3, K-suit=3)
            "board": [40, 28, 4], # Jh, 8h, 2c
            "active_opps": 1,
            "pot_odds": 0.50,
            "agg_level": 1.5
        },
        {
            "name": "Case 2: Turn Top Pair Weak Kicker (Kd 7s on Kc 9h 4d 2s) - Marginal",
            "hero": [45, 21], # Kd, 7s
            "board": [44, 33, 13, 5], # Kc, 9h, 4d, 2s
            "active_opps": 1,
            "pot_odds": 0.45,
            "agg_level": 1.5
        },
        {
            "name": "Case 3: Flop Open-Ended Straight Draw (Qh Jh on Th 9c 3d)",
            "hero": [42, 38], # Qh, Jh
            "board": [36, 32, 9], # Th, 9c, 3d
            "active_opps": 2,
            "pot_odds": 0.48,
            "agg_level": 1.2
        },
        {
            "name": "Case 4: River Middle Pair Bluff Catcher (Jc Tc on As Kd 8c 4h Ts)",
            "hero": [37, 33], # Jc, Tc
            "board": [50, 45, 29, 14, 35], # As, Kd, 8c, 4h, Ts
            "active_opps": 1,
            "pot_odds": 0.52,
            "agg_level": 1.5
        }
    ]

    sim_levels = [30, 100, 300, 1000, 3000]
    num_trials_per_case = 200 # 각 케이스별 200번 반복 추정하여 분산 및 플립 레이트 산출
    ground_truth_sims = 100000

    results = {}

    for idx, tc in enumerate(test_cases):
        print(f"\n--- Testing {tc['name']} ---")
        hero = tc["hero"]
        board = tc["board"]
        active_opps = tc["active_opps"]
        pot_odds = tc["pot_odds"]

        combos, probs = RobustBayesianRangeModel.compute_smoothed_weights(
            hero, board, mode='temperature', temperature=2.0,
            opp_aggression_level=tc["agg_level"], is_raised=True
        )

        # Ground Truth 에쿼티 산출 (100,000 sims)
        print(f"  * Calculating Ground Truth Equity ({ground_truth_sims:,} sims)...")
        t0 = time.time()
        gt_equity = estimate_range_aware_equity(hero, board, combos, probs, active_opps, num_sims=ground_truth_sims, rng_seed=42)
        gt_time = time.time() - t0
        print(f"    -> Ground Truth Equity: {gt_equity * 100:.2f}% (computed in {gt_time:.2f}s)")

        case_res = {
            "gt_equity": gt_equity,
            "sim_stats": {}
        }

        # 올바른 결정 (Call if gt_equity >= pot_odds - 0.05 else Fold)
        ideal_action = "CALL" if gt_equity >= pot_odds - 0.05 else "FOLD"

        for sims in sim_levels:
            estimates = []
            flip_count = 0
            t_start = time.time()
            for trial in range(num_trials_per_case):
                # 독립 시드로 추정
                eq = estimate_range_aware_equity(hero, board, combos, probs, active_opps, num_sims=sims, rng_seed=trial * 1000 + sims)
                estimates.append(eq)
                trial_act = "CALL" if eq >= pot_odds - 0.05 else "FOLD"
                if trial_act != ideal_action:
                    flip_count += 1
            t_elapsed = time.time() - t_start

            estimates = np.array(estimates)
            mean_est = float(np.mean(estimates))
            std_est = float(np.std(estimates))
            mae = float(np.mean(np.abs(estimates - gt_equity)))
            rmse = float(np.sqrt(np.mean((estimates - gt_equity) ** 2)))
            bias = float(mean_est - gt_equity)
            flip_rate = float(flip_count / num_trials_per_case)
            time_per_state_ms = (t_elapsed / num_trials_per_case) * 1000.0
            ops_per_sec = num_trials_per_case / t_elapsed

            print(f"  [Sims = {sims:4d}] Mean: {mean_est*100:5.2f}% | Std: {std_est*100:4.2f}%p | MAE: {mae*100:4.2f}%p | RMSE: {rmse*100:4.2f}%p | Flip: {flip_rate*100:4.1f}% | {time_per_state_ms:5.2f}ms/state ({ops_per_sec:5.1f} ops/s)")

            case_res["sim_stats"][str(sims)] = {
                "mean": mean_est,
                "std": std_est,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "flip_rate": flip_rate,
                "time_per_state_ms": time_per_state_ms,
                "ops_per_sec": ops_per_sec
            }

        results[f"case_{idx+1}"] = case_res

    # 전체 요약 통계 계산
    summary = {}
    for sims in sim_levels:
        s_key = str(sims)
        avg_std = np.mean([results[c]["sim_stats"][s_key]["std"] for c in results])
        avg_mae = np.mean([results[c]["sim_stats"][s_key]["mae"] for c in results])
        avg_rmse = np.mean([results[c]["sim_stats"][s_key]["rmse"] for c in results])
        avg_flip = np.mean([results[c]["sim_stats"][s_key]["flip_rate"] for c in results])
        avg_time = np.mean([results[c]["sim_stats"][s_key]["time_per_state_ms"] for c in results])
        avg_ops = np.mean([results[c]["sim_stats"][s_key]["ops_per_sec"] for c in results])
        summary[s_key] = {
            "avg_std": float(avg_std),
            "avg_mae": float(avg_mae),
            "avg_rmse": float(avg_rmse),
            "avg_flip_rate": float(avg_flip),
            "avg_time_ms": float(avg_time),
            "avg_ops_sec": float(avg_ops)
        }

    output_data = {
        "summary": summary,
        "cases": results
    }

    out_file = "v108/data/phase1_equity_precision.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[+] Phase 1 벤치마크 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    run_phase1_benchmark()
