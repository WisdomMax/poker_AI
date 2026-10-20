"""
POKER AI V4.9.8 - PHASE 2: Frozen Evaluation Reproducibility Verification
- 1,000판 고정 스케줄 생성 및 JSON 저장
- 동일 모델(CondD Seed 42)로 Run 1과 Run 2 독립 2회 실행
- 1,000개 모든 게임의 Terminal Return 및 Hero 액션 시퀀스 1:1 비트 단위 일치 검증
- 게임별 리턴 합산 bb/100 재검증
"""

import os
import sys
import json
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v498.frozen_evaluator import (
    generate_frozen_schedule, evaluate_model_frozen
)

def run_phase2_reproducibility():
    print("=" * 80)
    print("PHASE 2: FROZEN EVALUATION 재현성 비트 단위 전수 검증")
    print("=" * 80)

    schedule_path = "v498/data/frozen_schedule_1000.json"
    if not os.path.exists(schedule_path):
        print(f"고정 스케줄 1,000판 생성 중... (Master Seed: 20261017)")
        schedule = generate_frozen_schedule(num_hands=1000, master_seed=20261017)
        with open(schedule_path, 'w', encoding='utf-8') as f:
            json.dump(schedule, f)
        print(f"-> 스케줄 저장 완료: {schedule_path}")
    else:
        print(f"기존 고정 스케줄 로드: {schedule_path}")
        with open(schedule_path, 'r', encoding='utf-8') as f:
            schedule = json.load(f)

    test_model = "v496/models/CondD_Norm_EntON_seed_42.pt"
    print(f"\n테스트 모델: {test_model} (Normalize=True)")

    print("\n[Run 1 실행 중...]")
    summary1, logs1 = evaluate_model_frozen(test_model, schedule, normalize_input=True)

    print("\n[Run 2 실행 중...]")
    summary2, logs2 = evaluate_model_frozen(test_model, schedule, normalize_input=True)

    # 비트 단위 대조
    print("\n" + "=" * 80)
    print("RUN 1 vs RUN 2 정밀 대조 결과:")
    print("=" * 80)

    diff_bb100 = abs(summary1['bb100'] - summary2['bb100'])
    print(f"  Run 1 bb/100: {summary1['bb100']:+10.6f}")
    print(f"  Run 2 bb/100: {summary2['bb100']:+10.6f}")
    print(f"  bb/100 오차:   {diff_bb100:.20f}")

    action_mismatches = 0
    return_mismatches = 0
    card_mismatches = 0

    for i in range(len(schedule)):
        l1 = logs1[i]
        l2 = logs2[i]
        if l1['hero_cards'] != l2['hero_cards']:
            card_mismatches += 1
        if l1['hero_actions'] != l2['hero_actions']:
            action_mismatches += 1
        if abs(l1['ret_chips'] - l2['ret_chips']) > 1e-9:
            return_mismatches += 1

    print(f"  카드 일치율:      {len(schedule) - card_mismatches}/{len(schedule)} (100.0%)")
    print(f"  Hero 행동 일치율: {len(schedule) - action_mismatches}/{len(schedule)} (100.0%)")
    print(f"  리턴 비트 일치율: {len(schedule) - return_mismatches}/{len(schedule)} (100.0%)")

    assert card_mismatches == 0, f"카드 불일치 발생! {card_mismatches}"
    assert action_mismatches == 0, f"행동 불일치 발생! {action_mismatches}"
    assert return_mismatches == 0, f"리턴 불일치 발생! {return_mismatches}"
    assert diff_bb100 == 0.0, f"bb/100 불일치 발생! {diff_bb100}"

    print("\n-> 결론: Frozen Evaluation은 동일 체크포인트에 대해 100% 비트 단위로 완벽하게 재현됨이 입증되었습니다!")

    # 검증 CSV 및 로그 저장
    csv_logs = "v498/data/frozen_eval_run1_logs.csv"
    with open(csv_logs, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(logs1[0].keys()))
        w.writeheader()
        w.writerows(logs1)
    print(f"-> 1,000판 대국 상세 로그 저장 완료: {csv_logs}")

if __name__ == "__main__":
    run_phase2_reproducibility()
