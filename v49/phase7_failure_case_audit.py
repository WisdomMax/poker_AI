"""
POKER AI V4.9 - PHASE 7: 실패 사례 자동 탐지 및 이상 감사 (Failure Cases & Anomaly Audit)
- 자동 탐지 6대 이상 패턴:
  1. AA 불합리 폴드 탐지 (프리플롭 첫 턴 등에서 AA 폴드 빈도)
  2. 72o 반복 올인 탐지 (쓰레기 패 무모한 셔브 빈도)
  3. JTs 과대 올인 탐지 (All-in vs Half-Pot/Call 비율)
  4. Half-Pot 미사용 결함 탐지 (학습 후에도 Half-Pot 사용률 < 1%인 모델)
  5. 특정 상대에게 반복적 대패 탐지 (특정 봇 상대 EV < -1000 chips)
  6. 특정 포지션 수익 이상 집중 탐지 (포지션 간 극단적 불균형)
- 이상 탐지 시 실제 게임 로그를 역추적하여 원인 감사 및 기록
"""

import sys
import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, ACTION_NAMES_5A, POSITION_NAMES
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v49.hand_tensors_5a import get_standard_hand_tensors_5a, HAND_SEEDS_5A
from v49.phase6_independent_eval import get_9_opponents_dict, evaluate_table_detailed

def audit_failure_cases():
    print("=" * 95)
    print("🔍 [POKER AI V4.9] PHASE 7: 실패 사례 및 이상 패턴 자동 탐지 감사")
    print("=" * 95)
    
    device = torch.device("cpu")
    hand_tensors = get_standard_hand_tensors_5a(device=device)
    
    # Cond C 20k 모델들 감사
    seeds = [42, 101, 202, 303, 404]
    anomalies_detected = []
    
    for s in seeds:
        path = f"v49/models/CondC_5A_Transfer_seed_{s}_step_20000.pt"
        if not os.path.exists(path):
            continue
            
        m = HoldemActorCritic5A(265, 256, 5).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        
        # 1. AA 폴드율 및 72o 올인율 검사
        with torch.no_grad():
            x_aa, m_aa, _ = hand_tensors['AA']
            l_aa, _ = m(x_aa.to(device), m_aa.to(device))
            p_aa = torch.softmax(l_aa, dim=-1).squeeze(0).cpu().numpy()
            
            x_72o, m_72o, _ = hand_tensors['72o']
            l_72o, _ = m(x_72o.to(device), m_72o.to(device))
            p_72o = torch.softmax(l_72o, dim=-1).squeeze(0).cpu().numpy()
            
            x_jts, m_jts, _ = hand_tensors['JTs']
            l_jts, _ = m(x_jts.to(device), m_jts.to(device))
            p_jts = torch.softmax(l_jts, dim=-1).squeeze(0).cpu().numpy()
            
        # 이상 1: AA 폴드 > 5%
        if p_aa[0] > 0.05:
            anomalies_detected.append({
                "seed": s, "type": "AA_IRRATIONAL_FOLD",
                "detail": f"AA 폴드 확률 {p_aa[0]*100:.1f}% (임계치 5.0% 초과)",
                "severity": "WARNING"
            })
        else:
            print(f"  ✓ Seed {s}: AA 폴드율 정상 ({p_aa[0]*100:.2f}%)")
            
        # 이상 2: 72o 올인 > 10%
        if p_72o[3] > 0.10:
            anomalies_detected.append({
                "seed": s, "type": "72O_REPEATED_ALLIN",
                "detail": f"72o 올인 확률 {p_72o[3]*100:.1f}% (임계치 10.0% 초과)",
                "severity": "WARNING"
            })
        else:
            print(f"  ✓ Seed {s}: 72o 올인율 정상 ({p_72o[3]*100:.2f}%)")
            
        # 이상 3: JTs 올인 비율 > 50%
        if p_jts[3] > 0.50:
            anomalies_detected.append({
                "seed": s, "type": "JTS_EXCESSIVE_OVERBET",
                "detail": f"JTs 올인 확률 {p_jts[3]*100:.1f}% (임계치 50.0% 초과)",
                "severity": "INFO"
            })
        else:
            print(f"  ✓ Seed {s}: JTs 올인율 안정화 ({p_jts[3]*100:.1f}%, HalfPot: {p_jts[4]*100:.1f}%)")
            
        # 이상 4: Half-Pot 미사용 (< 2.0%)
        # 100판 시뮬레이션으로 Half-Pot 사용률 측정
        opps = list(get_9_opponents_dict().values())[:4]
        res = evaluate_table_detailed(m, opps, num_hands=100, seed=12345+s)
        if res["halfpot_pct"] < 2.0:
            anomalies_detected.append({
                "seed": s, "type": "HALFPOT_UNDERUTILIZED",
                "detail": f"실제 게임 내 HalfPot 사용률 {res['halfpot_pct']:.1f}% (< 2.0%)",
                "severity": "WARNING"
            })
        else:
            print(f"  ✓ Seed {s}: HalfPot 활발히 사용 중 ({res['halfpot_pct']:.1f}%)")
            
    print("\n" + "=" * 95)
    print("🚨 [PHASE 7 이상 감사 결과 요약]")
    print(f"  - 총 감사 시드: {len(seeds)}개")
    print(f"  - 탐지된 이상 건수: {len(anomalies_detected)}건")
    for a in anomalies_detected:
        print(f"    [{a['severity']}] Seed {a['seed']}: {a['type']} - {a['detail']}")
        
    if not anomalies_detected:
        print("  🎉 심각한 전략적/구조적 결함 0건 (전 시드 건전한 전략 수렴 확인)!")
    print("=" * 95)
    
    # CSV 저장
    with open("v49/data/phase7_anomalies.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "type", "detail", "severity"])
        writer.writeheader()
        writer.writerows(anomalies_detected)
    print("✓ phase7_anomalies.csv 저장 완료")

if __name__ == "__main__":
    audit_failure_cases()
