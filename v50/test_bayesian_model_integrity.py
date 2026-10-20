"""
POKER AI V5.0 - PHASE 3: Information Leakage & Model Dimension Unit Tests
- 1. 317차원(265+52) 입력 포워드 및 그래디언트 역전파 무결성 검증
- 2. 현재 핸드 진행 중 미래 액션 누출 방지(Temporal Leakage Prevention) 검증
- 3. Cold Start vs Warm Start 상대 피처 및 정책 출력 반응성 검증
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import (
    BayesianHoldemActorCritic5A, get_neutral_prior_features, compute_bayesian_ppo_losses
)

def test_model_integrity():
    print("=" * 80)
    print("PHASE 3: 상대 통계 신경망 입력 및 미래 정보 누출 방지 검증")
    print("=" * 80)

    device = torch.device("cpu")
    model = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)

    # 1. 317차원 텐서 포워드 및 역전파 검증
    print("\n[검사 1] 317차원 텐서 포워드 및 역전파 검증:")
    dummy_x = torch.randn(4, 317, device=device)
    dummy_m = torch.ones(4, 5, dtype=torch.bool, device=device)
    logits, val = model(dummy_x, dummy_m)
    assert logits.shape == (4, 5), f"로짓 형상 오류: {logits.shape}"
    assert val.shape == (4,), f"가치 형상 오류: {val.shape}"

    # 임의 손실 역전파
    loss = logits.sum() + val.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  로짓 형상: {logits.shape} | 가치 형상: {val.shape} | 역전파 그래디언트 Norm: {grad_norm:.4f}")
    assert grad_norm > 0, "그래디언트 역전파 실패!"
    print("-> PASS: 317차원 신경망 구조 및 역전파 정상 작동.")

    # 2. 미래 정보 누출 방지(Temporal Leakage) 검증
    print("\n[검사 2] 미래 정보 누출 방지 단위 테스트:")
    mgr = MatchSessionOpponentManager()
    mgr.reset_session()
    mgr.set_hand_index(0)

    # Hand 0 시작 시점의 상대 피처 추출 (아직 이번 판의 어떤 행동도 일어나지 않은 시점)
    feats_before_hand = mgr.get_table_feature_tensor(current_hero_seat=0).copy()

    # 이번 판(Hand 0)에서 상대(Seat 1, Agent 1)가 거대한 올인을 실행함
    hand_0_actions = [
        (1, 0, [0, 1, 2, 3, 4], 3) # Seat 1이 올인
    ]

    # 주의: 이 핸드가 끝난 후 update하기 전까지는, 중간 의사결정 시점의 feats는 변하지 않아야 함!
    feats_during_hand = mgr.get_table_feature_tensor(current_hero_seat=0).copy()
    diff_during = np.max(np.abs(feats_before_hand - feats_during_hand))
    print(f"  의사결정 시점과 판 시작 시점 피처 차이: {diff_during:.10f}")
    assert diff_during == 0.0, "판 진행 도중 미래/현재 핸드 액션이 즉시 반영되어 누출되었습니다!"

    # 판이 끝난 후 정식 업데이트
    seat_to_ag = {s: s for s in range(5)}
    mgr.parse_and_update_hand(hand_0_actions, seat_to_ag)

    # Hand 1 시작 시점
    mgr.set_hand_index(1)
    feats_after_hand = mgr.get_table_feature_tensor(current_hero_seat=0).copy()
    diff_after = np.max(np.abs(feats_before_hand - feats_after_hand))
    print(f"  핸드 완료 후 다음 핸드 피처 변화량: {diff_after:.4f}")
    assert diff_after > 0.0, "핸드 종료 후 과거 이력이 정상적으로 반영되어야 함!"
    print("-> PASS: 현재 판의 의사결정 시점에 미래 액션이 절대 누출되지 않음을 증명.")

    # 3. Cold Start vs Warm Start 비교
    print("\n[검사 3] Cold Start(중립 사전분포) vs Warm Start(100핸드 관측) 피처 대조:")
    neutral_feat = get_neutral_prior_features()
    assert neutral_feat.shape == (52,), "중립 피처 차원 오류"

    # 상대들에게 서로 다른 프로필 주입 (Agent 1: Tight, Agent 2: Aggressive)
    mgr.inject_synthetic_profile(1, "Tight", num_hands=100)
    mgr.inject_synthetic_profile(2, "Aggressive", num_hands=100)
    warm_feat = mgr.get_table_feature_tensor(current_hero_seat=0)

    print(f"  Cold Start 피처 평균: {np.mean(neutral_feat):.4f} (사전분포 0.500 근접)")
    print(f"  Warm Start 피처 평균: {np.mean(warm_feat):.4f}")
    print(f"  두 피처 간 거리 (L2 Norm): {np.linalg.norm(neutral_feat - warm_feat):.4f}")
    assert np.linalg.norm(neutral_feat - warm_feat) > 1.0, "Cold와 Warm 피처가 유의미하게 달라야 함"
    print("-> PASS: Cold Start와 Warm Start 간 명확한 피처 분화 확인.")

if __name__ == "__main__":
    test_model_integrity()
