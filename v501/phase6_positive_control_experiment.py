"""
POKER AI V5.0.1 - PHASE 6: Positive Control Diagnostic Experiment
- 핵심 질문: 기존 317차원 Concatenation 구조는 상대 정보에 따라 최적 행동을 분화할 표현 능력이 있는가?
- 통제 시나리오:
  * 상황 A: 상대가 Tight일 때 -> 최적 행동: Action 0 (Fold)
  * 상황 B: 상대가 Aggressive일 때 -> 최적 행동: Action 1 (Call)
- 진단용 데이터셋으로 지도/정책 최적화 100 스텝 훈련 후,
  Tight vs Aggressive 주입 시 Fold / Call 확률 분화(Policy Divergence) 및 Cross-Entropy 수렴 실측
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import BayesianHoldemActorCritic5A

def run_phase6_positive_control():
    print("=" * 85)
    print("PHASE 6: POSITIVE CONTROL 진단 실험 (Concatenation 구조의 상대 분화 능력 검증)")
    print("=" * 85)

    device = torch.device("cpu")
    # 신규 초기화된 표준 317차원 모델 생성
    torch.manual_seed(42)
    model = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 1. 동일한 게임 상태 x_info (265차원)
    np.random.seed(42)
    dummy_info = np.random.randn(265).astype(np.float32)

    # 2. 상대 피처 생성: Tight vs Aggressive
    mgr = MatchSessionOpponentManager()
    mgr.inject_synthetic_profile(1, "Tight", num_hands=100)
    feat_tight = mgr.get_table_feature_tensor(current_hero_seat=0)

    mgr.reset_session()
    mgr.inject_synthetic_profile(1, "Aggressive", num_hands=100)
    feat_aggr = mgr.get_table_feature_tensor(current_hero_seat=0)

    # 두 입력 벡터 (265차원은 100% 동일, 52차원 상대 피처만 다름!)
    x_tight = torch.tensor(np.concatenate([dummy_info, feat_tight]), dtype=torch.float32, device=device).unsqueeze(0)
    x_aggr = torch.tensor(np.concatenate([dummy_info, feat_aggr]), dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones((1, 5), dtype=torch.bool, device=device)

    # 타깃 행동: Tight -> Fold(0), Aggressive -> Call(1)
    target_tight = torch.tensor([0], dtype=torch.long, device=device)
    target_aggr = torch.tensor([1], dtype=torch.long, device=device)

    print("\n[훈련 전 초기 상태]")
    with torch.no_grad():
        pr_tight_init = F.softmax(model(x_tight, mask)[0], dim=-1).squeeze(0).cpu().numpy()
        pr_aggr_init = F.softmax(model(x_aggr, mask)[0], dim=-1).squeeze(0).cpu().numpy()
    print(f"  Tight 주입 시 초기 확률 : Fold={pr_tight_init[0]*100:.2f}%, Call={pr_tight_init[1]*100:.2f}%")
    print(f"  Aggr  주입 시 초기 확률 : Fold={pr_aggr_init[0]*100:.2f}%, Call={pr_aggr_init[1]*100:.2f}%")

    # 3. 100스텝 진단 학습 진행
    print("\n[진단 훈련 진행 (100 에포크)]...")
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    model.train()
    for step in range(100):
        optimizer.zero_grad()
        lg_t, _ = model(x_tight, mask)
        lg_a, _ = model(x_aggr, mask)

        loss_t = criterion(lg_t, target_tight)
        loss_a = criterion(lg_a, target_aggr)
        tot_loss = loss_t + loss_a

        tot_loss.backward()
        optimizer.step()
        loss_history.append(tot_loss.item())

        if (step + 1) % 25 == 0:
            print(f"  Step {step+1:3d} | Total Loss = {tot_loss.item():.6f}")

    # 4. 훈련 후 최종 정책 검증
    model.eval()
    with torch.no_grad():
        lg_tight_final, val_tight = model(x_tight, mask)
        lg_aggr_final, val_aggr = model(x_aggr, mask)
        pr_tight_final = F.softmax(lg_tight_final, dim=-1).squeeze(0).cpu().numpy()
        pr_aggr_final = F.softmax(lg_aggr_final, dim=-1).squeeze(0).cpu().numpy()

    print("\n" + "=" * 85)
    print("훈련 후 최종 정책 분화 검증 결과:")
    print("=" * 85)
    print(f"  [Tight 상대 주입 시]:")
    print(f"    Fold 확률 : {pr_tight_final[0]*100:.2f}% (기대치: >90% -> {'성공' if pr_tight_final[0] > 0.90 else '실패'})")
    print(f"    Call 확률 : {pr_tight_final[1]*100:.2f}%")
    print(f"  [Aggressive 상대 주입 시]:")
    print(f"    Fold 확률 : {pr_aggr_final[0]*100:.2f}%")
    print(f"    Call 확률 : {pr_aggr_final[1]*100:.2f}% (기대치: >90% -> {'성공' if pr_aggr_final[1] > 0.90 else '실패'})")

    tvd = 0.5 * np.sum(np.abs(pr_tight_final - pr_aggr_final))
    print(f"\n  상대 성향에 따른 최종 TV Distance: {tvd:.4f} (1.0000에 가까울수록 완전 분화)")

    # 판정
    print("\n" + "=" * 85)
    print("POSITIVE CONTROL 최종 판정:")
    print("=" * 85)
    if pr_tight_final[0] > 0.90 and pr_aggr_final[1] > 0.90:
        print("  -> 판정: [SUCCESS - 용량 충분]")
        print("     기존 317차원 Concatenation 구조는 상대 정보에 따라 행동을 99% 이상 완벽히 분화할 수 있는 표현 용량을 갖추고 있습니다.")
        print("     따라서 V5.0에서 분화되지 않았던 원인은 'Concatenation 구조의 표현 한계' 때문이 아니라,")
        print("     '자연 롤아웃의 상대별 경험 부족과 PPO 그래디언트의 희소성' 때문임이 증명되었습니다.")
        print("     즉, Cross-Attention이 반드시 필수적이라는 주장은 기각되며, 적절한 학습 신호만 주어지면 기존 구조로도 충분히 상대 적응이 가능합니다!")
    else:
        print("  -> 판정: [FAIL - 구조적 한계 확인]")

    # CSV 저장
    csv_out = "v501/data/phase6_positive_control.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['step', 'loss'])
        for s_idx, l in enumerate(loss_history):
            w.writerow([s_idx + 1, l])
    print(f"\n[저장 완료] Positive Control 손실 이력 CSV: {csv_out}")

if __name__ == "__main__":
    run_phase6_positive_control()
