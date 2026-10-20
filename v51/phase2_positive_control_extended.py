"""
POKER AI V5.1 - PHASE 2: Extended Positive Control Convergence Study
- 스텝 확장: 100, 300, 500, 1000, 2000, 5000 스텝
- 학습률 비교: lr=1e-3 vs lr=3e-3
- 측정 지표:
  * Training Loss & Validation Loss
  * Tight Fold% & Aggressive Call%
  * Total Variation Distance (TVD)
  * Actor Logit Difference (Logit_0 - Logit_1)
- 판정: Concatenation 구조가 90% 이상 완전 분화에 도달 가능한지 최종 실증
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

def run_positive_control_extended(lr: float = 1e-3, label: str = "lr1e-3"):
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    model = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 1. 고정 게임 상태 (Train x_info vs Val x_info)
    train_info = np.random.randn(265).astype(np.float32)
    val_info = np.random.randn(265).astype(np.float32)

    # 2. 상대 피처 (Tight vs Aggressive)
    mgr = MatchSessionOpponentManager()
    mgr.inject_synthetic_profile(1, "Tight", num_hands=100)
    feat_tight = mgr.get_table_feature_tensor(current_hero_seat=0)

    mgr.reset_session()
    mgr.inject_synthetic_profile(1, "Aggressive", num_hands=100)
    feat_aggr = mgr.get_table_feature_tensor(current_hero_seat=0)

    # 훈련용 텐서
    x_train_tight = torch.tensor(np.concatenate([train_info, feat_tight]), dtype=torch.float32, device=device).unsqueeze(0)
    x_train_aggr = torch.tensor(np.concatenate([train_info, feat_aggr]), dtype=torch.float32, device=device).unsqueeze(0)

    # 검증용 텐서 (정보 상태가 약간 다른 상태에서도 상대 피처에 반응하는지 검증)
    x_val_tight = torch.tensor(np.concatenate([val_info, feat_tight]), dtype=torch.float32, device=device).unsqueeze(0)
    x_val_aggr = torch.tensor(np.concatenate([val_info, feat_aggr]), dtype=torch.float32, device=device).unsqueeze(0)

    mask = torch.ones((1, 5), dtype=torch.bool, device=device)
    target_tight = torch.tensor([0], dtype=torch.long, device=device)
    target_aggr = torch.tensor([1], dtype=torch.long, device=device)

    checkpoints = [100, 300, 500, 1000, 2000, 5000]
    records = []

    print(f"\n--- [Positive Control 확장 학습: {label} (최대 5,000 스텝)] ---")

    for step in range(1, 5001):
        model.train()
        optimizer.zero_grad()
        lg_t, _ = model(x_train_tight, mask)
        lg_a, _ = model(x_train_aggr, mask)

        loss_t = criterion(lg_t, target_tight)
        loss_a = criterion(lg_a, target_aggr)
        train_loss = loss_t + loss_a
        train_loss.backward()
        optimizer.step()

        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                # Train 검증
                pr_t = F.softmax(lg_t, dim=-1).squeeze(0).cpu().numpy()
                pr_a = F.softmax(lg_a, dim=-1).squeeze(0).cpu().numpy()

                # Val 검증
                lg_vt, _ = model(x_val_tight, mask)
                lg_va, _ = model(x_val_aggr, mask)
                val_loss = (criterion(lg_vt, target_tight) + criterion(lg_va, target_aggr)).item()
                pr_vt = F.softmax(lg_vt, dim=-1).squeeze(0).cpu().numpy()
                pr_va = F.softmax(lg_va, dim=-1).squeeze(0).cpu().numpy()

                tvd = 0.5 * np.sum(np.abs(pr_t - pr_a))
                tvd_val = 0.5 * np.sum(np.abs(pr_vt - pr_va))

                logit_diff_t = float(lg_t[0, 0] - lg_t[0, 1]) # Tight에서 Fold - Call
                logit_diff_a = float(lg_a[0, 1] - lg_a[0, 0]) # Aggr에서 Call - Fold

            rec = {
                'step': step,
                'lr': lr,
                'train_loss': float(train_loss.item()),
                'val_loss': float(val_loss),
                'tight_fold_pct': float(pr_t[0] * 100.0),
                'aggr_call_pct': float(pr_a[1] * 100.0),
                'val_tight_fold_pct': float(pr_vt[0] * 100.0),
                'val_aggr_call_pct': float(pr_va[1] * 100.0),
                'tvd_train': float(tvd),
                'tvd_val': float(tvd_val),
                'logit_diff_tight': logit_diff_t,
                'logit_diff_aggr': logit_diff_a
            }
            records.append(rec)

            print(f"  Step {step:4d} | Train Loss={rec['train_loss']:.5f} | Val Loss={rec['val_loss']:.5f} | Tight Fold={rec['tight_fold_pct']:5.2f}% | Aggr Call={rec['aggr_call_pct']:5.2f}% | TVD={rec['tvd_train']:.4f}")

    return records

def run_phase2_audit():
    print("=" * 85)
    print("PHASE 2: POSITIVE CONTROL 확장 수렴 한계 전수 검증 (100 ~ 5,000 스텝)")
    print("=" * 85)

    res_1e3 = run_positive_control_extended(lr=1e-3, label="lr=1e-3 (기본)")
    res_3e3 = run_positive_control_extended(lr=3e-3, label="lr=3e-3 (가속)")

    all_records = res_1e3 + res_3e3

    # CSV 저장
    csv_out = "v51/data/phase2_positive_control_extended.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader()
        w.writerows(all_records)
    print(f"\n[저장 완료] Positive Control 확장 감사 CSV: {csv_out}")

    # 최종 판정
    final_1e3 = res_1e3[-1]
    final_3e3 = res_3e3[-1]

    print("\n" + "=" * 85)
    print("POSITIVE CONTROL 확장 학습 최종 판정:")
    print("=" * 85)
    print(f"  [lr=1e-3] 5,000스텝 최종: Tight Fold={final_1e3['tight_fold_pct']:.2f}%, Aggr Call={final_1e3['aggr_call_pct']:.2f}%, TVD={final_1e3['tvd_train']:.4f}")
    print(f"  [lr=3e-3] 5,000스텝 최종: Tight Fold={final_3e3['tight_fold_pct']:.2f}%, Aggr Call={final_3e3['aggr_call_pct']:.2f}%, TVD={final_3e3['tvd_train']:.4f}")

    if final_1e3['tight_fold_pct'] >= 90.0 and final_1e3['aggr_call_pct'] >= 90.0:
        print("\n  -> 판정: [기존 결론 철회 및 Concatenation 용량 입증!]")
        print("     Concatenation 구조는 1,000~5,000스텝의 충분한 학습을 거치면")
        print(f"     Tight Fold {final_1e3['tight_fold_pct']:.2f}%, Aggressive Call {final_1e3['aggr_call_pct']:.2f}% (TVD > 0.90)에 도달합니다.")
        print("     따라서 'Concatenation 구조가 원천적으로 상대 정보를 분화할 수 없다'는 주장은 사실이 아니며 공식 철회합니다!")
    else:
        print("\n  -> 판정: [5,000스텝에서도 90% 미달, 구조적 저항 확인]")

if __name__ == "__main__":
    run_phase2_audit()
