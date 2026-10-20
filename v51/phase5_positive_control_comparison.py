"""
POKER AI V5.1 - PHASE 5: OPPONENT-AWARE ARCHITECTURE BENCHMARK ON POSITIVE CONTROL
- 3대 모델(Concatenation, Two-Tower, FiLM)의 통제된 조건(동일 데이터, 동일 최적화기, 동일 시드) 하에 성능 비교:
  1. 수렴 속도 (0, 50, 100, 200, 300, 500, 1000, 2000 스텝)
  2. 정책 분화도 (TVD: Total Variation Distance)
  3. 상대 피처에 대한 정책 그래디언트 민감도 ||d(Logit)/d(opp_feat)||
  4. Cross Entropy 손실
- 원시 데이터를 v51/data/phase5_architecture_positive_control.csv에 저장
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v51.models_comparison import ConcatenationActorCritic, TwoTowerActorCritic, FiLMActorCritic

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def compute_tvd(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p1 - p2)))

def run_phase5_benchmark():
    print("=" * 85)
    print("PHASE 5: 3대 아키텍처 POSITIVE CONTROL 지도 학습 벤치마크")
    print("=" * 85)

    # 1. 대상 상태 구축 (JTs Facing 3-Bet All-in)
    gen = TargetedSubgameGenerator()
    st, _ = gen.generate_subgame_state("SitB_Facing_3Bet_AllIn", "JTs", hero_seat=2, rng=np.random.RandomState(42))

    hero_feat_np = normalize_vec(st.information_state_tensor(2))
    assert len(hero_feat_np) == 265, f"Hero feature length should be 265, got {len(hero_feat_np)}"

    legal_mask_np = np.zeros(5, dtype=bool)
    for a in st.legal_actions(2):
        legal_mask_np[a] = True

    # 2. 상대 프로필 구축 (Tight vs Aggressive)
    mgr = MatchSessionOpponentManager()
    mgr.inject_synthetic_profile(3, "Tight", num_hands=100)
    opp_feat_tight = mgr.get_table_feature_tensor(current_hero_seat=2)

    mgr.reset_session()
    mgr.inject_synthetic_profile(3, "Aggressive", num_hands=100)
    opp_feat_aggr = mgr.get_table_feature_tensor(current_hero_seat=2)

    assert len(opp_feat_tight) == 52, f"Opponent feature length should be 52, got {len(opp_feat_tight)}"

    hero_t = torch.tensor(hero_feat_np, dtype=torch.float32)
    opp_tight_t = torch.tensor(opp_feat_tight, dtype=torch.float32)
    opp_aggr_t = torch.tensor(opp_feat_aggr, dtype=torch.float32)
    mask_t = torch.tensor(legal_mask_np, dtype=torch.bool)

    # 3. 벤치마크 구성
    architectures = [
        ("Model A (Concatenation)", ConcatenationActorCritic),
        ("Model B (Two-Tower MLP)", TwoTowerActorCritic),
        ("Model C (FiLM Gating)", FiLMActorCritic)
    ]

    steps_to_eval = [0, 50, 100, 200, 300, 500, 1000, 2000]
    results = []

    print(f"\n{'모델':<25} | {'스텝':<5} | {'Tight Fold%':<12} | {'Aggr Call%':<11} | {'TVD':<8} | {'Loss':<8} | {'Opp 민감도':<10}")
    print("-" * 90)

    for model_name, model_cls in architectures:
        torch.manual_seed(12345)
        model = model_cls()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 스텝별 평가 루프
        step_idx = 0
        for target_step in steps_to_eval:
            # 훈련 진행
            while step_idx < target_step:
                model.train()
                optimizer.zero_grad()

                # Batch: [Tight sample (target: Fold=0), Aggressive sample (target: Call=1)]
                b_hero = torch.stack([hero_t, hero_t])
                b_opp = torch.stack([opp_tight_t, opp_aggr_t])
                b_mask = torch.stack([mask_t, mask_t])
                targets = torch.tensor([0, 1], dtype=torch.long)

                raw_logits, probs, _ = model(b_hero, b_opp, b_mask)
                loss = criterion(raw_logits, targets)
                loss.backward()
                optimizer.step()
                step_idx += 1

            # 평가 수행
            model.eval()

            # 민감도 계산을 위해 opp_feat에 requires_grad 설정
            t_hero = hero_t.unsqueeze(0)
            t_opp_tight = opp_tight_t.unsqueeze(0).clone().detach().requires_grad_(True)
            t_mask = mask_t.unsqueeze(0)

            logits_t, probs_t, _ = model(t_hero, t_opp_tight, t_mask)
            prob_tight_np = probs_t.squeeze(0).detach().cpu().numpy()

            # 민감도: Fold logit (action 0)의 opp_feat에 대한 그래디언트 노름
            logits_t[0, 0].backward(retain_graph=True)
            grad_norm = t_opp_tight.grad.norm().item()

            # Aggressive 평가
            t_opp_aggr = opp_aggr_t.unsqueeze(0)
            _, probs_a, _ = model(t_hero, t_opp_aggr, t_mask)
            prob_aggr_np = probs_a.squeeze(0).detach().cpu().numpy()

            # 손실 계산
            b_hero = torch.stack([hero_t, hero_t])
            b_opp = torch.stack([opp_tight_t, opp_aggr_t])
            b_mask = torch.stack([mask_t, mask_t])
            targets = torch.tensor([0, 1], dtype=torch.long)
            with torch.no_grad():
                raw_logits_eval, _, _ = model(b_hero, b_opp, b_mask)
                eval_loss = criterion(raw_logits_eval, targets).item()

            tight_fold_pct = prob_tight_np[0] * 100.0
            aggr_call_pct = prob_aggr_np[1] * 100.0
            tvd = compute_tvd(prob_tight_np, prob_aggr_np)

            print(f"{model_name:<25} | {target_step:<5} | {tight_fold_pct:9.2f}% | {aggr_call_pct:9.2f}% | {tvd:6.4f} | {eval_loss:6.4f} | {grad_norm:8.4f}")

            results.append({
                'model': model_name,
                'step': target_step,
                'tight_fold_pct': tight_fold_pct,
                'aggr_call_pct': aggr_call_pct,
                'tvd': tvd,
                'loss': eval_loss,
                'opp_gradient_norm': grad_norm
            })

    # CSV 저장
    csv_path = "v51/data/phase5_architecture_positive_control.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[저장 완료] Phase 5 벤치마크 결과 CSV: {csv_path}")

    # 요약 분석 출력
    print("\n" + "=" * 85)
    print("PHASE 5 판정 및 요약 분석:")
    print("=" * 85)
    for model_name, _ in architectures:
        m_res = [r for r in results if r['model'] == model_name]
        step100 = next(r for r in m_res if r['step'] == 100)
        step300 = next(r for r in m_res if r['step'] == 300)
        step2000 = next(r for r in m_res if r['step'] == 2000)
        print(f"  [{model_name}]")
        print(f"    - Step 100:  Tight Fold {step100['tight_fold_pct']:.2f}% | Aggr Call {step100['aggr_call_pct']:.2f}% | TVD {step100['tvd']:.4f} | 민감도 {step100['opp_gradient_norm']:.4f}")
        print(f"    - Step 300:  Tight Fold {step300['tight_fold_pct']:.2f}% | Aggr Call {step300['aggr_call_pct']:.2f}% | TVD {step300['tvd']:.4f} | 민감도 {step300['opp_gradient_norm']:.4f}")
        print(f"    - Step 2000: Tight Fold {step2000['tight_fold_pct']:.2f}% | Aggr Call {step2000['aggr_call_pct']:.2f}% | TVD {step2000['tvd']:.4f} | 민감도 {step2000['opp_gradient_norm']:.4f}")
    print()

if __name__ == '__main__':
    run_phase5_benchmark()
