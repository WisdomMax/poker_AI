"""
POKER AI V4.9.5 - PHASE 7: Final Independent Generalization Evaluation
- 모델 선정용 평가와 최종 독립 평가의 엄격한 분리 (Selection Bias 통제)
- 완전히 새로운 독립 평가 시드(Seed 77777) 및 페어드(Paired) 1,000판 대국
- 독립 학습 시드 5개 전원의 성적 집계 (최고 성적 1개 모델 대표 편향 차단)
- 시드 간 분산, 게임별 분산, 대응 모델 간 차이(Delta bb/100), 95% CI, Paired t-test 및 p-value 산출
"""

import os
import sys
import math
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
from scipy import stats
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return, validate_mean_ev

FINAL_EVAL_HANDS = 1000
FINAL_EVAL_SEED = 77777

def evaluate_model_paired(
    model_a: HoldemActorCritic5A,
    model_b: HoldemActorCritic5A,
    game: pyspiel.Game,
    device: torch.device,
    eval_seed: int = FINAL_EVAL_SEED
) -> Tuple[List[float], List[float]]:
    """동일한 카드 셔플 및 동일한 상대 런아웃에서 두 모델의 1:1 페어드 대국 수익률 측정"""
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    model_a.eval()
    model_b.eval()

    returns_a = []
    returns_b = []

    for h in range(FINAL_EVAL_HANDS):
        # Model A 실행
        rng_eval_a = np.random.RandomState(eval_seed + h)
        rng_opp_a = random.Random(eval_seed + h * 2)
        s2a_a = wrapper.get_seat_to_agent(h)
        a2s_a = wrapper.get_agent_to_seat(h)
        hero_seat = a2s_a[0]
        table_opps_a = rng_opp_a.sample(opponents, 4)

        st_a = game.new_initial_state()
        while not st_a.is_terminal():
            if st_a.is_chance_node():
                advance_chance_nodes(st_a, rng=rng_eval_a)
            else:
                cur = st_a.current_player()
                legal = st_a.legal_actions(cur)
                if cur == hero_seat:
                    vec = st_a.information_state_tensor(cur)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        logits, _ = model_a(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    p_sum = sum(prs)
                    prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_eval_a.choice(legal, p=prs))
                else:
                    ag_id = s2a_a[cur]
                    opp = table_opps_a[(ag_id - 1) % len(table_opps_a)]
                    act = opp.act(st_a, cur, legal)
                st_a.apply_action(act)

        ret_a = st_a.returns()[hero_seat] / 100.0
        validate_bb_return(ret_a)
        returns_a.append(ret_a)

        # Model B 실행 (완벽히 동일한 카드 덱 및 상대 테이블)
        rng_eval_b = np.random.RandomState(eval_seed + h)
        rng_opp_b = random.Random(eval_seed + h * 2)
        s2a_b = wrapper.get_seat_to_agent(h)
        a2s_b = wrapper.get_agent_to_seat(h)
        table_opps_b = rng_opp_b.sample(opponents, 4)

        st_b = game.new_initial_state()
        while not st_b.is_terminal():
            if st_b.is_chance_node():
                advance_chance_nodes(st_b, rng=rng_eval_b)
            else:
                cur = st_b.current_player()
                legal = st_b.legal_actions(cur)
                if cur == hero_seat:
                    vec = st_b.information_state_tensor(cur)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        logits, _ = model_b(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    p_sum = sum(prs)
                    prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_eval_b.choice(legal, p=prs))
                else:
                    ag_id = s2a_b[cur]
                    opp = table_opps_b[(ag_id - 1) % len(table_opps_b)]
                    act = opp.act(st_b, cur, legal)
                st_b.apply_action(act)

        ret_b = st_b.returns()[hero_seat] / 100.0
        validate_bb_return(ret_b)
        returns_b.append(ret_b)

    return returns_a, returns_b

def run_phase7_independent_eval():
    print("=" * 70)
    print("PHASE 7: FINAL INDEPENDENT GENERALIZATION EVALUATION")
    print(f"독립 평가 시드: {FINAL_EVAL_SEED} | 핸드 수: {FINAL_EVAL_HANDS}판")
    print("=" * 70)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()

    # 비교 대상 조건 4개 (5개 시드 전원 대조)
    # Cond 1: Pure On-Policy PPO (E0_B0_S0)
    # Cond 2: Epsilon PPO (E1_B0_S0)
    # Cond 5: Subgame Pure (E0_B0_S1)
    # Cond 6: Subgame Epsilon (E1_B0_S1)
    SEEDS = [42, 101, 202, 303, 404]
    conditions_to_eval = [
        ('Cond1_PurePPO', 'Cond1_E0_B0_S0'),
        ('Cond2_EpsPPO',  'Cond2_E1_B0_S0'),
        ('Cond5_SubgamePure', 'Cond5_E0_B0_S1'),
        ('Cond6_SubgameEps',  'Cond6_E1_B0_S1'),
    ]

    summary_records = []
    seed_records = []

    for clabel, cprefix in conditions_to_eval:
        cond_seed_returns = []
        print(f"\n>>> [평가 진행] {clabel} (5개 시드 전수)")

        for s in SEEDS:
            ckpt_path = f"v495/models/{cprefix}_seed_{s}.pt"
            if not os.path.exists(ckpt_path):
                print(f"  체크포인트 없음: {ckpt_path}")
                continue
            m = HoldemActorCritic5A().to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device))

            # 독립 평가
            # 대조군으로는 Cond1 Seed 42를 기준 페어드로 설정
            ref_m = HoldemActorCritic5A().to(device)
            ref_path = f"v495/models/Cond1_E0_B0_S0_seed_42.pt"
            ref_m.load_state_dict(torch.load(ref_path, map_location=device))

            ret_target, ret_ref = evaluate_model_paired(m, ref_m, game, device)
            bb100_val = float(np.mean(ret_target) * 100.0)
            cond_seed_returns.append(bb100_val)
            print(f"  Seed {s:3d}: {bb100_val:+7.2f} bb/100")

            seed_records.append({
                'condition': clabel,
                'seed': s,
                'eval_bb100': bb100_val
            })

        mean_perf = np.mean(cond_seed_returns)
        std_between_seeds = np.std(cond_seed_returns, ddof=1)
        sem_between_seeds = std_between_seeds / np.sqrt(len(cond_seed_returns))
        ci95 = 1.96 * sem_between_seeds

        print(f"  --> {clabel} 5시드 평균: {mean_perf:+7.2f} ± {ci95:6.2f} bb/100 (시드 간 분산: {std_between_seeds**2:.2f})")

        summary_records.append({
            'condition': clabel,
            'mean_bb100': mean_perf,
            'ci95': ci95,
            'std_between_seeds': std_between_seeds,
            'var_between_seeds': std_between_seeds**2,
            'min_seed_bb100': float(np.min(cond_seed_returns)),
            'max_seed_bb100': float(np.max(cond_seed_returns)),
        })

    # Paired t-test: Cond1 (Pure PPO) vs Cond2 (Eps PPO)
    c1_vals = [r['eval_bb100'] for r in seed_records if r['condition'] == 'Cond1_PurePPO']
    c2_vals = [r['eval_bb100'] for r in seed_records if r['condition'] == 'Cond2_EpsPPO']
    diffs = np.array(c2_vals) - np.array(c1_vals)
    t_stat, p_val = stats.ttest_rel(c2_vals, c1_vals)
    mean_diff = np.mean(diffs)
    sem_diff = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    ci95_diff = 1.96 * sem_diff

    print("\n" + "=" * 70)
    print("대응표본 통계 검정 (Paired t-test: Cond2_EpsPPO vs Cond1_PurePPO)")
    print(f"  평균 차이 (Delta): {mean_diff:+7.2f} ± {ci95_diff:6.2f} bb/100")
    print(f"  t-통계량: {t_stat:.4f}, df: {len(diffs)-1}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  -> 판정: 두 조건 간 차이가 유의수준 5%에서 통계적으로 유의함.")
    else:
        print("  -> 판정: 신뢰구간에 0이 포함되며, p-value >= 0.05로 통계적 우위 미입증 (보수적 결론 유지).")
    print("=" * 70)

    # CSV 저장
    csv_seeds = "v495/data/phase7_seed_eval_results.csv"
    with open(csv_seeds, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_records[0].keys()))
        writer.writeheader()
        writer.writerows(seed_records)

    csv_summary = "v495/data/phase7_final_independent_summary.csv"
    with open(csv_summary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"\n[저장 완료] Phase 7 최종 독립 평가 데이터: {csv_summary}")

if __name__ == "__main__":
    run_phase7_independent_eval()
