"""
POKER AI V4.9.3 - PHASE 2: 통계적 독립성 검사 및 클러스터 부트스트랩
- 분석 과제:
  1. 단일 게임 단위(N=5000, df=4999) vs 독립 학습 시드 단위(N=5, df=4)의 통계적 차이 규명
  2. 시드별(42, 101, 202, 303, 404) 1,000판 독립 평가 데이터 추출
  3. 시드 단위 대응표본 t-검정:
     - Treatment vs Control
     - Curriculum vs Baseline
     - Curriculum vs Treatment
     - 시드별 차이, 평균 차이, 표준편차, 표준오차, df=4 기준 t값, 정확한 p-value, 95% CI
  4. 시드 단위 Cluster Bootstrap (B=10,000) 95% 신뢰구간 산출
  5. 수정된 기준선 통계 확정 및 CSV 저장
"""

import os
import sys
import math
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot

SEEDS = [42, 101, 202, 303, 404]
EVAL_HANDS_PER_SEED = 1000
BASE_EVAL_SEED = 77777

def t_distribution_cdf(t: float, df: int) -> float:
    """t-분포의 누적분포함수 CDF(t, df) 근사 계산 (scipy 의존성 제거)"""
    # df=4일 때 정확한 수치 적분 또는 정밀 근사
    # t_pdf(x) = Gamma((df+1)/2) / (sqrt(df*pi)*Gamma(df/2)) * (1 + x^2/df)^(-(df+1)/2)
    # df=4일 때: Gamma(2.5)/ (sqrt(4*pi)*Gamma(2)) = (0.75 * sqrt(pi)) / (2*sqrt(pi)*1) = 3/8 = 0.375
    # f(x) = 0.375 * (1 + x^2/4)^(-2.5)
    # F(t) = 0.5 + 0.375 * integral_0^t (1 + x^2/4)^(-2.5) dx
    if t == 0:
        return 0.5
    # Simpson rule 적분
    n_steps = 1000
    x_vals = np.linspace(0, abs(t), n_steps + 1)
    y_vals = 0.375 * (1.0 + (x_vals**2) / 4.0)**(-2.5)
    integral = (abs(t) / (3.0 * n_steps)) * (y_vals[0] + y_vals[-1] + 4.0 * np.sum(y_vals[1:-1:2]) + 2.0 * np.sum(y_vals[2:-1:2]))
    cdf = 0.5 + integral if t > 0 else 0.5 - integral
    return float(np.clip(cdf, 0.0, 1.0))

def run_phase2_audit():
    print("=" * 80)
    print("POKER AI V4.9.3 - PHASE 2: 통계적 독립성 검사 (시드 단위 vs 게임 단위)")
    print("=" * 80)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    models_config = {
        'Control_Masked': {
            'path_template': "v49/models/CondC_5A_Transfer_seed_{seed}_step_20000.pt",
            'mask_halfpot': True
        },
        'Treatment_5A': {
            'path_template': "v49/models/CondC_5A_Transfer_seed_{seed}_step_20000.pt",
            'mask_halfpot': False
        },
        'Baseline_40k': {
            'path_template': "v492/models/Baseline_seed_{seed}_step_40000.pt",
            'mask_halfpot': False
        },
        'Curriculum_40k': {
            'path_template': "v492/models/Treatment_Curriculum_seed_{seed}_step_40000.pt",
            'mask_halfpot': False
        }
    }

    # 1. 시드별 bb/100 데이터 수집 (5개 시드 x 4개 조건)
    # V4.9.2와 100% 동일한 평가 스케줄 실행하여 시드별 정확한 bb/100 산출
    seed_condition_bb100 = defaultdict(dict) # {seed: {cond: bb100}}

    for s_idx, seed_val in enumerate(SEEDS):
        print(f"  시드 {seed_val} (1,000판 페어드 대결) 평가 중...")
        loaded_models = {}
        for cond_name, cfg in models_config.items():
            path = cfg['path_template'].format(seed=seed_val)
            mask_hp = cfg['mask_halfpot']
            m = HoldemActorCritic5A().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            loaded_models[cond_name] = (m, mask_hp)

        cond_chips = {cond: 0.0 for cond in models_config.keys()}

        for h in range(EVAL_HANDS_PER_SEED):
            s2a = wrapper.get_seat_to_agent(h)
            a2s = wrapper.get_agent_to_seat(h)
            hero_seat = a2s[0]

            rng_opp = random.Random(BASE_EVAL_SEED + s_idx * 10000 + h)
            table_opps = rng_opp.sample(opponents, 4)
            hand_card_seed = BASE_EVAL_SEED + 50000 + s_idx * 10000 + h

            for cond_name, (model, mask_hp) in loaded_models.items():
                rng_cards = np.random.RandomState(hand_card_seed)
                rng_act = np.random.RandomState(hand_card_seed + 1)
                st = game.new_initial_state()

                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng=rng_cards)
                    else:
                        cur = st.current_player()
                        legal = st.legal_actions(cur)
                        if cur == hero_seat:
                            vec = st.information_state_tensor(cur)
                            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                            m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                            for a in legal: m[0, a] = True
                            if mask_hp and 4 in legal:
                                m[0, 4] = False
                                if not m.any(): m[0, 4] = True
                            with torch.no_grad():
                                logits, _ = model(x, m)
                                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                            valid = [a for a in legal if m[0, a]]
                            prs = [probs[a] for a in valid]
                            p_sum = sum(prs)
                            prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(valid))/len(valid)
                            act = int(rng_act.choice(valid, p=prs))
                        else:
                            ag_id = s2a[cur]
                            opp = table_opps[(ag_id - 1) % len(table_opps)]
                            act = opp.act(st, cur, legal)
                        st.apply_action(act)

                cond_chips[cond_name] += st.returns()[hero_seat]

        # bb/100 = 100 * total_chips / (100 * 1000) = total_chips / 1000
        for cond_name in models_config.keys():
            bb100_val = cond_chips[cond_name] / float(EVAL_HANDS_PER_SEED)
            seed_condition_bb100[seed_val][cond_name] = bb100_val

    # 2. 시드별 성적 출력
    print("\n[시드별 독립 성적 (bb/100)]")
    print(f"{'Seed':8s} | " + " | ".join([f"{c:16s}" for c in models_config.keys()]))
    print("-" * 80)
    for s in SEEDS:
        row_str = f"Seed {s:<4d} | "
        for c in models_config.keys():
            row_str += f"{seed_condition_bb100[s][c]:+14.2f} bb | "
        print(row_str)

    # 3. 독립 학습 시드 단위(N=5, df=4) 대응표본 t-검정
    pairs = [
        ('Treatment_5A', 'Control_Masked'),
        ('Curriculum_40k', 'Baseline_40k'),
        ('Curriculum_40k', 'Treatment_5A')
    ]

    t_crit_df4 = 2.7764 # t_crit for df=4, two-tailed alpha=0.05
    test_results = []

    print("\n[시드 단위(N=5, df=4) 대응표본 가설 검정 결과]")
    print("=" * 80)

    for treat_name, ctrl_name in pairs:
        diffs = []
        for s in SEEDS:
            d = seed_condition_bb100[s][treat_name] - seed_condition_bb100[s][ctrl_name]
            diffs.append(d)
        diffs = np.array(diffs)

        n = len(diffs)
        mean_d = float(np.mean(diffs))
        s_d = float(np.std(diffs, ddof=1))
        se_d = s_d / math.sqrt(n)
        t_stat = mean_d / max(1e-9, se_d)
        
        # 정확한 양측 p-value (df=4)
        p_val = 2.0 * (1.0 - t_distribution_cdf(abs(t_stat), df=4))
        ci95_margin = t_crit_df4 * se_d

        # 4. Cluster Bootstrap (B=10,000)
        rng_boot = np.random.RandomState(493)
        boot_diffs = []
        for _ in range(10000):
            sample_indices = rng_boot.choice(n, size=n, replace=True)
            boot_diffs.append(np.mean(diffs[sample_indices]))
        boot_ci_low = float(np.percentile(boot_diffs, 2.5))
        boot_ci_high = float(np.percentile(boot_diffs, 97.5))

        # 판정
        sig = "유의미함 (p < 0.05)" if p_val < 0.05 else "유의미하지 않음 (p >= 0.05)"

        print(f"\n비교: {treat_name} vs {ctrl_name}")
        print(f"  시드별 차이: {[round(x, 2) for x in diffs]}")
        print(f"  평균 차이 (Mean Diff):    {mean_d:+8.2f} bb/100")
        print(f"  차이 표준편차 (Std Diff):  {s_d:8.2f} bb/100")
        print(f"  표준오차 (SE):            {se_d:8.2f} bb/100")
        print(f"  자유도 (Degrees of Freedom): 4")
        print(f"  t-statistic:              {t_stat:+8.3f}")
        print(f"  정확한 p-value:           {p_val:8.4f} -> {sig}")
        print(f"  시드 t분포 95% CI:        [{mean_d - ci95_margin:+7.2f}, {mean_d + ci95_margin:+7.2f}] bb/100")
        print(f"  클러스터 부트스트랩 95% CI: [{boot_ci_low:+7.2f}, {boot_ci_high:+7.2f}] bb/100")

        test_results.append({
            'treatment': treat_name,
            'control': ctrl_name,
            'mean_diff_bb100': mean_d,
            'std_diff_bb100': s_d,
            'se_diff_bb100': se_d,
            'df': 4,
            't_stat': t_stat,
            'exact_p_value': p_val,
            'ci95_lower': mean_d - ci95_margin,
            'ci95_upper': mean_d + ci95_margin,
            'boot_ci95_lower': boot_ci_low,
            'boot_ci95_upper': boot_ci_high,
            'significant': sig
        })

    # 5. 결과 CSV 저장
    os.makedirs("v493/data", exist_ok=True)
    out_csv = "v493/data/phase2_seed_level_statistics.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(test_results[0].keys()))
        writer.writeheader()
        writer.writerows(test_results)

    # 시드별 성적 CSV 저장
    seed_csv = "v493/data/phase2_seed_by_condition_bb100.csv"
    seed_rows = []
    for s in SEEDS:
        row = {'seed': s}
        row.update(seed_condition_bb100[s])
        seed_rows.append(row)
    with open(seed_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)

    print(f"\nPhase 2 시드 단위 통계 검정 데이터 저장 완료: {out_csv}")
    print(f"Phase 2 시드별 bb/100 데이터 저장 완료: {seed_csv}")

if __name__ == '__main__':
    run_phase2_audit()
