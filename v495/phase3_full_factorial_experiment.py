"""
POKER AI V4.9.5 - PHASE 3: Full 2x2x2 Factorial Controlled Experiment
(Factor E: Epsilon, Factor B: Bias Clamp, Factor S: Subgame)
- 8개 완전 직교 요인 조건 (2x2x2 Design)
- 5개 독립 학습 시드 (42, 101, 202, 303, 404)
- 각 조건 10,000 에피소드 학습 및 체크포인트 보존
- 수집 메트릭:
  * Fullgame bb/100 (독립 600판 대국, 물리적 상한 Assertion 준수)
  * AA / JTs / 72o 3-Bet 직면 Fold% 및 Call%
  * Actor Entropy, Approx KL, Clip Fraction
  * 3-Way ANOVA (F-stat, p-value, eta_p^2)
"""

import os
import sys
import time
import math
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A, compute_ppo_losses_5a
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator, TARGET_HOLE_CARDS
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return, validate_mean_ev

SEEDS = [42, 101, 202, 303, 404]
TRAIN_EPISODES = 10000
EVAL_HANDS = 600

SUBGAME_SITUATIONS = [
    ('SitA_Facing_OpenRaise', 3),
    ('SitB_Facing_3Bet_AllIn', 2),
    ('SitB_Facing_3Bet_Pot', 2),
    ('SitC_Facing_4Bet', 3),
    ('SitD_Flop_Facing_Bet', 2),
    ('SitE_Turn_Facing_Raise', 2),
    ('SitF_River_Facing_AllIn', 2)
]
HAND_KEYS = list(TARGET_HOLE_CARDS.keys())

# 8개 완전 직교 조건 정의
CONDITIONS = [
    {'name': 'Cond1_E0_B0_S0', 'eps': 0.0,  'clamp': False, 'subgame': 0.0},
    {'name': 'Cond2_E1_B0_S0', 'eps': 0.15, 'clamp': False, 'subgame': 0.0},
    {'name': 'Cond3_E0_B1_S0', 'eps': 0.0,  'clamp': True,  'subgame': 0.0},
    {'name': 'Cond4_E1_B1_S0', 'eps': 0.15, 'clamp': True,  'subgame': 0.0},
    {'name': 'Cond5_E0_B0_S1', 'eps': 0.0,  'clamp': False, 'subgame': 0.30},
    {'name': 'Cond6_E1_B0_S1', 'eps': 0.15, 'clamp': False, 'subgame': 0.30},
    {'name': 'Cond7_E0_B1_S1', 'eps': 0.0,  'clamp': True,  'subgame': 0.30},
    {'name': 'Cond8_E1_B1_S1', 'eps': 0.15, 'clamp': True,  'subgame': 0.30},
]

def evaluate_hand_policy(model: HoldemActorCritic5A, gen: TargetedSubgameGenerator, hand_name: str, device: torch.device) -> Tuple[float, float]:
    """특정 핸드로 3-Bet 직면 시 Fold%와 Call% 측정"""
    st, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', hand_name, hero_seat=2)
    vec = st.information_state_tensor(2)
    legal = st.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, m)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    fold_p = float(probs[0] * 100.0)
    call_p = float(probs[1] * 100.0) if len(probs) > 1 else 0.0
    return fold_p, call_p

def evaluate_fullgame(model: HoldemActorCritic5A, game: pyspiel.Game, device: torch.device, eval_seed: int = 99999) -> float:
    """비학습 독립 평가 시드로 풀-게임 600판 대국 평가"""
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    rng_eval = np.random.RandomState(eval_seed)
    rng_opp = random.Random(eval_seed + 1)
    total_chips = 0.0
    model.eval()

    for h in range(EVAL_HANDS):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        table_opps = rng_opp.sample(opponents, 4)

        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_eval)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    vec = st.information_state_tensor(cur)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    p_sum = sum(prs)
                    prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng_eval.choice(legal, p=prs))
                else:
                    ag_id = s2a[cur]
                    opp = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp.act(st, cur, legal)
                st.apply_action(act)

        ret_chips = st.returns()[hero_seat]
        validate_chips_return(ret_chips, hand_idx=h)
        validate_bb_return(ret_chips / 100.0, hand_idx=h)
        total_chips += ret_chips

    bb100 = float(total_chips / float(EVAL_HANDS))
    validate_mean_ev(bb100 / 100.0, metric_name="Fullgame_Eval_bb100")
    return bb100

def train_single_run(cond: Dict, seed_val: int, device: torch.device = torch.device("cpu")) -> Dict:
    cname = cond['name']
    eps_val = cond['eps']
    clamp_val = cond['clamp']
    subgame_r = cond['subgame']

    save_path = f"v495/models/{cname}_seed_{seed_val}.pt"
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    gen = TargetedSubgameGenerator(game)
    model = HoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)

    # 이전 체크포인트가 있으면 즉시 로드
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        aa_fold, aa_call = evaluate_hand_policy(model, gen, 'AA', device)
        jts_fold, _ = evaluate_hand_policy(model, gen, 'JTs', device)
        t72_fold, _ = evaluate_hand_policy(model, gen, '72o', device)
        eval_bb100 = evaluate_fullgame(model, game, device)
        return {
            'condition': cname,
            'factor_E': eps_val > 0,
            'factor_B': clamp_val,
            'factor_S': subgame_r > 0,
            'seed': seed_val,
            'eval_bb100': eval_bb100,
            'aa_fold_pct': aa_fold,
            'aa_call_pct': aa_call,
            'jts_fold_pct': jts_fold,
            '72o_fold_pct': t72_fold,
            'actor_entropy': 0.0,
            'approx_kl': 0.0,
            'clip_frac': 0.0,
            'updates': 0,
            'elapsed_sec': 0.0
        }

    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    # 초기 가중치: v49 Transfer 시드 가중치 로드
    init_path = f"v49/models/CondC_5A_Transfer_seed_{seed_val}_step_20000.pt"
    if os.path.exists(init_path):
        model.load_state_dict(torch.load(init_path, map_location=device))
        if clamp_val:
            with torch.no_grad():
                model.actor_head.bias.data.clamp_(-3.0, 3.0)

    optimizer = optim.Adam(model.parameters(), lr=0.0002, eps=1e-5)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed_val + 100)
    rng_choice = random.Random(seed_val + 200)

    batch_trajectories = []
    t_start = time.time()
    n_updates = 0
    running_entropy = 0.0
    running_kl = 0.0
    running_clip = 0.0

    for ep in range(1, TRAIN_EPISODES + 1):
        is_subgame = (rng_choice.random() < subgame_r)
        hero_seat = 2

        if is_subgame:
            sit_name, target_seat = rng_choice.choice(SUBGAME_SITUATIONS)
            hand_name = rng_choice.choice(HAND_KEYS)
            hero_seat = target_seat
            st, _ = gen.generate_subgame_state(sit_name, hand_name, hero_seat=hero_seat, rng=rng_env)
            table_opps = {p: rng_choice.choice(opponents) for p in range(5) if p != hero_seat}
        else:
            s2a = wrapper.get_seat_to_agent(ep)
            a2s = wrapper.get_agent_to_seat(ep)
            hero_seat = a2s[0]
            sampled_opps = rng_choice.sample(opponents, 4)
            table_opps = {s: sampled_opps[(s2a[s] - 1) % 4] for s in range(5) if s != hero_seat}
            st = game.new_initial_state()

        traj = []
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_env)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    vec = st.information_state_tensor(cur)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True

                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                    # 행동 선택
                    if eps_val > 0 and rng_env.random() < eps_val:
                        act = int(rng_env.choice(legal))
                    else:
                        prs = [probs[a] for a in legal]
                        p_sum = sum(prs)
                        prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                        act = int(rng_env.choice(legal, p=prs))

                    traj.append((vec, m.squeeze(0).cpu().numpy(), act, log_probs[act]))
                else:
                    act = table_opps[cur].act(st, cur, legal)
                st.apply_action(act)

        ret_chips = st.returns()[hero_seat]
        ret_bb = ret_chips / 100.0
        for v, m, a, olp in traj:
            batch_trajectories.append((v, m, a, olp, ret_bb))

        if ep % 64 == 0 and batch_trajectories:
            b_v, b_m, b_a, b_olp, b_r = zip(*batch_trajectories)
            t_states = torch.tensor(np.array(b_v), dtype=torch.float32, device=device)
            t_masks = torch.tensor(np.array(b_m), dtype=torch.bool, device=device)
            t_actions = torch.tensor(b_a, dtype=torch.int64, device=device)
            t_returns = torch.tensor(b_r, dtype=torch.float32, device=device)
            t_old_log_p = torch.tensor(b_olp, dtype=torch.float32, device=device)

            with torch.no_grad():
                _, values = model(t_states, t_masks)
                adv = t_returns - values
                adv_std = adv.std()
                if adv_std > 1e-8:
                    adv = (adv - adv.mean()) / adv_std

            model.train()
            for _ in range(4):
                tot_loss, pol_loss, val_loss, ent = compute_ppo_losses_5a(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv
                )
                optimizer.zero_grad()
                tot_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if clamp_val:
                    with torch.no_grad():
                        model.actor_head.bias.data.clamp_(-3.0, 3.0)

            with torch.no_grad():
                lg, _ = model(t_states, t_masks)
                new_lps = torch.log_softmax(lg, dim=-1).gather(1, t_actions.unsqueeze(-1)).squeeze(-1)
                ratios = torch.exp(new_lps - t_old_log_p)
                kl = ((ratios - 1.0) - torch.log(ratios)).mean().item()
                clip_f = ((ratios < 0.8) | (ratios > 1.2)).float().mean().item()

            running_entropy += ent.item()
            running_kl += kl
            running_clip += clip_f
            n_updates += 1
            batch_trajectories = []

    elapsed = time.time() - t_start
    torch.save(model.state_dict(), save_path)

    # 정책 평가
    aa_fold, aa_call = evaluate_hand_policy(model, gen, 'AA', device)
    jts_fold, _ = evaluate_hand_policy(model, gen, 'JTs', device)
    t72_fold, _ = evaluate_hand_policy(model, gen, '72o', device)
    eval_bb100 = evaluate_fullgame(model, game, device)

    avg_entropy = running_entropy / max(1, n_updates)
    avg_kl = running_kl / max(1, n_updates)
    avg_clip = running_clip / max(1, n_updates)

    print(f"  [{cname}] Seed {seed_val:3d} 완료 | bb/100: {eval_bb100:+6.1f} | AA Fold: {aa_fold:5.1f}%, Call: {aa_call:5.1f}% | KL: {avg_kl:.4f} ({elapsed:.1f}s)")

    return {
        'condition': cname,
        'factor_E': eps_val > 0,
        'factor_B': clamp_val,
        'factor_S': subgame_r > 0,
        'seed': seed_val,
        'eval_bb100': eval_bb100,
        'aa_fold_pct': aa_fold,
        'aa_call_pct': aa_call,
        'jts_fold_pct': jts_fold,
        '72o_fold_pct': t72_fold,
        'actor_entropy': avg_entropy,
        'approx_kl': avg_kl,
        'clip_frac': avg_clip,
        'updates': n_updates,
        'elapsed_sec': elapsed
    }

def run_phase3_experiment():
    print("=" * 70)
    print("PHASE 3: 2x2x2 FULL FACTORIAL EXPERIMENT (8 CONDITIONS x 5 SEEDS)")
    print("=" * 70)

    raw_results = []
    total_runs = len(CONDITIONS) * len(SEEDS)
    current_run = 0

    for cond in CONDITIONS:
        print(f"\n>>> 조건: {cond['name']} (Eps: {cond['eps']}, Clamp: {cond['clamp']}, Subgame: {cond['subgame']*100:.0f}%)")
        for seed in SEEDS:
            current_run += 1
            res = train_single_run(cond, seed)
            raw_results.append(res)

    # 1. 원시 결과 CSV 저장
    raw_csv_path = "v495/data/phase3_factorial_8cond_raw.csv"
    keys = list(raw_results[0].keys())
    with open(raw_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(raw_results)
    print(f"\n[저장 완료] 원시 실험 데이터: {raw_csv_path}")

    # 2. 3-Way Factorial ANOVA 분석
    print("\n" + "=" * 70)
    print("3-WAY FACTORIAL ANOVA ANALYSIS (E x B x S)")
    print("=" * 70)

    # 데이터 추출
    Y = np.array([r['eval_bb100'] for r in raw_results])
    E = np.array([1 if r['factor_E'] else -1 for r in raw_results])
    B = np.array([1 if r['factor_B'] else -1 for r in raw_results])
    S = np.array([1 if r['factor_S'] else -1 for r in raw_results])

    N = len(Y)
    SS_total = np.sum((Y - np.mean(Y))**2)

    # 직교 설계를 이용한 제곱합(Sum of Squares) 계산
    # 주효과
    SS_E = (np.sum(E * Y)**2) / N
    SS_B = (np.sum(B * Y)**2) / N
    SS_S = (np.sum(S * Y)**2) / N

    # 2차 상호작용
    SS_EB = (np.sum(E * B * Y)**2) / N
    SS_ES = (np.sum(E * S * Y)**2) / N
    SS_BS = (np.sum(B * S * Y)**2) / N

    # 3차 상호작용
    SS_EBS = (np.sum(E * B * S * Y)**2) / N

    # 오차 제곱합
    SS_model = SS_E + SS_B + SS_S + SS_EB + SS_ES + SS_BS + SS_EBS
    SS_error = max(1e-6, SS_total - SS_model)

    df_effect = 1
    df_error = N - 8 # 40 - 8 = 32
    MS_error = SS_error / df_error

    effects = [
        ('Factor E (Epsilon 15%)', SS_E),
        ('Factor B (Bias Clamp)', SS_B),
        ('Factor S (Subgame 30%)', SS_S),
        ('Interaction E x B', SS_EB),
        ('Interaction E x S', SS_ES),
        ('Interaction B x S', SS_BS),
        ('Interaction E x B x S', SS_EBS),
    ]

    anova_rows = []
    print(f"{'Source':<26} | {'SS':>10} | {'df':>3} | {'MS':>10} | {'F-value':>8} | {'p-value':>8} | {'eta_p^2':>7}")
    print("-" * 85)

    from scipy import stats
    for name, ss in effects:
        ms = ss / df_effect
        f_val = ms / MS_error
        # p-value 계산 (scipy f distribution)
        p_val = 1.0 - stats.f.cdf(f_val, df_effect, df_error)
        eta_p2 = ss / (ss + SS_error)
        print(f"{name:<26} | {ss:10.2f} | {df_effect:3d} | {ms:10.2f} | {f_val:8.3f} | {p_val:8.4f} | {eta_p2:7.4f}")
        anova_rows.append({
            'source': name,
            'sum_of_squares': ss,
            'df': df_effect,
            'mean_square': ms,
            'f_value': f_val,
            'p_value': p_val,
            'partial_eta_sq': eta_p2
        })

    print(f"{'Error (Residual)':<26} | {SS_error:10.2f} | {df_error:3d} | {MS_error:10.2f} | {'-':>8} | {'-':>8} | {'-':>7}")
    print(f"{'Total':<26} | {SS_total:10.2f} | {N-1:3d} | {'-':>10} | {'-':>8} | {'-':>8} | {'-':>7}")

    # ANOVA 요약 테이블 CSV 저장
    anova_csv_path = "v495/data/phase3_factorial_anova_table.csv"
    with open(anova_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(anova_rows[0].keys()))
        writer.writeheader()
        writer.writerows(anova_rows)
    print(f"[저장 완료] ANOVA 요약 테이블: {anova_csv_path}")

    # 3. 8개 조건별 집계 요약
    print("\n" + "=" * 70)
    print("8대 조건별 집계 요약 (Mean ± Std, 95% CI)")
    print("=" * 70)
    cond_summary = []
    for cond in CONDITIONS:
        cname = cond['name']
        vals = [r['eval_bb100'] for r in raw_results if r['condition'] == cname]
        aa_folds = [r['aa_fold_pct'] for r in raw_results if r['condition'] == cname]
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        ci95 = 1.96 * (std_v / np.sqrt(len(vals)))
        mean_aa_f = np.mean(aa_folds)
        print(f"{cname:<18} | bb/100: {mean_v:+7.2f} ± {ci95:6.2f} (Std: {std_v:6.2f}) | AA Fold: {mean_aa_f:5.1f}%")
        cond_summary.append({
            'condition': cname,
            'mean_bb100': mean_v,
            'ci95': ci95,
            'std_bb100': std_v,
            'aa_fold_mean': mean_aa_f
        })

    summary_csv_path = "v495/data/phase3_condition_summary.csv"
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(cond_summary[0].keys()))
        writer.writeheader()
        writer.writerows(cond_summary)
    print(f"[저장 완료] 조건별 요약 데이터: {summary_csv_path}")

if __name__ == "__main__":
    run_phase3_experiment()
