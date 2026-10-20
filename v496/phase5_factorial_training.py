"""
POKER AI V4.9.6 - PHASE 4 & 5: Stable On-Policy PPO 2x2 Factorial Controlled Experiment
(Factor Norm: Raw Input vs Normalized Input | Factor Ent: Entropy OFF vs ON)
- 4대 요인 조건 (2x2 Factorial Design):
  * CondA_Raw_EntOFF:  Raw Input (10000 chips) | Entropy Bonus 0.00
  * CondB_Norm_EntOFF: Normalized Input (0~1)  | Entropy Bonus 0.00
  * CondC_Raw_EntON:   Raw Input (10000 chips) | Entropy Bonus 0.01
  * CondD_Norm_EntON:  Normalized Input (0~1)  | Entropy Bonus 0.01
- 5개 독립 시드 (42, 101, 202, 303, 404), 각 20,000 에피소드 학습
- 순수 On-Policy PPO (No Epsilon, No Bias Clamp, Masked Categorical)
- 실시간 로깅: Actor/Critic Loss, Entropy, Approx KL, Clip Fraction, Grad Norm (Pre/Post Clip), Logits
- 비학습 독립 평가 600판 대국 & 상한 [-100, +400] BB 자동 Assertion
- Two-Way ANOVA (Norm 주효과, Ent 주효과, 상호작용) 분석
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
from scipy import stats
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return, validate_mean_ev
from v496.stable_ppo import StableHoldemActorCritic5A, compute_stable_ppo_losses

SEEDS = [42, 101, 202, 303, 404]
TRAIN_EPISODES = 20000
EVAL_HANDS = 600

CONDITIONS = [
    {'name': 'CondA_Raw_EntOFF',  'normalize': False, 'ent_coef': 0.00},
    {'name': 'CondB_Norm_EntOFF', 'normalize': True,  'ent_coef': 0.00},
    {'name': 'CondC_Raw_EntON',   'normalize': False, 'ent_coef': 0.01},
    {'name': 'CondD_Norm_EntON',  'normalize': True,  'ent_coef': 0.01},
]

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """인덱스 223~264 금액 정보를 10,000.0으로 나누어 정규화"""
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def evaluate_hand_policy(
    model: StableHoldemActorCritic5A,
    gen: TargetedSubgameGenerator,
    hand_name: str,
    normalize_input: bool,
    device: torch.device
) -> Tuple[float, float, float]:
    """SitB 3-Bet 직면 시 Fold%, Call%, Entropy 측정"""
    st, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', hand_name, hero_seat=2)
    vec = st.information_state_tensor(2)
    if normalize_input:
        vec = normalize_vec(vec)
    legal = st.legal_actions(2)
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True

    model.eval()
    with torch.no_grad():
        lg, _ = model(x, m)
        probs = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()

    fold_p = float(probs[0] * 100.0)
    call_p = float(probs[1] * 100.0) if len(probs) > 1 else 0.0
    safe_p = probs[probs > 1e-12]
    ent = float(-np.sum(safe_p * np.log(safe_p)))
    return fold_p, call_p, ent

def evaluate_fullgame(
    model: StableHoldemActorCritic5A,
    game: pyspiel.Game,
    normalize_input: bool,
    device: torch.device,
    eval_seed: int = 88888
) -> float:
    """비학습 독립 평가 시드로 600판 풀-게임 대국 진행"""
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
                    if normalize_input:
                        vec = normalize_vec(vec)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    act, _, _, _ = model.sample_action(x, m, deterministic=False)
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
    validate_mean_ev(bb100 / 100.0, metric_name="Fullgame_bb100")
    return bb100

def train_single_run(cond: Dict, seed_val: int, device: torch.device = torch.device("cpu")) -> Dict:
    cname = cond['name']
    normalize_input = cond['normalize']
    ent_coef = cond['ent_coef']

    save_path = f"v496/models/{cname}_seed_{seed_val}.pt"
    log_path = f"v496/logs/{cname}_seed_{seed_val}.csv"

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    gen = TargetedSubgameGenerator(game)
    model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)

    # 이전 체크포인트 존재 시 빠른 재평가
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        aa_fold, aa_call, aa_ent = evaluate_hand_policy(model, gen, 'AA', normalize_input, device)
        jts_fold, _, _ = evaluate_hand_policy(model, gen, 'JTs', normalize_input, device)
        t72_fold, _, _ = evaluate_hand_policy(model, gen, '72o', normalize_input, device)
        eval_bb100 = evaluate_fullgame(model, game, normalize_input, device)
        return {
            'condition': cname,
            'factor_norm': normalize_input,
            'factor_ent': ent_coef > 0,
            'seed': seed_val,
            'eval_bb100': eval_bb100,
            'aa_fold_pct': aa_fold,
            'aa_call_pct': aa_call,
            'jts_fold_pct': jts_fold,
            '72o_fold_pct': t72_fold,
            'policy_entropy': aa_ent,
            'approx_kl': 0.0,
            'clip_frac': 0.0,
            'grad_norm_pre': 0.0,
            'grad_norm_post': 0.0,
            'elapsed_sec': 0.0
        }

    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    optimizer = optim.Adam(model.parameters(), lr=0.0002, eps=1e-5)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed_val + 500)
    rng_choice = random.Random(seed_val + 600)

    batch_trajectories = []
    t_start = time.time()
    n_updates = 0
    running_entropy = 0.0
    running_kl = 0.0
    running_clip = 0.0
    running_gnorm_pre = 0.0
    running_gnorm_post = 0.0

    step_logs = []

    for ep in range(1, TRAIN_EPISODES + 1):
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
                    if normalize_input:
                        vec = normalize_vec(vec)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True

                    # 순수 On-Policy Masked Categorical 샘플링
                    act, log_prob, _, val_est = model.sample_action(x, m, deterministic=False)
                    traj.append((vec, m.squeeze(0).cpu().numpy(), act, log_prob))
                else:
                    act = table_opps[cur].act(st, cur, legal)
                st.apply_action(act)

        ret_chips = st.returns()[hero_seat]
        ret_bb = ret_chips / 100.0 # BB 단위 스케일링
        validate_bb_return(ret_bb)

        for v, m, a, olp in traj:
            batch_trajectories.append((v, m, a, olp, ret_bb))

        # 64 에피소드마다 PPO 업데이트
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
            ep_kl = 0.0
            ep_clip = 0.0
            ep_ent = 0.0
            ep_gnorm_pre = 0.0
            ep_gnorm_post = 0.0

            for _ in range(4):
                tot_l, pol_l, val_l, ent_l, kl_val, clip_f = compute_stable_ppo_losses(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv,
                    clip_eps=0.2, vf_coef=0.5, ent_coef=ent_coef
                )

                if math.isnan(tot_l.item()) or math.isinf(tot_l.item()):
                    print(f"\n[치명적 오류] NaN/Inf Loss 감지! (Cond: {cname}, Ep: {ep}) -> 학습 중단")
                    break

                optimizer.zero_grad()
                tot_l.backward()

                # 클리핑 전 Gradient Norm 기록
                total_norm_pre = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_pre += param_norm.item() ** 2
                total_norm_pre = total_norm_pre ** 0.5

                # Gradient Clipping (max_norm = 1.0)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                total_norm_post = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_post += param_norm.item() ** 2
                total_norm_post = total_norm_post ** 0.5

                optimizer.step()

                ep_kl += kl_val
                ep_clip += clip_f
                ep_ent += ent_l.item()
                ep_gnorm_pre += total_norm_pre
                ep_gnorm_post += total_norm_post

            running_kl += ep_kl / 4.0
            running_clip += ep_clip / 4.0
            running_entropy += ep_ent / 4.0
            running_gnorm_pre += ep_gnorm_pre / 4.0
            running_gnorm_post += ep_gnorm_post / 4.0
            n_updates += 1
            batch_trajectories = []

            # 5,000 에피소드마다 스텝 로깅
            if ep % 5000 == 0:
                with torch.no_grad():
                    # AA 상황 로짓 프로브
                    f_p, c_p, p_ent = evaluate_hand_policy(model, gen, 'AA', normalize_input, device)
                step_logs.append({
                    'episode': ep,
                    'entropy': running_entropy / max(1, n_updates),
                    'approx_kl': running_kl / max(1, n_updates),
                    'gnorm_pre': running_gnorm_pre / max(1, n_updates),
                    'gnorm_post': running_gnorm_post / max(1, n_updates),
                    'aa_fold_pct': f_p,
                    'aa_call_pct': c_p
                })

    elapsed = time.time() - t_start
    torch.save(model.state_dict(), save_path)

    # 최종 정책 및 대국 평가
    aa_fold, aa_call, aa_ent = evaluate_hand_policy(model, gen, 'AA', normalize_input, device)
    jts_fold, _, _ = evaluate_hand_policy(model, gen, 'JTs', normalize_input, device)
    t72_fold, _, _ = evaluate_hand_policy(model, gen, '72o', normalize_input, device)
    eval_bb100 = evaluate_fullgame(model, game, normalize_input, device)

    avg_ent = running_entropy / max(1, n_updates)
    avg_kl = running_kl / max(1, n_updates)
    avg_clip = running_clip / max(1, n_updates)
    avg_gnorm_pre = running_gnorm_pre / max(1, n_updates)
    avg_gnorm_post = running_gnorm_post / max(1, n_updates)

    print(f"  [{cname}] Seed {seed_val:3d} 완료 | bb/100: {eval_bb100:+6.1f} | AA Fold: {aa_fold:5.1f}%, Call: {aa_call:5.1f}% | KL: {avg_kl:.4f} | GNorm: {avg_gnorm_pre:.2f}->{avg_gnorm_post:.2f} ({elapsed:.1f}s)")

    return {
        'condition': cname,
        'factor_norm': normalize_input,
        'factor_ent': ent_coef > 0,
        'seed': seed_val,
        'eval_bb100': eval_bb100,
        'aa_fold_pct': aa_fold,
        'aa_call_pct': aa_call,
        'jts_fold_pct': jts_fold,
        '72o_fold_pct': t72_fold,
        'policy_entropy': avg_ent,
        'approx_kl': avg_kl,
        'clip_frac': avg_clip,
        'grad_norm_pre': avg_gnorm_pre,
        'grad_norm_post': avg_gnorm_post,
        'elapsed_sec': elapsed
    }

def run_phase5_experiment():
    print("=" * 75)
    print("PHASE 4 & 5: STABLE ON-POLICY PPO 2x2 FACTORIAL EXPERIMENT")
    print("4개 조건 x 5개 독립 시드 (총 20회 학습, 각 20,000 에피소드)")
    print("=" * 75)

    raw_results = []
    for cond in CONDITIONS:
        print(f"\n>>> [조건 시작] {cond['name']} (Norm: {cond['normalize']}, EntCoef: {cond['ent_coef']})")
        for seed in SEEDS:
            res = train_single_run(cond, seed)
            raw_results.append(res)

    # 1. 원시 데이터 CSV 저장
    raw_csv = "v496/data/phase5_factorial_raw.csv"
    with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
        writer.writeheader()
        writer.writerows(raw_results)
    print(f"\n[저장 완료] 원시 실험 데이터: {raw_csv}")

    # 2. 조건별 요약 통계
    print("\n" + "=" * 75)
    print("4대 조건별 집계 요약 (Mean ± 95% CI)")
    print("=" * 75)
    cond_summary = []
    for cond in CONDITIONS:
        cname = cond['name']
        vals = [r['eval_bb100'] for r in raw_results if r['condition'] == cname]
        aa_folds = [r['aa_fold_pct'] for r in raw_results if r['condition'] == cname]
        aa_calls = [r['aa_call_pct'] for r in raw_results if r['condition'] == cname]
        t72_folds = [r['72o_fold_pct'] for r in raw_results if r['condition'] == cname]
        kls = [r['approx_kl'] for r in raw_results if r['condition'] == cname]
        gnorms = [r['grad_norm_pre'] for r in raw_results if r['condition'] == cname]

        m_v = float(np.mean(vals))
        s_v = float(np.std(vals, ddof=1))
        ci95 = 1.96 * (s_v / np.sqrt(len(vals)))

        print(f"{cname:<20} | bb/100: {m_v:+7.2f} ± {ci95:6.2f} (Std: {s_v:6.2f}) | AA Fold: {np.mean(aa_folds):5.1f}%, Call: {np.mean(aa_calls):5.1f}% | 72o Fold: {np.mean(t72_folds):5.1f}% | KL: {np.mean(kls):.4f} | GNormPre: {np.mean(gnorms):.2f}")
        cond_summary.append({
            'condition': cname,
            'mean_bb100': m_v,
            'ci95': ci95,
            'std_bb100': s_v,
            'aa_fold_pct': float(np.mean(aa_folds)),
            'aa_call_pct': float(np.mean(aa_calls)),
            '72o_fold_pct': float(np.mean(t72_folds)),
            'approx_kl': float(np.mean(kls)),
            'grad_norm_pre': float(np.mean(gnorms))
        })

    sum_csv = "v496/data/phase5_condition_summary.csv"
    with open(sum_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(cond_summary[0].keys()))
        writer.writeheader()
        writer.writerows(cond_summary)
    print(f"[저장 완료] 조건별 요약 데이터: {sum_csv}")

    # 3. Two-Way ANOVA 분석
    print("\n" + "=" * 75)
    print("TWO-WAY ANOVA ANALYSIS (Factor Normalization x Factor Entropy)")
    print("=" * 75)
    Y = np.array([r['eval_bb100'] for r in raw_results])
    Factor_N = np.array([1 if r['factor_norm'] else -1 for r in raw_results])
    Factor_E = np.array([1 if r['factor_ent'] else -1 for r in raw_results])

    N_tot = len(Y)
    SS_total = np.sum((Y - np.mean(Y))**2)
    SS_Norm = (np.sum(Factor_N * Y)**2) / N_tot
    SS_Ent = (np.sum(Factor_E * Y)**2) / N_tot
    SS_Inter = (np.sum(Factor_N * Factor_E * Y)**2) / N_tot
    SS_error = max(1e-6, SS_total - (SS_Norm + SS_Ent + SS_Inter))

    df_effect = 1
    df_error = N_tot - 4 # 20 - 4 = 16
    MS_error = SS_error / df_error

    effects = [
        ('Factor Norm (Input Normalization)', SS_Norm),
        ('Factor Ent (Entropy Regularization)', SS_Ent),
        ('Interaction Norm x Ent', SS_Inter),
    ]

    anova_rows = []
    print(f"{'Source':<35} | {'SS':>10} | {'df':>3} | {'MS':>10} | {'F-value':>8} | {'p-value':>8} | {'eta_p^2':>7}")
    print("-" * 95)
    for name, ss in effects:
        ms = ss / df_effect
        f_val = ms / MS_error
        p_val = 1.0 - stats.f.cdf(f_val, df_effect, df_error)
        eta_p2 = ss / (ss + SS_error)
        print(f"{name:<35} | {ss:10.2f} | {df_effect:3d} | {ms:10.2f} | {f_val:8.3f} | {p_val:8.4f} | {eta_p2:7.4f}")
        anova_rows.append({
            'source': name,
            'sum_of_squares': ss,
            'df': df_effect,
            'mean_square': ms,
            'f_value': f_val,
            'p_value': p_val,
            'partial_eta_sq': eta_p2
        })

    print(f"{'Error (Residual)':<35} | {SS_error:10.2f} | {df_error:3d} | {MS_error:10.2f} | {'-':>8} | {'-':>8} | {'-':>7}")
    print(f"{'Total':<35} | {SS_total:10.2f} | {N_tot-1:3d} | {'-':>10} | {'-':>8} | {'-':>8} | {'-':>7}")

    anova_csv = "v496/data/phase5_anova_table.csv"
    with open(anova_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(anova_rows[0].keys()))
        writer.writeheader()
        writer.writerows(anova_rows)
    print(f"[저장 완료] Two-Way ANOVA 테이블: {anova_csv}")

if __name__ == "__main__":
    run_phase5_experiment()
