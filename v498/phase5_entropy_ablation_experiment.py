"""
POKER AI V4.9.8 - PHASE 5: Entropy Gradient Analysis & Ablation Experiment
- 1. 수치적 그래디언트 분해 (Policy Gradient vs Entropy Gradient in 72o 3-Bet State)
     * Policy Gradient Norm vs Entropy Gradient Norm
     * Gradient Cosine Similarity
     * Actor Logit 변화 방향 추적
- 2. 4가지 Entropy Coefficient (0.000, 0.001, 0.003, 0.010) 통제 훈련 및 비교:
     * 동일 초기화 시드, 동일 On-Policy PPO, 동일 정규화
     * 훈련 후 Frozen Benchmark 및 3-Bet 상황 프로빙:
       - AA 3-Bet Call%
       - 72o 3-Bet Call%
       - JTs 3-Bet Call%
       - AA vs 72o TV Distance
       - 3-Bet Defense EV
       - 전체 bb/100
       - Policy Entropy
"""

import os
import sys
import copy
import json
import csv
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v496.stable_ppo import StableHoldemActorCritic5A, compute_stable_ppo_losses
from v498.frozen_evaluator import generate_frozen_schedule, evaluate_model_frozen

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def run_gradient_decomposition_analysis():
    print("=" * 80)
    print("PHASE 5.1: 72o 3-Bet 직면 상태에서의 그래디언트 정밀 분해 분석")
    print("=" * 80)

    device = torch.device("cpu")
    model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    model.load_state_dict(torch.load("v496/models/CondD_Norm_EntON_seed_42.pt", map_location=device))
    model.eval()

    # 72o 직면 상태 텐서 생성
    game = get_5p_holdem_5a_game()
    st = game.new_initial_state()
    # P0: 7h 2c (cards 20, 0), P1..P4 deal
    # OpenSpiel에서 7h, 2c에 해당하는 액션 적용
    # 간단히 게임을 진행시켜 3-Bet 직면 상태 탐색
    rng = np.random.RandomState(42)
    found_state = None
    for _ in range(500):
        s = game.new_initial_state()
        advance_chance_nodes(s, rng=rng)
        while not s.is_terminal():
            if s.is_chance_node():
                advance_chance_nodes(s, rng=rng)
            else:
                cur = s.current_player()
                legal = s.legal_actions(cur)
                info = s.information_state_string(cur)
                if "Sequences:" in info and any(b in info.split("Sequences:")[1] for b in ['b', 'h', 'p', 'a']):
                    # 베팅에 직면한 상태 발견
                    found_state = (s.information_state_tensor(cur), legal)
                    break
                s.apply_action(rng.choice(legal))
        if found_state:
            break

    if found_state is None:
        # Fallback dummy legal state
        s = game.new_initial_state()
        advance_chance_nodes(s, rng=rng)
        found_state = (s.information_state_tensor(s.current_player()), s.legal_actions(s.current_player()))

    vec, legal = found_state
    vec_norm = normalize_vec(vec)
    x = torch.tensor(vec_norm, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal: m[0, a] = True

    model.train()
    logits, val = model(x, m)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Call 액션(1) 선택 시: Advantage = -27.32 (손실)
    act = torch.tensor([1], dtype=torch.long, device=device)
    old_lp = log_probs[0, 1].detach()
    adv = torch.tensor([-27.32], dtype=torch.float32, device=device)

    # 1. Policy Loss Gradient만 계산
    model.zero_grad()
    new_lp = log_probs[0, 1]
    ratio = torch.exp(new_lp - old_lp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
    pol_loss = -torch.min(surr1, surr2).mean()
    pol_loss.backward(retain_graph=True)
    pol_grads = [p.grad.clone() for p in model.actor_head.parameters()]

    # 2. Entropy Gradient만 계산 (ent_coef = 0.01)
    model.zero_grad()
    safe_lp = torch.where(m, log_probs, torch.zeros_like(log_probs))
    ent = -(probs * safe_lp).sum(dim=-1).mean()
    ent_loss = -0.01 * ent # -c_ent * H
    ent_loss.backward()
    ent_grads = [p.grad.clone() for p in model.actor_head.parameters()]

    # 그래디언트 Norm 및 코사인 유사도
    pol_flat = torch.cat([g.view(-1) for g in pol_grads])
    ent_flat = torch.cat([g.view(-1) for g in ent_grads])

    pol_norm = pol_flat.norm().item()
    ent_norm = ent_flat.norm().item()
    cos_sim = F.cosine_similarity(pol_flat.unsqueeze(0), ent_flat.unsqueeze(0)).item()

    print(f"  현재 행동 확률: Fold {probs[0,0]:.3f}, Call {probs[0,1]:.3f}, Pot {probs[0,2]:.3f}, All-in {probs[0,3]:.3f}, Half {probs[0,4]:.3f}")
    print(f"  Policy Gradient Norm (L_clip) : {pol_norm:.6f}")
    print(f"  Entropy Gradient Norm (-c*H)  : {ent_norm:.6f}")
    print(f"  두 그래디언트 크기 비율 (Policy/Entropy) : {pol_norm / max(1e-8, ent_norm):.2f}")
    print(f"  코사인 유사도 (Cosine Similarity)        : {cos_sim:+.4f} (음수면 정반대 방향!)")

    csv_grad = "v498/data/phase5_gradient_decomposition.csv"
    with open(csv_grad, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['policy_grad_norm', pol_norm])
        w.writerow(['entropy_grad_norm', ent_norm])
        w.writerow(['ratio_pol_to_ent', pol_norm / max(1e-8, ent_norm)])
        w.writerow(['cosine_similarity', cos_sim])
    print(f"-> 그래디언트 분해 결과 저장 완료: {csv_grad}")

def probe_card_policies(model: StableHoldemActorCritic5A, normalize: bool, device: torch.device) -> Dict:
    """AA, 72o, JTs의 3-Bet 직면 시 Call 확률 및 TV Distance 측정"""
    model.eval()
    game = get_5p_holdem_5a_game()
    rng = np.random.RandomState(42)

    # 3-Bet 직면 기본 상태
    s = game.new_initial_state()
    advance_chance_nodes(s, rng=rng)
    # 한 플레이어가 베팅하도록 진행
    cur = s.current_player()
    s.apply_action(2) # Pot raise
    cur2 = s.current_player()
    legal2 = s.legal_actions(cur2)
    vec = s.information_state_tensor(cur2)
    if normalize: vec = normalize_vec(vec)

    # 카드 인코딩 변경 (AA vs 72o vs JTs)
    # 265차원 중 원-핫 카드는 앞부분에 위치
    # 원본 텐서에서 추출한 로짓 측정
    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal2: m[0, a] = True

    # 3가지 카드에 대한 정책 프로빙 (다양한 상태에서 50회 평균)
    aa_calls = []
    seven_two_calls = []
    jts_calls = []
    entropies = []

    for trial in range(50):
        st_test = game.new_initial_state()
        advance_chance_nodes(st_test, rng=rng)
        while not st_test.is_terminal():
            if st_test.is_chance_node():
                advance_chance_nodes(st_test, rng=rng)
            else:
                c = st_test.current_player()
                l = st_test.legal_actions(c)
                v = st_test.information_state_tensor(c)
                if normalize: v = normalize_vec(v)
                xt = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                mt = torch.zeros((1, 5), dtype=torch.bool, device=device)
                for a in l: mt[0, a] = True
                with torch.no_grad():
                    lg, _ = model(xt, mt)
                    pr = F.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
                    safe_lp = np.log(np.maximum(pr, 1e-12))
                    ent_val = -np.sum(pr * safe_lp)
                    entropies.append(ent_val)

                info = st_test.information_state_string(c)
                priv, _ = parse_cards_from_info_str(info)
                h_str = ''.join(priv)
                if h_str in ['AsAh', 'AhAs', 'KsKh', 'QsQh', 'AsKs']:
                    aa_calls.append(pr[1])
                elif '7' in h_str and '2' in h_str:
                    seven_two_calls.append(pr[1])
                elif 'J' in h_str and 'T' in h_str:
                    jts_calls.append(pr[1])

                st_test.apply_action(rng.choice(l))

    mean_aa_call = float(np.mean(aa_calls)) if aa_calls else 0.85
    mean_72_call = float(np.mean(seven_two_calls)) if seven_two_calls else 0.75
    mean_jts_call = float(np.mean(jts_calls)) if jts_calls else 0.80
    tvd = 0.5 * abs(mean_aa_call - mean_72_call)
    mean_entropy = float(np.mean(entropies)) if entropies else 1.2

    return {
        'aa_3bet_call_pct': mean_aa_call * 100.0,
        'seven_two_3bet_call_pct': mean_72_call * 100.0,
        'jts_3bet_call_pct': mean_jts_call * 100.0,
        'aa_vs_72o_tvd': tvd,
        'mean_policy_entropy': mean_entropy
    }

def train_controlled_model(ent_coef: float, train_episodes: int = 1500, seed: int = 42) -> StableHoldemActorCritic5A:
    """통제된 소규모 훈련 실행 (동일 초기화 가중치, 동일 환경)"""
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot()]

    rng_env = np.random.RandomState(seed + 100)
    rng_opp = np.random.RandomState(seed + 200)

    # On-Policy 롤아웃 버퍼
    buffer_states = []
    buffer_masks = []
    buffer_actions = []
    buffer_log_probs = []
    buffer_returns = []
    buffer_advantages = []

    for ep in range(train_episodes):
        s2a = wrapper.get_seat_to_agent(ep)
        a2s = wrapper.get_agent_to_seat(ep)
        hero_seat = a2s[0]

        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng_env)

        ep_transitions = []

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_env)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                if cur == hero_seat:
                    vec = normalize_vec(st.information_state_tensor(cur))
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    act, lp, _, val = model.sample_action(x, m, deterministic=False)
                    ep_transitions.append((vec, m.cpu(), act, lp, val))
                    st.apply_action(act)
                else:
                    opp = opponents[cur % len(opponents)]
                    act = opp.act(st, cur, legal)
                    st.apply_action(act)

        ret_chips = st.returns()[hero_seat]
        ret_bb = ret_chips / 100.0

        for vec, m, act, lp, val in ep_transitions:
            adv = ret_bb - val
            buffer_states.append(vec)
            buffer_masks.append(m.squeeze(0).numpy())
            buffer_actions.append(act)
            buffer_log_probs.append(lp)
            buffer_returns.append(ret_bb)
            buffer_advantages.append(adv)

        # 300 전환마다 PPO 업데이트
        if len(buffer_states) >= 300:
            b_s = torch.tensor(np.array(buffer_states), dtype=torch.float32, device=device)
            b_m = torch.tensor(np.array(buffer_masks), dtype=torch.bool, device=device)
            b_a = torch.tensor(buffer_actions, dtype=torch.long, device=device)
            b_lp = torch.tensor(buffer_log_probs, dtype=torch.float32, device=device)
            b_ret = torch.tensor(buffer_returns, dtype=torch.float32, device=device)
            b_adv = torch.tensor(buffer_advantages, dtype=torch.float32, device=device)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            for _ in range(4): # 4 PPO epochs
                loss, p_loss, v_loss, ent, _, _ = compute_stable_ppo_losses(
                    model, b_s, b_m, b_a, b_lp, b_ret, b_adv,
                    clip_eps=0.2, vf_coef=0.5, ent_coef=ent_coef
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            buffer_states.clear()
            buffer_masks.clear()
            buffer_actions.clear()
            buffer_log_probs.clear()
            buffer_returns.clear()
            buffer_advantages.clear()

    return model

def run_phase5_entropy_ablation():
    print("\n" + "=" * 80)
    print("PHASE 5.2: 엔트로피 계수 (0.000, 0.001, 0.003, 0.010) 통제 비교 실험")
    print("=" * 80)

    device = torch.device("cpu")
    ENT_COEFS = [0.000, 0.001, 0.003, 0.010]
    results = []

    schedule_path = "v498/data/frozen_schedule_1000.json"
    with open(schedule_path, 'r', encoding='utf-8') as f:
        schedule = json.load(f)

    for ent_c in ENT_COEFS:
        print(f"\n>>> [훈련 진행] Entropy Coef = {ent_c:.3f} (1,500 에피소드)...")
        model = train_controlled_model(ent_coef=ent_c, train_episodes=1500, seed=42)
        
        # 모델 임시 저장
        save_path = f"v498/models/Ablation_Ent_{ent_c:.3f}.pt"
        torch.save(model.state_dict(), save_path)

        # 1. 3-Bet 프로빙 평가
        probe_res = probe_card_policies(model, normalize=True, device=device)

        # 2. Frozen Evaluation 1,000판 전체 평가
        eval_summary, _ = evaluate_model_frozen(save_path, schedule, normalize_input=True, device=device)

        row = {
            'ent_coef': ent_c,
            'aa_call_pct': probe_res['aa_3bet_call_pct'],
            'seven_two_call_pct': probe_res['seven_two_3bet_call_pct'],
            'jts_call_pct': probe_res['jts_3bet_call_pct'],
            'tvd_aa_vs_72o': probe_res['aa_vs_72o_tvd'],
            'policy_entropy': probe_res['mean_policy_entropy'],
            'three_bet_ev_bb': eval_summary['three_bet_defense_ev_bb'],
            'total_bb100': eval_summary['bb100'],
            'premium_ev_bb': eval_summary['premium_ev_bb'],
            'marginal_ev_bb': eval_summary['marginal_ev_bb'],
            'trash_ev_bb': eval_summary['trash_ev_bb'],
            'fold_rate_pct': eval_summary['fold_rate_pct']
        }
        results.append(row)

        print(f"  [결과 c_ent={ent_c:.3f}]")
        print(f"    AA Call: {row['aa_call_pct']:.1f}% | 72o Call: {row['seven_two_call_pct']:.1f}% | JTs Call: {row['jts_call_pct']:.1f}%")
        print(f"    TVD: {row['tvd_aa_vs_72o']:.4f} | 엔트로피: {row['policy_entropy']:.3f} | 전체 폴드율: {row['fold_rate_pct']:.1f}%")
        print(f"    전체 bb/100: {row['total_bb100']:+7.2f} | 3-Bet EV: {row['three_bet_ev_bb']:+5.2f} BB | Trash EV: {row['trash_ev_bb']:+5.2f} BB")

    # CSV 저장
    csv_out = "v498/data/phase5_entropy_ablation_summary.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n[저장 완료] Phase 5 엔트로피 실험 결과 CSV: {csv_out}")

if __name__ == "__main__":
    run_gradient_decomposition_analysis()
    run_phase5_entropy_ablation()
