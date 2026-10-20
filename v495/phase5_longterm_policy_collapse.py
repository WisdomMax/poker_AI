"""
POKER AI V4.9.5 - PHASE 5: Long-Term Policy Collapse & Gradient Explosion Tracking
- 0k -> 10k -> 20k -> 30k -> 40k 장기 학습 궤적 전수 추적
- Fold/Call/Raise 로짓, 엔트로피, Actor/Critic 손실, Advantage Mean/Std, Gradient Norm, Approx KL, Clip Fraction 추적
- 올-폴더(All-Folder)로 붕괴하는 임계 시점 및 근본 원인(Off-policy Ratio 폭발 vs 보상 구조) 규명
"""

import os
import sys
import time
import math
import random
import csv
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
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return

def track_longterm_collapse(seed_val: int = 42, max_steps: int = 40000) -> List[Dict]:
    print("=" * 70)
    print(f"PHASE 5: LONG-TERM POLICY COLLAPSE TRACKING (0k -> 40k, Seed {seed_val})")
    print("=" * 70)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    gen = TargetedSubgameGenerator(game)

    # 1. Probe State (AA vs 3-Bet 올인)
    st_probe, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', 'AA', hero_seat=2)
    vec_probe = st_probe.information_state_tensor(2)
    legal_probe = st_probe.legal_actions(2)
    x_probe = torch.tensor(vec_probe, dtype=torch.float32, device=device).unsqueeze(0)
    m_probe = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for a in legal_probe: m_probe[0, a] = True

    model = HoldemActorCritic5A().to(device)
    init_path = f"v49/models/CondC_5A_Transfer_seed_{seed_val}_step_20000.pt"
    if os.path.exists(init_path):
        model.load_state_dict(torch.load(init_path, map_location=device))
        print(f"[초기 모델 로드] {init_path}")

    optimizer = optim.Adam(model.parameters(), lr=0.0002, eps=1e-5)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed_val + 777)
    rng_choice = random.Random(seed_val + 888)

    checkpoints_to_record = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    collapse_logs = []

    batch_trajectories = []
    running_actor_loss = 0.0
    running_critic_loss = 0.0
    running_entropy = 0.0
    running_kl = 0.0
    running_clip = 0.0
    running_grad_norm = 0.0
    running_adv_mean = 0.0
    running_adv_std = 0.0
    n_updates = 0

    t_start = time.time()

    for ep in range(0, max_steps + 1):
        # 체크포인트 기록
        if ep in checkpoints_to_record:
            model.eval()
            with torch.no_grad():
                lg, val_est = model(x_probe, m_probe)
                probs = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
                raw_lg = lg.squeeze(0).cpu().numpy()

            fold_lg = float(raw_lg[0])
            call_lg = float(raw_lg[1])
            pot_lg = float(raw_lg[2])
            allin_lg = float(raw_lg[3])
            half_lg = float(raw_lg[4])

            avg_a_loss = running_actor_loss / max(1, n_updates)
            avg_c_loss = running_critic_loss / max(1, n_updates)
            avg_ent = running_entropy / max(1, n_updates)
            avg_kl = running_kl / max(1, n_updates)
            avg_clip = running_clip / max(1, n_updates)
            avg_gnorm = running_grad_norm / max(1, n_updates)
            avg_adv_m = running_adv_mean / max(1, n_updates)
            avg_adv_s = running_adv_std / max(1, n_updates)

            print(f"Step {ep:5d} | Fold Logit: {fold_lg:+6.2f}, Call: {call_lg:+6.2f} | Fold%: {probs[0]*100:5.1f}%, Call%: {probs[1]*100:5.1f}% | Entropy: {avg_ent:.4f} | KL: {avg_kl:.4f} | GradNorm: {avg_gnorm:.3f}")

            collapse_logs.append({
                'step': ep,
                'fold_logit': fold_lg,
                'call_logit': call_lg,
                'pot_logit': pot_lg,
                'allin_logit': allin_lg,
                'half_logit': half_lg,
                'fold_prob_pct': float(probs[0] * 100.0),
                'call_prob_pct': float(probs[1] * 100.0),
                'actor_entropy': avg_ent,
                'actor_loss': avg_a_loss,
                'critic_loss': avg_c_loss,
                'adv_mean': avg_adv_m,
                'adv_std': avg_adv_s,
                'grad_norm': avg_gnorm,
                'approx_kl': avg_kl,
                'clip_frac': avg_clip,
            })

            # 누적 카운터 리셋
            running_actor_loss = 0.0
            running_critic_loss = 0.0
            running_entropy = 0.0
            running_kl = 0.0
            running_clip = 0.0
            running_grad_norm = 0.0
            running_adv_mean = 0.0
            running_adv_std = 0.0
            n_updates = 0

        if ep == max_steps:
            break

        # 게임 롤아웃 (풀-게임 On-Policy PPO)
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

        if (ep + 1) % 64 == 0 and batch_trajectories:
            b_v, b_m, b_a, b_olp, b_r = zip(*batch_trajectories)
            t_states = torch.tensor(np.array(b_v), dtype=torch.float32, device=device)
            t_masks = torch.tensor(np.array(b_m), dtype=torch.bool, device=device)
            t_actions = torch.tensor(b_a, dtype=torch.int64, device=device)
            t_returns = torch.tensor(b_r, dtype=torch.float32, device=device)
            t_old_log_p = torch.tensor(b_olp, dtype=torch.float32, device=device)

            with torch.no_grad():
                _, values = model(t_states, t_masks)
                adv = t_returns - values
                adv_m = adv.mean().item()
                adv_s = adv.std().item()
                if adv_s > 1e-8:
                    adv = (adv - adv.mean()) / adv_s

            model.train()
            for _ in range(4):
                tot_loss, pol_loss, val_loss, ent = compute_ppo_losses_5a(
                    model, t_states, t_masks, t_actions, t_old_log_p, t_returns, adv
                )
                optimizer.zero_grad()
                tot_loss.backward()
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()

            with torch.no_grad():
                lg, _ = model(t_states, t_masks)
                new_lps = torch.log_softmax(lg, dim=-1).gather(1, t_actions.unsqueeze(-1)).squeeze(-1)
                ratios = torch.exp(new_lps - t_old_log_p)
                kl = ((ratios - 1.0) - torch.log(ratios)).mean().item()
                clip_f = ((ratios < 0.8) | (ratios > 1.2)).float().mean().item()

            running_actor_loss += pol_loss.item()
            running_critic_loss += val_loss.item()
            running_entropy += ent.item()
            running_kl += kl
            running_clip += clip_f
            running_grad_norm += gnorm
            running_adv_mean += adv_m
            running_adv_std += adv_s
            n_updates += 1
            batch_trajectories = []

    csv_path = f"v495/data/phase5_longterm_collapse_seed_{seed_val}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(collapse_logs[0].keys()))
        writer.writeheader()
        writer.writerows(collapse_logs)
    print(f"\n[저장 완료] 장기 붕괴 추적 데이터: {csv_path}")
    return collapse_logs

if __name__ == "__main__":
    track_longterm_collapse(seed_val=42, max_steps=40000)
