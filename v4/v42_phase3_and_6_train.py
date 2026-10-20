"""
POKER AI V4.2 - PHASE 3 & 6: 최적 체크포인트 자동 선택 및 대규모 학습 (100k -> 300k)
- 최적화된 하이퍼파라미터 반영 (Optimized NFSP)
- 매 주기마다 NashConv, Exploitability, Loss, Policy Entropy 등 기록
- Best Validation Checkpoint 자동 감지 및 최적 모델 채택
"""

import sys
import os
import time
import copy
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from v4.openspiel_adapter import DictPolicyWrapper
from kuhn_engine_v3 import evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from v4.v42_phase2_ablation import OptimizedNFSPAgent, encode_kuhn_state


def compute_policy_entropy(policy_dict: dict) -> float:
    """12개 정보집합의 평균 섀넌 엔트로피 (탐색 다양성 지표)"""
    entropies = []
    for k, (p_pass, p_bet) in policy_dict.items():
        p_p = max(p_pass, 1e-8)
        p_b = max(p_bet, 1e-8)
        ent = -(p_p * np.log2(p_p) + p_b * np.log2(p_b))
        entropies.append(ent)
    return float(np.mean(entropies))


def train_optimized_nfsp(
    total_episodes: int = 100000,
    eval_interval: int = 5000,
    seed: int = 42
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    game = pyspiel.load_game("kuhn_poker")
    device = torch.device("cpu")
    models_dir = "v4/models"
    os.makedirs(models_dir, exist_ok=True)

    # 최적 하이퍼파라미터
    a0 = OptimizedNFSPAgent(0, hidden_dim=64, anticipatory_param=0.1, rl_learning_rate=0.005,
                            sl_learning_rate=0.002, batch_size=128, target_update_interval=100,
                            epsilon_decay_steps=total_episodes, use_action_probs_sl=True,
                            use_huber_loss=False, device=device)
    a1 = OptimizedNFSPAgent(1, hidden_dim=64, anticipatory_param=0.1, rl_learning_rate=0.005,
                            sl_learning_rate=0.002, batch_size=128, target_update_interval=100,
                            epsilon_decay_steps=total_episodes, use_action_probs_sl=True,
                            use_huber_loss=False, device=device)

    history_log = []
    best_nash_conv = 999.0
    best_ep = 0
    best_policy_dict = None

    print("=" * 80)
    print(f"🚀 [V4.2 Optimized NFSP 학습] 총 {total_episodes:,} 판 (평가 주기: {eval_interval} 판)")
    print("=" * 80)

    start_time = time.time()
    deck = [1, 2, 3]

    for ep in range(total_episodes + 1):
        if ep % eval_interval == 0:
            # 평가 수행
            p0_d = a0.get_average_policy_dict()
            p1_d = a1.get_average_policy_dict()
            comb = {}
            for c in ['J', 'Q', 'K']:
                comb[f"{c}:"] = p0_d[f"{c}:"]; comb[f"{c}:pb"] = p0_d[f"{c}:pb"]
                comb[f"{c}:p"] = p1_d[f"{c}:p"]; comb[f"{c}:b"] = p1_d[f"{c}:b"]

            nc = float(exploitability.nash_conv(game, DictPolicyWrapper(game, comb)))
            exp = nc / 2.0
            entropy = compute_policy_entropy(comb)
            elapsed = time.time() - start_time

            is_best = False
            if nc < best_nash_conv:
                best_nash_conv = nc
                best_ep = ep
                best_policy_dict = copy.deepcopy(comb)
                is_best = True
                torch.save(a0.avg_policy_net.state_dict(), f"{models_dir}/v42_best_nfsp_p0.pt")
                torch.save(a1.avg_policy_net.state_dict(), f"{models_dir}/v42_best_nfsp_p1.pt")

            star = " 🌟 [NEW BEST!]" if is_best else ""
            print(f"  * Ep {ep:>6} | NashConv: {nc:.6f} (Exp: {exp*100:>5.2f} bb/100) | "
                  f"Entropy: {entropy:.3f} | Elapsed: {elapsed:>5.1f}s{star}")

            history_log.append({
                "episode": ep, "nash_conv": nc, "exploitability": exp,
                "entropy": entropy, "elapsed": elapsed, "is_best": is_best
            })

            if ep == total_episodes:
                break

        # 에피소드 진행
        a0.sample_episode_mode()
        a1.sample_episode_mode()

        cards = np.random.permutation(deck)
        c0, c1 = cards[0], cards[1]
        traj0, traj1 = [], []

        s0 = encode_kuhn_state(c0, "", device)
        act0, pr0 = a0.act(c0, "")
        traj0.append((s0, act0))
        if a0.current_mode == 'best_response': a0.sl_buffer.add((s0.cpu(), pr0))

        if act0 == PASS:
            s1 = encode_kuhn_state(c1, "p", device)
            act1, pr1 = a1.act(c1, "p")
            traj1.append((s1, act1))
            if a1.current_mode == 'best_response': a1.sl_buffer.add((s1.cpu(), pr1))

            if act1 == PASS: fin = "pp"
            else:
                s0_2 = encode_kuhn_state(c0, "pb", device)
                act0_2, pr0_2 = a0.act(c0, "pb")
                traj0.append((s0_2, act0_2))
                if a0.current_mode == 'best_response': a0.sl_buffer.add((s0_2.cpu(), pr0_2))
                fin = "pbp" if act0_2 == PASS else "pbb"
        else:
            s1 = encode_kuhn_state(c1, "b", device)
            act1, pr1 = a1.act(c1, "b")
            traj1.append((s1, act1))
            if a1.current_mode == 'best_response': a1.sl_buffer.add((s1.cpu(), pr1))
            fin = "bp" if act1 == PASS else "bb"

        pay0, pay1 = evaluate_payoff(fin, c0, c1)

        if len(traj0) == 1:
            st, ac = traj0[0]; a0.rl_buffer.push(st.cpu(), ac, pay0, st.cpu(), True)
        elif len(traj0) == 2:
            st1, ac1 = traj0[0]; st2, ac2 = traj0[1]
            a0.rl_buffer.push(st1.cpu(), ac1, 0.0, st2.cpu(), False)
            a0.rl_buffer.push(st2.cpu(), ac2, pay0, st2.cpu(), True)

        if len(traj1) == 1:
            st, ac = traj1[0]; a1.rl_buffer.push(st.cpu(), ac, pay1, st.cpu(), True)

        if ep > 0 and ep % 64 == 0:
            a0.train_step()
            a1.train_step()

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"🎯 [V4.2 최적 모델 자동 선택 완료]")
    print(f"  - 최고의 체크포인트 에피소드: {best_ep:,} 판")
    print(f"  - 최소 NashConv:             {best_nash_conv:.6f} (Exploitability: {best_nash_conv*50:.2f} bb/100)")
    print(f"  - V4.1 NFSP 최종값(0.3111) 대비 개선: {0.311127 - best_nash_conv:+.6f}")
    print(f"  - 가중치 저장 완료: v4/models/v42_best_nfsp_p0.pt / p1.pt")
    print("=" * 80)

    return history_log, best_nash_conv, best_ep, best_policy_dict


if __name__ == "__main__":
    train_optimized_nfsp(total_episodes=100000, eval_interval=5000)
