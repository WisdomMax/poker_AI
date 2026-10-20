"""
POKER AI V4.1 - STEP 2 & 4: DeepMind NFSP 전체 학습 및 체크포인트 평가
- 체크포인트: 0, 1,000, 5,000, 10,000, 25,000, 50,000, 100,000
- Best Response vs Average Policy 분리 측정
- NashConv < 0.05 목표 도달 여부 및 체크포인트 가중치 저장
"""

import sys
import os
import time
import copy
from typing import Dict, Tuple, List
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from open_spiel.python.algorithms import exploitability
from v4.openspiel_adapter import DictPolicyWrapper
from kuhn_exact_solver import compute_exact_exploitability, get_kuhn_analytical_nash
from kuhn_engine_v3 import is_terminal, evaluate_payoff, PASS, BET, ACTIONS, CARD_NAMES
from v4.v41_nfsp import NFSPAgent, encode_kuhn_state


def combine_policies(agent0: NFSPAgent, agent1: NFSPAgent, mode: str = "avg") -> Dict[str, Tuple[float, float]]:
    """agent0의 P0 정책과 agent1의 P1 정책을 결합한 12개 정보집합 통합 정책 반환"""
    strat0 = agent0.get_average_policy_dict() if mode == "avg" else agent0.get_best_response_policy_dict()
    strat1 = agent1.get_average_policy_dict() if mode == "avg" else agent1.get_best_response_policy_dict()

    combined = {}
    for c_name in ['J', 'Q', 'K']:
        # P0 노드
        combined[f"{c_name}:"] = strat0[f"{c_name}:"]
        combined[f"{c_name}:pb"] = strat0[f"{c_name}:pb"]
        # P1 노드
        combined[f"{c_name}:p"] = strat1[f"{c_name}:p"]
        combined[f"{c_name}:b"] = strat1[f"{c_name}:b"]
    return combined


def train_nfsp(
    total_episodes: int = 100000,
    checkpoints: List[int] = [0, 1000, 5000, 10000, 25000, 50000, 100000],
    seed: int = 42,
    device_str: str = "cpu"
) -> Dict[str, any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str)

    game = pyspiel.load_game("kuhn_poker")
    models_dir = "v4/models"
    os.makedirs(models_dir, exist_ok=True)

    agent0 = NFSPAgent(player_id=0, anticipatory_param=0.1, device=device)
    agent1 = NFSPAgent(player_id=1, anticipatory_param=0.1, device=device)

    history_records = []
    ckpt_idx = 0

    print("=" * 70)
    print(f"🚀 [NFSP 학습 시작] 총 {total_episodes:,} 에피소드 (Device: {device_str})")
    print(f"체크포인트: {checkpoints}")
    print("=" * 70)

    start_time = time.time()
    deck = [1, 2, 3]

    for ep in range(total_episodes + 1):
        # 1. 체크포인트 도달 시 평가 및 저장
        if ckpt_idx < len(checkpoints) and ep == checkpoints[ckpt_idx]:
            ckpt_ep = checkpoints[ckpt_idx]
            ckpt_idx += 1

            # 평균 정책 평가
            avg_policy = combine_policies(agent0, agent1, mode="avg")
            os_avg_policy = DictPolicyWrapper(game, avg_policy)
            nc_avg = exploitability.nash_conv(game, os_avg_policy)
            exp_avg = nc_avg / 2.0

            # 최신 BR 정책 평가
            br_policy = combine_policies(agent0, agent1, mode="br")
            os_br_policy = DictPolicyWrapper(game, br_policy)
            nc_br = exploitability.nash_conv(game, os_br_policy)

            elapsed = time.time() - start_time
            history_records.append({
                "episode": ckpt_ep,
                "elapsed": elapsed,
                "nc_avg": nc_avg,
                "exp_avg": exp_avg,
                "nc_br": nc_br,
                "policy": copy.deepcopy(avg_policy)
            })

            # 가중치 파일 저장
            torch.save(agent0.avg_policy_net.state_dict(), f"{models_dir}/nfsp_p0_avg_ckpt_{ckpt_ep}.pt")
            torch.save(agent1.avg_policy_net.state_dict(), f"{models_dir}/nfsp_p1_avg_ckpt_{ckpt_ep}.pt")

            print(f"  💾 [체크포인트 {ckpt_ep:>6} 판] 소요시간: {elapsed:>6.1f}s | "
                  f"Average NashConv: {nc_avg:.6f} (Exp: {exp_avg*100:>5.2f} bb/100) | "
                  f"BR NashConv: {nc_br:.6f}")

            if ep == total_episodes:
                break

        # 2. 에피소드 모드 샘플링
        agent0.sample_episode_mode()
        agent1.sample_episode_mode()

        cards = np.random.permutation(deck)
        c0, c1 = cards[0], cards[1]

        # 궤적 추적
        traj0, traj1 = [], [] # list of (state_tensor, action)

        # Step 1: P0 액션
        s0_root = encode_kuhn_state(c0, "", device)
        a0 = agent0.act(c0, "")
        traj0.append((s0_root, a0, ""))
        if agent0.current_mode == 'best_response':
            agent0.sl_buffer.add((s0_root.cpu(), a0))

        if a0 == PASS:
            # Step 2: P1 액션 (history 'p')
            s1_p = encode_kuhn_state(c1, "p", device)
            a1 = agent1.act(c1, "p")
            traj1.append((s1_p, a1, "p"))
            if agent1.current_mode == 'best_response':
                agent1.sl_buffer.add((s1_p.cpu(), a1))

            if a1 == PASS:
                # Terminal pp
                final_history = "pp"
            else:
                # Step 3: P0 액션 (history 'pb')
                s0_pb = encode_kuhn_state(c0, "pb", device)
                a0_2 = agent0.act(c0, "pb")
                traj0.append((s0_pb, a0_2, "pb"))
                if agent0.current_mode == 'best_response':
                    agent0.sl_buffer.add((s0_pb.cpu(), a0_2))
                final_history = "pbp" if a0_2 == PASS else "pbb"
        else:
            # Step 2: P1 액션 (history 'b')
            s1_b = encode_kuhn_state(c1, "b", device)
            a1 = agent1.act(c1, "b")
            traj1.append((s1_b, a1, "b"))
            if agent1.current_mode == 'best_response':
                agent1.sl_buffer.add((s1_b.cpu(), a1))
            final_history = "bp" if a1 == PASS else "bb"

        pay0, pay1 = evaluate_payoff(final_history, c0, c1)

        # 3. RL Replay Buffer 업데이트
        # Agent 0 경험 저장
        if len(traj0) == 1:
            st, act, h = traj0[0]
            agent0.rl_buffer.push(st.cpu(), act, pay0, st.cpu(), True)
        elif len(traj0) == 2:
            st1, act1, _ = traj0[0]
            st2, act2, _ = traj0[1]
            # 중간 전이: 보상 0
            agent0.rl_buffer.push(st1.cpu(), act1, 0.0, st2.cpu(), False)
            agent0.rl_buffer.push(st2.cpu(), act2, pay0, st2.cpu(), True)

        # Agent 1 경험 저장
        if len(traj1) == 1:
            st, act, _ = traj1[0]
            agent1.rl_buffer.push(st.cpu(), act, pay1, st.cpu(), True)

        # 4. 신경망 학습 스텝 (매 64판마다 배치 업데이트)
        if ep > 0 and ep % 64 == 0:
            agent0.train_step()
            agent1.train_step()

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"✅ NFSP 학습 완료: 총 소요시간 {total_time:.1f}초 ({total_episodes/total_time:.1f} games/sec)")

    final_rec = history_records[-1]
    final_nc = final_rec["nc_avg"]
    print(f"🎯 최종 Average Policy NashConv: {final_nc:.6f}")
    if final_nc < 0.05:
        print("🎉 [목표 달성] 신경망 평균 정책 NashConv < 0.05 성공!")
    else:
        print(f"⚠️ [목표 미달] NashConv ({final_nc:.4f} >= 0.05)")

    return {
        "history": history_records,
        "total_time": total_time,
        "agent0": agent0,
        "agent1": agent1
    }


if __name__ == "__main__":
    train_nfsp(total_episodes=100000)
