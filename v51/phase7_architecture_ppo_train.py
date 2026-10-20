"""
POKER AI V5.1 - PHASE 7: OPPONENT-AWARE ARCHITECTURE PPO TRAINING
- 3대 모델(Concatenation, Two-Tower, FiLM)의 공정 매치 세션 PPO 학습:
  * 3개 독립 시드: 42, 101, 202 (총 9개 체크포인트)
  * 세션 길이 50핸드, 총 1,000핸드 (20세션)
  * On-Policy PPO 무결성 (Legal Mask, Old Log Prob 정합성, Advantage Normalization)
  * 체크포인트 저장 위치: v51/models/{arch}_seed_{seed}.pt
  * 학습 곡선 로그 저장 위치: v51/logs/ppo_train_{arch}_seed_{seed}.csv
"""

import os
import sys
import csv
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
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
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v51.models_comparison import ConcatenationActorCritic, TwoTowerActorCritic, FiLMActorCritic

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def train_architecture_ppo(
    arch_name: str,
    model_cls,
    seed: int,
    total_episodes: int = 1000,
    session_length: int = 50,
    update_interval: int = 200,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: Optional[torch.device] = None
) -> str:
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    opponents_master = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed + 100)
    rng_opp = np.random.RandomState(seed + 200)

    # 롤아웃 버퍼
    buf_hero_feats = []
    buf_opp_feats = []
    buf_masks = []
    buf_actions = []
    buf_old_log_probs = []
    buf_returns = []
    buf_values = []

    log_records = []
    num_sessions = total_episodes // session_length

    start_time = time.time()
    print(f"\n>>> [학습 시작] {arch_name} (Seed: {seed}, {total_episodes} 핸드)")

    for sess_idx in range(num_sessions):
        # 세션 초기화
        opp_indices = rng_opp.choice(len(opponents_master), 4, replace=False)
        table_opponents = [opponents_master[i] for i in opp_indices]
        tracker_mgr = MatchSessionOpponentManager()
        tracker_mgr.reset_session()

        for h_in_sess in range(session_length):
            ep_global = sess_idx * session_length + h_in_sess
            tracker_mgr.set_hand_index(h_in_sess)

            s2a = wrapper.get_seat_to_agent(h_in_sess)
            a2s = wrapper.get_agent_to_seat(h_in_sess)
            hero_seat = a2s[0]

            # 상대 피처 획득 (공개 통계만 활용)
            opp_feat = tracker_mgr.get_table_feature_tensor(hero_seat)

            st = game.new_initial_state()
            advance_chance_nodes(st, rng=rng_env)

            # 희귀 상태 노출 (20% 확률로 레이즈 직면 상황 주입)
            if rng_env.rand() < 0.20:
                first_bettor = st.current_player()
                if first_bettor != hero_seat and not st.is_terminal():
                    leg = st.legal_actions(first_bettor)
                    if 2 in leg: st.apply_action(2)
                    elif 4 in leg: st.apply_action(4)

            ep_transitions = []
            hand_action_log = []

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng=rng_env)
                else:
                    cur = st.current_player()
                    legal = st.legal_actions(cur)
                    round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                    if cur == hero_seat:
                        hero_info = normalize_vec(st.information_state_tensor(cur))
                        h_t = torch.tensor(hero_info, dtype=torch.float32, device=device).unsqueeze(0)
                        o_t = torch.tensor(opp_feat, dtype=torch.float32, device=device).unsqueeze(0)
                        m_t = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in legal: m_t[0, a] = True

                        with torch.no_grad():
                            raw_logits, probs, val = model(h_t, o_t, m_t)
                            dist = torch.distributions.Categorical(probs)
                            act = dist.sample().item()
                            lp = dist.log_prob(torch.tensor(act, device=device)).item()
                            v = val.item()

                        ep_transitions.append((hero_info, opp_feat, m_t.squeeze(0).cpu().numpy(), act, lp, v))
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        ag_id = s2a[cur]
                        opp = table_opponents[(ag_id - 1) % len(table_opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            # 세션 트래커 업데이트
            tracker_mgr.parse_and_update_hand(hand_action_log, s2a)

            # 리턴 계산
            ret_bb = st.returns()[hero_seat] / 100.0

            for h_info, o_info, mask_arr, act, lp, val in ep_transitions:
                buf_hero_feats.append(h_info)
                buf_opp_feats.append(o_info)
                buf_masks.append(mask_arr)
                buf_actions.append(act)
                buf_old_log_probs.append(lp)
                buf_returns.append(ret_bb)
                buf_values.append(val)

            # PPO 업데이트 주기 도달 시
            if len(buf_hero_feats) >= update_interval:
                b_hero = torch.tensor(np.array(buf_hero_feats), dtype=torch.float32, device=device)
                b_opp = torch.tensor(np.array(buf_opp_feats), dtype=torch.float32, device=device)
                b_mask = torch.tensor(np.array(buf_masks), dtype=torch.bool, device=device)
                b_act = torch.tensor(np.array(buf_actions), dtype=torch.long, device=device)
                b_old_lp = torch.tensor(np.array(buf_old_log_probs), dtype=torch.float32, device=device)
                b_ret = torch.tensor(np.array(buf_returns), dtype=torch.float32, device=device)
                b_val = torch.tensor(np.array(buf_values), dtype=torch.float32, device=device)

                advantages = b_ret - b_val
                if len(advantages) > 1 and advantages.std() > 1e-6:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dataset_size = len(buf_hero_feats)
                indices = np.arange(dataset_size)

                for epoch in range(ppo_epochs):
                    np.random.shuffle(indices)
                    for start_idx in range(0, dataset_size, batch_size):
                        batch_idx = indices[start_idx : start_idx + batch_size]

                        mb_hero = b_hero[batch_idx]
                        mb_opp = b_opp[batch_idx]
                        mb_mask = b_mask[batch_idx]
                        mb_act = b_act[batch_idx]
                        mb_old_lp = b_old_lp[batch_idx]
                        mb_ret = b_ret[batch_idx]
                        mb_adv = advantages[batch_idx]

                        raw_lgs, probs, values = model(mb_hero, mb_opp, mb_mask)
                        dist = torch.distributions.Categorical(probs)
                        new_lp = dist.log_prob(mb_act)
                        entropy = dist.entropy().mean()

                        # PPO Clipped Policy Loss
                        ratios = torch.exp(new_lp - mb_old_lp)
                        surr1 = ratios * mb_adv
                        surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value Loss
                        val_loss = F.mse_loss(values.squeeze(-1), mb_ret)

                        loss = policy_loss + vf_coef * val_loss - ent_coef * entropy

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()

                # 버퍼 초기화
                buf_hero_feats.clear()
                buf_opp_feats.clear()
                buf_masks.clear()
                buf_actions.clear()
                buf_old_log_probs.clear()
                buf_returns.clear()
                buf_values.clear()

            if (ep_global + 1) % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  [에피소드 {ep_global + 1:>4}/{total_episodes}] 최근 리턴: {ret_bb:+.2f} BB | 소요 시간: {elapsed:.1f}초")

    # 모델 체크포인트 저장
    model_dir = "v51/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{arch_name}_seed_{seed}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"-> [체크포인트 저장 완료] {model_path} (소요 시간: {time.time() - start_time:.1f}초)")

    return model_path

def run_phase7_training():
    print("=" * 85)
    print("PHASE 7: 3대 아키텍처 매치 세션 PPO 독립 시드 학습")
    print("=" * 85)

    architectures = [
        ("ModelA_Concat", ConcatenationActorCritic),
        ("ModelB_TwoTower", TwoTowerActorCritic),
        ("ModelC_FiLM", FiLMActorCritic)
    ]
    seeds = [42, 101, 202]

    saved_paths = []
    for arch_name, cls_obj in architectures:
        for s in seeds:
            p = train_architecture_ppo(arch_name, cls_obj, seed=s, total_episodes=1000)
            saved_paths.append(p)

    print("\n" + "=" * 85)
    print("PHASE 7 완료: 총 9개 체크포인트 학습 및 저장 완료:")
    for p in saved_paths:
        print(f"  * {p} (크기: {os.path.getsize(p):,} bytes)")
    print("=" * 85)

if __name__ == '__main__':
    run_phase7_training()
