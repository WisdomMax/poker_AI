"""
POKER AI V5.0 - PHASE 4: 2x2 Factorial Controlled Training (Pilot: 3 Seeds x 4 Conditions)
- 4대 통제 조건 (모두 317차원 동일 입력 구조):
  * CondA: 상대통계 OFF (중립값 패딩) + 일반 Full-Game
  * CondB: 상대통계 ON (베이지안 추적) + 일반 Full-Game
  * CondC: 상대통계 OFF (중립값 패딩) + 희귀 상태 노출 강화 (Rare Exposure)
  * CondD: 상대통계 ON (베이지안 추적) + 희귀 상태 노출 강화 (Rare Exposure)
- 3개 독립 시드: 42, 101, 202 (총 12개 모델)
- Match Session 기반 지속 관측: 1세션당 50핸드, 총 40세션 (2,000 에피소드)
- On-Policy PPO 정합성 준수
"""

import os
import sys
import copy
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
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
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import (
    BayesianHoldemActorCritic5A, get_neutral_prior_features, compute_bayesian_ppo_losses
)

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def train_single_model(
    cond_name: str,
    use_opponent_features: bool,
    use_rare_exposure: bool,
    seed: int,
    total_episodes: int = 2000,
    session_length: int = 50,
    device: Optional[torch.device] = None
) -> str:
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    opponents_master = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    rng_env = np.random.RandomState(seed + 100)
    rng_opp = np.random.RandomState(seed + 200)

    neutral_opp_feats = get_neutral_prior_features() # 52차원

    buffer_states = []
    buffer_masks = []
    buffer_actions = []
    buffer_log_probs = []
    buffer_returns = []
    buffer_advantages = []

    num_sessions = total_episodes // session_length

    # 매치 세션 루프
    for sess_idx in range(num_sessions):
        # 세션 시작: 상대 4인 테이블 고정 및 상대 추적기 리셋
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

            # 현재 판 시작 전의 상대 피처 추출 (미래 정보 누출 0%)
            if use_opponent_features:
                opp_feats = tracker_mgr.get_table_feature_tensor(hero_seat)
            else:
                opp_feats = neutral_opp_feats

            st = game.new_initial_state()
            advance_chance_nodes(st, rng=rng_env)

            # 희귀 상태 노출 강화 (Cond C, Cond D): 20% 확률로 상대가 베팅을 이미 감행한 상황 주입
            if use_rare_exposure and rng_env.rand() < 0.20:
                # 첫 번째 베팅 턴에서 상대가 오픈 레이즈(Action 2 or 4)를 하도록 유도
                first_bettor = st.current_player()
                if first_bettor != hero_seat and not st.is_terminal():
                    leg = st.legal_actions(first_bettor)
                    if 2 in leg: st.apply_action(2)
                    elif 4 in leg: st.apply_action(4)

            ep_transitions = []
            hand_action_log = [] # 공개 시퀀스 추적용

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng=rng_env)
                else:
                    cur = st.current_player()
                    legal = st.legal_actions(cur)
                    round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                    if cur == hero_seat:
                        info_vec = normalize_vec(st.information_state_tensor(cur))
                        # 265 info + 52 opp = 317차원 결합
                        full_vec = np.concatenate([info_vec, opp_feats])
                        x = torch.tensor(full_vec, dtype=torch.float32, device=device).unsqueeze(0)
                        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in legal: m[0, a] = True

                        act, lp, _, val = model.sample_action(x, m, deterministic=False)
                        ep_transitions.append((full_vec, m.cpu(), act, lp, val))
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        ag_id = s2a[cur]
                        opp = table_opponents[(ag_id - 1) % len(table_opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            # 핸드 종료 후: 공개 액션 이력으로 상대 통계 업데이트
            tracker_mgr.parse_and_update_hand(hand_action_log, s2a)

            # 보상 수집
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

            # PPO 배치 업데이트 (300 전환마다)
            if len(buffer_states) >= 300:
                b_s = torch.tensor(np.array(buffer_states), dtype=torch.float32, device=device)
                b_m = torch.tensor(np.array(buffer_masks), dtype=torch.bool, device=device)
                b_a = torch.tensor(buffer_actions, dtype=torch.long, device=device)
                b_lp = torch.tensor(buffer_log_probs, dtype=torch.float32, device=device)
                b_ret = torch.tensor(buffer_returns, dtype=torch.float32, device=device)
                b_adv = torch.tensor(buffer_advantages, dtype=torch.float32, device=device)
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                for _ in range(4):
                    loss, _, _, _, _, _ = compute_bayesian_ppo_losses(
                        model, b_s, b_m, b_a, b_lp, b_ret, b_adv,
                        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
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

    # 모델 저장
    out_path = f"v50/models/{cond_name}_seed_{seed}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"  [저장 완료] {cond_name} Seed {seed} -> {out_path}")
    return out_path

def run_phase4_pilot():
    print("=" * 80)
    print("PHASE 4: 2x2 Factorial Pilot 훈련 (3개 시드 x 4개 조건 = 12개 모델)")
    print("=" * 80)

    device = torch.device("cpu")
    SEEDS = [42, 101, 202]
    CONDITIONS = [
        # (cond_name, use_opp, use_rare)
        ('CondA_Base_Standard', False, False),
        ('CondB_Bayes_Standard', True, False),
        ('CondC_Base_RareExp', False, True),
        ('CondD_Bayes_RareExp', True, True),
    ]

    trained_models = []
    for cname, use_opp, use_rare in CONDITIONS:
        print(f"\n>>> [조건 시작] {cname} (상대통계={use_opp}, 희귀노출={use_rare})")
        for s in SEEDS:
            mpath = train_single_model(
                cond_name=cname,
                use_opponent_features=use_opp,
                use_rare_exposure=use_rare,
                seed=s,
                total_episodes=1500, # 파일럿 1,500 에피소드
                session_length=50,
                device=device
            )
            trained_models.append((cname, s, mpath))

    print("\n-> 파일럿 12개 모델 훈련 전수 완료.")

if __name__ == "__main__":
    run_phase4_pilot()
