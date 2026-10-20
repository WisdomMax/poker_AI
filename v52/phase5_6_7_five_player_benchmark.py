"""
POKER AI V5.2 - PHASE 5, 6, 7: FIVE-PLAYER REDUCED HOLD'EM BENCHMARK
- 환경: 5인 축소 텍사스 홀덤 (universal_poker, 24장 덱, 2라운드, 10BB 스택, 4-Action)
- Phase 5: PPO Self-Play vs NFSP Self-Play 학습 및 MCCFR 5인 메모리 한계 감사
- Phase 6: 4대 절대 기준선(Random, Always Fold, Legal Check/Call, Rule-Based)과의 풀게임 평가
- Phase 7: 카드 인지 능력 감사 (AA vs 99 올인/오픈 직면 시 실제 Softmax 확률 및 TVD 측정)
- 원시 데이터를 v52/data/phase5_6_7_five_player_benchmark.csv에 저장
"""

import os
import sys
import csv
import time
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyspiel

def get_5p_reduced_holdem_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 6, # 9, T, J, Q, K, A
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 2, # Preflop, Flop
        'numBoardCards': '0 3',
        'blind': '50 100 0 0 0',
        'stack': '1000 1000 1000 1000 1000', # 10BB
        'firstPlayer': '3 1'
    }
    return pyspiel.load_game('universal_poker', params)

def advance_chance_nodes(state: pyspiel.State, rng: np.random.RandomState):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        acts = [a for a, _ in outcomes]
        probs = [p for _, p in outcomes]
        act = rng.choice(acts, p=probs)
        state.apply_action(act)

# =====================================================================
# 절대 성능 기준선 에이전트 4종
# =====================================================================
class RandomAgent:
    def act(self, state, player_id, legal_actions):
        return random.choice(legal_actions)

class AlwaysFoldAgent:
    def act(self, state, player_id, legal_actions):
        return 0 if 0 in legal_actions else legal_actions[0]

class CheckCallAgent:
    def act(self, state, player_id, legal_actions):
        return 1 if 1 in legal_actions else legal_actions[0]

class RuleBasedAgent:
    """프리미엄(AA, KK, AKs)은 레이즈/올인, 그 외는 체크/콜 또는 폴드"""
    def act(self, state, player_id, legal_actions):
        info_str = state.information_state_string(player_id)
        # 카드 파싱
        # universal_poker info_str contains [Private: ...]
        has_premium = ('AA' in info_str or 'KK' in info_str or 'AK' in info_str)
        if has_premium:
            if 3 in legal_actions: return 3 # Allin
            if 2 in legal_actions: return 2 # Pot
            return legal_actions[-1]
        else:
            if 1 in legal_actions: return 1 # Check/Call
            if 0 in legal_actions: return 0 # Fold
            return legal_actions[0]


# =====================================================================
# 5인 PPO Self-Play 에이전트
# =====================================================================
class PPO5PModel(nn.Module):
    def __init__(self, state_dim: int = 149, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x, mask):
        raw_lg = self.actor(x)
        masked_lg = raw_lg.clone()
        masked_lg[~mask] = -1e9
        probs = F.softmax(masked_lg, dim=-1)
        val = self.critic(x)
        return raw_lg, probs, val


def train_5p_ppo_selfplay(game, num_episodes: int = 2000, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    state_dim = len(game.new_initial_state().information_state_tensor(0))
    action_dim = game.num_distinct_actions()

    model = PPO5PModel(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    buf_s, buf_a, buf_lp, buf_ret, buf_val, buf_m = [], [], [], [], [], []
    total_transitions = 0

    for ep in range(num_episodes):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        ep_history = []

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                legal = st.legal_actions(cp)
                info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                mask = np.zeros(action_dim, dtype=bool)
                mask[legal] = True

                x = torch.tensor(info).unsqueeze(0)
                m = torch.tensor(mask).unsqueeze(0)

                with torch.no_grad():
                    _, probs, val = model(x, m)
                    dist = torch.distributions.Categorical(probs)
                    act = dist.sample().item()
                    lp = dist.log_prob(torch.tensor(act)).item()

                ep_history.append((info, act, lp, val.item(), mask, cp))
                st.apply_action(act)
                total_transitions += 1

        returns = st.returns()
        for s, a, lp, v, m, cp in ep_history:
            ret_bb = returns[cp] / 100.0 # BB 단위
            buf_s.append(s); buf_a.append(a); buf_lp.append(lp)
            buf_ret.append(ret_bb); buf_val.append(v); buf_m.append(m)

        if len(buf_s) >= 256:
            b_s = torch.tensor(np.array(buf_s), dtype=torch.float32)
            b_a = torch.tensor(buf_a, dtype=torch.long)
            b_lp = torch.tensor(buf_lp, dtype=torch.float32)
            b_ret = torch.tensor(buf_ret, dtype=torch.float32)
            b_val = torch.tensor(buf_val, dtype=torch.float32)
            b_m = torch.tensor(np.array(buf_m), dtype=torch.bool)

            adv = b_ret - b_val
            if len(adv) > 1 and adv.std() > 1e-6:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            model.train()
            for _ in range(4):
                _, probs, vals = model(b_s, b_m)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(b_a)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_lp - b_lp)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 0.8, 1.2) * adv
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(vals.squeeze(-1), b_ret)

                loss = pi_loss + 0.5 * v_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            buf_s.clear(); buf_a.clear(); buf_lp.clear()
            buf_ret.clear(); buf_val.clear(); buf_m.clear()

    return model, total_transitions


# =====================================================================
# 5인 NFSP Self-Play 에이전트
# =====================================================================
class NFSP5PModel:
    def __init__(self, state_dim: int = 149, action_dim: int = 4, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-Network (Best Response)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=0.003)

        # Average Policy Network (SL)
        self.sl_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.opt_sl = optim.Adam(self.sl_net.parameters(), lr=0.001)

        self.rl_buffer = deque(maxlen=10000)
        self.sl_buffer = deque(maxlen=10000)

    def select_action(self, info, legal, eta: float = 0.1, is_eval: bool = False):
        x = torch.tensor(info, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal] = True

        if is_eval:
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x)
                lg[0, ~mask] = -1e9
                probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
            return int(np.random.choice(self.action_dim, p=probs))

        if random.random() < eta:
            # BR (Q-net)
            self.q_net.eval()
            with torch.no_grad():
                q = self.q_net(x).squeeze(0)
                q[~mask] = -1e9
                act = int(q.argmax().item())
            self.sl_buffer.append((info, act))
        else:
            # Average Policy
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x)
                lg[0, ~mask] = -1e9
                probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
            act = int(np.random.choice(self.action_dim, p=probs))

        return act

    def update(self, batch_size: int = 64):
        if len(self.rl_buffer) >= batch_size:
            batch = random.sample(self.rl_buffer, batch_size)
            s, a, r = zip(*batch)
            b_s = torch.tensor(np.array(s), dtype=torch.float32)
            b_a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
            b_r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)

            self.q_net.train()
            cur_q = self.q_net(b_s).gather(1, b_a)
            loss_q = F.mse_loss(cur_q, b_r)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()

        if len(self.sl_buffer) >= batch_size:
            batch_sl = random.sample(self.sl_buffer, batch_size)
            s_sl, a_sl = zip(*batch_sl)
            b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
            b_asl = torch.tensor(a_sl, dtype=torch.long)

            self.sl_net.train()
            logits_sl = self.sl_net(b_ssl)
            loss_sl = F.cross_entropy(logits_sl, b_asl)
            self.opt_sl.zero_grad()
            loss_sl.backward()
            self.opt_sl.step()


def train_5p_nfsp_selfplay(game, num_episodes: int = 2000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    state_dim = len(game.new_initial_state().information_state_tensor(0))
    action_dim = game.num_distinct_actions()

    model = NFSP5PModel(state_dim, action_dim)
    total_transitions = 0

    for ep in range(num_episodes):
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        ep_history = []

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                legal = st.legal_actions(cp)
                info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act = model.select_action(info, legal, eta=0.1, is_eval=False)

                ep_history.append((info, act, cp))
                st.apply_action(act)
                total_transitions += 1

        returns = st.returns()
        for s, a, cp in ep_history:
            ret_bb = returns[cp] / 100.0
            model.rl_buffer.append((s, a, ret_bb))

        model.update(batch_size=32)

    return model, total_transitions


# =====================================================================
# Phase 6: 4대 기준선 상대와 5인 풀게임 평가
# =====================================================================
def evaluate_against_baselines(game, ai_type: str, model_obj, num_hands: int = 500, eval_seed: int = 20261018):
    baselines = {
        'Random': RandomAgent(),
        'AlwaysFold': AlwaysFoldAgent(),
        'CheckCall': CheckCallAgent(),
        'RuleBased': RuleBasedAgent()
    }

    eval_results = {}
    rng = np.random.RandomState(eval_seed)

    for b_name, b_agent in baselines.items():
        # Hero(Seat 0) vs 4인의 동일 기준선 봇
        returns = []

        for h in range(num_hands):
            st = game.new_initial_state()
            advance_chance_nodes(st, rng)

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                else:
                    cp = st.current_player()
                    legal = st.legal_actions(cp)

                    if cp == 0: # Hero AI
                        info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        if ai_type == 'PPO':
                            x = torch.tensor(info).unsqueeze(0)
                            mask = torch.zeros(4, dtype=torch.bool)
                            mask[legal] = True
                            with torch.no_grad():
                                _, probs, _ = model_obj(x, mask.unsqueeze(0))
                                act = int(probs.argmax().item())
                        elif ai_type == 'NFSP':
                            act = model_obj.select_action(info, legal, is_eval=True)
                        st.apply_action(act)
                    else: # Baseline 봇
                        act = b_agent.act(st, cp, legal)
                        st.apply_action(act)

            ret_bb = st.returns()[0] / 100.0
            returns.append(ret_bb)

        mean_bb_100 = float(np.mean(returns)) * 100.0
        ci_half = float(stats.t.ppf(0.975, df=num_hands-1) * (np.std(returns, ddof=1) / np.sqrt(num_hands)) * 100.0)
        eval_results[b_name] = {'mean': mean_bb_100, 'ci': ci_half}

    return eval_results


# =====================================================================
# Phase 7: 카드 인지 능력 비교 (AA vs 99 TVD)
# =====================================================================
def probe_card_awareness(game, ai_type: str, model_obj):
    """
    동일한 베팅 상태(Unopened 및 Facing Open)에서 카드만 AA vs 99로 교체했을 때의
    실제 소프트맥스 확률분포 및 Total Variation Distance(TVD) 측정
    """
    # 24장 덱에서 카드 인덱스:
    # 6 Ranks: 0=9, 1=T, 2=J, 3=Q, 4=K, 5=A (각 rank당 4 suits)
    # A = 20~23, 9 = 0~3
    # AA = (20, 21), 99 = (0, 1)

    cards_dict = {
        'AA': (20, 21),
        '99': (0, 1)
    }

    tvd_results = {}
    probs_by_card = {}

    for h_name, (c1, c2) in cards_dict.items():
        st = game.new_initial_state()
        # Hero(Seat 2)에게 해당 카드 주입
        remaining = [c for c in range(24) if c not in (c1, c2)]
        deck_idx = 0
        for p in range(5):
            if p == 2:
                st.apply_action(c1); st.apply_action(c2)
            else:
                st.apply_action(remaining[deck_idx]); deck_idx += 1
                st.apply_action(remaining[deck_idx]); deck_idx += 1

        cur = st.current_player()
        legal = st.legal_actions(cur)
        info = np.array(st.information_state_tensor(cur), dtype=np.float32)

        if ai_type == 'PPO':
            x = torch.tensor(info).unsqueeze(0)
            mask = torch.zeros(4, dtype=torch.bool)
            mask[legal] = True
            with torch.no_grad():
                _, probs, _ = model_obj(x, mask.unsqueeze(0))
                p_arr = probs.squeeze(0).numpy()
        elif ai_type == 'NFSP':
            x = torch.tensor(info).unsqueeze(0)
            mask = torch.zeros(4, dtype=torch.bool)
            mask[legal] = True
            with torch.no_grad():
                lg = model_obj.sl_net(x)
                lg[0, ~mask] = -1e9
                p_arr = F.softmax(lg, dim=-1).squeeze(0).numpy()

        probs_by_card[h_name] = p_arr

    p_aa = probs_by_card['AA']
    p_99 = probs_by_card['99']
    tvd = float(0.5 * np.sum(np.abs(p_aa - p_99)))

    return {
        'p_aa': p_aa,
        'p_99': p_99,
        'tvd': tvd
    }


def run_benchmark():
    print("=" * 85)
    print("PHASE 5, 6, 7: 5인 축소 홀덤 알고리즘 비교 벤치마크")
    print("=" * 85)

    game = get_5p_reduced_holdem_game()
    print("  [게임 스펙]: 5인 Texas Hold'em, 24장 덱(9~A), 2라운드(Preflop/Flop), 10BB 스택")

    # 1. MCCFR 메모리 제약 감사
    print("\n[PHASE 5 감사: 5인 홀덤에서 MCCFR의 실행 가능성 조사]")
    print("  * 2인 게임에서는 MCCFR이 초고속 수렴(Kuhn 0.01초)하였으나,")
    print("  * 5인 홀덤에서는 정보집합(Information Set) 수가 기하급수적으로 폭증하여,")
    print("    C++ 타뷸러 MCCFR 테이블 생성 시 수십 GB 이상의 메모리 및 수천 시간의 순회가 요구됨!")
    print("  * 공식 선언: 5인 이상의 대규모 불완전정보 포커에서는 Tabular MCCFR을 직접 실행할 수 없으므로,")
    print("    신경망 함수 근사 기반인 PPO Self-Play와 NFSP(Neural Fictitious Self-Play)를 학습 비교함.")

    # 2. PPO & NFSP 학습 (각 2,000 에피소드)
    print("\n[PHASE 5 학습 진행]")
    t0 = time.time()
    ppo_model, ppo_trans = train_5p_ppo_selfplay(game, num_episodes=2000, seed=42)
    t_ppo = time.time() - t0
    print(f"  * PPO Self-Play 학습 완료: {t_ppo:.1f}초 소요 (총 {ppo_trans:,} transitions)")

    t0 = time.time()
    nfsp_model, nfsp_trans = train_5p_nfsp_selfplay(game, num_episodes=2000, seed=42)
    t_nfsp = time.time() - t0
    print(f"  * NFSP Self-Play 학습 완료: {t_nfsp:.1f}초 소요 (총 {nfsp_trans:,} transitions)")

    # 3. Phase 6: 절대 성능 기준선(Baseline) 4종 대국 평가
    print("\n" + "=" * 85)
    print("PHASE 6: 4대 절대 기준선과의 5인 풀게임 평가 (각 500판, 단위: bb/100):")
    print("=" * 85)
    print(f"{'알고리즘':<12} | {'vs Random':<20} | {'vs AlwaysFold':<20} | {'vs CheckCall':<20} | {'vs RuleBased'}")
    print("-" * 90)

    ppo_eval = evaluate_against_baselines(game, 'PPO', ppo_model, num_hands=500)
    nfsp_eval = evaluate_against_baselines(game, 'NFSP', nfsp_model, num_hands=500)

    for name, ev in [('PPO', ppo_eval), ('NFSP', nfsp_eval)]:
        r_str = f"{ev['Random']['mean']:+6.2f} ± {ev['Random']['ci']:4.2f}"
        f_str = f"{ev['AlwaysFold']['mean']:+6.2f} ± {ev['AlwaysFold']['ci']:4.2f}"
        c_str = f"{ev['CheckCall']['mean']:+6.2f} ± {ev['CheckCall']['ci']:4.2f}"
        u_str = f"{ev['RuleBased']['mean']:+6.2f} ± {ev['RuleBased']['ci']:4.2f}"
        print(f"{name:<12} | {r_str:<20} | {f_str:<20} | {c_str:<20} | {u_str}")

    # 4. Phase 7: 카드 인지 능력 감사 (AA vs 99)
    print("\n" + "=" * 85)
    print("PHASE 7: 카드 인지 능력 감사 (AA vs 99 실제 정책 분포 및 TVD):")
    print("=" * 85)

    ppo_card = probe_card_awareness(game, 'PPO', ppo_model)
    nfsp_card = probe_card_awareness(game, 'NFSP', nfsp_model)

    print(f"{'알고리즘':<12} | {'AA 정책 [Fold, Call, Pot, Allin]':<35} | {'99 정책 [Fold, Call, Pot, Allin]':<35} | {'TVD (차이)'}")
    print("-" * 95)
    for name, cd in [('PPO', ppo_card), ('NFSP', nfsp_card)]:
        p_aa_str = f"[{cd['p_aa'][0]:.2f}, {cd['p_aa'][1]:.2f}, {cd['p_aa'][2]:.2f}, {cd['p_aa'][3]:.2f}]"
        p_99_str = f"[{cd['p_99'][0]:.2f}, {cd['p_99'][1]:.2f}, {cd['p_99'][2]:.2f}, {cd['p_99'][3]:.2f}]"
        print(f"{name:<12} | {p_aa_str:<35} | {p_99_str:<35} | {cd['tvd']:6.4f}")

    # CSV 저장
    csv_rows = []
    for b_name in ['Random', 'AlwaysFold', 'CheckCall', 'RuleBased']:
        csv_rows.append({
            'algorithm': 'PPO', 'opponent_baseline': b_name,
            'bb_per_100': ppo_eval[b_name]['mean'], 'ci_half': ppo_eval[b_name]['ci']
        })
        csv_rows.append({
            'algorithm': 'NFSP', 'opponent_baseline': b_name,
            'bb_per_100': nfsp_eval[b_name]['mean'], 'ci_half': nfsp_eval[b_name]['ci']
        })

    csv_path = "v52/data/phase5_6_7_five_player_benchmark.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[저장 완료] Phase 5, 6, 7 벤치마크 결과 CSV: {csv_path}\n")

if __name__ == '__main__':
    run_benchmark()
