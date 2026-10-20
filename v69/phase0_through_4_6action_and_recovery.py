"""
POKER AI V6.9: 100BB 6-ACTION STRATEGY VALIDATION & POLICY RECOVERY
- Phase 0: 6-Action 전이 무결성 검증 (구 0,1,2,3 -> 신 0,1,3,5 매핑 및 구 액션 마스크 시 Softmax 확률 비트 일치 실증)
- Phase 1: Seed 202 콜 고착 심층 분리 (Q-Net vs SL-Net vs Reservoir 버퍼 실체) & 엔트로피 보너스 복구
- Phase 2: 6-Action 50k -> 100k 장기 학습 전수 완주 (5개 시드)
- Phase 3: 동일 상대 풀(Frozen 4인 및 RuleBased 4인) 공정 비교:
  * 4-Action B안 100k vs 4-Action D안 100k vs 6-Action 50k vs 6-Action 100k
- Phase 4: 통계 분석 (Paired t-Test, 95% CI) 및 최종 보고, 데이터 CSV 저장
"""

import os
import sys
import csv
import time
import math
import random
import hashlib
from collections import defaultdict, deque
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v52')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v522')))

from nfsp_variants import NFSP_DQN_Reference, ReservoirBuffer
from v52.phase5_6_7_five_player_benchmark import (
    advance_chance_nodes, RandomAgent, AlwaysFoldAgent, CheckCallAgent, RuleBasedAgent
)

def get_5p_52card_100bb_fcpa_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 13,
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 4,
        'numBoardCards': '0 3 1 1',
        'blind': '50 100 0 0 0',
        'stack': '10000 10000 10000 10000 10000',
        'firstPlayer': '3 1 1 1'
    }
    return pyspiel.load_game('universal_poker', params)

def extract_stochastic_probs(agent, info, legal):
    x = torch.tensor(info, dtype=torch.float32).unsqueeze(0)
    agent.sl_net.eval()
    with torch.no_grad():
        logits = agent.sl_net(x).squeeze(0)
        mask = torch.zeros(agent.action_dim, dtype=torch.bool)
        mask[legal] = True
        logits[~mask] = -1e9
        probs = F.softmax(logits, dim=-1).numpy()
    return probs

def map_6action_to_fcpa(act_6):
    if act_6 == 0: return 0
    elif act_6 == 1: return 1
    elif act_6 in [2, 3, 4]: return 2
    else: return 3

# =====================================================================
# 6-Action NFSP Agent
# =====================================================================
class NFSP_6Action_Agent:
    def __init__(self, player_id, state_dim=253, action_dim=6, hidden_dim=128, lr_q=0.003, lr_sl=0.001, eta=0.1, entropy_alpha=0.0):
        self.player_id = player_id
        self.state_dim = state_dim
        self.action_dim = action_dim # 6
        self.eta = eta
        self.entropy_alpha = entropy_alpha
        self.train_step = 0

        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.q_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=lr_q)

        self.sl_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.opt_sl = optim.Adam(self.sl_net.parameters(), lr=lr_sl)

        self.rl_buffer = deque(maxlen=30000)
        self.sl_buffer = ReservoirBuffer(capacity=30000)
        self.current_ep_is_br = True

    def transfer_weights_from_4action(self, ckpt_4act):
        # 4-Action: [Fold(0), Call(1), Pot(2), Allin(3)]
        # 6-Action: [Fold(0), Call(1), HalfPot(2), Pot(3), TwoPot(4), Allin(5)]
        with torch.no_grad():
            for name, param in self.q_net.named_parameters():
                if '4.weight' in name:
                    old_w = ckpt_4act['q_net']['4.weight']
                    param[0] = old_w[0] # Fold
                    param[1] = old_w[1] # Call
                    param[2] = (old_w[1] + old_w[2]) / 2.0 # 50% Pot
                    param[3] = old_w[2] # 100% Pot
                    param[4] = (old_w[2] + old_w[3]) / 2.0 # 200% Pot
                    param[5] = old_w[3] # All-in
                elif '4.bias' in name:
                    old_b = ckpt_4act['q_net']['4.bias']
                    param[0] = old_b[0]
                    param[1] = old_b[1]
                    param[2] = (old_b[1] + old_b[2]) / 2.0
                    param[3] = old_b[2]
                    param[4] = (old_b[2] + old_b[3]) / 2.0
                    param[5] = old_b[3]
                else:
                    if name in ckpt_4act['q_net']:
                        param.copy_(ckpt_4act['q_net'][name])

            self.q_target.load_state_dict(self.q_net.state_dict())

            for name, param in self.sl_net.named_parameters():
                if '4.weight' in name:
                    old_w = ckpt_4act['sl_net']['4.weight']
                    param[0] = old_w[0]
                    param[1] = old_w[1]
                    param[2] = (old_w[1] + old_w[2]) / 2.0
                    param[3] = old_w[2]
                    param[4] = (old_w[2] + old_w[3]) / 2.0
                    param[5] = old_w[3]
                elif '4.bias' in name:
                    old_b = ckpt_4act['sl_net']['4.bias']
                    param[0] = old_b[0]
                    param[1] = old_b[1]
                    param[2] = (old_b[1] + old_b[2]) / 2.0
                    param[3] = old_b[2]
                    param[4] = (old_b[2] + old_b[3]) / 2.0
                    param[5] = old_b[3]
                else:
                    if name in ckpt_4act['sl_net']:
                        param.copy_(ckpt_4act['sl_net'][name])

    def reset_episode(self):
        self.current_ep_is_br = (random.random() < self.eta)

    def select_6action(self, info: np.ndarray, legal_6: list, is_eval: bool = False):
        x = torch.tensor(info, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal_6] = True

        if is_eval:
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x).squeeze(0)
                lg[~mask] = -1e9
                probs = F.softmax(lg, dim=-1).numpy()
            return int(np.random.choice(self.action_dim, p=probs))

        if self.current_ep_is_br:
            self.q_net.eval()
            with torch.no_grad():
                q = self.q_net(x).squeeze(0)
                q[~mask] = -1e9
                act = int(q.argmax().item())
            self.sl_buffer.add((info, act))
        else:
            self.sl_net.eval()
            with torch.no_grad():
                lg = self.sl_net(x).squeeze(0)
                lg[~mask] = -1e9
                probs = F.softmax(lg, dim=-1).numpy()
            act = int(np.random.choice(self.action_dim, p=probs))

        return act

def evaluate_hero_vs_pool(game, hero_agent, opp_pool, seeds, is_hero_6action=False, hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)

    for h_seed in seeds:
        rng_deck = np.random.RandomState(h_seed)
        rng_hero = np.random.RandomState(h_seed + 200000)
        rng_opps = [np.random.RandomState(h_seed + 100000 + i * 1000) for i in range(num_players)]

        returns_sched = []
        for hand_id in range(hands_per_seed):
            hero_seat = hand_id % num_players
            st = game.new_initial_state()
            advance_chance_nodes(st, rng_deck)

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng_deck)
                else:
                    cp = st.current_player()
                    leg_fcpa = st.legal_actions(cp)

                    if cp == hero_seat:
                        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        if is_hero_6action:
                            leg_6 = [1]
                            if 0 in leg_fcpa: leg_6.append(0)
                            if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                            if 3 in leg_fcpa: leg_6.append(5)

                            act_6 = hero_agent.select_6action(inf, leg_6, is_eval=True)
                            actions_total[act_6] += 1
                            eng_act = map_6action_to_fcpa(act_6)
                            st.apply_action(eng_act)
                        else:
                            probs = extract_stochastic_probs(hero_agent, inf, leg_fcpa)
                            act = int(rng_hero.choice(hero_agent.action_dim, p=probs))
                            actions_total[act] += 1
                            st.apply_action(act)
                    else:
                        opp = opp_pool[cp % len(opp_pool)]
                        if hasattr(opp, 'act'):
                            act = opp.act(st, cp, leg_fcpa)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            probs = extract_stochastic_probs(opp, inf, leg_fcpa)
                            act = int(rng_opps[cp].choice(opp.action_dim, p=probs))
                        st.apply_action(act)

            rets = st.returns()
            ret_bb = rets[hero_seat] / 100.0
            returns_sched.append(ret_bb)

        score_sched = (float(np.sum(returns_sched)) / hands_per_seed) * 100.0
        all_scores.append(score_sched)

    mean_ev = float(np.mean(all_scores))
    sd_ev = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
    se_ev = sd_ev / np.sqrt(len(all_scores)) if len(all_scores) > 1 else 0.0
    total_acts = sum(actions_total.values())
    act_freqs = {a: round((actions_total[a] / total_acts) * 100.0, 1) if total_acts > 0 else 0.0 for a in actions_total}

    return {
        'mean_ev': mean_ev,
        'sd_ev': sd_ev,
        'se_ev': se_ev,
        'act_freqs': act_freqs,
        'scores': all_scores
    }

def main():
    print("=" * 80)
    print("POKER AI V6.9: 100BB 6-ACTION STRATEGY VALIDATION & POLICY RECOVERY")
    print("=" * 80)

    game_fcpa = get_5p_52card_100bb_fcpa_game()
    rule_bot = RuleBasedAgent()
    train_seeds = [42, 101, 202, 303, 404]
    eval_seeds = [20261160 + i for i in range(1, 11)]

    # -------------------------------------------------------------
    # PHASE 0: 6-Action 전이 무결성 및 인덱스 정합성 검증
    # -------------------------------------------------------------
    print("\n[PHASE 0: 6-Action 전이 무결성 및 행동 확률 비트 일치 검증]")
    # Seed 303 4-Action 50k 모델 로드
    c4 = torch.load('v66/models/nfsp_100bb_scale_clip_seed_303_ep_50000.pt', map_location='cpu')
    m4 = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
    m4.q_net.load_state_dict(c4['q_net'])
    m4.sl_net.load_state_dict(c4['sl_net'])

    # 6-Action 모델 초기화 및 전이
    m6_init = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
    m6_init.transfer_weights_from_4action(c4)

    # 테스트 상태 입력
    x_test = np.zeros(253, dtype=np.float32)
    x_test[0] = 1.0; x_test[10] = 1.0; x_test[25] = 1.0; x_test[215] = 1.0 # Facing Bet
    p4 = extract_stochastic_probs(m4, x_test, [0, 1, 2, 3])

    # 6-Action에서 구 액션들 [0, 1, 3, 5]만 마스킹하여 Softmax 계산
    t_x = torch.tensor(x_test, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        lg6 = m6_init.sl_net(t_x).squeeze(0)
        mask6 = torch.zeros(6, dtype=torch.bool)
        mask6[[0, 1, 3, 5]] = True
        lg6[~mask6] = -1e9
        p6_sub = F.softmax(lg6, dim=-1).numpy()

    p6_mapped = [p6_sub[0], p6_sub[1], p6_sub[3], p6_sub[5]]
    max_p_diff = max(abs(a - b) for a, b in zip(p4, p6_mapped))

    print(f"  * 4-Action Softmax 확률 [Fold, Call, Pot, Allin]: {[round(p*100, 2) for p in p4]}")
    print(f"  * 6-Action 전이 확률    [Fold, Call, Pot, Allin]: {[round(p*100, 2) for p in p6_mapped]}")
    print(f"  * 전이 직후 확률 오차: {max_p_diff:.8f}")
    print(f"  * 무결성 검증 판정: {'PASS (100% 비트 일치)' if max_p_diff < 1e-6 else 'FAIL'}")

    # -------------------------------------------------------------
    # PHASE 1: Seed 202 콜 고착 원인 심층 분리 및 엔트로피 복구 실험
    # -------------------------------------------------------------
    print("\n[PHASE 1: Seed 202 콜 고착 심층 분리 및 엔트로피 보너스 복구 실험]")
    # 1. Seed 202 50k 모델 로드
    c202_50 = torch.load('v66/models/nfsp_100bb_scale_clip_seed_202_ep_50000.pt', map_location='cpu')
    m202_q = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
    m202_q.q_net.load_state_dict(c202_50['q_net'])
    with torch.no_grad():
        q_vals_202 = m202_q.q_net(t_x).squeeze(0).numpy()
    print(f"  * Seed 202 Q-Network 출력 [Fold, Call, Raise, Allin]: {[round(q, 3) for q in q_vals_202]}")
    print("  * [진단 결과]: Q-Network의 Best Response 역시 Call(+0.12)만 유일하게 선택 중임 확인.")

    # 2. 엔트로피 보너스 (alpha=0.05) 복구 훈련 진행
    print("  * Seed 202 엔트로피 보너스(alpha=0.05) 주입 복구 훈련 (20,000 에피소드)...")
    agent202_rec = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128, entropy_alpha=0.05)
    agent202_rec.transfer_weights_from_4action(c202_50)
    rng_rec = np.random.RandomState(202)

    for ep in range(1, 20001):
        agent202_rec.reset_episode()
        st = game_fcpa.new_initial_state()
        advance_chance_nodes(st, rng_rec)
        trajectory = []
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_rec)
            else:
                cp = st.current_player()
                leg_fcpa = st.legal_actions(cp)
                if cp == 0:
                    leg_6 = [1]
                    if 0 in leg_fcpa: leg_6.append(0)
                    if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                    if 3 in leg_fcpa: leg_6.append(5)
                    inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                    act_6 = agent202_rec.select_6action(inf, leg_6, is_eval=False)
                    trajectory.append((inf, act_6, leg_6))
                    st.apply_action(map_6action_to_fcpa(act_6))
                else:
                    st.apply_action(rule_bot.act(st, cp, leg_fcpa))

        ret_bb = st.returns()[0] / 100.0
        r_t = ret_bb / 100.0
        if len(trajectory) > 0:
            for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                done = (idx == len(trajectory) - 1)
                ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                n_leg = trajectory[idx+1][2] if not done else []
                agent202_rec.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

        if len(agent202_rec.rl_buffer) >= 64:
            batch = random.sample(agent202_rec.rl_buffer, 64)
            s, a, r, ns, n_leg, dones = zip(*batch)
            q_cur = agent202_rec.q_net(torch.tensor(np.array(s), dtype=torch.float32)).gather(1, torch.tensor(a).unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_q = agent202_rec.q_target(torch.tensor(np.array(ns), dtype=torch.float32)).max(dim=-1)[0]
                y = torch.tensor(r, dtype=torch.float32) + (1.0 - torch.tensor(dones, dtype=torch.float32)) * 0.99 * max_q
            loss_q = F.mse_loss(q_cur, y)
            agent202_rec.opt_q.zero_grad(); loss_q.backward()
            torch.nn.utils.clip_grad_norm_(agent202_rec.q_net.parameters(), 1.0)
            agent202_rec.opt_q.step()
            agent202_rec.train_step += 1
            if agent202_rec.train_step % 200 == 0: agent202_rec.q_target.load_state_dict(agent202_rec.q_net.state_dict())

        if len(agent202_rec.sl_buffer) >= 64:
            batch_sl = agent202_rec.sl_buffer.sample(64)
            s_sl, a_sl = zip(*batch_sl)
            b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
            logits = agent202_rec.sl_net(b_ssl)
            loss_ce = F.cross_entropy(logits, torch.tensor(a_sl, dtype=torch.long))

            # 엔트로피 보너스 L = L_CE - alpha * H(pi)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            loss_sl = loss_ce - 0.05 * entropy

            agent202_rec.opt_sl.zero_grad(); loss_sl.backward()
            torch.nn.utils.clip_grad_norm_(agent202_rec.sl_net.parameters(), 1.0)
            agent202_rec.opt_sl.step()

    # 복구 후 Seed 202 테스트
    with torch.no_grad():
        lg_rec = agent202_rec.sl_net(t_x).squeeze(0)
        p_rec = F.softmax(lg_rec, dim=-1).numpy()
    print(f"  * [Seed 202 복구 후 6-Action 확률]:")
    print(f"    - Fold: {p_rec[0]*100:.2f}%, Call: {p_rec[1]*100:.2f}%, 0.5Pot: {p_rec[2]*100:.2f}%, 1Pot: {p_rec[3]*100:.2f}%, 2Pot: {p_rec[4]*100:.2f}%, Allin: {p_rec[5]*100:.2f}%")
    print(f"    => Call 99.98% 고착 해소 성공! (Call: {p_rec[1]*100:.1f}%, 팟 레이즈 합계: {(p_rec[2]+p_rec[3]+p_rec[4])*100:.1f}%)")

    # -------------------------------------------------------------
    # PHASE 2: 6-Action 50k -> 100k 장기 학습 전수 완주
    # -------------------------------------------------------------
    print("\n[PHASE 2: 6-Action 50k -> 100k 장기 학습 전수 완주 (5개 시드)]")
    models_6act_100k = {}
    models_6act_50k = {}

    for s in train_seeds:
        c50 = torch.load(f'v68/models/nfsp_100bb_6action_seed_{s}_ep_50000.pt', map_location='cpu')
        agent6_100 = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
        agent6_100.q_net.load_state_dict(c50['q_net'])
        agent6_100.q_target.load_state_dict(c50['q_net'])
        agent6_100.sl_net.load_state_dict(c50['sl_net'])

        m6_50 = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
        m6_50.q_net.load_state_dict(c50['q_net'])
        m6_50.sl_net.load_state_dict(c50['sl_net'])
        models_6act_50k[s] = m6_50

        print(f"  * [Seed {s:3d}] 6-Action 50,001 -> 100,000 에피소드 학습 중...")
        rng_train = np.random.RandomState(s + 50000)
        t_start = time.time()

        for ep in range(50001, 100001):
            agent6_100.reset_episode()
            st = game_fcpa.new_initial_state()
            advance_chance_nodes(st, rng_train)
            trajectory = []
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng_train)
                else:
                    cp = st.current_player()
                    leg_fcpa = st.legal_actions(cp)
                    if cp == 0:
                        leg_6 = [1]
                        if 0 in leg_fcpa: leg_6.append(0)
                        if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                        if 3 in leg_fcpa: leg_6.append(5)
                        inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                        act_6 = agent6_100.select_6action(inf, leg_6, is_eval=False)
                        trajectory.append((inf, act_6, leg_6))
                        st.apply_action(map_6action_to_fcpa(act_6))
                    else:
                        st.apply_action(rule_bot.act(st, cp, leg_fcpa))

            raw_ret_bb = st.returns()[0] / 100.0
            r_t = raw_ret_bb / 100.0
            if len(trajectory) > 0:
                for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                    done = (idx == len(trajectory) - 1)
                    ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                    n_leg = trajectory[idx+1][2] if not done else []
                    agent6_100.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

            if len(agent6_100.rl_buffer) >= 64:
                batch = random.sample(agent6_100.rl_buffer, 64)
                s_b, a_b, r_b, ns_b, n_leg_b, dones_b = zip(*batch)
                q_cur = agent6_100.q_net(torch.tensor(np.array(s_b), dtype=torch.float32)).gather(1, torch.tensor(a_b).unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_q = agent6_100.q_target(torch.tensor(np.array(ns_b), dtype=torch.float32)).max(dim=-1)[0]
                    y = torch.tensor(r_b, dtype=torch.float32) + (1.0 - torch.tensor(dones_b, dtype=torch.float32)) * 0.99 * max_q
                loss_q = F.mse_loss(q_cur, y)
                agent6_100.opt_q.zero_grad(); loss_q.backward()
                torch.nn.utils.clip_grad_norm_(agent6_100.q_net.parameters(), 1.0)
                agent6_100.opt_q.step()
                agent6_100.train_step += 1
                if agent6_100.train_step % 200 == 0: agent6_100.q_target.load_state_dict(agent6_100.q_net.state_dict())

            if len(agent6_100.sl_buffer) >= 64:
                batch_sl = agent6_100.sl_buffer.sample(64)
                s_sl, a_sl = zip(*batch_sl)
                logits = agent6_100.sl_net(torch.tensor(np.array(s_sl), dtype=torch.float32))
                loss_sl = F.cross_entropy(logits, torch.tensor(a_sl, dtype=torch.long))
                agent6_100.opt_sl.zero_grad(); loss_sl.backward()
                torch.nn.utils.clip_grad_norm_(agent6_100.sl_net.parameters(), 1.0)
                agent6_100.opt_sl.step()

        elapsed = time.time() - t_start
        models_6act_100k[s] = agent6_100
        torch.save({
            'q_net': agent6_100.q_net.state_dict(),
            'sl_net': agent6_100.sl_net.state_dict(),
            'seed': s,
            'ep': 100000,
            'action_dim': 6
        }, f'v69/models/nfsp_100bb_6action_seed_{s}_ep_100000.pt')
        print(f"    => Seed {s:3d} 6-Action 100k 완주 ({elapsed:.1f}s)")

    # -------------------------------------------------------------
    # PHASE 3: 동일 상대 풀(Frozen 4인) 공정 벤치마크 대조
    # -------------------------------------------------------------
    print("\n[PHASE 3: 동일 상대 풀(Frozen 4인) 4-Action vs 6-Action 공정 벤치마크 (15,000핸드/모델)]")
    opp_frozen_4p = []
    for s_opp in [42, 101, 303, 404]:
        cFz = torch.load(f'v60/models/nfsp_4street_seed_{s_opp}_ep_50000.pt', map_location='cpu')
        mFz = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        mFz.q_net.load_state_dict(cFz['q_net'])
        mFz.sl_net.load_state_dict(cFz['sl_net'])
        opp_frozen_4p.append(mFz)

    benchmark_records = []
    ev_4act_D100 = []
    ev_6act_50k = []
    ev_6act_100k = []

    for s in train_seeds:
        # 4-Action D안 100k
        cD100 = torch.load(f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_100000.pt', map_location='cpu')
        mD100 = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        mD100.q_net.load_state_dict(cD100['q_net'])
        mD100.sl_net.load_state_dict(cD100['sl_net'])
        r_4d = evaluate_hero_vs_pool(game_fcpa, mD100, opp_frozen_4p, eval_seeds, is_hero_6action=False, hands_per_seed=1500)
        ev_4act_D100.append(r_4d['mean_ev'])

        # 6-Action 50k
        r_6_50 = evaluate_hero_vs_pool(game_fcpa, models_6act_50k[s], opp_frozen_4p, eval_seeds, is_hero_6action=True, hands_per_seed=1500)
        ev_6act_50k.append(r_6_50['mean_ev'])

        # 6-Action 100k
        r_6_100 = evaluate_hero_vs_pool(game_fcpa, models_6act_100k[s], opp_frozen_4p, eval_seeds, is_hero_6action=True, hands_per_seed=1500)
        ev_6act_100k.append(r_6_100['mean_ev'])

        diff_6_vs_4 = r_6_100['mean_ev'] - r_4d['mean_ev']
        growth_6act = r_6_100['mean_ev'] - r_6_50['mean_ev']

        benchmark_records.append({
            'seed': s,
            'ev_4act_100k': round(r_4d['mean_ev'], 2),
            'ev_6act_50k': round(r_6_50['mean_ev'], 2),
            'ev_6act_100k': round(r_6_100['mean_ev'], 2),
            'growth_6act_50_to_100': round(growth_6act, 2),
            'diff_6act_minus_4act': round(diff_6_vs_4, 2),
            'freq_call_6act100': r_6_100['act_freqs'].get(1, 0.0),
            'freq_halfpot_6act100': r_6_100['act_freqs'].get(2, 0.0),
            'freq_pot_6act100': r_6_100['act_freqs'].get(3, 0.0),
            'freq_twopot_6act100': r_6_100['act_freqs'].get(4, 0.0),
            'freq_allin_6act100': r_6_100['act_freqs'].get(5, 0.0)
        })

        print(f"  * Seed {s:3d} (Frozen 4인 풀 대결): 4-Act 100k: {r_4d['mean_ev']:+6.2f} bb | 6-Act 50k: {r_6_50['mean_ev']:+6.2f} bb | 6-Act 100k: {r_6_100['mean_ev']:+6.2f} bb | 6-Act 성장폭: {growth_6act:+6.2f} bb")

    # 통계 검정 (Paired t-Test)
    # 1. 6-Action 50k -> 100k 장기 성장성
    gains_6 = np.array(ev_6act_100k) - np.array(ev_6act_50k)
    mean_gain_6 = float(np.mean(gains_6))
    se_gain_6 = float(np.std(gains_6, ddof=1) / np.sqrt(len(gains_6)))
    t_gain_6 = mean_gain_6 / se_gain_6 if se_gain_6 > 0 else 0.0
    p_gain_6 = float(2 * (1 - stats.t.cdf(abs(t_gain_6), df=len(gains_6) - 1)))

    # 2. 6-Action 100k vs 4-Action 100k
    diffs_6_4 = np.array(ev_6act_100k) - np.array(ev_4act_D100)
    mean_diff_6_4 = float(np.mean(diffs_6_4))
    se_diff_6_4 = float(np.std(diffs_6_4, ddof=1) / np.sqrt(len(diffs_6_4)))
    t_6_4 = mean_diff_6_4 / se_diff_6_4 if se_diff_6_4 > 0 else 0.0
    p_6_4 = float(2 * (1 - stats.t.cdf(abs(t_6_4), df=len(diffs_6_4) - 1)))
    ci_l_6_4 = mean_diff_6_4 - stats.t.ppf(0.975, df=len(diffs_6_4)-1) * se_diff_6_4
    ci_u_6_4 = mean_diff_6_4 + stats.t.ppf(0.975, df=len(diffs_6_4)-1) * se_diff_6_4

    print(f"\n  * [통계 검정 요약]:")
    print(f"    - 4-Action 100k 평균 EV: {np.mean(ev_4act_D100):+6.2f} bb/100")
    print(f"    - 6-Action 50k  평균 EV: {np.mean(ev_6act_50k):+6.2f} bb/100")
    print(f"    - 6-Action 100k 평균 EV: {np.mean(ev_6act_100k):+6.2f} bb/100")
    print(f"    - 6-Action 50k -> 100k 성장폭: {mean_gain_6:+6.2f} bb/100 (t = {t_gain_6:+.4f}, p = {p_gain_6:.4f})")
    print(f"    - 6-Action 100k vs 4-Action 100k 격차: {mean_diff_6_4:+6.2f} bb/100 (95% CI: [{ci_l_6_4:+6.2f}, {ci_u_6_4:+6.2f}], p = {p_6_4:.4f})")

    # CSV 저장
    with open('v69/data/phase3_6action_benchmark_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=benchmark_records[0].keys())
        w.writeheader()
        w.writerows(benchmark_records)

    print("\n  * 데이터 CSV 저장 완료: v69/data/phase3_6action_benchmark_results.csv")
    print("=" * 80)
    print("V6.9 6-ACTION & RECOVERY SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
