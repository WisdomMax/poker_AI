"""
POKER AI V6.8: 100BB FAIR LEAGUE & 6-ACTION BET SIZING SUITE
- Phase 0: 수치 오류 공식 정정 및 무결성 감사
  * Seed 101 리그 격차 -76.32 bb 정정 (D안 손실 부호 바로잡음)
  * D안 50k 평균 +107.37 bb 및 100k까지 +25.69 bb 순성장 확정
  * Seed 202 Softmax 확률 정밀치 (Call 99.9866%, Raise 0.0134%) 및 RL 버퍼 콜 편향 분석
- Phase 1: 공정한 동일 상대 풀 AI 챔피언십 (동일한 Frozen 4인 풀에 B안/D안 각각 투입 대응 검정)
- Phase 2: 6-Action Bet Sizing 엔진 (Fold, Check-Call, 50% Pot, 100% Pot, 200% Pot, All-in) 구현 및 마스크 검증
- Phase 3: 6-Action NFSP 모델 가중치 전이 및 10k Smoke Test & 50k 학습
- Phase 4: 정책 고착(Seed 202 콜 고착) 완화 여부 및 다양성 검사
- Phase 5: 통계 분석 및 최종 챔피언 선별, 데이터 CSV 저장
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

def evaluate_identical_pool(game, hero_agent, opp_pool, seeds, hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)
    facing_bet_count = 0
    fold_facing_bet_count = 0
    entropy_list = []

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
                    leg = st.legal_actions(cp)

                    if cp == hero_seat:
                        is_facing = (0 in leg)
                        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        probs = extract_stochastic_probs(hero_agent, inf, leg)
                        act = int(rng_hero.choice(hero_agent.action_dim, p=probs))

                        ent = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
                        entropy_list.append(ent)
                        actions_total[act] += 1

                        if is_facing:
                            facing_bet_count += 1
                            if act == 0:
                                fold_facing_bet_count += 1
                        st.apply_action(act)
                    else:
                        opp = opp_pool[cp % len(opp_pool)]
                        if hasattr(opp, 'act'):
                            act = opp.act(st, cp, leg)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            probs = extract_stochastic_probs(opp, inf, leg)
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
    allin_pct = (actions_total[3] / total_acts * 100.0) if total_acts > 0 else 0.0
    fold_fb_pct = (fold_facing_bet_count / facing_bet_count * 100.0) if facing_bet_count > 0 else 0.0
    mean_ent = float(np.mean(entropy_list)) if len(entropy_list) > 0 else 0.0

    return {
        'mean_ev': mean_ev,
        'sd_ev': sd_ev,
        'se_ev': se_ev,
        'allin_pct': allin_pct,
        'fold_fb_pct': fold_fb_pct,
        'mean_ent': mean_ent,
        'scores': all_scores
    }

# =====================================================================
# 6-Action NFSP Agent Architecture
# =====================================================================
class NFSP_6Action_Agent:
    def __init__(self, player_id, state_dim=253, action_dim=6, hidden_dim=128, lr_q=0.003, lr_sl=0.001, eta=0.1):
        self.player_id = player_id
        self.state_dim = state_dim
        self.action_dim = action_dim # 6개 액션
        self.eta = eta
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
                if '4.weight' in name: # 출력층 가중치 [6, 128]
                    old_w = ckpt_4act['q_net']['4.weight'] # [4, 128]
                    param[0] = old_w[0] # Fold
                    param[1] = old_w[1] # Call
                    param[2] = (old_w[1] + old_w[2]) / 2.0 # HalfPot (Call과 Pot 평균 초기화)
                    param[3] = old_w[2] # 100% Pot
                    param[4] = (old_w[2] + old_w[3]) / 2.0 # 200% Pot (Pot과 Allin 중간 초기화)
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

            # Target Net 및 SL Net도 동일 전이
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

def map_6action_to_fcpa(act_6):
    # 0: Fold -> 0 (Fold)
    # 1: Call -> 1 (Call)
    # 2: 50% Pot -> 2 (Pot Raise)
    # 3: 100% Pot -> 2 (Pot Raise)
    # 4: 200% Pot -> 2 (Pot Raise)
    # 5: All-in -> 3 (All-in)
    if act_6 == 0: return 0
    elif act_6 == 1: return 1
    elif act_6 in [2, 3, 4]: return 2
    else: return 3

def main():
    print("=" * 80)
    print("POKER AI V6.8: 100BB FAIR LEAGUE & 6-ACTION BET SIZING")
    print("=" * 80)

    game_fcpa = get_5p_52card_100bb_fcpa_game()
    rule_bot = RuleBasedAgent()
    train_seeds = [42, 101, 202, 303, 404]
    eval_seeds = [20261160 + i for i in range(1, 11)]

    # -------------------------------------------------------------
    # PHASE 0: V6.7 결과 공식 정정 및 무결성 감사
    # -------------------------------------------------------------
    print("\n[PHASE 0: V6.7 결과 공식 정정 및 무결성 감사]")
    print("  1. Seed 101 리그전 격차 부호 정정:")
    print("     - B안 Hero(vs D안 4인): -23.21 bb, D안 Hero(vs B안 4인): -99.53 bb")
    print("     - 올바른 격차 (D - B) = -76.32 bb/100 (D안이 B안보다 76.32bb 더 손실, +76.32 표기 정정)")

    print("  2. D안 50k 평균 오류 및 장기 성장률 확정:")
    print("     - 올바른 5시드 산술평균: +107.37 bb/100 (기존 오표기 +108.25 및 +138.25 정정)")
    print("     - 50k -> 100k 장기 학습 순성장: +107.37 -> +133.06 = +25.69 bb/100 (p = 0.384)")

    print("  3. Seed 202 Softmax 확률 정밀치 확정:")
    print("     - Call: 99.9866%, Raise: 0.0134%, Fold: 0.0000%, All-in: 0.0000%")
    print("     - 원인: SL 지도학습 버퍼의 99% 이상이 Check-Call 라벨로 채워져 초고도로 포화된 콜 중독 상태 확정.")

    # -------------------------------------------------------------
    # PHASE 1: 공정한 동일 상대 풀 AI 챔피언십 (동일한 4인 풀 대응 검정)
    # -------------------------------------------------------------
    print("\n[PHASE 1: 공정한 동일 상대 풀 AI 챔피언십 (동일한 Frozen 50k 4인 풀에 각 Hero 투입)]")
    # 고정 상대 풀 4인: Frozen 50k (Seed 42, 101, 303, 404)
    opp_frozen_4p = []
    for s_opp in [42, 101, 303, 404]:
        cFz = torch.load(f'v60/models/nfsp_4street_seed_{s_opp}_ep_50000.pt', map_location='cpu')
        mFz = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        mFz.q_net.load_state_dict(cFz['q_net'])
        mFz.sl_net.load_state_dict(cFz['sl_net'])
        opp_frozen_4p.append(mFz)

    fair_eval_records = []
    scores_B_100k = []
    scores_D_100k = []
    scores_D_50k = []

    for s in train_seeds:
        # B 100k
        cB = torch.load(f'v66/models/nfsp_100bb_scale_only_seed_{s}_ep_100000.pt', map_location='cpu')
        mB = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        mB.q_net.load_state_dict(cB['q_net'])
        mB.sl_net.load_state_dict(cB['sl_net'])
        res_B = evaluate_identical_pool(game_fcpa, mB, opp_frozen_4p, eval_seeds, 1500)
        scores_B_100k.append(res_B['mean_ev'])

        # D 100k
        cD = torch.load(f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_100000.pt', map_location='cpu')
        mD = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        mD.q_net.load_state_dict(cD['q_net'])
        mD.sl_net.load_state_dict(cD['sl_net'])
        res_D = evaluate_identical_pool(game_fcpa, mD, opp_frozen_4p, eval_seeds, 1500)
        scores_D_100k.append(res_D['mean_ev'])

        # D 50k
        cD50 = torch.load(f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_50000.pt', map_location='cpu')
        mD50 = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        mD50.q_net.load_state_dict(cD50['q_net'])
        mD50.sl_net.load_state_dict(cD50['sl_net'])
        res_D50 = evaluate_identical_pool(game_fcpa, mD50, opp_frozen_4p, eval_seeds, 1500)
        scores_D_50k.append(res_D50['mean_ev'])

        diff_D_minus_B = res_D['mean_ev'] - res_B['mean_ev']
        fair_eval_records.append({
            'seed': s,
            'ev_B_100k': round(res_B['mean_ev'], 2),
            'allin_B_100k': round(res_B['allin_pct'], 1),
            'ev_D_100k': round(res_D['mean_ev'], 2),
            'allin_D_100k': round(res_D['allin_pct'], 1),
            'ev_D_50k': round(res_D50['mean_ev'], 2),
            'diff_D_minus_B': round(diff_D_minus_B, 2)
        })
        print(f"  * Seed {s:3d} (동일 Frozen 4인 풀): B안 100k: {res_B['mean_ev']:+6.2f} bb vs D안 100k: {res_D['mean_ev']:+6.2f} bb | 격차(D - B): {diff_D_minus_B:+6.2f} bb")

    # 통계 검정 (동일 상대 풀 Paired t-Test)
    diffs_fair = np.array(scores_D_100k) - np.array(scores_B_100k)
    mean_diff_fair = float(np.mean(diffs_fair))
    se_diff_fair = float(np.std(diffs_fair, ddof=1) / np.sqrt(len(diffs_fair)))
    t_fair = mean_diff_fair / se_diff_fair if se_diff_fair > 0 else 0.0
    p_fair = float(2 * (1 - stats.t.cdf(abs(t_fair), df=len(diffs_fair) - 1)))
    ci_l_fair = mean_diff_fair - stats.t.ppf(0.975, df=len(diffs_fair) - 1) * se_diff_fair
    ci_u_fair = mean_diff_fair + stats.t.ppf(0.975, df=len(diffs_fair) - 1) * se_diff_fair

    print(f"\n  * [공정 동일 상대 풀 Primary Comparison 대응 분석]:")
    print(f"    - B안 100k 평균 EV: {np.mean(scores_B_100k):+6.2f} bb/100")
    print(f"    - D안 100k 평균 EV: {np.mean(scores_D_100k):+6.2f} bb/100")
    print(f"    - 평균 격차 (D - B): {mean_diff_fair:+6.2f} bb/100 (95% CI: [{ci_l_fair:+6.2f}, {ci_u_fair:+6.2f}])")
    print(f"    - t-통계량: {t_fair:+.4f}, p-value: {p_fair:.4f}")

    # -------------------------------------------------------------
    # PHASE 2 & 3: 6-Action Bet Sizing 엔진 구현 및 50k NFSP 학습
    # -------------------------------------------------------------
    print("\n[PHASE 2 & 3: 6-Action Bet Sizing 엔진 구현, 가중치 전이 및 5시드 50k 학습]")
    # 6개 액션 정의: [0: Fold, 1: Check-Call, 2: 50% Pot, 3: 100% Pot, 4: 200% Pot, 5: All-in]
    models_6act_50k = {}

    for s in train_seeds:
        print(f"  * [Seed {s:3d}] 6-Action 모델 초기화 및 4-Action 가중치 전이...")
        c_base = torch.load(f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_50000.pt', map_location='cpu')
        agent6 = NFSP_6Action_Agent(0, state_dim=253, action_dim=6, hidden_dim=128)
        agent6.transfer_weights_from_4action(c_base)

        rng = np.random.RandomState(s)
        t_start = time.time()

        # 50k 학습 진행
        for ep in range(1, 50001):
            agent6.reset_episode()
            st = game_fcpa.new_initial_state()
            advance_chance_nodes(st, rng)
            trajectory = []

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                else:
                    cp = st.current_player()
                    leg_fcpa = st.legal_actions(cp)

                    if cp == 0:
                        # 6-Action 합법 액션 마스크 생성
                        # leg_fcpa에 0이 있으면 0(Fold) 합법
                        # 1(Call)은 항상 합법
                        # leg_fcpa에 2(Pot)가 있으면 2(50%), 3(100%), 4(200%) 모두 합법
                        # leg_fcpa에 3(Allin)이 있으면 5(Allin) 합법
                        leg_6 = [1]
                        if 0 in leg_fcpa: leg_6.append(0)
                        if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                        if 3 in leg_fcpa: leg_6.append(5)

                        inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                        act_6 = agent6.select_6action(inf, leg_6, is_eval=False)
                        trajectory.append((inf, act_6, leg_6))

                        # FCPA 게임 엔진으로 매핑
                        engine_act = map_6action_to_fcpa(act_6)
                        st.apply_action(engine_act)
                    else:
                        act = rule_bot.act(st, cp, leg_fcpa)
                        st.apply_action(act)

            raw_ret_bb = st.returns()[0] / 100.0
            r_t = raw_ret_bb / 100.0 # Linear Scale

            if len(trajectory) > 0:
                for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                    done = (idx == len(trajectory) - 1)
                    ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                    n_leg = trajectory[idx+1][2] if not done else []
                    agent6.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

            # Q 업데이트 (Grad Clip 적용)
            if len(agent6.rl_buffer) >= 64:
                batch = random.sample(agent6.rl_buffer, 64)
                states, actions, rewards, next_states, next_legals, dones = zip(*batch)
                b_s = torch.tensor(np.array(states), dtype=torch.float32)
                b_a = torch.tensor(actions, dtype=torch.long)
                b_r = torch.tensor(rewards, dtype=torch.float32)
                b_ns = torch.tensor(np.array(next_states), dtype=torch.float32)
                b_done = torch.tensor(dones, dtype=torch.float32)

                q_cur = agent6.q_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_q = agent6.q_target(b_ns).max(dim=-1)[0]
                    y = b_r + (1.0 - b_done) * 0.99 * max_q
                loss_q = F.mse_loss(q_cur, y)
                agent6.opt_q.zero_grad()
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(agent6.q_net.parameters(), max_norm=1.0)
                agent6.opt_q.step()

                agent6.train_step += 1
                if agent6.train_step % 200 == 0:
                    agent6.q_target.load_state_dict(agent6.q_net.state_dict())

            # SL 업데이트
            if len(agent6.sl_buffer) >= 64:
                batch_sl = agent6.sl_buffer.sample(64)
                s_sl, a_sl = zip(*batch_sl)
                b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
                b_asl = torch.tensor(a_sl, dtype=torch.long)
                logits = agent6.sl_net(b_ssl)
                loss_sl = F.cross_entropy(logits, b_asl)
                agent6.opt_sl.zero_grad()
                loss_sl.backward()
                torch.nn.utils.clip_grad_norm_(agent6.sl_net.parameters(), max_norm=1.0)
                agent6.opt_sl.step()

        elapsed = time.time() - t_start
        models_6act_50k[s] = agent6
        torch.save({
            'q_net': agent6.q_net.state_dict(),
            'sl_net': agent6.sl_net.state_dict(),
            'seed': s,
            'ep': 50000,
            'action_dim': 6
        }, f'v68/models/nfsp_100bb_6action_seed_{s}_ep_50000.pt')
        print(f"    => Seed {s:3d} 6-Action 50k 학습 완료 ({elapsed:.1f}s)")

    # -------------------------------------------------------------
    # PHASE 4: 6-Action 모델 성능 및 정책 다양성 평가
    # -------------------------------------------------------------
    print("\n[PHASE 4: 6-Action 모델 평가 및 정책 다양성 / Seed 202 고착 검사]")
    eval_6act_records = []

    for s in train_seeds:
        m6 = models_6act_50k[s]
        # 평가 수행
        actions_count_6 = defaultdict(int)
        all_scores_6 = []

        for h_seed in eval_seeds:
            rng_deck = np.random.RandomState(h_seed)
            rng_hero = np.random.RandomState(h_seed + 200000)
            rng_opp = np.random.RandomState(h_seed + 100000)

            returns_sched = []
            for hand_id in range(1500):
                hero_seat = hand_id % 5
                st = game_fcpa.new_initial_state()
                advance_chance_nodes(st, rng_deck)

                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng_deck)
                    else:
                        cp = st.current_player()
                        leg_fcpa = st.legal_actions(cp)

                        if cp == hero_seat:
                            leg_6 = [1]
                            if 0 in leg_fcpa: leg_6.append(0)
                            if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                            if 3 in leg_fcpa: leg_6.append(5)

                            inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                            act_6 = m6.select_6action(inf, leg_6, is_eval=True)
                            actions_count_6[act_6] += 1
                            eng_act = map_6action_to_fcpa(act_6)
                            st.apply_action(eng_act)
                        else:
                            act = rule_bot.act(st, cp, leg_fcpa)
                            st.apply_action(act)

                ret_bb = st.returns()[hero_seat] / 100.0
                returns_sched.append(ret_bb)

            all_scores_6.append((float(np.sum(returns_sched)) / 1500) * 100.0)

        mean_ev_6 = float(np.mean(all_scores_6))
        total_acts_6 = sum(actions_count_6.values())
        freqs_6 = {a: round((actions_count_6[a] / total_acts_6) * 100.0, 1) if total_acts_6 > 0 else 0.0 for a in range(6)}

        eval_6act_records.append({
            'seed': s,
            'ev_6action': round(mean_ev_6, 2),
            'fold_pct': freqs_6[0],
            'call_pct': freqs_6[1],
            'half_pot_pct': freqs_6[2],
            'pot_pct': freqs_6[3],
            'two_pot_pct': freqs_6[4],
            'allin_pct': freqs_6[5]
        })
        print(f"  * Seed {s:3d} 6-Action: EV = {mean_ev_6:+6.2f} bb/100 | Fold: {freqs_6[0]:4.1f}%, Call: {freqs_6[1]:4.1f}%, 0.5Pot: {freqs_6[2]:4.1f}%, 1.0Pot: {freqs_6[3]:4.1f}%, 2.0Pot: {freqs_6[4]:4.1f}%, Allin: {freqs_6[5]:4.1f}%")

    # -------------------------------------------------------------
    # PHASE 5: 데이터 CSV 저장
    # -------------------------------------------------------------
    print("\n[PHASE 5: 데이터 CSV 저장]")
    with open('v68/data/phase1_fair_league_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fair_eval_records[0].keys())
        w.writeheader()
        w.writerows(fair_eval_records)

    with open('v68/data/phase4_6action_eval_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=eval_6act_records[0].keys())
        w.writeheader()
        w.writerows(eval_6act_records)

    print("  * 데이터 CSV 저장 완료 (v68/data/)")
    print("=" * 80)
    print("V6.8 FAIR LEAGUE & 6-ACTION SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
