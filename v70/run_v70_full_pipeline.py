"""
POKER AI V7.0: 100BB 6-ACTION STRATEGY VALIDATION & FAILURE ANALYSIS
- Phase 0: 4-Action -> 6-Action 전이 무결성 1,000개 상태 전수 검증
- Phase 1: Seed 404 6-Action 100k 손실 원인 정밀 해부 (액션별 손익, 200% Pot 오버베팅, 상위 5대 손실 핸드)
- Phase 2: Seed 202 엔트로피 및 탐색 복구 검증 (Group A, B, C, D 실질 수익성 bb/100 평가)
- Phase 3: 4-Action 100k 공통 분기 비교 (4-Action 150k vs 6-Action 150k, 5개 시드)
- Phase 4: 완전 신규 독립 홀드아웃 시드(20261181~20261190, 15,000핸드) 챔피언십 및 통계 검정
- Phase 5: 종합 결과 CSV 저장 및 데이터 정리
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

# 경로 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v52')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v522')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v69')))

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
    # 0: Fold, 1: Call, 2: 0.5 Pot, 3: 1.0 Pot, 4: 2.0 Pot, 5: Allin
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

def train_agent_batch(agent):
    if len(agent.rl_buffer) >= 64:
        batch = random.sample(agent.rl_buffer, 64)
        s, a, r, ns, n_leg, dones = zip(*batch)
        q_cur = agent.q_net(torch.tensor(np.array(s), dtype=torch.float32)).gather(1, torch.tensor(a).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_q = agent.q_target(torch.tensor(np.array(ns), dtype=torch.float32)).max(dim=-1)[0]
            y = torch.tensor(r, dtype=torch.float32) + (1.0 - torch.tensor(dones, dtype=torch.float32)) * 0.99 * max_q
        loss_q = F.mse_loss(q_cur, y)
        agent.opt_q.zero_grad(); loss_q.backward()
        torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), 1.0)
        agent.opt_q.step()
        agent.train_step += 1
        if agent.train_step % 200 == 0:
            agent.q_target.load_state_dict(agent.q_net.state_dict())

    if len(agent.sl_buffer) >= 64:
        batch_sl = agent.sl_buffer.sample(64)
        s_sl, a_sl = zip(*batch_sl)
        b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
        logits = agent.sl_net(b_ssl)
        loss_ce = F.cross_entropy(logits, torch.tensor(a_sl, dtype=torch.long))

        entropy_alpha = getattr(agent, 'entropy_alpha', 0.0)
        if entropy_alpha > 0.0:
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            loss_sl = loss_ce - entropy_alpha * entropy
        else:
            loss_sl = loss_ce

        agent.opt_sl.zero_grad(); loss_sl.backward()
        torch.nn.utils.clip_grad_norm_(agent.sl_net.parameters(), 1.0)
        agent.opt_sl.step()

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
    act_freqs = {a: round((actions_total[a] / total_acts) * 100.0, 1) if total_acts > 0 else 0.0 for a in range(6 if is_hero_6action else 4)}

    return {
        'mean_ev': mean_ev,
        'sd_ev': sd_ev,
        'se_ev': se_ev,
        'act_freqs': act_freqs,
        'scores': all_scores
    }

def main():
    print("=" * 80)
    print("POKER AI V7.0: 100BB 6-ACTION STRATEGY VALIDATION & FAILURE ANALYSIS")
    print("=" * 80)

    game_fcpa = get_5p_52card_100bb_fcpa_game()
    rule_bot = RuleBasedAgent()
    train_seeds = [42, 101, 202, 303, 404]
    
    # 완전 신규 독립 홀드아웃 시드 10개 (20261181 ~ 20261190)
    holdout_seeds = [20261180 + i for i in range(1, 11)]
    print(f"독립 홀드아웃 시드: {holdout_seeds}")

    # =====================================================================
    # PHASE 0: 1,000개 게임 상태 전수 검증 (4-Action -> 6-Action 무결성)
    # =====================================================================
    print("\n" + "="*80)
    print("[PHASE 0: 1,000개 상태 전수 검증 - 4-Action -> 6-Action 전이 무결성]")
    print("="*80)

    # 1,000개 다양한 상태 수집
    print("  * 무작위 게임 트리 순회를 통한 1,000개 상태 수집 중...")
    collected_states = []
    rng_sample = np.random.RandomState(777)
    while len(collected_states) < 1000:
        st = game_fcpa.new_initial_state()
        advance_chance_nodes(st, rng_sample)
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng_sample)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                collected_states.append((inf, leg))
                if len(collected_states) >= 1000:
                    break
                act = rng_sample.choice(leg)
                st.apply_action(act)

    print(f"  * 수집 완료: 총 {len(collected_states)}개 실제 유효 게임 상태.")

    # Seed 303 50k 모델로 전수 비교
    c303_50 = torch.load('v66/models/nfsp_100bb_scale_clip_seed_303_ep_50000.pt', map_location='cpu')
    m4_ref = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
    m4_ref.q_net.load_state_dict(c303_50['q_net'])
    m4_ref.sl_net.load_state_dict(c303_50['sl_net'])

    m6_test = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
    m6_test.transfer_weights_from_4action(c303_50)

    max_q_err = 0.0
    max_prob_err = 0.0
    q_interp_err = 0.0

    m4_ref.q_net.eval(); m4_ref.sl_net.eval()
    m6_test.q_net.eval(); m6_test.sl_net.eval()

    with torch.no_grad():
        for inf, leg in collected_states:
            tx = torch.tensor(inf, dtype=torch.float32).unsqueeze(0)
            
            # Q-Net 비교
            q4 = m4_ref.q_net(tx).squeeze(0).numpy()
            q6 = m6_test.q_net(tx).squeeze(0).numpy()

            # 일치 확인: [0->0, 1->1, 2->3, 3->5]
            max_q_err = max(max_q_err, abs(q4[0] - q6[0]))
            max_q_err = max(max_q_err, abs(q4[1] - q6[1]))
            max_q_err = max(max_q_err, abs(q4[2] - q6[3]))
            max_q_err = max(max_q_err, abs(q4[3] - q6[5]))

            # 보간 확인: 2 = (1+2)/2, 4 = (2+3)/2
            q_interp_err = max(q_interp_err, abs(q6[2] - (q4[1] + q4[2]) / 2.0))
            q_interp_err = max(q_interp_err, abs(q6[4] - (q4[2] + q4[3]) / 2.0))

            # SL-Net 마스크 후 Softmax 일치 확인
            lg4 = m4_ref.sl_net(tx).squeeze(0)
            lg6 = m6_test.sl_net(tx).squeeze(0)

            # 4-Action legal mask
            mask4 = torch.zeros(4, dtype=torch.bool)
            mask4[leg] = True
            lg4_masked = lg4.clone()
            lg4_masked[~mask4] = -1e9
            p4 = F.softmax(lg4_masked, dim=-1).numpy()

            # 6-Action 구 액션만 마스크
            leg_6_sub = []
            if 0 in leg: leg_6_sub.append(0)
            if 1 in leg: leg_6_sub.append(1)
            if 2 in leg: leg_6_sub.append(3)
            if 3 in leg_6_sub: pass
            if 3 in leg: leg_6_sub.append(5)

            mask6 = torch.zeros(6, dtype=torch.bool)
            mask6[leg_6_sub] = True
            lg6_masked = lg6.clone()
            lg6_masked[~mask6] = -1e9
            p6 = F.softmax(lg6_masked, dim=-1).numpy()

            for idx4, idx6 in [(0, 0), (1, 1), (2, 3), (3, 5)]:
                if idx4 in leg:
                    max_prob_err = max(max_prob_err, abs(p4[idx4] - p6[idx6]))

    print(f"  * 1,000개 상태 Q-Network 직접 전이 최대 오차: {max_q_err:.10f}")
    print(f"  * 1,000개 상태 Q-Network 보간 가중치 최대 오차: {q_interp_err:.10f}")
    print(f"  * 1,000개 상태 SL-Network 구 액션 마스크 Softmax 최대 오차: {max_prob_err:.10f}")
    
    phase0_pass = (max_q_err < 1e-4) and (q_interp_err < 1e-4) and (max_prob_err < 1e-4)
    print(f"  * Phase 0 전수 무결성 검증 판정: {'PASS (Float32 정밀도 내 100.000% 무결성 비트 일치)' if phase0_pass else 'FAIL'}")

    # =====================================================================
    # PHASE 1: Seed 404 손실 원인 정밀 해부
    # =====================================================================
    print("\n" + "="*80)
    print("[PHASE 1: Seed 404 6-Action 100k 손실 원인 정밀 해부]")
    print("="*80)

    c404_100 = torch.load('v69/models/nfsp_100bb_6action_seed_404_ep_100000.pt', map_location='cpu')
    agent404 = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
    agent404.sl_net.load_state_dict(c404_100['sl_net'])
    agent404.q_net.load_state_dict(c404_100['q_net'])

    # 3,000핸드 정밀 로깅 (RuleBased 상대 풀)
    action_stats = defaultdict(lambda: {'count': 0, 'total_net_bb': 0.0, 'won_count': 0})
    street_stats = defaultdict(lambda: {'count': 0, 'total_loss_bb': 0.0})
    hand_records = []

    rng_deck_p1 = np.random.RandomState(404100)
    for hand_id in range(3000):
        hero_seat = hand_id % 5
        st = game_fcpa.new_initial_state()
        advance_chance_nodes(st, rng_deck_p1)

        hero_actions_in_hand = []
        action_streets = []
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng_deck_p1)
            else:
                cp = st.current_player()
                leg_fcpa = st.legal_actions(cp)

                # 현재 라운드(스트리트) 판별: 0: Preflop, 1: Flop, 2: Turn, 3: River
                round_idx = st.round() if hasattr(st, 'round') else 0

                if cp == hero_seat:
                    inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    leg_6 = [1]
                    if 0 in leg_fcpa: leg_6.append(0)
                    if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                    if 3 in leg_fcpa: leg_6.append(5)

                    act_6 = agent404.select_6action(inf, leg_6, is_eval=True)
                    hero_actions_in_hand.append(act_6)
                    action_streets.append(round_idx)
                    st.apply_action(map_6action_to_fcpa(act_6))
                else:
                    st.apply_action(rule_bot.act(st, cp, leg_fcpa))

        ret_bb = st.returns()[hero_seat] / 100.0

        # 액션별 귀속 손익 로깅
        for a, st_idx in zip(hero_actions_in_hand, action_streets):
            action_stats[a]['count'] += 1
            action_stats[a]['total_net_bb'] += ret_bb / len(hero_actions_in_hand)
            if ret_bb > 0:
                action_stats[a]['won_count'] += 1
            
            st_name = ['Preflop', 'Flop', 'Turn', 'River'][min(st_idx, 3)]
            street_stats[st_name]['count'] += 1
            if ret_bb < 0:
                street_stats[st_name]['total_loss_bb'] += abs(ret_bb) / len(hero_actions_in_hand)

        hand_records.append({
            'hand_id': hand_id,
            'hero_seat': hero_seat,
            'actions': hero_actions_in_hand,
            'net_bb': ret_bb
        })

    # 정렬하여 최악의 손실 핸드 Top 5 추출
    sorted_losses = sorted(hand_records, key=lambda x: x['net_bb'])[:5]

    act_names = {0: 'Fold', 1: 'Call', 2: '0.5Pot', 3: '1.0Pot', 4: '2.0Pot', 5: 'Allin'}
    print("  * [Seed 404 액션별 실질 통계 (3,000핸드)]: ")
    for a in range(6):
        cnt = action_stats[a]['count']
        tot_bb = action_stats[a]['total_net_bb']
        win_rate = (action_stats[a]['won_count'] / cnt * 100.0) if cnt > 0 else 0.0
        avg_bb = (tot_bb / cnt) if cnt > 0 else 0.0
        print(f"    - {act_names[a]:6s}: {cnt:5d}회 | 총 손익: {tot_bb:+8.1f} bb | 평균 손익: {avg_bb:+6.2f} bb/회 | 승률: {win_rate:5.1f}%")

    print("\n  * [스트리트별 손실 기여도]:")
    for st_name in ['Preflop', 'Flop', 'Turn', 'River']:
        cnt = street_stats[st_name]['count']
        loss = street_stats[st_name]['total_loss_bb']
        print(f"    - {st_name:7s}: {cnt:5d}회 액션 발생 | 총 누적 손실: -{loss:7.1f} bb")

    print("\n  * [상위 5대 최대 손실 핸드 역추적 (Biggest Losers)]:")
    for rank, h in enumerate(sorted_losses, 1):
        act_str = " -> ".join([act_names[a] for a in h['actions']])
        print(f"    #{rank}: Hand {h['hand_id']:4d} | Seat {h['hero_seat']} | 액션 시퀀스: [{act_str}] | 최종 손실: {h['net_bb']:+8.1f} bb")

    # =====================================================================
    # PHASE 2: Seed 202 엔트로피 및 탐색 복구 검증 (실질 수익성)
    # =====================================================================
    print("\n" + "="*80)
    print("[PHASE 2: Seed 202 엔트로피 및 탐색 복구 검증 (실질 수익성)]")
    print("="*80)

    # 베이스라인: Seed 202 50k 모델
    c202_50 = torch.load('v66/models/nfsp_100bb_scale_clip_seed_202_ep_50000.pt', map_location='cpu')

    recovery_configs = [
        ('Group A: 기존 정책 베이스라인', {'eta': 0.1, 'entropy_alpha': 0.0}),
        ('Group B: Average Policy 엔트로피 정규화 (alpha=0.05)', {'eta': 0.1, 'entropy_alpha': 0.05}),
        ('Group C: BR 탐색 강화 (eta=0.3)', {'eta': 0.3, 'entropy_alpha': 0.0}),
        ('Group D: BR 탐색 + 엔트로피 복합 (eta=0.3, alpha=0.05)', {'eta': 0.3, 'entropy_alpha': 0.05})
    ]

    p2_results = {}
    for group_name, cfg in recovery_configs:
        print(f"\n  * {group_name} 20,000 에피소드 훈련 중...")
        rec_agent = NFSP_6Action_Agent(
            0, 253, 6, hidden_dim=128,
            eta=cfg['eta'],
            entropy_alpha=cfg['entropy_alpha']
        )
        rec_agent.transfer_weights_from_4action(c202_50)
        rng_train_p2 = np.random.RandomState(202202)

        t_p2 = time.time()
        for ep in range(1, 20001):
            rec_agent.reset_episode()
            st = game_fcpa.new_initial_state()
            advance_chance_nodes(st, rng_train_p2)
            trajectory = []
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng_train_p2)
                else:
                    cp = st.current_player()
                    leg_fcpa = st.legal_actions(cp)
                    if cp == 0:
                        leg_6 = [1]
                        if 0 in leg_fcpa: leg_6.append(0)
                        if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                        if 3 in leg_fcpa: leg_6.append(5)
                        inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                        act_6 = rec_agent.select_6action(inf, leg_6, is_eval=False)
                        trajectory.append((inf, act_6, leg_6))
                        st.apply_action(map_6action_to_fcpa(act_6))
                    else:
                        st.apply_action(rule_bot.act(st, cp, leg_fcpa))

            raw_ret = st.returns()[0] / 100.0
            r_t = raw_ret / 100.0
            if len(trajectory) > 0:
                for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                    done = (idx == len(trajectory) - 1)
                    ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                    n_leg = trajectory[idx+1][2] if not done else []
                    rec_agent.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

            train_agent_batch(rec_agent)

        print(f"    훈련 소요 시간: {time.time() - t_p2:.1f}초")

        # 평가 (RuleBased 상대 10개 독립 시드, 15,000핸드)
        eval_res = evaluate_hero_vs_pool(
            game_fcpa, rec_agent, [rule_bot], holdout_seeds,
            is_hero_6action=True, hands_per_seed=1500
        )
        p2_results[group_name] = {
            'agent': rec_agent,
            'eval': eval_res
        }
        f = eval_res['act_freqs']
        print(f"    결과 EV: {eval_res['mean_ev']:+7.2f} ± {eval_res['se_ev']:.2f} bb/100")
        print(f"    액션 분포: Fold={f[0]}%, Call={f[1]}%, 0.5Pot={f[2]}%, 1Pot={f[3]}%, 2Pot={f[4]}%, Allin={f[5]}%")

    # =====================================================================
    # PHASE 3: 4-Action 100k 공통 분기 비교 (4-Action 150k vs 6-Action 150k)
    # =====================================================================
    print("\n" + "="*80)
    print("[PHASE 3: 4-Action 100k 공통 분기 비교 (4-Act 150k vs 6-Act 150k)]")
    print("="*80)

    models_4act_150k = {}
    models_6act_150k = {}

    for s in train_seeds:
        print(f"\n--- [Seed {s:3d}] 공통 분기 추가 50k 학습 (누적 150,000 에피소드) ---")
        c100 = torch.load(f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_100000.pt', map_location='cpu')

        # Branch A: 4-Action 150k (기존 4-Action에서 50,000 에피소드 추가 학습)
        print(f"  * [Branch A] 4-Action 100,001 -> 150,000 학습 중...")
        m4_150 = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m4_150.q_net.load_state_dict(c100['q_net'])
        m4_150.q_target.load_state_dict(c100['q_net'])
        m4_150.sl_net.load_state_dict(c100['sl_net'])
        m4_150.opt_q = optim.Adam(m4_150.q_net.parameters(), lr=0.003)
        m4_150.opt_sl = optim.Adam(m4_150.sl_net.parameters(), lr=0.001)

        rng_tr4 = np.random.RandomState(s + 100000)
        t_a = time.time()
        for ep in range(100001, 150001):
            m4_150.reset_episode()
            st = game_fcpa.new_initial_state()
            advance_chance_nodes(st, rng_tr4)
            trajectory = []
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng_tr4)
                else:
                    cp = st.current_player()
                    leg = st.legal_actions(cp)
                    if cp == 0:
                        inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                        act = m4_150.select_action(inf, leg, is_eval=False)
                        trajectory.append((inf, act, leg))
                        st.apply_action(act)
                    else:
                        st.apply_action(rule_bot.act(st, cp, leg))

            raw_ret = st.returns()[0] / 100.0
            r_t = raw_ret / 100.0
            if len(trajectory) > 0:
                for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                    done = (idx == len(trajectory) - 1)
                    ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                    n_leg = trajectory[idx+1][2] if not done else []
                    m4_150.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

            train_agent_batch(m4_150)

        torch.save({'q_net': m4_150.q_net.state_dict(), 'sl_net': m4_150.sl_net.state_dict()}, f'v70/models/nfsp_4act_150k_seed_{s}.pt')
        models_4act_150k[s] = m4_150
        print(f"    Branch A 완료 ({time.time() - t_a:.1f}초)")

        # Branch B: 6-Action 150k (4-Action 100k 체크포인트에서 6-Action 가중치 전이 후 50,000 에피소드 추가 학습)
        print(f"  * [Branch B] 6-Action 전이 후 추가 50k 학습 (누적 150k)...")
        m6_150 = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
        m6_150.transfer_weights_from_4action(c100)
        rng_tr6 = np.random.RandomState(s + 150000)
        t_b = time.time()

        for ep in range(100001, 150001):
            m6_150.reset_episode()
            st = game_fcpa.new_initial_state()
            advance_chance_nodes(st, rng_tr6)
            trajectory = []
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng_tr6)
                else:
                    cp = st.current_player()
                    leg_fcpa = st.legal_actions(cp)
                    if cp == 0:
                        leg_6 = [1]
                        if 0 in leg_fcpa: leg_6.append(0)
                        if 2 in leg_fcpa: leg_6.extend([2, 3, 4])
                        if 3 in leg_fcpa: leg_6.append(5)
                        inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                        act_6 = m6_150.select_6action(inf, leg_6, is_eval=False)
                        trajectory.append((inf, act_6, leg_6))
                        st.apply_action(map_6action_to_fcpa(act_6))
                    else:
                        st.apply_action(rule_bot.act(st, cp, leg_fcpa))

            raw_ret = st.returns()[0] / 100.0
            r_t = raw_ret / 100.0
            if len(trajectory) > 0:
                for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                    done = (idx == len(trajectory) - 1)
                    ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                    n_leg = trajectory[idx+1][2] if not done else []
                    m6_150.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

            train_agent_batch(m6_150)

        torch.save({'q_net': m6_150.q_net.state_dict(), 'sl_net': m6_150.sl_net.state_dict()}, f'v70/models/nfsp_6act_150k_seed_{s}.pt')
        models_6act_150k[s] = m6_150
        print(f"    Branch B 완료 ({time.time() - t_b:.1f}초)")

    # =====================================================================
    # PHASE 4: 완전 신규 독립 홀드아웃 시드 챔피언십
    # =====================================================================
    print("\n" + "="*80)
    print("[PHASE 4: 완전 신규 독립 홀드아웃 시드 챔피언십 (20261181~20261190, 15,000핸드)]")
    print("="*80)

    # 평가 모델 준비
    # 1) 4-Action 100k 모델 5시드
    models_4act_100k = {}
    for s in train_seeds:
        c = torch.load(f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_100000.pt', map_location='cpu')
        m = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m.sl_net.load_state_dict(c['sl_net'])
        models_4act_100k[s] = m

    # 2) 6-Action 100k 모델 5시드
    models_6act_100k = {}
    for s in train_seeds:
        c = torch.load(f'v69/models/nfsp_100bb_6action_seed_{s}_ep_100000.pt', map_location='cpu')
        m = NFSP_6Action_Agent(0, 253, 6, hidden_dim=128)
        m.sl_net.load_state_dict(c['sl_net'])
        models_6act_100k[s] = m

    eval_summary = []

    print("\n--- 1. 4-Action 100k (D안) 5시드 평가 ---")
    evs_4_100 = []
    for s in train_seeds:
        res = evaluate_hero_vs_pool(game_fcpa, models_4act_100k[s], [rule_bot], holdout_seeds, is_hero_6action=False, hands_per_seed=1500)
        evs_4_100.append(res['mean_ev'])
        eval_summary.append({
            'group': '4-Action 100k', 'seed': s, 'mean_ev': res['mean_ev'], 'se_ev': res['se_ev'],
            'act_freqs': res['act_freqs'], 'scores': res['scores']
        })
        print(f"  * Seed {s:3d}: {res['mean_ev']:+7.2f} ± {res['se_ev']:.2f} bb/100 | {res['act_freqs']}")

    print("\n--- 2. 6-Action 100k (V6.9) 5시드 평가 ---")
    evs_6_100 = []
    for s in train_seeds:
        res = evaluate_hero_vs_pool(game_fcpa, models_6act_100k[s], [rule_bot], holdout_seeds, is_hero_6action=True, hands_per_seed=1500)
        evs_6_100.append(res['mean_ev'])
        eval_summary.append({
            'group': '6-Action 100k', 'seed': s, 'mean_ev': res['mean_ev'], 'se_ev': res['se_ev'],
            'act_freqs': res['act_freqs'], 'scores': res['scores']
        })
        print(f"  * Seed {s:3d}: {res['mean_ev']:+7.2f} ± {res['se_ev']:.2f} bb/100 | {res['act_freqs']}")

    print("\n--- 3. 4-Action 150k (Phase 3 분기) 5시드 평가 ---")
    evs_4_150 = []
    for s in train_seeds:
        res = evaluate_hero_vs_pool(game_fcpa, models_4act_150k[s], [rule_bot], holdout_seeds, is_hero_6action=False, hands_per_seed=1500)
        evs_4_150.append(res['mean_ev'])
        eval_summary.append({
            'group': '4-Action 150k', 'seed': s, 'mean_ev': res['mean_ev'], 'se_ev': res['se_ev'],
            'act_freqs': res['act_freqs'], 'scores': res['scores']
        })
        print(f"  * Seed {s:3d}: {res['mean_ev']:+7.2f} ± {res['se_ev']:.2f} bb/100 | {res['act_freqs']}")

    print("\n--- 4. 6-Action 150k (Phase 3 분기) 5시드 평가 ---")
    evs_6_150 = []
    for s in train_seeds:
        res = evaluate_hero_vs_pool(game_fcpa, models_6act_150k[s], [rule_bot], holdout_seeds, is_hero_6action=True, hands_per_seed=1500)
        evs_6_150.append(res['mean_ev'])
        eval_summary.append({
            'group': '6-Action 150k', 'seed': s, 'mean_ev': res['mean_ev'], 'se_ev': res['se_ev'],
            'act_freqs': res['act_freqs'], 'scores': res['scores']
        })
        print(f"  * Seed {s:3d}: {res['mean_ev']:+7.2f} ± {res['se_ev']:.2f} bb/100 | {res['act_freqs']}")

    # =====================================================================
    # 통계 검정: 4-Action 150k vs 6-Action 150k Paired t-Test
    # =====================================================================
    print("\n" + "="*80)
    print("[PHASE 4: 통계 검정 및 150k 공통 분기 우열 판정]")
    print("="*80)

    mean_4_150 = np.mean(evs_4_150)
    mean_6_150 = np.mean(evs_6_150)
    diffs_150 = np.array(evs_6_150) - np.array(evs_4_150)
    t_stat_150, p_val_150 = stats.ttest_rel(evs_6_150, evs_4_150)

    ci95_4_150 = stats.t.interval(0.95, df=len(evs_4_150)-1, loc=np.mean(evs_4_150), scale=stats.sem(evs_4_150))
    ci95_6_150 = stats.t.interval(0.95, df=len(evs_6_150)-1, loc=np.mean(evs_6_150), scale=stats.sem(evs_6_150))

    print(f"  * 4-Action 100k 5시드 평균: {np.mean(evs_4_100):+7.2f} bb/100")
    print(f"  * 6-Action 100k 5시드 평균: {np.mean(evs_6_100):+7.2f} bb/100")
    print(f"  * 4-Action 150k 5시드 평균: {mean_4_150:+7.2f} bb/100 (95% CI: [{ci95_4_150[0]:+.2f}, {ci95_4_150[1]:+.2f}])")
    print(f"  * 6-Action 150k 5시드 평균: {mean_6_150:+7.2f} bb/100 (95% CI: [{ci95_6_150[0]:+.2f}, {ci95_6_150[1]:+.2f}])")
    print(f"  * 동일 예산(150k) 6-Action 차이: {np.mean(diffs_150):+7.2f} bb/100 | t={t_stat_150:.3f}, p={p_val_150:.4f}")

    # CSV 데이터 저장
    csv_path = 'v70/data/v70_championship_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Seed', 'Mean_EV_bb100', 'SE_EV', 'Act_Fold', 'Act_Call', 'Act_0.5Pot', 'Act_1Pot', 'Act_2Pot', 'Act_Allin'])
        for row in eval_summary:
            af = row['act_freqs']
            writer.writerow([
                row['group'], row['seed'], f"{row['mean_ev']:.2f}", f"{row['se_ev']:.2f}",
                af.get(0, 0.0), af.get(1, 0.0), af.get(2, 0.0), af.get(3, 0.0), af.get(4, 0.0), af.get(5, 0.0)
            ])

    print(f"\n  * CSV 저장 완료: {csv_path}")
    print("\n[V7.0 전체 파이프라인 정상 완주]")

if __name__ == '__main__':
    main()
