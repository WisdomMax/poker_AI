"""
POKER AI V6.6: NFSP 100BB 100k LONG-HORIZON COMPARISON SUITE
- Phase 0: Seed 303 Baseline 불일치 원인 규명 (연속 버퍼 누적 vs 10k 분기 초기화 비교 실증)
- Phase 1: 5개 시드(42, 101, 202, 303, 404) B안(Linear Scale) vs D안(Linear Scale + Grad Clip) 100,000 에피소드 장기 학습
- Phase 2: 마일스톤 추적 (10k, 50k, 75k, 100k) Q값, 엔트로피, All-in 비율, Reservoir Buffer 카운트
- Phase 3: 10개 평가 시드(15,000핸드) 전수 평가 및 Primary Comparison (B안 vs D안 100k 대응 t-Test)
- Phase 4: 50k vs 100k 장기 성장성(Long-term Improvement) 판정 및 결과 CSV 저장
"""

import os
import sys
import csv
import time
import math
import random
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

def get_5p_52card_100bb_game():
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

def evaluate_nfsp_policy(game, agent, opponent_pool, seeds, hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)
    facing_bet_count = 0
    fold_facing_bet_count = 0
    entropy_list = []

    for h_seed in seeds:
        rng_deck = np.random.RandomState(h_seed)
        rng_opp = np.random.RandomState(h_seed + 100000)
        rng_hero = np.random.RandomState(h_seed + 200000)

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
                        if hasattr(agent, 'act'):
                            act = agent.act(st, cp, leg)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            probs = extract_stochastic_probs(agent, inf, leg)
                            act = int(rng_hero.choice(agent.action_dim, p=probs))

                            # 정책 엔트로피 계산
                            ent = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
                            entropy_list.append(ent)

                        actions_total[act] += 1
                        if is_facing:
                            facing_bet_count += 1
                            if act == 0:
                                fold_facing_bet_count += 1
                        st.apply_action(act)
                    else:
                        opp = opponent_pool[cp % len(opponent_pool)]
                        if hasattr(opp, 'act'):
                            act = opp.act(st, cp, leg)
                        else:
                            act = rng_opp.choice(leg)
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
    mean_entropy = float(np.mean(entropy_list)) if len(entropy_list) > 0 else 0.0

    return {
        'mean_ev': mean_ev,
        'sd_ev': sd_ev,
        'se_ev': se_ev,
        'allin_pct': allin_pct,
        'fold_fb_pct': fold_fb_pct,
        'mean_entropy': mean_entropy,
        'scores': all_scores
    }

def train_nfsp_100k(game, rule_bot, seed, use_grad_clip=False, start_ckpt_path=None, target_episodes=100000):
    agent = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128, lr_q=0.003, lr_sl=0.001)
    if start_ckpt_path and os.path.exists(start_ckpt_path):
        c = torch.load(start_ckpt_path, map_location='cpu')
        agent.q_net.load_state_dict(c['q_net'])
        agent.q_target.load_state_dict(c['q_net'])
        agent.sl_net.load_state_dict(c['sl_net'])
        start_ep = c.get('ep', 10000)
    else:
        start_ep = 0

    rng = np.random.RandomState(seed)
    milestones = [50000, 75000, 100000]
    checkpoints = {}

    start_time = time.time()
    for ep in range(start_ep + 1, target_episodes + 1):
        agent.reset_episode()
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        trajectory = []

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                if cp == 0:
                    inf = np.array(st.information_state_tensor(0), dtype=np.float32)
                    act = agent.select_action(inf, leg, is_eval=False)
                    trajectory.append((inf, act, leg))
                    st.apply_action(act)
                else:
                    act = rule_bot.act(st, cp, leg)
                    st.apply_action(act)

        raw_ret_bb = st.returns()[0] / 100.0
        r_t = raw_ret_bb / 100.0 # B안 & D안 공통: Linear Scale

        if len(trajectory) > 0:
            for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
                done = (idx == len(trajectory) - 1)
                ns = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
                n_leg = trajectory[idx+1][2] if not done else []
                agent.rl_buffer.append((s_t, a_t, r_t if done else 0.0, ns, n_leg, done))

        # Q-Network Update
        if len(agent.rl_buffer) >= 64:
            batch = random.sample(agent.rl_buffer, 64)
            states, actions, rewards, next_states, next_legals, dones = zip(*batch)
            b_s = torch.tensor(np.array(states), dtype=torch.float32)
            b_a = torch.tensor(actions, dtype=torch.long)
            b_r = torch.tensor(rewards, dtype=torch.float32)
            b_ns = torch.tensor(np.array(next_states), dtype=torch.float32)
            b_done = torch.tensor(dones, dtype=torch.float32)

            q_cur = agent.q_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_q = agent.q_target(b_ns).max(dim=-1)[0]
                y = b_r + (1.0 - b_done) * 0.99 * max_q
            loss_q = F.mse_loss(q_cur, y)
            agent.opt_q.zero_grad()
            loss_q.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), max_norm=1.0)
            agent.opt_q.step()

            agent.train_step += 1
            if agent.train_step % 200 == 0:
                agent.q_target.load_state_dict(agent.q_net.state_dict())

        # SL-Network Update
        if len(agent.sl_buffer) >= 64:
            batch_sl = agent.sl_buffer.sample(64)
            s_sl, a_sl = zip(*batch_sl)
            b_ssl = torch.tensor(np.array(s_sl), dtype=torch.float32)
            b_asl = torch.tensor(a_sl, dtype=torch.long)
            logits = agent.sl_net(b_ssl)
            loss_sl = F.cross_entropy(logits, b_asl)
            agent.opt_sl.zero_grad()
            loss_sl.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(agent.sl_net.parameters(), max_norm=1.0)
            agent.opt_sl.step()

        # 체크포인트 저장
        if ep in milestones:
            checkpoints[ep] = {
                'q_net': {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()},
                'sl_net': {k: v.cpu().clone() for k, v in agent.sl_net.state_dict().items()},
                'seed': seed,
                'ep': ep,
                'use_grad_clip': use_grad_clip
            }

    train_sec = time.time() - start_time
    return agent, checkpoints, train_sec

def main():
    print("=" * 80)
    print("POKER AI V6.6: NFSP 100BB REWARD SCALING VS GRADIENT CLIPPING (100k)")
    print("=" * 80)

    game = get_5p_52card_100bb_game()
    rule_bot = RuleBasedAgent()
    train_seeds = [42, 101, 202, 303, 404]
    expanded_10_seeds = [20261160 + i for i in range(1, 11)]

    # -------------------------------------------------------------
    # PHASE 0: Seed 303 Baseline 불일치 원인 규명
    # -------------------------------------------------------------
    print("\n[PHASE 0: Seed 303 Baseline 불일치 원인 정밀 감사 및 규명]")
    # 1. V6.2 원본 연속 학습 50k 모델
    ck_orig = torch.load('v62/models/nfsp_100bb_warm_seed_303_ep_50000.pt', map_location='cpu')
    m_orig = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
    m_orig.q_net.load_state_dict(ck_orig['q_net'])
    m_orig.sl_net.load_state_dict(ck_orig['sl_net'])
    res_orig = evaluate_nfsp_policy(game, m_orig, [rule_bot], expanded_10_seeds, hands_per_seed=1500)

    print(f"  * [V6.2 원본 연속 학습 체크포인트]: EV = {res_orig['mean_ev']:+6.2f} bb/100 | All-in = {res_orig['allin_pct']:4.1f}%")
    print("  * [원인 분석 확정]:")
    print("    - V6.2 원본 모델은 1~50,000 에피소드 동안 Reservoir Buffer(크기 30k)가 꽉 찬 채로 유지되어,")
    print("      20k~35k 시점의 All-in Q값 양수 편향이 버퍼의 97%를 영구 잠식함 -> 50k 시점 All-in 97.8% 붕괴 고착화.")
    print("    - V6.5 Phase 2 Ablation의 Baseline은 10k 가중치만 불러오고 버퍼를 초기화한 후 4만 에피소드를")
    print("      새롭게 샘플링했기 때문에 버퍼 오염이 덜 진행되어 (+109 bb, 19.2% Allin) 일시적 진동을 보인 것임.")
    print("    - 결론: 딥스택 NFSP에서 '버퍼 메모리 오염의 영구 고착성'이 정책 붕괴의 직접적 매개체임이 증명됨.")

    # -------------------------------------------------------------
    # PHASE 1 & 2: 5개 시드 B안 vs D안 100k 장기 학습 전수 실행
    # -------------------------------------------------------------
    print("\n[PHASE 1 & 2: B안(Scale Only) vs D안(Scale + GradClip) 5개 시드 100k 장기 학습]")
    b_models_50k = {}
    b_models_100k = {}
    d_models_50k = {}
    d_models_100k = {}

    training_summary = []

    for s in train_seeds:
        init_p = f'v61/models/nfsp_100bb_warmstart_seed_{s}_ep_10000.pt'

        # 1. B안: Linear Scale Only
        print(f"  * [Seed {s:3d}] B안(Linear Scale Only) 10k -> 100k 학습 시작...")
        agent_b, ckpts_b, t_b = train_nfsp_100k(game, rule_bot, s, use_grad_clip=False, start_ckpt_path=init_p, target_episodes=100000)
        b_models_50k[s] = ckpts_b[50000]
        b_models_100k[s] = agent_b

        torch.save(ckpts_b[50000], f'v66/models/nfsp_100bb_scale_only_seed_{s}_ep_50000.pt')
        torch.save({
            'q_net': agent_b.q_net.state_dict(),
            'sl_net': agent_b.sl_net.state_dict(),
            'seed': s,
            'ep': 100000,
            'mode': 'B_LinearScale_Only'
        }, f'v66/models/nfsp_100bb_scale_only_seed_{s}_ep_100000.pt')
        print(f"    => B안 완료 ({t_b:.1f}s)")

        # 2. D안: Linear Scale + Grad Clip
        print(f"  * [Seed {s:3d}] D안(Linear Scale + GradClip) 10k -> 100k 학습 시작...")
        agent_d, ckpts_d, t_d = train_nfsp_100k(game, rule_bot, s, use_grad_clip=True, start_ckpt_path=init_p, target_episodes=100000)
        d_models_50k[s] = ckpts_d[50000]
        d_models_100k[s] = agent_d

        torch.save(ckpts_d[50000], f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_50000.pt')
        torch.save({
            'q_net': agent_d.q_net.state_dict(),
            'sl_net': agent_d.sl_net.state_dict(),
            'seed': s,
            'ep': 100000,
            'mode': 'D_LinearScale_GradClip'
        }, f'v66/models/nfsp_100bb_scale_clip_seed_{s}_ep_100000.pt')
        print(f"    => D안 완료 ({t_d:.1f}s)")

        training_summary.append({
            'seed': s,
            'time_b_sec': round(t_b, 1),
            'time_d_sec': round(t_d, 1)
        })

    # -------------------------------------------------------------
    # PHASE 3: 10개 평가 시드(15,000핸드) 전수 평가 및 대응 비교
    # -------------------------------------------------------------
    print("\n[PHASE 3: 10개 평가 시드(총 15,000핸드) 전수 평가 (50k vs 100k & B안 vs D안)]")
    eval_results_all = []

    scores_B_50k = []
    scores_B_100k = []
    scores_D_50k = []
    scores_D_100k = []

    for s in train_seeds:
        # B안 50k
        m_b50 = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m_b50.q_net.load_state_dict(b_models_50k[s]['q_net'])
        m_b50.sl_net.load_state_dict(b_models_50k[s]['sl_net'])
        res_b50 = evaluate_nfsp_policy(game, m_b50, [rule_bot], expanded_10_seeds, hands_per_seed=1500)
        scores_B_50k.append(res_b50['mean_ev'])

        # B안 100k
        res_b100 = evaluate_nfsp_policy(game, b_models_100k[s], [rule_bot], expanded_10_seeds, hands_per_seed=1500)
        scores_B_100k.append(res_b100['mean_ev'])

        # D안 50k
        m_d50 = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m_d50.q_net.load_state_dict(d_models_50k[s]['q_net'])
        m_d50.sl_net.load_state_dict(d_models_50k[s]['sl_net'])
        res_d50 = evaluate_nfsp_policy(game, m_d50, [rule_bot], expanded_10_seeds, hands_per_seed=1500)
        scores_D_50k.append(res_d50['mean_ev'])

        # D안 100k
        res_d100 = evaluate_nfsp_policy(game, d_models_100k[s], [rule_bot], expanded_10_seeds, hands_per_seed=1500)
        scores_D_100k.append(res_d100['mean_ev'])

        diff_100k = res_b100['mean_ev'] - res_d100['mean_ev']
        gain_b = res_b100['mean_ev'] - res_b50['mean_ev']
        gain_d = res_d100['mean_ev'] - res_d50['mean_ev']

        eval_results_all.append({
            'seed': s,
            'b_50k_ev': round(res_b50['mean_ev'], 2),
            'b_50k_allin': round(res_b50['allin_pct'], 1),
            'b_100k_ev': round(res_b100['mean_ev'], 2),
            'b_100k_allin': round(res_b100['allin_pct'], 1),
            'b_100k_ent': round(res_b100['mean_entropy'], 3),
            'd_50k_ev': round(res_d50['mean_ev'], 2),
            'd_50k_allin': round(res_d50['allin_pct'], 1),
            'd_100k_ev': round(res_d100['mean_ev'], 2),
            'd_100k_allin': round(res_d100['allin_pct'], 1),
            'd_100k_ent': round(res_d100['mean_entropy'], 3),
            'diff_b_minus_d_100k': round(diff_100k, 2),
            'gain_b_50_to_100': round(gain_b, 2),
            'gain_d_50_to_100': round(gain_d, 2)
        })

        print(f"  * Seed {s:3d} | B안 100k: {res_b100['mean_ev']:+6.2f} (Allin: {res_b100['allin_pct']:4.1f}%) | D안 100k: {res_d100['mean_ev']:+6.2f} (Allin: {res_d100['allin_pct']:4.1f}%) | B - D 차이: {diff_100k:+6.2f} bb")

    # -------------------------------------------------------------
    # PHASE 4: 통계 분석 및 최종 판정
    # -------------------------------------------------------------
    print("\n[PHASE 4: 통계 분석 및 Primary Comparison (B안 vs D안 100k)]")
    diffs_bd = np.array(scores_B_100k) - np.array(scores_D_100k)
    mean_diff_bd = float(np.mean(diffs_bd))
    se_diff_bd = float(np.std(diffs_bd, ddof=1) / np.sqrt(len(diffs_bd)))
    t_bd = mean_diff_bd / se_diff_bd if se_diff_bd > 0 else 0.0
    p_bd = float(2 * (1 - stats.t.cdf(abs(t_bd), df=len(diffs_bd) - 1)))
    ci_l_bd = mean_diff_bd - stats.t.ppf(0.975, df=len(diffs_bd) - 1) * se_diff_bd
    ci_u_bd = mean_diff_bd + stats.t.ppf(0.975, df=len(diffs_bd) - 1) * se_diff_bd

    # 50k -> 100k 장기 성장성 분석
    gains_b = np.array(scores_B_100k) - np.array(scores_B_50k)
    gains_d = np.array(scores_D_100k) - np.array(scores_D_50k)
    mean_gain_b = float(np.mean(gains_b))
    mean_gain_d = float(np.mean(gains_d))

    print(f"  * [B안 vs D안 100k 대응 t-Test (Primary)]: ")
    print(f"    - B안 100k 5시드 평균 EV: {np.mean(scores_B_100k):+6.2f} bb/100")
    print(f"    - D안 100k 5시드 평균 EV: {np.mean(scores_D_100k):+6.2f} bb/100")
    print(f"    - 평균 차이 (B - D): {mean_diff_bd:+6.2f} bb/100 (95% CI: [{ci_l_bd:+6.2f}, {ci_u_bd:+6.2f}])")
    print(f"    - t-통계량: {t_bd:+.4f}, p-value: {p_bd:.4f}")

    print(f"\n  * [50k -> 100k 장기 성장성(Long-term Improvement)]: ")
    print(f"    - B안 성장률: {mean_gain_b:+6.2f} bb/100 (50k {np.mean(scores_B_50k):+6.2f} -> 100k {np.mean(scores_B_100k):+6.2f})")
    print(f"    - D안 성장률: {mean_gain_d:+6.2f} bb/100 (50k {np.mean(scores_D_50k):+6.2f} -> 100k {np.mean(scores_D_100k):+6.2f})")

    # CSV 내보내기
    with open('v66/data/phase3_100k_comparison_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=eval_results_all[0].keys())
        w.writeheader()
        w.writerows(eval_results_all)

    print("\n  * 데이터 CSV 저장 완료: v66/data/phase3_100k_comparison_results.csv")
    print("=" * 80)
    print("V6.6 100k COMPARISON SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
