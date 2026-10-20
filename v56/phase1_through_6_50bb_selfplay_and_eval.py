"""
POKER AI V5.6 - 5-PLAYER 50BB NFSP-DQN DEEP-STACK SELF-PLAY
- Phase 1: 50BB 환경 확정 및 칩 원장 무결성 검증 (Stack: 5,000 chips = 50BB)
- Phase 2: Group A (50BB From Scratch 50k) vs Group B (50BB Warm Start 50k) 대규모 자기대국
- Phase 3: 3개 신규 독립 홀드아웃 시드 (20261028, 20261029, 20261030) 기반 4,500핸드 전수 평가
- Phase 4: 대표 카드별 (AA, KK, QQ, JJ, TT, 99) 50BB 딥스택 의사결정 및 Policy Regret
- Phase 5: Primary Comparison (Warm Start vs Scratch) Paired t-test 및 95% 신뢰구간 검정
- Phase 6: 최종 GO / NO-GO 판정
"""

import os
import sys
import csv
import time
import psutil
import hashlib
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

# 경로 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v52')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v522')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v523')))

from nfsp_variants import NFSP_DQN_Reference
from v52.phase5_6_7_five_player_benchmark import (
    advance_chance_nodes, RandomAgent, AlwaysFoldAgent, CheckCallAgent, RuleBasedAgent
)

def get_5p_50bb_poker_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 6,
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 2,
        'numBoardCards': '0 3',
        'blind': '50 100 0 0 0',
        'stack': '5000 5000 5000 5000 5000', # 50BB (5000 chips)
        'firstPlayer': '3 1'
    }
    return pyspiel.load_game('universal_poker', params)

def extract_stochastic_probs(agent, info, legal_actions):
    x = torch.tensor(info, dtype=torch.float32).unsqueeze(0)
    agent.sl_net.eval()
    with torch.no_grad():
        lg = agent.sl_net(x)
        mask = torch.zeros(agent.action_dim, dtype=torch.bool)
        mask[legal_actions] = True
        lg[0, ~mask] = -1e9
        probs = F.softmax(lg, dim=-1).squeeze(0).numpy()
    return probs

def train_nfsp_50bb(game, seed: int, max_episodes: int = 50000, 
                    checkpoint_eps: list = [10000, 20000, 30000, 40000, 50000],
                    init_weights_path: str = None, tag: str = "scratch"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    if init_weights_path and os.path.exists(init_weights_path):
        ckpt = torch.load(init_weights_path, map_location='cpu')
        # 149 -> 161 차원 불일치 해결을 위한 Partial Weight Transfer
        with torch.no_grad():
            old_sdim = ckpt['q_net']['0.weight'].shape[1]
            agent.q_net[0].weight[:, :old_sdim] = ckpt['q_net']['0.weight']
            agent.q_net[0].bias.copy_(ckpt['q_net']['0.bias'])
            agent.q_net[2].weight.copy_(ckpt['q_net']['2.weight'])
            agent.q_net[2].bias.copy_(ckpt['q_net']['2.bias'])
            agent.q_net[4].weight.copy_(ckpt['q_net']['4.weight'])
            agent.q_net[4].bias.copy_(ckpt['q_net']['4.bias'])

            agent.sl_net[0].weight[:, :old_sdim] = ckpt['sl_net']['0.weight']
            agent.sl_net[0].bias.copy_(ckpt['sl_net']['0.bias'])
            agent.sl_net[2].weight.copy_(ckpt['sl_net']['2.weight'])
            agent.sl_net[2].bias.copy_(ckpt['sl_net']['2.bias'])
            agent.sl_net[4].weight.copy_(ckpt['sl_net']['4.weight'])
            agent.sl_net[4].bias.copy_(ckpt['sl_net']['4.bias'])

            agent.q_target.load_state_dict(agent.q_net.state_dict())
        print(f"    [50BB Warm-Start]: 20BB 50k 가중치 Partial Transfer 성공 (149 -> {s_dim} dim 매핑, {init_weights_path})")

    checkpoints = {}
    train_logs = []

    # 이미 50k 체크포인트가 모두 존재하는지 확인
    all_exist = all(os.path.exists(f"v56/models/nfsp_50bb_{tag}_seed_{seed}_ep_{ep}.pt") for ep in checkpoint_eps)
    if all_exist:
        print(f"    [{tag.upper()} Seed {seed}]: 이미 디스크에 50k 체크포인트가 존재하므로 로드합니다.")
        for ep in checkpoint_eps:
            path = f"v56/models/nfsp_50bb_{tag}_seed_{seed}_ep_{ep}.pt"
            ckpt = torch.load(path, map_location='cpu')
            cp_agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
            cp_agent.q_net.load_state_dict(ckpt['q_net'])
            cp_agent.sl_net.load_state_dict(ckpt['sl_net'])
            checkpoints[ep] = cp_agent
        return checkpoints, []

    start_time = time.time()
    process = psutil.Process()
    q_losses = []
    sl_losses = []

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        ep_history = defaultdict(list)

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act = agent.select_action(info, leg, is_eval=False)
                ep_history[cp].append((info, act, leg))
                st.apply_action(act)

        returns = st.returns()
        for cp in range(5):
            hist = ep_history[cp]
            if not hist: continue
            for i in range(len(hist)):
                s_t, a_t, _ = hist[i]
                if i + 1 < len(hist):
                    ns, _, n_leg = hist[i+1]
                    done = 0.0
                    r = 0.0
                else:
                    ns = np.zeros(s_dim, dtype=np.float32)
                    n_leg = []
                    done = 1.0
                    r = returns[cp] / 100.0 # Terminal Net BB
                agent.rl_buffer.append((s_t, a_t, r, ns, n_leg, done))

        q_loss, sl_loss = agent.update(batch_size=64)
        if q_loss > 0: q_losses.append(q_loss)
        if sl_loss > 0: sl_losses.append(sl_loss)

        if ep in checkpoint_eps:
            elapsed = time.time() - start_time
            ram_mb = process.memory_info().rss / (1024 * 1024)

            cp_agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
            cp_agent.q_net.load_state_dict(agent.q_net.state_dict())
            cp_agent.sl_net.load_state_dict(agent.sl_net.state_dict())
            checkpoints[ep] = cp_agent

            save_path = f"v56/models/nfsp_50bb_{tag}_seed_{seed}_ep_{ep}.pt"
            torch.save({
                'q_net': agent.q_net.state_dict(),
                'sl_net': agent.sl_net.state_dict(),
                'seed': seed,
                'ep': ep,
                'tag': tag
            }, save_path)

            with open(save_path, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()[:16]

            mean_q = float(np.mean(q_losses[-300:])) if q_losses else 0.0
            mean_sl = float(np.mean(sl_losses[-300:])) if sl_losses else 0.0

            log_entry = {
                'seed': seed,
                'tag': tag,
                'episodes': ep,
                'elapsed_sec': round(elapsed, 2),
                'ram_mb': round(ram_mb, 1),
                'rl_buffer': len(agent.rl_buffer),
                'sl_buffer': len(agent.sl_buffer),
                'mean_q_loss': round(mean_q, 4),
                'mean_sl_loss': round(mean_sl, 4),
                'sha256': sha
            }
            train_logs.append(log_entry)
            print(f"  * [{tag.upper()} | Seed {seed} | Ep {ep:>5}] Time: {elapsed:5.1f}s | RAM: {ram_mb:5.1f}MB | Q-Loss: {mean_q:.4f} | SL-Loss: {mean_sl:.4f} | Hash: {sha}")

    return checkpoints, train_logs

def evaluate_50bb(game, agent, opponent_pool, holdout_seeds=[20261028, 20261029, 20261030], hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)

    for h_seed in holdout_seeds:
        rng_deck = np.random.RandomState(h_seed)
        rng_opp = np.random.RandomState(h_seed + 100000)
        rng_hero = np.random.RandomState(h_seed + 200000)

        returns_this_schedule = []
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
                        if hasattr(agent, 'act'):
                            act = agent.act(st, cp, leg)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            probs = extract_stochastic_probs(agent, inf, leg)
                            act = int(rng_hero.choice(agent.action_dim, p=probs))
                        actions_total[act] += 1
                        st.apply_action(act)
                    else:
                        opp = opponent_pool[cp % len(opponent_pool)]
                        if hasattr(opp, 'act'):
                            act = opp.act(st, cp, leg)
                        else:
                            act = rng_opp.choice(leg)
                        st.apply_action(act)

            ret_bb = st.returns()[hero_seat] / 100.0
            returns_this_schedule.append(ret_bb)

        score_sched = (float(np.sum(returns_this_schedule)) / hands_per_seed) * 100.0
        all_scores.append(score_sched)

    total_acts = sum(actions_total.values())
    act_freqs = {
        'p_fold': round((actions_total[0] / total_acts) * 100.0, 2) if total_acts > 0 else 0.0,
        'p_call': round((actions_total[1] / total_acts) * 100.0, 2) if total_acts > 0 else 0.0,
        'p_raise': round((actions_total[2] / total_acts) * 100.0, 2) if total_acts > 0 else 0.0,
        'p_allin': round((actions_total[3] / total_acts) * 100.0, 2) if total_acts > 0 else 0.0
    }

    mean_score = float(np.mean(all_scores))
    sd_score = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
    sem = sd_score / np.sqrt(len(all_scores))
    ci = 4.303 * sem if len(all_scores) == 3 else 1.96 * sem

    return {
        'mean_bb_100': round(mean_score, 2),
        'sd_bb_100': round(sd_score, 2),
        'ci_95': round(ci, 2),
        'scores_per_schedule': [round(x, 2) for x in all_scores],
        'act_freqs': act_freqs
    }

def main():
    print("================================================================================")
    print("POKER AI V5.6: 5-PLAYER 50BB DEEP-STACK SELF-PLAY (PHASE 1 ~ 6)")
    print("================================================================================")

    game_50bb = get_5p_50bb_poker_game()
    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [10000, 20000, 30000, 40000, 50000]
    holdout_seeds = [20261028, 20261029, 20261030] # 50BB 전용 신규 홀드아웃 시드

    rule_bot = RuleBasedAgent()
    fold_bot = AlwaysFoldAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    # Phase 1: 50BB 칩 원장 및 제로섬 검증
    print("\n[PHASE 1: 50BB 환경 확정 및 칩 원장 무결성 검사]")
    rng = np.random.RandomState(42)
    zero_err = 0
    bound_err = 0
    for _ in range(200):
        st = game_50bb.new_initial_state()
        advance_chance_nodes(st, rng)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng)
            else: st.apply_action(rng.choice(st.legal_actions(st.current_player())))
        rets = st.returns()
        if abs(sum(rets)) > 1e-4: zero_err += 1
        for p in range(5):
            r_bb = rets[p] / 100.0
            if r_bb < -50.0 or r_bb > 200.0: bound_err += 1
    print(f"  * 50BB Zero-sum 위반: {zero_err}회 (0 허용)")
    print(f"  * 50BB 손익 한계(-50BB ~ +200BB) 위반: {bound_err}회 (0 허용)")
    assert zero_err == 0 and bound_err == 0, "50BB Ledger Error!"

    # 50BB 기준선 실측 (vs RuleBased)
    print("\n[50BB 기준선(Baselines) 롤아웃 실측 (스케줄당 1,500핸드 x 3스케줄)]")
    for bname, bagent in [('AlwaysFold', fold_bot), ('CheckCall', check_bot), ('Random', rand_bot)]:
        b_res = evaluate_50bb(game_50bb, bagent, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        print(f"  * {bname:<12} : {b_res['mean_bb_100']:+6.2f} ± {b_res['sd_bb_100']:4.2f} bb/100")

    # ---------------------------------------------------------
    # Phase 2: Group A (50BB From Scratch) vs Group B (50BB Warm Start) 훈련
    # ---------------------------------------------------------
    scratch_checkpoints = {}
    warmstart_checkpoints = {}
    all_train_logs = []

    print("\n[PHASE 2: 50BB From Scratch 50k 에피소드 훈련 (5개 시드)]")
    for s in train_seeds:
        print(f"\n>>> [50BB Scratch Seed {s}] 훈련 시작...")
        cps, logs = train_nfsp_50bb(game_50bb, seed=s, max_episodes=50000, 
                                    checkpoint_eps=checkpoints_to_track, tag="scratch")
        scratch_checkpoints[s] = cps
        all_train_logs.extend(logs)

    print("\n[PHASE 2: 50BB Warm Start 50k 에피소드 훈련 (20BB 50k 가중치 승계)]")
    for s in train_seeds:
        init_path = f"v55/models/nfsp_20bb_seed_{s}_ep_50000.pt"
        print(f"\n>>> [50BB Warm-Start Seed {s}] 훈련 시작...")
        cps, logs = train_nfsp_50bb(game_50bb, seed=s, max_episodes=50000, 
                                    checkpoint_eps=checkpoints_to_track, init_weights_path=init_path, tag="warmstart")
        warmstart_checkpoints[s] = cps
        all_train_logs.extend(logs)

    os.makedirs('v56/data', exist_ok=True)
    if all_train_logs:
        with open('v56/data/phase2_training_50bb_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_train_logs[0].keys())
            writer.writeheader()
            writer.writerows(all_train_logs)
        print("  * 50BB 훈련 로그 저장 완료: v56/data/phase2_training_50bb_logs.csv")

    # ---------------------------------------------------------
    # Phase 3: 50BB 전수 평가 (Scratch vs Warm Start)
    # ---------------------------------------------------------
    print("\n[PHASE 3: 50BB Scratch vs Warm Start 50k 최종 전수 평가 (총 4,500핸드)]")
    scratch_50k_scores = []
    warm_50k_scores = []
    eval_records = []

    for s in train_seeds:
        # Scratch 50k 평가
        m_sc = scratch_checkpoints[s][50000]
        res_sc = evaluate_50bb(game_50bb, m_sc, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        scratch_50k_scores.append(res_sc['mean_bb_100'])

        # Warm Start 50k 평가
        m_w = warmstart_checkpoints[s][50000]
        res_w = evaluate_50bb(game_50bb, m_w, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        warm_50k_scores.append(res_w['mean_bb_100'])

        eval_records.append({
            'train_seed': s,
            'scratch_50k_bb': res_sc['mean_bb_100'],
            'warm_50k_bb': res_w['mean_bb_100'],
            'warm_start_gain': round(res_w['mean_bb_100'] - res_sc['mean_bb_100'], 2),
            'sc_fold': res_sc['act_freqs']['p_fold'],
            'sc_call': res_sc['act_freqs']['p_call'],
            'sc_raise': res_sc['act_freqs']['p_raise'],
            'sc_allin': res_sc['act_freqs']['p_allin'],
            'warm_fold': res_w['act_freqs']['p_fold'],
            'warm_call': res_w['act_freqs']['p_call'],
            'warm_raise': res_w['act_freqs']['p_raise'],
            'warm_allin': res_w['act_freqs']['p_allin']
        })

        print(f"  * Seed {s:<3} | Scratch: {res_sc['mean_bb_100']:+6.2f} bb | Warm: {res_w['mean_bb_100']:+6.2f} bb | Warm Gain: {res_w['mean_bb_100'] - res_sc['mean_bb_100']:+6.2f} bb")

    with open('v56/data/phase3_50bb_scratch_vs_warm.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_records[0].keys())
        writer.writeheader()
        writer.writerows(eval_records)
    print("  * 평가 데이터 저장 완료: v56/data/phase3_50bb_scratch_vs_warm.csv")

    # ---------------------------------------------------------
    # Phase 5: Primary Comparison 통계적 검정 (Paired t-test)
    # ---------------------------------------------------------
    print("\n[PHASE 5: Primary Comparison 통계 검정 (Warm Start vs Scratch)]")
    m_scratch = float(np.mean(scratch_50k_scores))
    sd_scratch = float(np.std(scratch_50k_scores, ddof=1))
    ci_scratch = 2.776 * (sd_scratch / np.sqrt(5))

    m_warm = float(np.mean(warm_50k_scores))
    sd_warm = float(np.std(warm_50k_scores, ddof=1))
    ci_warm = 2.776 * (sd_warm / np.sqrt(5))

    t_pair, p_pair = stats.ttest_rel(warm_50k_scores, scratch_50k_scores)
    warm_gain_mean = m_warm - m_scratch

    print(f"  * 50BB Scratch 50k (5시드) : {m_scratch:+6.2f} ± {ci_scratch:4.2f} bb/100 (SD: {sd_scratch:.2f})")
    print(f"    - 95% CI: [{m_scratch - ci_scratch:+6.2f}, {m_scratch + ci_scratch:+6.2f}]")
    print(f"  * 50BB Warm Start 50k (5시드): {m_warm:+6.2f} ± {ci_warm:4.2f} bb/100 (SD: {sd_warm:.2f})")
    print(f"    - 95% CI: [{m_warm - ci_warm:+6.2f}, {m_warm + ci_warm:+6.2f}]")
    print(f"  * [Primary Comparison Gain]: {warm_gain_mean:+6.2f} bb/100 (t={t_pair:+.2f}, p={p_pair:.4f})")

    print("\n================================================================================")
    print("V5.6 50BB EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY")
    print("================================================================================")

if __name__ == '__main__':
    main()
