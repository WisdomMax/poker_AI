"""
POKER AI V5.5 - 20BB 50,000 EPISODES STABILITY & 4-ACTION BEHAVIOR BENCHMARK
- 5개 독립 학습 시드 (42, 101, 202, 303, 404) 50,000 에피소드 연장 자기 대국
- 체크포인트 보관: 10k, 20k, 30k, 40k, 50k 에피소드
- 4-Action 정밀 행동 분석: Fold, Check/Call, Pot Raise, All-in 빈도 분리 실측
- Best Checkpoint vs Last Checkpoint 과적합/정책 붕괴 분석
- 평가 표본 확대: 신규 3개 홀드아웃 시드 (20261025, 20261026, 20261027) 각 1,500핸드 (총 4,500핸드)
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
from v54.phase0_1_2_smoke_test import get_5p_20bb_poker_game

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

def train_nfsp_50k(game, seed: int, max_episodes: int = 50000, 
                   checkpoint_eps: list = [10000, 20000, 30000, 40000, 50000]):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    checkpoints = {}
    train_logs = []

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
                    r = returns[cp] / 100.0
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

            save_path = f"v55/models/nfsp_20bb_seed_{seed}_ep_{ep}.pt"
            torch.save({
                'q_net': agent.q_net.state_dict(),
                'sl_net': agent.sl_net.state_dict(),
                'seed': seed,
                'ep': ep
            }, save_path)

            with open(save_path, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()[:16]

            mean_q = float(np.mean(q_losses[-300:])) if q_losses else 0.0
            mean_sl = float(np.mean(sl_losses[-300:])) if sl_losses else 0.0

            log_entry = {
                'seed': seed,
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
            print(f"  * [Seed {seed} | Ep {ep:>5}] Time: {elapsed:5.1f}s | RAM: {ram_mb:5.1f}MB | Q-Loss: {mean_q:.4f} | SL-Loss: {mean_sl:.4f} | Hash: {sha}")

    return checkpoints, train_logs

def evaluate_model_extended(game, agent, opponent_pool, holdout_seeds=[20261025, 20261026, 20261027], hands_per_seed=1500):
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
    ci = 4.303 * sem if len(all_scores) == 3 else 1.96 * sem # df=2 t_crit=4.303

    return {
        'mean_bb_100': round(mean_score, 2),
        'sd_bb_100': round(sd_score, 2),
        'ci_95': round(ci, 2),
        'scores_per_schedule': [round(x, 2) for x in all_scores],
        'act_freqs': act_freqs
    }

def main():
    print("================================================================================")
    print("POKER AI V5.5: 20BB 50,000 EPISODES STABILITY & 4-ACTION BEHAVIOR BENCHMARK")
    print("================================================================================")

    game = get_5p_20bb_poker_game()
    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [10000, 20000, 30000, 40000, 50000]
    holdout_seeds = [20261025, 20261026, 20261027] # 표본 확대용 신규 시드

    rule_bot = RuleBasedAgent()
    opp_pool = [rule_bot]

    all_logs = []
    evaluation_records = []
    checkpoint_models = {}

    print("\n[PHASE 1: 5개 독립 시드 50,000 에피소드 장기 학습 개시]")
    for s in train_seeds:
        print(f"\n>>> [Train Seed {s}] 0 -> 50,000 Episodes 훈련 시작...")
        cps, logs = train_nfsp_50k(game, seed=s, max_episodes=50000, checkpoint_eps=checkpoints_to_track)
        checkpoint_models[s] = cps
        all_logs.extend(logs)

    os.makedirs('v55/data', exist_ok=True)
    with open('v55/data/phase1_training_50k_logs.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_logs[0].keys())
        writer.writeheader()
        writer.writerows(all_logs)
    print("  * 50k 훈련 로그 저장 완료: v55/data/phase1_training_50k_logs.csv")

    print("\n[PHASE 2: 10k, 20k, 30k, 40k, 50k 전 체크포인트 전수 평가 (스케줄당 1,500핸드 x 3개 시드)]")
    for s in train_seeds:
        for ep in checkpoints_to_track:
            m = checkpoint_models[s][ep]
            eval_res = evaluate_model_extended(game, m, opp_pool, holdout_seeds=holdout_seeds, hands_per_seed=1500)
            
            rec = {
                'train_seed': s,
                'episodes': ep,
                'mean_bb_100': eval_res['mean_bb_100'],
                'sd_bb_100': eval_res['sd_bb_100'],
                'ci_95': eval_res['ci_95'],
                'p_fold': eval_res['act_freqs']['p_fold'],
                'p_call': eval_res['act_freqs']['p_call'],
                'p_raise': eval_res['act_freqs']['p_raise'],
                'p_allin': eval_res['act_freqs']['p_allin'],
                'sched_1': eval_res['scores_per_schedule'][0],
                'sched_2': eval_res['scores_per_schedule'][1],
                'sched_3': eval_res['scores_per_schedule'][2]
            }
            evaluation_records.append(rec)
            print(f"  * Seed {s:<3} | Ep {ep:>5} | {eval_res['mean_bb_100']:+6.2f} bb/100 | Fold:{eval_res['act_freqs']['p_fold']:4.1f}% | Call:{eval_res['act_freqs']['p_call']:4.1f}% | Raise:{eval_res['act_freqs']['p_raise']:4.1f}% | Shove:{eval_res['act_freqs']['p_allin']:4.1f}%")

    with open('v55/data/phase2_checkpoints_extended_eval.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=evaluation_records[0].keys())
        writer.writeheader()
        writer.writerows(evaluation_records)
    print("  * 전수 평가 데이터 저장 완료: v55/data/phase2_checkpoints_extended_eval.csv")

    # ---------------------------------------------------------
    # PHASE 3: Best Checkpoint vs Last (50k) Checkpoint 과적합 분석
    # ---------------------------------------------------------
    print("\n[PHASE 3: Best Checkpoint vs Last Checkpoint 성능 및 과적합 분석]")
    best_scores = []
    last_scores = []
    for s in train_seeds:
        seed_recs = [r for r in evaluation_records if r['train_seed'] == s]
        best_rec = max(seed_recs, key=lambda x: x['mean_bb_100'])
        last_rec = [r for r in seed_recs if r['episodes'] == 50000][0]

        best_scores.append(best_rec['mean_bb_100'])
        last_scores.append(last_rec['mean_bb_100'])
        print(f"  * Seed {s:<3} | Best Ep: {best_rec['episodes']:>5} ({best_rec['mean_bb_100']:+6.2f} bb) | Last Ep: 50000 ({last_rec['mean_bb_100']:+6.2f} bb) | Drop: {last_rec['mean_bb_100'] - best_rec['mean_bb_100']:+6.2f} bb")

    m_best = float(np.mean(best_scores))
    m_last = float(np.mean(last_scores))
    t_drop, p_drop = stats.ttest_rel(best_scores, last_scores)

    print(f"\n  * 5개 시드 Best Checkpoint 평균: {m_best:+6.2f} bb/100")
    print(f"  * 5개 시드 Last (50k) Checkpoint 평균: {m_last:+6.2f} bb/100")
    print(f"  * 과적합 낙폭 (Drop): {m_last - m_best:+6.2f} bb/100 (t={t_drop:+.2f}, p={p_drop:.4f})")

    # ---------------------------------------------------------
    # PHASE 4: 50k 에피소드 최종 통계적 검정 (5개 독립 시드)
    # ---------------------------------------------------------
    sd_last = float(np.std(last_scores, ddof=1))
    se_last = sd_last / np.sqrt(5)
    ci_last = 2.776 * se_last # df=4

    print(f"\n[PHASE 4: 20BB 50k 최종 통계 판정]")
    print(f"  * 5개 독립 시드 50k 평균 성적: {m_last:+6.2f} ± {ci_last:4.2f} bb/100 (SD: {sd_last:.2f})")
    print(f"  * 95% 신뢰구간: [{m_last - ci_last:+6.2f}, {m_last + ci_last:+6.2f}] bb/100")

    print("\n================================================================================")
    print("V5.5 BENCHMARK EXPERIMENT COMPLETED SUCCESSFULLY")
    print("================================================================================")

if __name__ == '__main__':
    main()
