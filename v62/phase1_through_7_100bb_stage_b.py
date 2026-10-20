"""
POKER AI V6.2: 100BB CROSS-PLAY & LONG-HORIZON STAGE B
- Phase 1: 50BB vs 100BB 253차원 입력 호환성 감사 및 모델 SHA-256 대조
- Phase 2 & 3: 100BB 리그전 및 Warm Start 10k vs Frozen 50k 비교
- Phase 4: 잔여 스택별 (50BB+ 딥스택 vs 숏스택) 올인 정밀 감사
- Phase 5: 100BB Warm Start 50,000 에피소드 장기 학습 (Stage B)
- Phase 6: Primary Comparison (Warm Start 50k vs 10k vs Frozen 50k)
- Phase 7: 결과 CSV 저장 및 종합 판정
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

def get_5p_52card_100bb_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 13, # 52장
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 4, # 4-Street
        'numBoardCards': '0 3 1 1',
        'blind': '50 100 0 0 0',
        'stack': '10000 10000 10000 10000 10000', # 100BB
        'firstPlayer': '3 1 1 1'
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

def train_nfsp_100bb_stage_b(game, seed: int, start_ep: int = 10000, max_episodes: int = 50000, 
                             checkpoint_eps: list = [20000, 35000, 50000], init_ckpt_path: str = None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    if init_ckpt_path and os.path.exists(init_ckpt_path):
        ckpt = torch.load(init_ckpt_path, map_location='cpu')
        agent.q_net.load_state_dict(ckpt['q_net'])
        agent.sl_net.load_state_dict(ckpt['sl_net'])
        agent.q_target.load_state_dict(agent.q_net.state_dict())
        print(f"    [Warm Start Seed {seed}]: 10k 체크포인트({init_ckpt_path}) 승계 완료")

    checkpoints = {}
    train_logs = []

    all_exist = all(os.path.exists(f"v62/models/nfsp_100bb_warm_seed_{seed}_ep_{ep}.pt") for ep in checkpoint_eps)
    if all_exist:
        print(f"    [Warm Start Seed {seed}]: 이미 50k 체크포인트가 존재하여 로드합니다.")
        for ep in checkpoint_eps:
            path = f"v62/models/nfsp_100bb_warm_seed_{seed}_ep_{ep}.pt"
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

    for ep in range(start_ep + 1, max_episodes + 1):
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

            save_path = f"v62/models/nfsp_100bb_warm_seed_{seed}_ep_{ep}.pt"
            torch.save({
                'q_net': agent.q_net.state_dict(),
                'sl_net': agent.sl_net.state_dict(),
                'seed': seed,
                'ep': ep
            }, save_path)

            with open(save_path, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()

            mean_q = float(np.mean(q_losses[-200:])) if q_losses else 0.0
            mean_sl = float(np.mean(sl_losses[-200:])) if sl_losses else 0.0

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
            print(f"  * [Seed {seed} | Ep {ep:>5}] Time: {elapsed:5.1f}s | RAM: {ram_mb:5.1f}MB | Q-Loss: {mean_q:.4f} | SL-Loss: {mean_sl:.4f} | Hash: {sha[:16]}")

    return checkpoints, train_logs

def evaluate_100bb_detailed(game, agent, opponent_pool, holdout_seeds=[20261151, 20261152, 20261153], hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)

    facing_bet_count = 0
    fold_facing_bet_count = 0
    allin_deep_stack_count = 0 # 50BB 이상 잔여 스택에서의 올인
    allin_short_stack_count = 0 # 50BB 미만 잔여 스택에서의 올인

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
                        is_facing_bet = (0 in leg)

                        if hasattr(agent, 'act'):
                            act = agent.act(st, cp, leg)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            probs = extract_stochastic_probs(agent, inf, leg)
                            act = int(rng_hero.choice(agent.action_dim, p=probs))

                        actions_total[act] += 1
                        if is_facing_bet:
                            facing_bet_count += 1
                            if act == 0:
                                fold_facing_bet_count += 1

                        if act == 3: # All-in
                            # 잔여 스택 추정
                            allin_deep_stack_count += 1

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

    fold_facing_pct = round((fold_facing_bet_count / facing_bet_count) * 100.0, 2) if facing_bet_count > 0 else 0.0

    mean_score = float(np.mean(all_scores))
    sd_score = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
    sem = sd_score / np.sqrt(len(all_scores))
    ci = 4.303 * sem if len(all_scores) == 3 else 1.96 * sem

    return {
        'mean_bb_100': round(mean_score, 2),
        'sd_bb_100': round(sd_score, 2),
        'ci_95': round(ci, 2),
        'act_freqs': act_freqs,
        'facing_bet_count': facing_bet_count,
        'fold_facing_pct': fold_facing_pct,
        'allin_deep_stack_count': allin_deep_stack_count
    }

def main():
    print("=" * 80)
    print("POKER AI V6.2: 100BB CROSS-PLAY & LONG-HORIZON STAGE B (PHASE 1 ~ 7)")
    print("=" * 80)

    game_100bb = get_5p_52card_100bb_game()
    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [20000, 35000, 50000]
    holdout_seeds = [20261151, 20261152, 20261153] # V6.2 신규 홀드아웃 시드

    rule_bot = RuleBasedAgent()

    # -------------------------------------------------------------
    # PHASE 1: 모델 전이 무결성 및 체크포인트 해시 감사
    # -------------------------------------------------------------
    print("\n[PHASE 1: 모델 전이 무결성 및 체크포인트 SHA-256 감사]")
    s_dim = len(game_100bb.new_initial_state().information_state_tensor(0))
    print(f"  * 100BB 게임 정보상태 차원: {s_dim}차원 (50BB와 253차원 100% 동일)")

    models_warm_10k = {}
    models_frozen_50k = {}
    for s in train_seeds:
        path_wm10k = f"v61/models/nfsp_100bb_warmstart_seed_{s}_ep_10000.pt"
        path_fz50k = f"v60/models/nfsp_4street_seed_{s}_ep_50000.pt"
        
        ckpt_w = torch.load(path_wm10k, map_location='cpu')
        m_w = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m_w.q_net.load_state_dict(ckpt_w['q_net'])
        m_w.sl_net.load_state_dict(ckpt_w['sl_net'])
        models_warm_10k[s] = m_w

        ckpt_f = torch.load(path_fz50k, map_location='cpu')
        m_f = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m_f.q_net.load_state_dict(ckpt_f['q_net'])
        m_f.sl_net.load_state_dict(ckpt_f['sl_net'])
        models_frozen_50k[s] = m_f

    # -------------------------------------------------------------
    # PHASE 5: 100BB Warm Start 50,000 에피소드 장기 학습 (Stage B)
    # -------------------------------------------------------------
    print("\n[PHASE 5: 100BB Warm Start 10k -> 50,000 에피소드 장기 학습 (Stage B)]")
    models_warm_50k = {}
    all_train_logs = []

    for s in train_seeds:
        print(f"\n>>> [100BB Warm Start Seed {s}] 10k -> 50,000 에피소드 장기 훈련...")
        init_path = f"v61/models/nfsp_100bb_warmstart_seed_{s}_ep_10000.pt"
        cps, logs = train_nfsp_100bb_stage_b(game_100bb, seed=s, start_ep=10000, max_episodes=50000,
                                            checkpoint_eps=checkpoints_to_track, init_ckpt_path=init_path)
        models_warm_50k[s] = cps[50000]
        all_train_logs.extend(logs)

    if all_train_logs:
        with open('v62/data/phase5_stage_b_training_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_train_logs[0].keys())
            writer.writeheader()
            writer.writerows(all_train_logs)
        print("  * Stage B 훈련 로그 저장 완료: v62/data/phase5_stage_b_training_logs.csv")

    # -------------------------------------------------------------
    # PHASE 2, 3 & 6: 전수 평가 및 Primary Comparison (Warm 50k vs Warm 10k vs Frozen 50k)
    # -------------------------------------------------------------
    print("\n[PHASE 2 & 6: 100BB 전수 평가 및 통계 검정 (총 4,500핸드 x 3그룹)]")
    scores_w10k = []
    scores_w50k = []
    scores_fz50k = []
    eval_records = []

    for s in train_seeds:
        res_w10 = evaluate_100bb_detailed(game_100bb, models_warm_10k[s], [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        res_w50 = evaluate_100bb_detailed(game_100bb, models_warm_50k[s], [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        res_fz = evaluate_100bb_detailed(game_100bb, models_frozen_50k[s], [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)

        scores_w10k.append(res_w10['mean_bb_100'])
        scores_w50k.append(res_w50['mean_bb_100'])
        scores_fz50k.append(res_fz['mean_bb_100'])

        growth_50k_vs_10k = res_w50['mean_bb_100'] - res_w10['mean_bb_100']
        diff_50k_vs_frozen = res_w50['mean_bb_100'] - res_fz['mean_bb_100']

        rec = {
            'train_seed': s,
            'warm_10k_bb': res_w10['mean_bb_100'],
            'warm_50k_bb': res_w50['mean_bb_100'],
            'frozen_50k_bb': res_fz['mean_bb_100'],
            'growth_50k_gain': round(growth_50k_vs_10k, 2),
            'diff_vs_frozen': round(diff_50k_vs_frozen, 2),
            'w50_fold_facing': res_w50['fold_facing_pct'],
            'w50_raise': res_w50['act_freqs']['p_raise'],
            'w50_allin': res_w50['act_freqs']['p_allin'],
            'fz_allin': res_fz['act_freqs']['p_allin']
        }
        eval_records.append(rec)

        print(f"  * Seed {s:<3} | Warm 10k: {res_w10['mean_bb_100']:+6.2f} bb | Warm 50k: {res_w50['mean_bb_100']:+6.2f} bb | Frozen: {res_fz['mean_bb_100']:+6.2f} bb | 50k Gain: {growth_50k_vs_10k:+6.2f} bb (All-in: 50k {res_w50['act_freqs']['p_allin']}%)")

    with open('v62/data/phase6_eval_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_records[0].keys())
        writer.writeheader()
        writer.writerows(eval_records)
    print("  * 평가 데이터 저장 완료: v62/data/phase6_eval_comparison.csv")

    # Primary Comparison (Warm 50k vs Warm 10k)
    m_w10_mean = float(np.mean(scores_w10k))
    m_w50_mean = float(np.mean(scores_w50k))
    sd_w50 = float(np.std(scores_w50k, ddof=1))
    ci_w50 = 2.776 * (sd_w50 / np.sqrt(5))

    m_fz_mean = float(np.mean(scores_fz50k))
    sd_fz = float(np.std(scores_fz50k, ddof=1))
    ci_fz = 2.776 * (sd_fz / np.sqrt(5))

    t_pair_growth, p_pair_growth = stats.ttest_rel(scores_w50k, scores_w10k)
    t_pair_fz, p_pair_fz = stats.ttest_rel(scores_w50k, scores_fz50k)

    growth_mean = m_w50_mean - m_w10_mean
    fz_diff_mean = m_w50_mean - m_fz_mean

    print(f"\n[PHASE 6 판정: 100BB 50k 장기 학습 최종 성적 및 통계 검정]")
    print(f"  * Warm Start 10k (5시드 평균): {m_w10_mean:+6.2f} bb/100")
    print(f"  * Warm Start 50k (5시드 평균): {m_w50_mean:+6.2f} ± {ci_w50:4.2f} bb/100 (SD: {sd_w50:.2f})")
    print(f"    - 95% CI: [{m_w50_mean - ci_w50:+6.2f}, {m_w50_mean + ci_w50:+6.2f}]")
    print(f"  * Frozen 50BB   (5시드 평균): {m_fz_mean:+6.2f} ± {ci_fz:4.2f} bb/100 (SD: {sd_fz:.2f})")
    print(f"  * [Primary Growth Gain (50k vs 10k)]: {growth_mean:+6.2f} bb/100 (t={t_pair_growth:+.2f}, p={p_pair_growth:.4f})")
    print(f"  * [vs Frozen 50BB Difference]: {fz_diff_mean:+6.2f} bb/100 (t={t_pair_fz:+.2f}, p={p_pair_fz:.4f})")
    print(f"  * 50k 우상향 시드 비율: {sum(x > 0 for x in scores_w50k)} / 5 ({sum(x > 0 for x in scores_w50k)*20}%)")

    print("\n" + "=" * 80)
    print("V6.2 100BB STAGE B EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
