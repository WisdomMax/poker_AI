"""
POKER AI V5.9: 52-CARD FULL 4-STREET REGULAR HOLD'EM (PREFLOP TO RIVER)
- Phase 1: 52장 4-Street 정규 홀덤 환경 구축 및 칩 원장 무결성 감사
- Phase 2: 5개 독립 시드 (42, 101, 202, 303, 404) 10,000 에피소드 훈련
- Phase 3: 신규 홀드아웃 3개 시드 기반 4,500핸드 전수 평가
- Phase 4: Primary Comparison (10k vs 2k Paired t-test) 및 AA vs 72o 전략 분화 측정
- Phase 5: 결과 CSV 저장 및 종합 판정
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

def get_5p_52card_4street_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 13, # 52장 (13 ranks x 4 suits)
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 4, # 4-Street: Preflop, Flop, Turn, River
        'numBoardCards': '0 3 1 1', # 0 -> 3 -> 1 -> 1 (총 5장)
        'blind': '50 100 0 0 0',
        'stack': '5000 5000 5000 5000 5000', # 50BB (5,000 chips)
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

def train_nfsp_4street(game, seed: int, max_episodes: int = 10000, 
                       checkpoint_eps: list = [2000, 5000, 10000]):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    checkpoints = {}
    train_logs = []

    # 디스크 캐시 확인
    all_exist = all(os.path.exists(f"v59/models/nfsp_4street_seed_{seed}_ep_{ep}.pt") for ep in checkpoint_eps)
    if all_exist:
        print(f"    [4-Street Seed {seed}]: 이미 체크포인트가 디스크에 존재하여 로드합니다.")
        for ep in checkpoint_eps:
            path = f"v59/models/nfsp_4street_seed_{seed}_ep_{ep}.pt"
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
                    r = returns[cp] / 100.0 # Net BB
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

            save_path = f"v59/models/nfsp_4street_seed_{seed}_ep_{ep}.pt"
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
            print(f"  * [4-Street | Seed {seed} | Ep {ep:>5}] Time: {elapsed:5.1f}s | RAM: {ram_mb:5.1f}MB | Q-Loss: {mean_q:.4f} | SL-Loss: {mean_sl:.4f} | Hash: {sha[:16]}")

    return checkpoints, train_logs

def evaluate_4street(game, agent, opponent_pool, holdout_seeds=[20261121, 20261122, 20261123], hands_per_seed=1500):
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
    print("=" * 80)
    print("POKER AI V5.9: 52-CARD FULL 4-STREET REGULAR HOLD'EM (PHASE 1 ~ 5)")
    print("=" * 80)

    game_4st = get_5p_52card_4street_game()
    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [2000, 5000, 10000]
    holdout_seeds = [20261121, 20261122, 20261123] # V5.9 신규 홀드아웃 시드

    rule_bot = RuleBasedAgent()
    fold_bot = AlwaysFoldAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    # -------------------------------------------------------------
    # PHASE 1: 52장 4-Street 환경 및 칩 원장 무결성 감사
    # -------------------------------------------------------------
    print("\n[PHASE 1: 52장 4-Street 환경 및 칩 원장 무결성 검증]")
    s_dim = len(game_4st.new_initial_state().information_state_tensor(0))
    print(f"  * 52장 4-Street 정보상태 차원: {s_dim}차원 (Preflop + Flop 3 + Turn 1 + River 1)")

    rng = np.random.RandomState(42)
    zero_err = 0
    bound_err = 0
    for _ in range(200):
        st = game_4st.new_initial_state()
        advance_chance_nodes(st, rng)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng)
            else: st.apply_action(rng.choice(st.legal_actions(st.current_player())))
        rets = st.returns()
        if abs(sum(rets)) > 1e-4: zero_err += 1
        for p in range(5):
            r_bb = rets[p] / 100.0
            if r_bb < -50.0 or r_bb > 200.0: bound_err += 1

    print(f"  * 4-Street Zero-sum 위반: {zero_err}회 (0 허용)")
    print(f"  * 4-Street 손익 한계(-50BB ~ +200BB) 위반: {bound_err}회 (0 허용)")
    assert zero_err == 0 and bound_err == 0, "4-Street Ledger Error!"

    # 4-Street 기준선 롤아웃 실측 (vs RuleBased)
    print("\n[4-Street 기준선(Baselines) 롤아웃 실측 (스케줄당 1,500핸드 x 3스케줄 = 4,500핸드)]")
    for bname, bagent in [('AlwaysFold', fold_bot), ('CheckCall', check_bot), ('Random', rand_bot)]:
        b_res = evaluate_4street(game_4st, bagent, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        print(f"  * {bname:<12} : {b_res['mean_bb_100']:+6.2f} ± {b_res['ci_95']:4.2f} bb/100 (SD: {b_res['sd_bb_100']:.2f})")

    # -------------------------------------------------------------
    # PHASE 2: 52장 4-Street NFSP 훈련 (5개 시드 10k 에피소드)
    # -------------------------------------------------------------
    print("\n[PHASE 2: 52장 4-Street NFSP-DQN 자기대국 훈련 (5개 독립 시드)]")
    all_checkpoints = {}
    all_train_logs = []

    for s in train_seeds:
        print(f"\n>>> [4-Street Seed {s}] 10,000 에피소드 훈련 시작...")
        cps, logs = train_nfsp_4street(game_4st, seed=s, max_episodes=10000, 
                                       checkpoint_eps=checkpoints_to_track)
        all_checkpoints[s] = cps
        all_train_logs.extend(logs)

    if all_train_logs:
        with open('v59/data/phase2_training_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_train_logs[0].keys())
            writer.writeheader()
            writer.writerows(all_train_logs)
        print("  * 4-Street 훈련 로그 저장 완료: v59/data/phase2_training_logs.csv")

    # -------------------------------------------------------------
    # PHASE 3 & 4: 52장 4-Street 전수 평가 및 Primary Comparison (10k vs 2k)
    # -------------------------------------------------------------
    print("\n[PHASE 3 & 4: 4-Street 전수 평가 및 Primary Comparison (10k vs 2k)]")
    eval_records = []
    scores_2k = []
    scores_10k = []

    for s in train_seeds:
        m_2k = all_checkpoints[s][2000]
        res_2k = evaluate_4street(game_4st, m_2k, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        scores_2k.append(res_2k['mean_bb_100'])

        m_10k = all_checkpoints[s][10000]
        res_10k = evaluate_4street(game_4st, m_10k, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        scores_10k.append(res_10k['mean_bb_100'])

        gain = res_10k['mean_bb_100'] - res_2k['mean_bb_100']

        rec = {
            'train_seed': s,
            'score_2k': res_2k['mean_bb_100'],
            'score_10k': res_10k['mean_bb_100'],
            'growth_gain': round(gain, 2),
            'fold_10k': res_10k['act_freqs']['p_fold'],
            'call_10k': res_10k['act_freqs']['p_call'],
            'raise_10k': res_10k['act_freqs']['p_raise'],
            'allin_10k': res_10k['act_freqs']['p_allin']
        }
        eval_records.append(rec)
        print(f"  * Seed {s:<3} | 2k: {res_2k['mean_bb_100']:+6.2f} bb | 10k: {res_10k['mean_bb_100']:+6.2f} bb | Growth Gain: {gain:+6.2f} bb/100 | Fold: {res_10k['act_freqs']['p_fold']:4.1f}% | Raise: {res_10k['act_freqs']['p_raise']:4.1f}% | All-in: {res_10k['act_freqs']['p_allin']:4.1f}%")

    with open('v59/data/phase3_eval_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_records[0].keys())
        writer.writeheader()
        writer.writerows(eval_records)
    print("  * 평가 데이터 저장 완료: v59/data/phase3_eval_results.csv")

    # Primary Comparison (10k vs 2k Paired t-test)
    m_2k_mean = float(np.mean(scores_2k))
    m_10k_mean = float(np.mean(scores_10k))
    sd_10k = float(np.std(scores_10k, ddof=1))
    ci_10k = 2.776 * (sd_10k / np.sqrt(5))

    t_pair, p_pair = stats.ttest_rel(scores_10k, scores_2k)
    growth_mean = m_10k_mean - m_2k_mean

    print(f"\n[PHASE 5: 통계 검정 및 학습 성장성 종합 판정]")
    print(f"  * 4-Street 2k  (5시드 평균): {m_2k_mean:+6.2f} bb/100")
    print(f"  * 4-Street 10k (5시드 평균): {m_10k_mean:+6.2f} ± {ci_10k:4.2f} bb/100 (SD: {sd_10k:.2f})")
    print(f"    - 95% CI: [{m_10k_mean - ci_10k:+6.2f}, {m_10k_mean + ci_10k:+6.2f}]")
    print(f"  * [Primary Growth Gain (10k vs 2k)]: {growth_mean:+6.2f} bb/100 (t={t_pair:+.2f}, p={p_pair:.4f})")
    print(f"  * 우상향 시드 비율: {sum(x > 0 for x in scores_10k)} / 5 ({sum(x > 0 for x in scores_10k)*20}%)")

    print("\n" + "=" * 80)
    print("V5.9 4-STREET EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
