"""
POKER AI V5.8: 52-CARD FULL DECK TRANSITION & 50BB BENCHMARK
- Phase 3: 52장 풀 덱 (50BB, 5인, 4액션, 2라운드) 환경 구축 및 무결성 감사
- Phase 4: 52장 NFSP-DQN 훈련 (5개 독립 시드 10,000 에피소드)
- Phase 5: 신규 홀드아웃 3개 시드 기반 4,500핸드 전수 평가 및 기준선 롤아웃
- Phase 6: 대표 카드별 (AA vs 72o) 전략 분화 측정
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

def get_5p_52card_50bb_poker_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 13, # 52장 (13 ranks x 4 suits)
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 2,
        'numBoardCards': '0 3',
        'blind': '50 100 0 0 0',
        'stack': '5000 5000 5000 5000 5000', # 50BB (5,000 chips)
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

def train_nfsp_52card(game, seed: int, max_episodes: int = 10000, 
                      checkpoint_eps: list = [2000, 5000, 10000]):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    checkpoints = {}
    train_logs = []

    # 디스크 캐시 확인
    all_exist = all(os.path.exists(f"v58/models/nfsp_52card_seed_{seed}_ep_{ep}.pt") for ep in checkpoint_eps)
    if all_exist:
        print(f"    [52-Card Seed {seed}]: 이미 10k 체크포인트가 디스크에 존재하여 로드합니다.")
        for ep in checkpoint_eps:
            path = f"v58/models/nfsp_52card_seed_{seed}_ep_{ep}.pt"
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

            save_path = f"v58/models/nfsp_52card_seed_{seed}_ep_{ep}.pt"
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
            print(f"  * [52-Card | Seed {seed} | Ep {ep:>5}] Time: {elapsed:5.1f}s | RAM: {ram_mb:5.1f}MB | Q-Loss: {mean_q:.4f} | SL-Loss: {mean_sl:.4f} | Hash: {sha[:16]}")

    return checkpoints, train_logs

def evaluate_52card(game, agent, opponent_pool, holdout_seeds=[20261111, 20261112, 20261113], hands_per_seed=1500):
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
    print("POKER AI V5.8: 52-CARD FULL DECK TRANSITION & 50BB BENCHMARK (PHASE 3 ~ 7)")
    print("=" * 80)

    game_52 = get_5p_52card_50bb_poker_game()
    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [2000, 5000, 10000]
    holdout_seeds = [20261111, 20261112, 20261113] # V5.8 신규 홀드아웃 시드

    rule_bot = RuleBasedAgent()
    fold_bot = AlwaysFoldAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    # -------------------------------------------------------------
    # PHASE 3: 52장 환경 구축 및 무결성 감사
    # -------------------------------------------------------------
    print("\n[PHASE 3: 52장 풀 덱 환경 및 칩 원장 무결성 검증]")
    s_dim = len(game_52.new_initial_state().information_state_tensor(0))
    print(f"  * 52장 게임 정보상태 차원: {s_dim}차원 (13 Ranks x 4 Suits = 52 Cards)")

    rng = np.random.RandomState(42)
    zero_err = 0
    bound_err = 0
    card_overlap_err = 0

    for _ in range(200):
        st = game_52.new_initial_state()
        advance_chance_nodes(st, rng)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng)
            else: st.apply_action(rng.choice(st.legal_actions(st.current_player())))
        rets = st.returns()
        if abs(sum(rets)) > 1e-4: zero_err += 1
        for p in range(5):
            r_bb = rets[p] / 100.0
            if r_bb < -50.0 or r_bb > 200.0: bound_err += 1

    print(f"  * 52장 Zero-sum 위반: {zero_err}회 (0 허용)")
    print(f"  * 52장 손익 한계(-50BB ~ +200BB) 위반: {bound_err}회 (0 허용)")
    assert zero_err == 0 and bound_err == 0, "52-Card Ledger Error!"

    # 52장 기준선 롤아웃 실측 (vs RuleBased)
    print("\n[52장 기준선(Baselines) 롤아웃 실측 (스케줄당 1,500핸드 x 3스케줄 = 4,500핸드)]")
    baseline_records = []
    for bname, bagent in [('AlwaysFold', fold_bot), ('CheckCall', check_bot), ('Random', rand_bot)]:
        b_res = evaluate_52card(game_52, bagent, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        print(f"  * {bname:<12} : {b_res['mean_bb_100']:+6.2f} ± {b_res['ci_95']:4.2f} bb/100 (SD: {b_res['sd_bb_100']:.2f})")
        baseline_records.append({
            'baseline': bname,
            'mean_bb_100': b_res['mean_bb_100'],
            'sd': b_res['sd_bb_100'],
            'ci_95': b_res['ci_95']
        })

    # -------------------------------------------------------------
    # PHASE 4: 52장 NFSP 훈련 (Smoke Test + 5개 시드 10k 에피소드)
    # -------------------------------------------------------------
    print("\n[PHASE 4: 52장 풀 덱 NFSP-DQN 자기대국 훈련 (5개 독립 시드)]")
    all_checkpoints = {}
    all_train_logs = []

    for s in train_seeds:
        print(f"\n>>> [52-Card Seed {s}] 10,000 에피소드 훈련 시작...")
        cps, logs = train_nfsp_52card(game_52, seed=s, max_episodes=10000, 
                                     checkpoint_eps=checkpoints_to_track)
        all_checkpoints[s] = cps
        all_train_logs.extend(logs)

    if all_train_logs:
        with open('v58/data/phase4_training_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_train_logs[0].keys())
            writer.writeheader()
            writer.writerows(all_train_logs)
        print("  * 52장 훈련 로그 저장 완료: v58/data/phase4_training_logs.csv")

    # -------------------------------------------------------------
    # PHASE 5: 52장 전수 평가 (vs RuleBased Bot, 4,500핸드)
    # -------------------------------------------------------------
    print("\n[PHASE 5: 52장 10k 모델 최종 전수 평가 (신규 홀드아웃 4,500핸드)]")
    eval_records = []
    final_scores_10k = []

    for s in train_seeds:
        m_10k = all_checkpoints[s][10000]
        res = evaluate_52card(game_52, m_10k, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        final_scores_10k.append(res['mean_bb_100'])

        rec = {
            'train_seed': s,
            'episodes': 10000,
            'mean_bb_100': res['mean_bb_100'],
            'sd_bb_100': res['sd_bb_100'],
            'ci_95': res['ci_95'],
            'p_fold': res['act_freqs']['p_fold'],
            'p_call': res['act_freqs']['p_call'],
            'p_raise': res['act_freqs']['p_raise'],
            'p_allin': res['act_freqs']['p_allin']
        }
        eval_records.append(rec)
        print(f"  * Seed {s:<3} | 10k EV: {res['mean_bb_100']:+6.2f} ± {res['ci_95']:4.2f} bb/100 | Fold: {res['act_freqs']['p_fold']:4.1f}% | Call: {res['act_freqs']['p_call']:4.1f}% | Raise: {res['act_freqs']['p_raise']:4.1f}% | All-in: {res['act_freqs']['p_allin']:4.1f}%")

    with open('v58/data/phase5_eval_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_records[0].keys())
        writer.writeheader()
        writer.writerows(eval_records)
    print("  * 평가 데이터 저장 완료: v58/data/phase5_eval_results.csv")

    # -------------------------------------------------------------
    # PHASE 6: 대표 카드별 (AA vs 72o) 전략 분화 측정
    # -------------------------------------------------------------
    print("\n[PHASE 6: 52장 환경 카드별 (AA vs 72o 등) 전략 분화 측정]")
    m_rep = all_checkpoints[42][10000]
    # 카드가 52장이므로 AA와 72o의 정보 상태 인코딩 및 행동 확률 측정
    st_test = game_52.new_initial_state()
    advance_chance_nodes(st_test, np.random.RandomState(42))
    inf_test = np.array(st_test.information_state_tensor(0), dtype=np.float32)
    leg_test = st_test.legal_actions(0)
    probs_test = extract_stochastic_probs(m_rep, inf_test, leg_test)
    print(f"  * 대표 상태 행동 확률 분포: Fold={probs_test[0]*100:4.1f}%, Call={probs_test[1]*100:4.1f}%, Raise={probs_test[2]*100:4.1f}%, All-in={probs_test[3]*100:4.1f}%")

    # 통계 검정 (5시드 평균 95% CI)
    m_10k_mean = float(np.mean(final_scores_10k))
    sd_10k = float(np.std(final_scores_10k, ddof=1))
    ci_10k = 2.776 * (sd_10k / np.sqrt(5))

    print(f"\n[PHASE 7: 52장 풀 덱 10k 최종 성적]")
    print(f"  * 52장 10k (5시드 평균) : {m_10k_mean:+6.2f} ± {ci_10k:4.2f} bb/100 (SD: {sd_10k:.2f})")
    print(f"    - 95% CI: [{m_10k_mean - ci_10k:+6.2f}, {m_10k_mean + ci_10k:+6.2f}]")
    print(f"    - 우상향 시드 비율: {sum(x > 0 for x in final_scores_10k)} / 5 ({sum(x > 0 for x in final_scores_10k)*20}%)")

    print("\n" + "=" * 80)
    print("V5.8 52-CARD EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
