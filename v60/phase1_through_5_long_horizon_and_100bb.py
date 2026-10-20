"""
POKER AI V6.0: 52-CARD 4-STREET 50BB LONG-HORIZON & 100BB READINESS
- Phase 1: Fold / Call 행동 정밀 감사 (Facing Bet vs Free Check 분리)
- Phase 2: 52장 4-Street 50BB 50,000 에피소드 장기 학습 (5개 독립 시드)
- Phase 3: 신규 홀드아웃 3개 시드 기반 4,500핸드 전수 평가 및 Primary Comparison (50k vs 10k)
- Phase 4: 100BB (10,000칩) 풀스택 환경 파일럿 감사 및 Smoke Test
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

def get_5p_52card_game(stack=5000):
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 13, # 52장 (13 ranks x 4 suits)
        'numSuits': 4,
        'numHoleCards': 2,
        'numRounds': 4, # 4-Street: Preflop, Flop, Turn, River
        'numBoardCards': '0 3 1 1',
        'blind': '50 100 0 0 0',
        'stack': f'{stack} {stack} {stack} {stack} {stack}',
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

def train_nfsp_long(game, seed: int, start_ep: int = 10000, max_episodes: int = 50000, 
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
        print(f"    [Seed {seed}]: 10k 체크포인트({init_ckpt_path}) 성공적 계승 완료")

    checkpoints = {}
    train_logs = []

    all_exist = all(os.path.exists(f"v60/models/nfsp_4street_seed_{seed}_ep_{ep}.pt") for ep in checkpoint_eps)
    if all_exist:
        print(f"    [Seed {seed}]: 이미 50k 체크포인트가 디스크에 존재하여 로드합니다.")
        for ep in checkpoint_eps:
            path = f"v60/models/nfsp_4street_seed_{seed}_ep_{ep}.pt"
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

            save_path = f"v60/models/nfsp_4street_seed_{seed}_ep_{ep}.pt"
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

def evaluate_with_detailed_audit(game, agent, opponent_pool, holdout_seeds=[20261131, 20261132, 20261133], hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)

    facing_bet_count = 0
    fold_facing_bet_count = 0
    free_check_count = 0
    paid_call_count = 0

    street_actions = defaultdict(lambda: defaultdict(int)) # round -> act -> count

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
                        # 라운드 파악 (0: Preflop, 1: Flop, 2: Turn, 3: River)
                        cur_round = st.round_from_state() if hasattr(st, 'round_from_state') else 0

                        # Fold가 합법인지 여부 (Fold가 합법이면 Facing Bet 상황임)
                        is_facing_bet = (0 in leg)

                        if hasattr(agent, 'act'):
                            act = agent.act(st, cp, leg)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            probs = extract_stochastic_probs(agent, inf, leg)
                            act = int(rng_hero.choice(agent.action_dim, p=probs))

                        actions_total[act] += 1
                        street_actions[cur_round][act] += 1

                        if is_facing_bet:
                            facing_bet_count += 1
                            if act == 0:
                                fold_facing_bet_count += 1
                            elif act == 1:
                                paid_call_count += 1
                        else:
                            if act == 1:
                                free_check_count += 1

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

    fold_facing_bet_pct = round((fold_facing_bet_count / facing_bet_count) * 100.0, 2) if facing_bet_count > 0 else 0.0

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
        'fold_facing_bet_count': fold_facing_bet_count,
        'fold_facing_bet_pct': fold_facing_bet_pct,
        'free_check_count': free_check_count,
        'paid_call_count': paid_call_count
    }

def main():
    print("=" * 80)
    print("POKER AI V6.0: 52-CARD 4-STREET 50BB LONG-HORIZON & 100BB READINESS")
    print("=" * 80)

    game_50bb = get_5p_52card_game(stack=5000)
    game_100bb = get_5p_52card_game(stack=10000)

    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [20000, 35000, 50000]
    holdout_seeds = [20261131, 20261132, 20261133] # V6.0 신규 홀드아웃 시드

    rule_bot = RuleBasedAgent()

    # -------------------------------------------------------------
    # PHASE 1: V5.9 10k 모델 대상 FOLD/CALL 행동 정밀 감사
    # -------------------------------------------------------------
    print("\n[PHASE 1: FOLD/CALL 정밀 행동 감사 (Facing Bet vs Free Check 분리)]")
    audit_records = []
    scores_10k = []
    models_10k = {}

    for s in train_seeds:
        path_10k = f"v59/models/nfsp_4street_seed_{s}_ep_10000.pt"
        ckpt = torch.load(path_10k, map_location='cpu')
        m_10k = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m_10k.q_net.load_state_dict(ckpt['q_net'])
        m_10k.sl_net.load_state_dict(ckpt['sl_net'])
        models_10k[s] = m_10k

        res_10k = evaluate_with_detailed_audit(game_50bb, m_10k, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        scores_10k.append(res_10k['mean_bb_100'])
        audit_records.append(res_10k)

        print(f"  * Seed {s:<3} 10k 감사 | Facing Bet: {res_10k['facing_bet_count']:>4}회 중 Fold {res_10k['fold_facing_bet_count']:>2}회 ({res_10k['fold_facing_bet_pct']:4.1f}%) | Free Check: {res_10k['free_check_count']}회 | Paid Call: {res_10k['paid_call_count']}회 | EV: {res_10k['mean_bb_100']:+6.2f} bb")

    # -------------------------------------------------------------
    # PHASE 2: 4-Street 50BB 50,000 에피소드 장기 학습
    # -------------------------------------------------------------
    print("\n[PHASE 2: 52장 4-Street 50,000 에피소드 장기 학습 (5개 독립 시드)]")
    models_50k = {}
    all_train_logs = []

    for s in train_seeds:
        print(f"\n>>> [Seed {s}] 10k -> 50,000 에피소드 장기 학습 시작...")
        init_path = f"v59/models/nfsp_4street_seed_{s}_ep_10000.pt"
        cps, logs = train_nfsp_long(game_50bb, seed=s, start_ep=10000, max_episodes=50000, 
                                   checkpoint_eps=checkpoints_to_track, init_ckpt_path=init_path)
        models_50k[s] = cps[50000]
        all_train_logs.extend(logs)

    if all_train_logs:
        with open('v60/data/phase2_long_training_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_train_logs[0].keys())
            writer.writeheader()
            writer.writerows(all_train_logs)
        print("  * 장기 훈련 로그 저장 완료: v60/data/phase2_long_training_logs.csv")

    # -------------------------------------------------------------
    # PHASE 3: 50k 모델 전수 평가 및 Primary Comparison (50k vs 10k)
    # -------------------------------------------------------------
    print("\n[PHASE 3: 50k 모델 전수 평가 및 Primary Comparison (50k vs 10k)]")
    scores_50k = []
    eval_records = []

    for s in train_seeds:
        m_50k = models_50k[s]
        res_50k = evaluate_with_detailed_audit(game_50bb, m_50k, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        scores_50k.append(res_50k['mean_bb_100'])
        gain = res_50k['mean_bb_100'] - scores_10k[train_seeds.index(s)]

        rec = {
            'train_seed': s,
            'score_10k': scores_10k[train_seeds.index(s)],
            'score_50k': res_50k['mean_bb_100'],
            'growth_gain': round(gain, 2),
            'fold_pct': res_50k['act_freqs']['p_fold'],
            'facing_bet_fold_pct': res_50k['fold_facing_bet_pct'],
            'call_pct': res_50k['act_freqs']['p_call'],
            'raise_pct': res_50k['act_freqs']['p_raise'],
            'allin_pct': res_50k['act_freqs']['p_allin']
        }
        eval_records.append(rec)
        print(f"  * Seed {s:<3} | 10k: {scores_10k[train_seeds.index(s)]:+6.2f} bb | 50k: {res_50k['mean_bb_100']:+6.2f} bb | Gain: {gain:+6.2f} bb/100 | Facing Fold: {res_50k['fold_facing_bet_pct']:4.1f}% | Raise: {res_50k['act_freqs']['p_raise']:4.1f}% | All-in: {res_50k['act_freqs']['p_allin']:4.1f}%")

    with open('v60/data/phase3_eval_50k_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_records[0].keys())
        writer.writeheader()
        writer.writerows(eval_records)
    print("  * 50k 평가 데이터 저장 완료: v60/data/phase3_eval_50k_results.csv")

    # Primary Comparison (50k vs 10k Paired t-test)
    m_10k_mean = float(np.mean(scores_10k))
    m_50k_mean = float(np.mean(scores_50k))
    sd_50k = float(np.std(scores_50k, ddof=1))
    ci_50k = 2.776 * (sd_50k / np.sqrt(5))

    t_pair, p_pair = stats.ttest_rel(scores_50k, scores_10k)
    long_gain = m_50k_mean - m_10k_mean

    print(f"\n[PHASE 3 판정: 50k 장기 학습 성과]")
    print(f"  * 10k 평균 성적: {m_10k_mean:+6.2f} bb/100")
    print(f"  * 50k 평균 성적: {m_50k_mean:+6.2f} ± {ci_50k:4.2f} bb/100 (SD: {sd_50k:.2f})")
    print(f"    - 95% CI: [{m_50k_mean - ci_50k:+6.2f}, {m_50k_mean + ci_50k:+6.2f}]")
    print(f"  * [Primary Gain (50k vs 10k)]: {long_gain:+6.2f} bb/100 (t={t_pair:+.2f}, p={p_pair:.4f})")
    print(f"  * 50k 우상향 시드 비율: {sum(x > 0 for x in scores_50k)} / 5 ({sum(x > 0 for x in scores_50k)*20}%)")

    # -------------------------------------------------------------
    # PHASE 4: 100BB (10,000칩) 풀스택 환경 파일럿 감사 및 Smoke Test
    # -------------------------------------------------------------
    print("\n[PHASE 4: 100BB (10,000칩) 풀스택 정규 환경 파일럿 무결성 감사]")
    s_dim_100bb = len(game_100bb.new_initial_state().information_state_tensor(0))
    print(f"  * 100BB 4-Street 정보상태 차원: {s_dim_100bb}차원")

    rng_pilot = np.random.RandomState(42)
    zero_err_100bb = 0
    bound_err_100bb = 0
    for _ in range(200):
        st = game_100bb.new_initial_state()
        advance_chance_nodes(st, rng_pilot)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_pilot)
            else: st.apply_action(rng_pilot.choice(st.legal_actions(st.current_player())))
        rets = st.returns()
        if abs(sum(rets)) > 1e-4: zero_err_100bb += 1
        for p in range(5):
            r_bb = rets[p] / 100.0
            if r_bb < -100.0 or r_bb > 400.0: bound_err_100bb += 1

    print(f"  * 100BB Zero-sum 위반: {zero_err_100bb}회 (0 허용)")
    print(f"  * 100BB 손익 한계(-100BB ~ +400BB) 위반: {bound_err_100bb}회 (0 허용)")
    assert zero_err_100bb == 0 and bound_err_100bb == 0, "100BB Ledger Error!"

    # 100BB 3개 시드 Smoke Test (1,000 에피소드)
    print("  * 100BB 3개 시드 Smoke Test (1,000 eps) 실행...")
    for s_pilot in [42, 101, 202]:
        train_nfsp_long(game_100bb, seed=s_pilot, start_ep=0, max_episodes=1000, 
                        checkpoint_eps=[1000])
    print("  * 100BB Smoke Test 통과: 100BB 풀스택 본격 학습 준비 완료 (READY)")

    print("\n" + "=" * 80)
    print("V6.0 50BB LONG-HORIZON & 100BB READINESS SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
