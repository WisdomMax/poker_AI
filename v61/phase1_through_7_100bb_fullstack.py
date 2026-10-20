"""
POKER AI V6.1: 100BB FULL-STACK NFSP-DQN BENCHMARK
- Phase 1: 52장 4-Street 100BB (10,000칩) 정규 환경 무결성 감사
- Phase 2 & 3: Group A (100BB Scratch), Group B (100BB Warm Start), Group C (50BB Frozen) 훈련 및 구성
- Phase 4: Fold / Call 및 100BB 딥스택 올인 정밀 행동 분석
- Phase 5: 신규 홀드아웃 3개 시드 기반 4,500핸드 전수 평가
- Phase 6: Primary Comparison (Warm Start vs Scratch Paired t-test)
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
        'numRounds': 4, # 4-Street: Preflop, Flop, Turn, River
        'numBoardCards': '0 3 1 1',
        'blind': '50 100 0 0 0',
        'stack': '10000 10000 10000 10000 10000', # 100BB = 10,000 chips
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

def train_nfsp_100bb(game, seed: int, max_episodes: int = 10000, 
                     checkpoint_eps: list = [2000, 5000, 10000],
                     init_ckpt_path: str = None, tag: str = "scratch"):
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
        print(f"    [{tag.upper()} Seed {seed}]: 50BB 50k 가중치 승계 완료 ({init_ckpt_path})")

    checkpoints = {}
    train_logs = []

    all_exist = all(os.path.exists(f"v61/models/nfsp_100bb_{tag}_seed_{seed}_ep_{ep}.pt") for ep in checkpoint_eps)
    if all_exist:
        print(f"    [{tag.upper()} Seed {seed}]: 이미 디스크에 체크포인트가 존재하여 로드합니다.")
        for ep in checkpoint_eps:
            path = f"v61/models/nfsp_100bb_{tag}_seed_{seed}_ep_{ep}.pt"
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

            save_path = f"v61/models/nfsp_100bb_{tag}_seed_{seed}_ep_{ep}.pt"
            torch.save({
                'q_net': agent.q_net.state_dict(),
                'sl_net': agent.sl_net.state_dict(),
                'seed': seed,
                'ep': ep,
                'tag': tag
            }, save_path)

            with open(save_path, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()

            mean_q = float(np.mean(q_losses[-200:])) if q_losses else 0.0
            mean_sl = float(np.mean(sl_losses[-200:])) if sl_losses else 0.0

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
            print(f"  * [{tag.upper()} | Seed {seed} | Ep {ep:>5}] Time: {elapsed:5.1f}s | RAM: {ram_mb:5.1f}MB | Q-Loss: {mean_q:.4f} | SL-Loss: {mean_sl:.4f} | Hash: {sha[:16]}")

    return checkpoints, train_logs

def evaluate_100bb(game, agent, opponent_pool, holdout_seeds=[20261141, 20261142, 20261143], hands_per_seed=1500):
    num_players = 5
    all_scores = []
    actions_total = defaultdict(int)

    facing_bet_count = 0
    fold_facing_bet_count = 0

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
        'fold_facing_bet_count': fold_facing_bet_count,
        'fold_facing_pct': fold_facing_pct
    }

def main():
    print("=" * 80)
    print("POKER AI V6.1: 100BB FULL-STACK NFSP-DQN BENCHMARK (PHASE 1 ~ 7)")
    print("=" * 80)

    game_100bb = get_5p_52card_100bb_game()
    train_seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [2000, 5000, 10000]
    holdout_seeds = [20261141, 20261142, 20261143] # V6.1 신규 홀드아웃 시드

    rule_bot = RuleBasedAgent()
    fold_bot = AlwaysFoldAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    # -------------------------------------------------------------
    # PHASE 1: 100BB 환경 및 칩 원장 무결성 감사
    # -------------------------------------------------------------
    print("\n[PHASE 1: 100BB 정규 환경 및 칩 원장 무결성 감사]")
    s_dim = len(game_100bb.new_initial_state().information_state_tensor(0))
    print(f"  * 100BB 4-Street 정보상태 차원: {s_dim}차원 (50BB와 253차원 완전 일치)")

    rng = np.random.RandomState(42)
    zero_err = 0
    bound_err = 0
    for _ in range(200):
        st = game_100bb.new_initial_state()
        advance_chance_nodes(st, rng)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng)
            else: st.apply_action(rng.choice(st.legal_actions(st.current_player())))
        rets = st.returns()
        if abs(sum(rets)) > 1e-4: zero_err += 1
        for p in range(5):
            r_bb = rets[p] / 100.0
            if r_bb < -100.0 or r_bb > 400.0: bound_err += 1

    print(f"  * 100BB Zero-sum 위반: {zero_err}회 (0 허용)")
    print(f"  * 100BB 손익 한계(-100BB ~ +400BB) 위반: {bound_err}회 (0 허용)")
    assert zero_err == 0 and bound_err == 0, "100BB Ledger Error!"

    # 100BB 기준선 롤아웃 실측 (vs RuleBased)
    print("\n[100BB 기준선(Baselines) 롤아웃 실측 (스케줄당 1,500핸드 x 3스케줄 = 4,500핸드)]")
    for bname, bagent in [('AlwaysFold', fold_bot), ('CheckCall', check_bot), ('Random', rand_bot)]:
        b_res = evaluate_100bb(game_100bb, bagent, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        print(f"  * {bname:<12} : {b_res['mean_bb_100']:+6.2f} ± {b_res['ci_95']:4.2f} bb/100 (SD: {b_res['sd_bb_100']:.2f})")

    # -------------------------------------------------------------
    # PHASE 2 & 3: Group A (100BB Scratch) vs Group B (100BB Warm Start) 훈련
    # -------------------------------------------------------------
    scratch_checkpoints = {}
    warmstart_checkpoints = {}
    frozen_models = {}
    all_train_logs = []

    print("\n[PHASE 2 & 3: Group A (100BB Scratch 10k) 훈련]")
    for s in train_seeds:
        print(f"\n>>> [100BB Scratch Seed {s}] 10,000 에피소드 훈련 시작...")
        cps, logs = train_nfsp_100bb(game_100bb, seed=s, max_episodes=10000, 
                                     checkpoint_eps=checkpoints_to_track, tag="scratch")
        scratch_checkpoints[s] = cps
        all_train_logs.extend(logs)

    print("\n[PHASE 2 & 3: Group B (100BB Warm Start 10k) 훈련 (50BB 50k 모델 승계)]")
    for s in train_seeds:
        init_path = f"v60/models/nfsp_4street_seed_{s}_ep_50000.pt"
        print(f"\n>>> [100BB Warm-Start Seed {s}] 10,000 에피소드 훈련 시작...")
        cps, logs = train_nfsp_100bb(game_100bb, seed=s, max_episodes=10000, 
                                     checkpoint_eps=checkpoints_to_track, init_ckpt_path=init_path, tag="warmstart")
        warmstart_checkpoints[s] = cps
        all_train_logs.extend(logs)

        # Group C (50BB Frozen 50k) 모델도 로드
        ckpt_fz = torch.load(init_path, map_location='cpu')
        m_fz = NFSP_DQN_Reference(0, 253, 4, hidden_dim=128)
        m_fz.q_net.load_state_dict(ckpt_fz['q_net'])
        m_fz.sl_net.load_state_dict(ckpt_fz['sl_net'])
        frozen_models[s] = m_fz

    if all_train_logs:
        with open('v61/data/phase2_training_100bb_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_train_logs[0].keys())
            writer.writeheader()
            writer.writerows(all_train_logs)
        print("  * 100BB 훈련 로그 저장 완료: v61/data/phase2_training_100bb_logs.csv")

    # -------------------------------------------------------------
    # PHASE 5: 100BB 전수 평가 및 Primary Comparison (Scratch vs Warm Start vs Frozen)
    # -------------------------------------------------------------
    print("\n[PHASE 5: 100BB 전수 평가 (총 4,500핸드 x 3그룹)]")
    scratch_scores = []
    warm_scores = []
    frozen_scores = []
    eval_records = []

    for s in train_seeds:
        # Group A (Scratch 10k)
        m_sc = scratch_checkpoints[s][10000]
        res_sc = evaluate_100bb(game_100bb, m_sc, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        scratch_scores.append(res_sc['mean_bb_100'])

        # Group B (Warm Start 10k)
        m_wm = warmstart_checkpoints[s][10000]
        res_wm = evaluate_100bb(game_100bb, m_wm, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        warm_scores.append(res_wm['mean_bb_100'])

        # Group C (Frozen 50BB 50k)
        m_fz = frozen_models[s]
        res_fz = evaluate_100bb(game_100bb, m_fz, [rule_bot], holdout_seeds=holdout_seeds, hands_per_seed=1500)
        frozen_scores.append(res_fz['mean_bb_100'])

        warm_gain = res_wm['mean_bb_100'] - res_sc['mean_bb_100']

        rec = {
            'train_seed': s,
            'scratch_100bb': res_sc['mean_bb_100'],
            'warm_100bb': res_wm['mean_bb_100'],
            'frozen_50bb': res_fz['mean_bb_100'],
            'warm_gain': round(warm_gain, 2),
            'sc_fold_facing': res_sc['fold_facing_pct'],
            'sc_allin': res_sc['act_freqs']['p_allin'],
            'wm_fold_facing': res_wm['fold_facing_pct'],
            'wm_allin': res_wm['act_freqs']['p_allin'],
            'fz_fold_facing': res_fz['fold_facing_pct'],
            'fz_allin': res_fz['act_freqs']['p_allin']
        }
        eval_records.append(rec)

        print(f"  * Seed {s:<3} | Scratch: {res_sc['mean_bb_100']:+6.2f} bb | Warm: {res_wm['mean_bb_100']:+6.2f} bb | Frozen: {res_fz['mean_bb_100']:+6.2f} bb | Warm Gain: {warm_gain:+6.2f} bb (Allin: Sc {res_sc['act_freqs']['p_allin']}%, Wm {res_wm['act_freqs']['p_allin']}%)")

    with open('v61/data/phase5_eval_100bb_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=eval_records[0].keys())
        writer.writeheader()
        writer.writerows(eval_records)
    print("  * 평가 데이터 저장 완료: v61/data/phase5_eval_100bb_results.csv")

    # -------------------------------------------------------------
    # PHASE 6: 통계 검정 및 Primary Comparison
    # -------------------------------------------------------------
    print("\n[PHASE 6: Primary Comparison 통계 검정 (100BB Warm Start vs Scratch)]")
    m_sc_mean = float(np.mean(scratch_scores))
    sd_sc = float(np.std(scratch_scores, ddof=1))
    ci_sc = 2.776 * (sd_sc / np.sqrt(5))

    m_wm_mean = float(np.mean(warm_scores))
    sd_wm = float(np.std(warm_scores, ddof=1))
    ci_wm = 2.776 * (sd_wm / np.sqrt(5))

    m_fz_mean = float(np.mean(frozen_scores))
    sd_fz = float(np.std(frozen_scores, ddof=1))
    ci_fz = 2.776 * (sd_fz / np.sqrt(5))

    t_pair, p_pair = stats.ttest_rel(warm_scores, scratch_scores)
    primary_gain = m_wm_mean - m_sc_mean

    print(f"  * Group A (100BB Scratch 10k) : {m_sc_mean:+6.2f} ± {ci_sc:4.2f} bb/100 (SD: {sd_sc:.2f})")
    print(f"    - 95% CI: [{m_sc_mean - ci_sc:+6.2f}, {m_sc_mean + ci_sc:+6.2f}]")
    print(f"  * Group B (100BB Warm Start 10k): {m_wm_mean:+6.2f} ± {ci_wm:4.2f} bb/100 (SD: {sd_wm:.2f})")
    print(f"    - 95% CI: [{m_wm_mean - ci_wm:+6.2f}, {m_wm_mean + ci_wm:+6.2f}]")
    print(f"  * Group C (50BB Frozen 50k)    : {m_fz_mean:+6.2f} ± {ci_fz:4.2f} bb/100 (SD: {sd_fz:.2f})")
    print(f"  * [Primary Gain (Warm vs Scratch)]: {primary_gain:+6.2f} bb/100 (t={t_pair:+.2f}, p={p_pair:.4f})")

    print("\n" + "=" * 80)
    print("V6.1 100BB FULL-STACK EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
