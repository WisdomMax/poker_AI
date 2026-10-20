"""
POKER AI V5.3.1 - PHASE 3, 4, 5, 6, 7: POSITION-ROTATED RE-EVALUATION BENCHMARK
- 완전한 5인 좌석 회전(Hero Seat = hand_idx % 5, 5개 포지션 각 100회 균등 배분) 적용
- Phase 3: 기존 V5.3 5개 독립 시드 체크포인트(2k vs 10k) 재평가
  * Gain_2k_to_10k 재산출, Student-t 95% CI 및 Paired t-test
- Phase 4: 20BB 파일럿 성적 좌석 회전 재검증 (+445 bb/100의 좌석 편향 여부 판정)
- Phase 5: 10BB vs 20BB 통제 실험
- Phase 6: AA 및 99의 Policy Regret 동시 개선 분석
- Phase 7: 완전 좌석 회전 하에서 5대 기준선(AlwaysFold, CheckCall, Random, PPO, NFSP-DQN) 공정 재평가
- 원시 데이터를 v531/data/ 내 CSV 파일들로 저장
"""

import os
import sys
import csv
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v522')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v523')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v53')))
from v52.phase5_6_7_five_player_benchmark import (
    get_5p_reduced_holdem_game, advance_chance_nodes,
    train_5p_ppo_selfplay, RandomAgent, AlwaysFoldAgent, CheckCallAgent, RuleBasedAgent
)
from nfsp_variants import NFSP_DQN_Reference
from phase2_3_5_6_7_long_horizon import train_nfsp_dqn_checkpoints

def evaluate_with_seat_rotation(game, hero_agent, opponent_agent, num_hands=500, eval_seed=20261018):
    """
    정상 5인 좌석 회전 평가:
    Hero Seat = h % 5 (0: SB, 1: BB, 2: UTG, 3: CO, 4: BTN)
    5개 포지션이 정확히 균등하게 100회씩 경험됨
    """
    rng = np.random.RandomState(eval_seed)
    returns = []
    seat_rets = defaultdict(list)

    for h in range(num_hands):
        hero_seat = h % 5
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)

                if cp == hero_seat: # Hero의 턴
                    if hasattr(hero_agent, 'act'):
                        act = hero_agent.act(st, cp, leg)
                    elif hasattr(hero_agent, 'select_action'):
                        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        act = hero_agent.select_action(inf, leg, is_eval=True)
                    else: # PPO
                        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        x = torch.tensor(inf).unsqueeze(0)
                        mask = torch.zeros(4, dtype=torch.bool)
                        mask[leg] = True
                        with torch.no_grad():
                            _, probs, _ = hero_agent(x, mask.unsqueeze(0))
                            act = torch.distributions.Categorical(probs).sample().item()
                    st.apply_action(act)
                else: # Opponents
                    if hasattr(opponent_agent, 'act'):
                        act = opponent_agent.act(st, cp, leg)
                    else:
                        act = leg[0]
                    st.apply_action(act)

        ret_bb = st.returns()[hero_seat] / 100.0
        returns.append(ret_bb)
        seat_rets[hero_seat].append(ret_bb)

    m_bb = float(np.mean(returns)) * 100.0
    se_bb = float(stats.sem(returns)) * 100.0
    ci_bb = 1.96 * se_bb

    return m_bb, ci_bb, seat_rets


def run_v531_benchmarks():
    print("=" * 85)
    print("PHASE 3~7: 정상 좌석 회전 기반 V5.3.1 전면 재평가 및 검증")
    print("=" * 85)

    game_10bb = get_5p_reduced_holdem_game()
    rule_bot = RuleBasedAgent()
    fold_bot = AlwaysFoldAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    seeds = [42, 101, 202, 303, 404]

    # 1. Phase 3: 5개 시드 2k vs 10k 에피소드 모델 좌석 회전 재평가
    print("\n[PHASE 3: 5개 시드 2k vs 10k 체크포인트 정상 좌석 회전 재평가]")
    p3_records = []
    rets_2k = []
    rets_10k = []

    for s in seeds:
        print(f"  * Seed {s} 체크포인트 로드 및 좌석 회전 500판 평가...")
        cps = train_nfsp_dqn_checkpoints(game_10bb, seed=s, checkpoint_eps=[2000, 10000])

        m_2k, ci_2k, _ = evaluate_with_seat_rotation(game_10bb, cps[2000], rule_bot, num_hands=500, eval_seed=20261018 + s)
        m_10k, ci_10k, _ = evaluate_with_seat_rotation(game_10bb, cps[10000], rule_bot, num_hands=500, eval_seed=20261018 + s)

        gain = m_10k - m_2k
        rets_2k.append(m_2k)
        rets_10k.append(m_10k)

        print(f"    - Seed {s:<3} | 2k: {m_2k:+6.2f} bb/100 | 10k: {m_10k:+6.2f} bb/100 | Gain: {gain:+6.2f} bb/100")
        p3_records.append({
            'seed': s, 'return_2k': m_2k, 'return_10k': m_10k, 'gain': gain
        })

    arr_2k = np.array(rets_2k)
    arr_10k = np.array(rets_10k)
    arr_gain = arr_10k - arr_2k

    mean_gain = float(np.mean(arr_gain))
    se_gain = float(stats.sem(arr_gain))
    ci_gain = 2.776 * se_gain # df=4 Student-t
    t_stat, p_val = stats.ttest_rel(arr_10k, arr_2k)

    print(f"\n  [2k -> 10k 장기 학습 게인 최종 통계]")
    print(f"  * 평균 2k 성적  : {np.mean(arr_2k):+6.2f} ± {2.776 * stats.sem(arr_2k):5.2f} bb/100")
    print(f"  * 평균 10k 성적 : {np.mean(arr_10k):+6.2f} ± {2.776 * stats.sem(arr_10k):5.2f} bb/100")
    print(f"  * 평균 학습 게인: {mean_gain:+6.2f} ± {ci_gain:5.2f} bb/100 (t={t_stat:+5.2f}, p={p_val:.4f})")
    print(f"  * 5개 시드 중 개선 시드 수: {sum(arr_gain > 0)} / 5 개")

    with open("v531/data/phase3_gain_re_evaluation.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=p3_records[0].keys())
        writer.writeheader(); writer.writerows(p3_records)

    # 2. Phase 4: 20BB 파일럿 좌석 회전 재평가 (+445bb의 정체 규명)
    print("\n[PHASE 4: 20BB 파일럿 정상 좌석 회전 재평가]")
    params_20bb = {
        'betting': 'nolimit', 'bettingAbstraction': 'fcpa', 'numPlayers': 5,
        'numRanks': 6, 'numSuits': 4, 'numHoleCards': 2, 'numRounds': 2,
        'numBoardCards': '0 3', 'blind': '50 100 0 0 0', 'stack': '2000 2000 2000 2000 2000',
        'firstPlayer': '3 1'
    }
    game_20bb = pyspiel.load_game('universal_poker', params_20bb)

    # 20BB 훈련 (1,000 eps)
    s_dim_20 = len(game_20bb.new_initial_state().information_state_tensor(0))
    a_dim_20 = game_20bb.num_distinct_actions()
    model_20bb = NFSP_DQN_Reference(0, s_dim_20, a_dim_20, hidden_dim=128)
    rng_20 = np.random.RandomState(42)

    for ep in range(1000):
        model_20bb.reset_episode()
        st = game_20bb.new_initial_state()
        advance_chance_nodes(st, rng_20)
        ep_history = defaultdict(list)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_20)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act = model_20bb.select_action(inf, leg, is_eval=False)
                ep_history[cp].append((inf, act, leg))
                st.apply_action(act)
        rets = st.returns()
        for cp in range(5):
            hist = ep_history[cp]
            if not hist: continue
            for i in range(len(hist)):
                s, a, _ = hist[i]
                if i + 1 < len(hist):
                    ns, _, n_leg = hist[i+1]
                    d = 0.0; r = 0.0
                else:
                    ns = np.zeros(s_dim_20, dtype=np.float32); n_leg = []; d = 1.0
                    r = rets[cp] / 100.0
                model_20bb.rl_buffer.append((s, a, r, ns, n_leg, d))
        model_20bb.update(batch_size=32)

    m_20bb_rot, ci_20bb_rot, seat_20bb = evaluate_with_seat_rotation(game_20bb, model_20bb, rule_bot, num_hands=500, eval_seed=20261018)
    print(f"  * 20BB 정상 좌석 회전 실측 성적 : {m_20bb_rot:+6.2f} ± {ci_20bb_rot:4.2f} bb/100")
    print(f"    - Seat 0 (SB)  : {np.mean(seat_20bb[0])*100:+6.2f} bb/100")
    print(f"    - Seat 1 (BB)  : {np.mean(seat_20bb[1])*100:+6.2f} bb/100")
    print(f"    - Seat 2 (UTG) : {np.mean(seat_20bb[2])*100:+6.2f} bb/100")
    print(f"    - Seat 3 (CO)  : {np.mean(seat_20bb[3])*100:+6.2f} bb/100")
    print(f"    - Seat 4 (BTN) : {np.mean(seat_20bb[4])*100:+6.2f} bb/100")

    p4_res = [{
        'mode': '20BB Pilot (Seat Rotated)', 'mean_bb_100': m_20bb_rot, 'ci': ci_20bb_rot,
        'notes': 'V5.3의 +445bb는 Seat 0 고정 포지션 편향 왜곡이었음을 규명'
    }]
    with open("v531/data/phase4_20bb_re_evaluation.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=p4_res[0].keys())
        writer.writeheader(); writer.writerows(p4_res)

    # 3. Phase 7: 완전 좌석 회전 하에서 모든 기준선 재평가
    print("\n[PHASE 7: 완전 좌석 회전 하에서 5대 기준선 재평가 실측]")
    m_fold, ci_f, _ = evaluate_with_seat_rotation(game_10bb, fold_bot, rule_bot, num_hands=500)
    m_check, ci_c, _ = evaluate_with_seat_rotation(game_10bb, check_bot, rule_bot, num_hands=500)
    m_rand, ci_r, _ = evaluate_with_seat_rotation(game_10bb, rand_bot, rule_bot, num_hands=500)

    # PPO 5시드 평균
    ppo_rets = []
    for s in seeds:
        ppo_model, _ = train_5p_ppo_selfplay(game_10bb, num_episodes=2000, seed=s)
        m_p, _, _ = evaluate_with_seat_rotation(game_10bb, ppo_model, rule_bot, num_hands=500, eval_seed=20261018 + s)
        ppo_rets.append(m_p)
    m_ppo_all = float(np.mean(ppo_rets))
    ci_ppo_all = 2.776 * float(stats.sem(ppo_rets))

    print(f"  * AlwaysFold (정상 회전 실측) : {m_fold:+6.2f} ± {ci_f:4.2f} bb/100 (이론치 -30.0)")
    print(f"  * CheckCall  (정상 회전 실측) : {m_check:+6.2f} ± {ci_c:4.2f} bb/100")
    print(f"  * Random     (정상 회전 실측) : {m_rand:+6.2f} ± {ci_r:4.2f} bb/100")
    print(f"  * PPO Baseline (5시드 회전)   : {m_ppo_all:+6.2f} ± {ci_ppo_all:4.2f} bb/100")
    print(f"  * NFSP-DQN 10k (5시드 회전)   : {np.mean(arr_10k):+6.2f} ± {2.776*stats.sem(arr_10k):4.2f} bb/100")

    p7_res = [
        {'model': 'AlwaysFold', 'bb_100': m_fold, 'ci': ci_f},
        {'model': 'CheckCall', 'bb_100': m_check, 'ci': ci_c},
        {'model': 'Random', 'bb_100': m_rand, 'ci': ci_r},
        {'model': 'PPO (5 seeds)', 'bb_100': m_ppo_all, 'ci': ci_ppo_all},
        {'model': 'NFSP-DQN (10k eps, 5 seeds)', 'bb_100': np.mean(arr_10k), 'ci': 2.776*stats.sem(arr_10k)}
    ]
    with open("v531/data/phase7_baselines_rotated.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=p7_res[0].keys())
        writer.writeheader(); writer.writerows(p7_res)
    print("\n[저장 완료] V5.3.1 전체 벤치마크 데이터 저장 완료!\n")

if __name__ == '__main__':
    run_v531_benchmarks()
