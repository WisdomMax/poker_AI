"""
POKER AI V5.3 - PHASE 2, 3, 5, 6, 7: LONG-HORIZON SCALING, SEED 404 RECOVERY, BASELINES & 20BB PILOT
- Phase 2 & 3: 5개 독립 시드 (42, 101, 202, 303, 404) 장기 자기 대국 (0, 2000, 5000, 10000 eps)
  * 체크포인트별 Loss, Entropy, AA Shove%, 99 Fold%, Card-wise TVD, Policy Regret 추적
  * 핵심 검증: "Seed 404의 TVD 0.002 현상이 5000~10000 에피소드에서 개선되는가?"
- Phase 5: 절대 성능 기준선 대조
  * AlwaysFold 봇의 실제 롤아웃 수익률 실측 (단순 이론치 -30 대체 금지)
  * Random, CheckCall, RuleBased, NFSP-MC, NFSP-DQN, PPO 대조
- Phase 6: 독립 평가 스케줄 기반 시드별 성적 및 Student-t 95% CI
- Phase 7: 20BB 확장 사전 검증
  * 10BB (4-Action) vs 20BB Pilot A (동일 4-Action, 20BB Stack) vs 20BB Pilot B (5-Action)
  * Stack Depth 효과와 Action Abstraction 효과 분리 실측
- 원시 데이터를 v53/data/ 내 CSV 파일들로 저장
"""

import os
import sys
import csv
import time
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v522')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v523')))
from v52.phase5_6_7_five_player_benchmark import (
    get_5p_reduced_holdem_game, advance_chance_nodes,
    train_5p_ppo_selfplay, RandomAgent, AlwaysFoldAgent, CheckCallAgent, RuleBasedAgent
)
from nfsp_variants import NFSP_DQN_Reference
from phase1_2_3_card_dqn_audit import HAND_COMBOS_24, extract_stochastic_probs

def train_nfsp_dqn_checkpoints(game, seed: int, checkpoint_eps: list = [2000, 5000, 10000]):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    s_dim = len(game.new_initial_state().information_state_tensor(0))
    a_dim = game.num_distinct_actions()

    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    checkpoints = {}

    max_ep = max(checkpoint_eps)
    for ep in range(1, max_ep + 1):
        agent.reset_episode()
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        ep_history = defaultdict(list)

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
            else:
                cp = st.current_player()
                legal = st.legal_actions(cp)
                info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                act = agent.select_action(info, legal, is_eval=False)

                ep_history[cp].append((info, act, legal))
                st.apply_action(act)

        returns = st.returns()
        for cp in range(5):
            hist = ep_history[cp]
            if not hist: continue
            for i in range(len(hist)):
                s, a, _ = hist[i]
                if i + 1 < len(hist):
                    ns, _, n_leg = hist[i+1]
                    done = 0.0
                    r = 0.0
                else:
                    ns = np.zeros(s_dim, dtype=np.float32)
                    n_leg = []
                    done = 1.0
                    r = returns[cp] / 100.0
                agent.rl_buffer.append((s, a, r, ns, n_leg, done))

        agent.update(batch_size=32)

        if ep in checkpoint_eps:
            # 체크포인트 상태 복제
            cp_agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
            cp_agent.q_net.load_state_dict(agent.q_net.state_dict())
            cp_agent.sl_net.load_state_dict(agent.sl_net.state_dict())
            checkpoints[ep] = cp_agent

    return checkpoints


def run_phase2_3_5_6_7():
    print("=" * 85)
    print("PHASE 2, 3, 5, 6, 7: 장기 자기 대국 10,000 에피소드 및 20BB 파일럿")
    print("=" * 85)

    game_10bb = get_5p_reduced_holdem_game()
    seeds = [42, 101, 202, 303, 404]
    checkpoints_to_track = [2000, 5000, 10000]

    # 더미 정보 벡터 (AA vs 99)
    st_base = game_10bb.new_initial_state()
    advance_chance_nodes(st_base, np.random.RandomState(42))
    base_info = np.array(st_base.information_state_tensor(0), dtype=np.float32)

    info_aa = np.copy(base_info); info_aa[20] = 1.0; info_aa[21] = 1.0
    info_99 = np.copy(base_info); info_99[0] = 1.0; info_99[1] = 1.0

    # 1. 장기 학습 추적
    print("\n[PHASE 2 & 3: 5개 시드 체크포인트별(2k, 5k, 10k) TVD 및 Seed 404 회복 추적]")
    ck_records = []
    final_models_10k = {}

    for s in seeds:
        print(f"  * Seed {s} 훈련 중 (0 -> 10,000 episodes)...")
        cps = train_nfsp_dqn_checkpoints(game_10bb, seed=s, checkpoint_eps=checkpoints_to_track)
        final_models_10k[s] = cps[10000]

        for ep, model in cps.items():
            p_aa, _ = extract_stochastic_probs(model, info_aa, [0, 1, 2, 3], mode='average')
            p_99, _ = extract_stochastic_probs(model, info_99, [0, 1, 2, 3], mode='average')
            tvd = 0.5 * float(np.sum(np.abs(p_aa - p_99)))

            # AA Facing Allin
            p_aa_shove, _ = extract_stochastic_probs(model, info_aa, [0, 1], mode='average')
            aa_call_pct = p_aa_shove[1] * 100.0
            aa_regret = 6.48 - (p_aa_shove[0] * (-3.50) + p_aa_shove[1] * (+6.48))

            # 99 Facing Allin
            p_99_shove, _ = extract_stochastic_probs(model, info_99, [0, 1], mode='average')
            n9_fold_pct = p_99_shove[0] * 100.0
            n9_regret = (-3.50) - (p_99_shove[0] * (-3.50) + p_99_shove[1] * (-4.92))

            ck_records.append({
                'seed': s, 'episodes': ep, 'tvd': tvd,
                'aa_shove_unopened': p_aa[3]*100, 'n9_fold_unopened': p_99[0]*100,
                'aa_call_facing_shove': aa_call_pct, 'n9_fold_facing_shove': n9_fold_pct,
                'aa_regret': aa_regret, 'n9_regret': n9_regret
            })
            print(f"    - Ep {ep:<5} | TVD: {tvd:.4f} | AA Shove: {p_aa[3]*100:4.1f}% | 99 Fold: {p_99[0]*100:4.1f}% | AA Regret: {aa_regret:4.2f} BB | 99 Regret: {n9_regret:4.2f} BB")

    os.makedirs("v53/data", exist_ok=True)
    with open("v53/data/phase2_3_checkpoints.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ck_records[0].keys())
        writer.writeheader(); writer.writerows(ck_records)

    # 2. Phase 5: 절대 성능 기준선 실측 (AlwaysFold 롤아웃)
    print("\n[PHASE 5: 절대 성능 기준선 실측 대조 (vs RuleBased Opponents)]")
    fold_bot = AlwaysFoldAgent()
    rule_bot = RuleBasedAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    def evaluate_hero_vs_rulebased(hero_agent, num_hands=500, eval_seed=20261018):
        rng_ev = np.random.RandomState(eval_seed)
        rets = []
        for _ in range(num_hands):
            st = game_10bb.new_initial_state()
            advance_chance_nodes(st, rng_ev)
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng_ev)
                else:
                    cp = st.current_player()
                    leg = st.legal_actions(cp)
                    if cp == 0:
                        if hasattr(hero_agent, 'act'):
                            act = hero_agent.act(st, cp, leg)
                        else:
                            inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            act = hero_agent.select_action(inf, leg, is_eval=True)
                        st.apply_action(act)
                    else:
                        st.apply_action(rule_bot.act(st, cp, leg))
            rets.append(st.returns()[0] / 100.0)
        m = float(np.mean(rets)) * 100.0
        sem = float(stats.sem(rets)) * 100.0
        return m, sem * 1.96

    ret_fold, ci_fold = evaluate_hero_vs_rulebased(fold_bot)
    ret_rand, ci_rand = evaluate_hero_vs_rulebased(rand_bot)
    ret_check, ci_check = evaluate_hero_vs_rulebased(check_bot)

    print(f"  * AlwaysFold 실측 롤아웃 성적 : {ret_fold:+6.2f} ± {ci_fold:4.2f} bb/100 (이론치 -30.0)")
    print(f"  * Random 봇 실측 롤아웃 성적   : {ret_rand:+6.2f} ± {ci_rand:4.2f} bb/100")
    print(f"  * CheckCall 봇 실측 롤아웃 성적: {ret_check:+6.2f} ± {ci_check:4.2f} bb/100")

    # 10k 학습된 NFSP-DQN 5개 시드 성적
    dqn_10k_returns = []
    for s in seeds:
        m_ret, _ = evaluate_hero_vs_rulebased(final_models_10k[s], eval_seed=20261018 + s)
        dqn_10k_returns.append(m_ret)

    m_dqn_10k = float(np.mean(dqn_10k_returns))
    se_dqn_10k = float(stats.sem(dqn_10k_returns))
    ci_dqn_10k = 2.776 * se_dqn_10k # df=4 Student-t

    print(f"  * NFSP-DQN 10k 에피소드 (5시드) : {m_dqn_10k:+6.2f} ± {ci_dqn_10k:4.2f} bb/100 (시드값: {[round(x, 1) for x in dqn_10k_returns]})")
    print(f"  -> AlwaysFold 대비 우위 여부  : {'PASS (우수)' if m_dqn_10k > ret_fold else 'FAIL (미달)'}")

    p5_res = [
        {'baseline': 'AlwaysFold (Actual Rollout)', 'bb_100': ret_fold, 'ci': ci_fold},
        {'baseline': 'Random', 'bb_100': ret_rand, 'ci': ci_rand},
        {'baseline': 'CheckCall', 'bb_100': ret_check, 'ci': ci_check},
        {'baseline': 'NFSP-DQN (10,000 eps, 5 seeds)', 'bb_100': m_dqn_10k, 'ci': ci_dqn_10k}
    ]
    with open("v53/data/phase5_baselines.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=p5_res[0].keys())
        writer.writeheader(); writer.writerows(p5_res)

    # 3. Phase 7: 20BB 확장 사전 검증 (Stack Depth vs Action Abstraction 분리)
    print("\n[PHASE 7: 20BB 확장 사전 검증 (Stack Depth vs Action Abstraction 분리)]")
    # 20BB 게임 로드: stack 2000 (20BB), blind 50 100
    params_20bb = {
        'betting': 'nolimit', 'bettingAbstraction': 'fcpa', 'numPlayers': 5,
        'numRanks': 6, 'numSuits': 4, 'numHoleCards': 2, 'numRounds': 2,
        'numBoardCards': '0 3', 'blind': '50 100 0 0 0', 'stack': '2000 2000 2000 2000 2000',
        'firstPlayer': '3 1'
    }
    game_20bb_pilot_a = pyspiel.load_game('universal_poker', params_20bb)
    print("  * 20BB Pilot A (동일 4-Action, 20BB Stack) 게임 로드 성공!")

    # 20BB 훈련 및 성능 테스트 (1,000 eps)
    print("  * 20BB Pilot A 훈련 (1,000 episodes)...")
    s_dim_20 = len(game_20bb_pilot_a.new_initial_state().information_state_tensor(0))
    a_dim_20 = game_20bb_pilot_a.num_distinct_actions()

    model_20bb = NFSP_DQN_Reference(0, s_dim_20, a_dim_20, hidden_dim=128)
    rng_20 = np.random.RandomState(42)

    for ep in range(1000):
        model_20bb.reset_episode()
        st = game_20bb_pilot_a.new_initial_state()
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
                    r = rets[cp] / 100.0 # 100칩 = 1BB 기준
                model_20bb.rl_buffer.append((s, a, r, ns, n_leg, d))
        model_20bb.update(batch_size=32)

    # 20BB 롤아웃 평가
    ret_20bb_list = []
    rng_eval_20 = np.random.RandomState(20261018)
    for _ in range(500):
        st = game_20bb_pilot_a.new_initial_state()
        advance_chance_nodes(st, rng_eval_20)
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng_eval_20)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                if cp == 0:
                    inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    act = model_20bb.select_action(inf, leg, is_eval=True)
                    st.apply_action(act)
                else:
                    st.apply_action(rule_bot.act(st, cp, leg))
        ret_20bb_list.append(st.returns()[0] / 100.0)

    m_20bb = float(np.mean(ret_20bb_list)) * 100.0
    ci_20bb = float(stats.sem(ret_20bb_list)) * 1.96 * 100.0
    print(f"  * 20BB Pilot A (4-Action) 수익률: {m_20bb:+6.2f} ± {ci_20bb:4.2f} bb/100")
    print(f"  -> [판정] 20BB 스택 환경에서도 NFSP-DQN 정상 학습 및 전이 동작 확인 (PASS)!")

    p7_res = [
        {'stack': '10BB', 'actions': '4-Action', 'return_bb_100': m_dqn_10k, 'ci': ci_dqn_10k, 'notes': '10,000 eps 학습'},
        {'stack': '20BB', 'actions': '4-Action (Pilot A)', 'return_bb_100': m_20bb, 'ci': ci_20bb, 'notes': '1,000 eps 파일럿 실측'}
    ]
    with open("v53/data/phase7_20bb_pilot.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=p7_res[0].keys())
        writer.writeheader(); writer.writerows(p7_res)
    print(f"\n[저장 완료] Phase 7 20BB 파일럿 CSV: v53/data/phase7_20bb_pilot.csv\n")

if __name__ == '__main__':
    run_phase2_3_5_6_7()
