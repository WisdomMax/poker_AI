"""
POKER AI V5.2.2 - PHASE 6: COMMON RANDOM NUMBERS (CRN) CROSS-PLAY BENCHMARK
- V5.2.1의 500판 분산 문제 해결:
  * 공통 난수(Common Random Numbers, CRN) 적용:
    동일한 카드 덱 분배(Hero 카드 및 보드 카드)와 동일한 좌석 조건에서 각 모델(Hero)을 대입하여 대응 차이(Paired Difference)를 직접 측정
  * 판 수: 각 매치업당 1,000판 (총 수천 판)
- 비교 Hero 모델:
  1. NFSP-MC (기존 MC Q 회귀)
  2. NFSP-DQN (원 논문 Reference)
  3. PPO (Baseline)
- 상대 봇 구성:
  1. vs 4 Random Bots
  2. vs 4 CheckCall Bots
  3. vs 4 RuleBased Bots
  4. vs 4 PPO Bots
  5. vs 4 NFSP Bots
- 측정 지표:
  * bb/100 및 95% CI
  * 승률 (Win Rate %)
  * 시드별 분산 (Seed 42, 101, 202)
  * Paired t-test (NFSP-MC vs PPO, NFSP-DQN vs PPO)
- 원시 데이터를 v522/data/phase6_cross_play_crn.csv에 저장
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
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v52.phase5_6_7_five_player_benchmark import (
    get_5p_reduced_holdem_game, advance_chance_nodes,
    train_5p_ppo_selfplay, RandomAgent, CheckCallAgent, RuleBasedAgent
)
from phase3_4_5_nfsp_comparison import train_5p_holdem

def evaluate_crn_hands(
    game,
    hero_models: dict, # {'NFSP-MC': model, 'NFSP-DQN': model, 'PPO': model}
    opponents: list,
    num_hands: int = 1000,
    crn_seed: int = 20261018
):
    """
    공통 난수(CRN)를 사용하여 동일한 카드 시퀀스를 각 모델에 정확히 동일하게 부여
    """
    hero_returns = {name: [] for name in hero_models.keys()}
    hero_wins = {name: [] for name in hero_models.keys()}

    for h in range(num_hands):
        hand_seed = crn_seed + h * 997

        for m_name, model in hero_models.items():
            rng = np.random.RandomState(hand_seed)
            st = game.new_initial_state()
            advance_chance_nodes(st, rng)

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                else:
                    cp = st.current_player()
                    legal = st.legal_actions(cp)

                    if cp == 0: # Hero
                        info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        if m_name == 'PPO':
                            x = torch.tensor(info).unsqueeze(0)
                            mask = torch.zeros(4, dtype=torch.bool)
                            mask[legal] = True
                            with torch.no_grad():
                                _, probs, _ = model(x, mask.unsqueeze(0))
                                act = torch.distributions.Categorical(probs).sample().item()
                        elif m_name in ['NFSP-MC', 'NFSP-DQN']:
                            act = model.select_action(info, legal, is_eval=True)

                        st.apply_action(act)
                    else: # Opponents
                        opp = opponents[cp - 1]
                        if hasattr(opp, 'act'):
                            act = opp.act(st, cp, legal)
                        elif hasattr(opp, 'select_action'):
                            info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            act = opp.select_action(info, legal, is_eval=True)
                        else: # PPO Opponent
                            info = np.array(st.information_state_tensor(cp), dtype=np.float32)
                            x = torch.tensor(info).unsqueeze(0)
                            mask = torch.zeros(4, dtype=torch.bool)
                            mask[legal] = True
                            with torch.no_grad():
                                _, probs, _ = opp(x, mask.unsqueeze(0))
                                act = torch.distributions.Categorical(probs).sample().item()
                        st.apply_action(act)

            ret_bb = st.returns()[0] / 100.0
            hero_returns[m_name].append(ret_bb)
            hero_wins[m_name].append(1.0 if ret_bb > 0 else 0.0)

    summary = {}
    for m_name in hero_models.keys():
        rets = hero_returns[m_name]
        m_bb = float(np.mean(rets)) * 100.0
        sem = float(np.std(rets, ddof=1) / np.sqrt(num_hands)) * 100.0
        ci = float(stats.t.ppf(0.975, df=num_hands-1) * sem)
        wr = float(np.mean(hero_wins[m_name])) * 100.0
        summary[m_name] = {
            'bb_100': m_bb,
            'ci': ci,
            'win_rate': wr,
            'raw_returns': rets
        }

    # Paired t-test
    t_mc_ppo, p_mc_ppo = stats.ttest_rel(hero_returns['NFSP-MC'], hero_returns['PPO'])
    t_dqn_ppo, p_dqn_ppo = stats.ttest_rel(hero_returns['NFSP-DQN'], hero_returns['PPO'])
    t_mc_dqn, p_mc_dqn = stats.ttest_rel(hero_returns['NFSP-MC'], hero_returns['NFSP-DQN'])

    paired_stats = {
        'mc_vs_ppo_p': float(p_mc_ppo),
        'dqn_vs_ppo_p': float(p_dqn_ppo),
        'mc_vs_dqn_p': float(p_mc_dqn)
    }

    return summary, paired_stats

def run_phase6():
    print("=" * 85)
    print("PHASE 6: 공통 난수(CRN) 1,000판 대응 교차 대국 벤치마크")
    print("=" * 85)

    game = get_5p_reduced_holdem_game()

    print("\n[1. 모델 준비: NFSP-MC, NFSP-DQN, PPO (Seed 42 학습)]")
    mc_agent, _, _, _ = train_5p_holdem('NFSP-MC', num_episodes=2000, seed=42)
    dqn_agent, _, _, _ = train_5p_holdem('NFSP-DQN', num_episodes=2000, seed=42)
    ppo_agent, _ = train_5p_ppo_selfplay(game, num_episodes=2000, seed=42)

    hero_models = {
        'NFSP-MC': mc_agent,
        'NFSP-DQN': dqn_agent,
        'PPO': ppo_agent
    }

    rule_bot = RuleBasedAgent()
    check_bot = CheckCallAgent()
    rand_bot = RandomAgent()

    matchups = [
        ("vs 4 RuleBased Bots", [rule_bot]*4),
        ("vs 4 CheckCall Bots", [check_bot]*4),
        ("vs 4 Random Bots",    [rand_bot]*4),
        ("vs 4 PPO Opponents",  [ppo_agent]*4),
        ("vs 4 NFSP Opponents", [mc_agent]*4)
    ]

    all_rows = []

    print(f"\n{'상대 구성':<22} | {'NFSP-MC (bb/100)':<22} | {'NFSP-DQN (bb/100)':<22} | {'PPO (bb/100)':<22} | Paired p-val (MC vs PPO)")
    print("-" * 115)

    for desc, opp_list in matchups:
        summary, paired = evaluate_crn_hands(game, hero_models, opp_list, num_hands=1000, crn_seed=20261018)

        str_mc = f"{summary['NFSP-MC']['bb_100']:+6.2f} ± {summary['NFSP-MC']['ci']:4.2f} ({summary['NFSP-MC']['win_rate']:4.1f}%)"
        str_dqn = f"{summary['NFSP-DQN']['bb_100']:+6.2f} ± {summary['NFSP-DQN']['ci']:4.2f} ({summary['NFSP-DQN']['win_rate']:4.1f}%)"
        str_ppo = f"{summary['PPO']['bb_100']:+6.2f} ± {summary['PPO']['ci']:4.2f} ({summary['PPO']['win_rate']:4.1f}%)"

        p_str = f"p = {paired['mc_vs_ppo_p']:.4f}"

        print(f"{desc:<22} | {str_mc:<22} | {str_dqn:<22} | {str_ppo:<22} | {p_str}")

        all_rows.append({
            'opponent': desc,
            'mc_bb_100': summary['NFSP-MC']['bb_100'],
            'mc_ci': summary['NFSP-MC']['ci'],
            'mc_wr': summary['NFSP-MC']['win_rate'],
            'dqn_bb_100': summary['NFSP-DQN']['bb_100'],
            'dqn_ci': summary['NFSP-DQN']['ci'],
            'dqn_wr': summary['NFSP-DQN']['win_rate'],
            'ppo_bb_100': summary['PPO']['bb_100'],
            'ppo_ci': summary['PPO']['ci'],
            'ppo_wr': summary['PPO']['win_rate'],
            'p_mc_vs_ppo': paired['mc_vs_ppo_p'],
            'p_dqn_vs_ppo': paired['dqn_vs_ppo_p']
        })

    os.makedirs("v522/data", exist_ok=True)
    csv_p6 = "v522/data/phase6_cross_play_crn.csv"
    with open(csv_p6, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[저장 완료] Phase 6 CRN 교차 대국 CSV: {csv_p6}\n")

if __name__ == '__main__':
    run_phase6()
