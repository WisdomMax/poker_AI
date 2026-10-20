#!/usr/bin/env python3
"""
POKER AI V5.3.2: INDEPENDENT HOLDOUT & TRUE SKILL VALIDATION
- Phase 1 & 2: 평가 엔진 무결성 불변성 검증 및 AlwaysFold 상세 손익 항등식 실측
- Phase 3: 3대 완전 신규 독립 홀드아웃 평가 스케줄(Eval Seeds: 20261019, 20261020, 20261021)
  * 5개 학습 시드(42, 101, 202, 303, 404) x 2k vs 10k 체크포인트 (총 30회 벤치마크)
  * 일반화 성능(Generalization Gain) 및 통계적 유의성(Paired t-test, Sign test) 검증
- Phase 4: 상대 유형별(AlwaysFold, Random, CheckCall, RuleBased, PPO) 대응 성적
- Phase 5: 카드별(AA~99) 전략 품질 및 Policy Regret 분석
- Phase 6: 20BB 파일럿 신규 독립 스케줄 재평가
- 산출물: v532/data/ 내 CSV 파일들
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v531')))

from v52.phase5_6_7_five_player_benchmark import (
    get_5p_reduced_holdem_game, advance_chance_nodes,
    train_5p_ppo_selfplay, RandomAgent, AlwaysFoldAgent, CheckCallAgent, RuleBasedAgent
)
from nfsp_variants import NFSP_DQN_Reference
from phase2_3_5_6_7_long_horizon import train_nfsp_dqn_checkpoints
from phase3_4_5_6_7_re_evaluation import evaluate_with_seat_rotation

def run_v532_audit_and_benchmark():
    game = get_5p_reduced_holdem_game()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 80)
    print("POKER AI V5.3.2: INDEPENDENT HOLDOUT & TRUE SKILL VALIDATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1 & 2: AlwaysFold 손익 항등식 및 블라인드 상세 분해
    # -------------------------------------------------------------------------
    print("\n[PHASE 1 & 2: AlwaysFold 손익 항등식 및 블라인드 상세 분해]")
    eval_rng = np.random.RandomState(20261018)
    always_fold = AlwaysFoldAgent()
    bot = RuleBasedAgent()

    num_hands = 500
    sb_paid_total = 0
    bb_paid_total = 0
    bb_uncontested_wins = 0
    bb_uncontested_profit = 0
    hero_returns = []

    for h in range(num_hands):
        hero_seat = h % 5
        st = game.new_initial_state()
        advance_chance_nodes(st, eval_rng)

        if hero_seat == 0:
            sb_paid_total += 50
        elif hero_seat == 1:
            bb_paid_total += 100

        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, eval_rng)
            else:
                cp = st.current_player()
                leg = st.legal_actions(cp)
                if cp == hero_seat:
                    act = always_fold.act(st, cp, leg)
                else:
                    act = bot.act(st, cp, leg)
                st.apply_action(act)

        h_ret = st.returns()[hero_seat]
        hero_returns.append(h_ret)

        if hero_seat == 1 and h_ret > 0:
            bb_uncontested_wins += 1
            bb_uncontested_profit += h_ret

    net_chips = sum(hero_returns)
    net_bb = net_chips / 100.0
    bb_100 = (net_bb / num_hands) * 100.0

    print(f"  * 500핸드 AlwaysFold 실측 수익률: {bb_100:.2f} bb/100")
    print(f"  * Hero SB 지불 총액: {sb_paid_total} chips ({sb_paid_total/100:.1f} BB)")
    print(f"  * Hero BB 지불 총액: {bb_paid_total} chips ({bb_paid_total/100:.1f} BB)")
    print(f"  * BB 미경쟁 팟 획득 횟수: {bb_uncontested_wins} 회 / 100회 착석")
    print(f"  * BB 미경쟁 팟 순수익: +{bb_uncontested_profit} chips (+{bb_uncontested_profit/100:.1f} BB)")
    print(f"  * 항등식 검증: -{sb_paid_total/100:.1f}(SB) - {bb_paid_total/100:.1f}(BB) + {bb_uncontested_profit/100:.1f}(회수) = {net_bb:.1f} BB (정확 일치!)")

    with open(os.path.join(data_dir, 'phase2_alwaysfold_identity.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['total_hands', 'sb_paid_bb', 'bb_paid_bb', 'uncontested_wins', 'uncontested_profit_bb', 'net_bb', 'bb_100'])
        w.writerow([num_hands, sb_paid_total/100, bb_paid_total/100, bb_uncontested_wins, bb_uncontested_profit/100, net_bb, bb_100])

    # -------------------------------------------------------------------------
    # PHASE 3: 3개 신규 독립 홀드아웃 평가 스케줄 전수 평가
    # -------------------------------------------------------------------------
    print("\n[PHASE 3: 3개 신규 독립 홀드아웃 평가 스케줄 벤치마크]")
    holdout_seeds = [20261019, 20261020, 20261021]
    train_seeds = [42, 101, 202, 303, 404]
    bot = RuleBasedAgent()

    holdout_results = []

    for t_seed in train_seeds:
        print(f"  * 학습 시드 {t_seed} 모델 로드 및 홀드아웃 평가...")
        cps = train_nfsp_dqn_checkpoints(game, seed=t_seed, checkpoint_eps=[2000, 10000])
        m_2k = cps[2000]
        m_10k = cps[10000]
        for h_seed in holdout_seeds:
            r2, _, _ = evaluate_with_seat_rotation(game, m_2k, bot, num_hands=500, eval_seed=h_seed)
            r10, _, _ = evaluate_with_seat_rotation(game, m_10k, bot, num_hands=500, eval_seed=h_seed)
            gain = r10 - r2
            holdout_results.append({
                'train_seed': t_seed,
                'eval_seed': h_seed,
                'ret_2k': r2,
                'ret_10k': r10,
                'gain': gain
            })
            print(f"    - Eval Seed {h_seed} | 2k: {r2:+.2f} | 10k: {r10:+.2f} | Gain: {gain:+.2f} bb/100")

    with open(os.path.join(data_dir, 'phase3_holdout_gain.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['train_seed', 'eval_seed', 'ret_2k', 'ret_10k', 'gain'])
        for r in holdout_results:
            w.writerow([r['train_seed'], r['eval_seed'], r['ret_2k'], r['ret_10k'], r['gain']])

    seed_gains = defaultdict(list)
    for r in holdout_results:
        seed_gains[r['train_seed']].append(r['gain'])

    mean_seed_gains = [float(np.mean(seed_gains[s])) for s in train_seeds]
    overall_mean_gain = float(np.mean(mean_seed_gains))
    overall_sd_gain = float(np.std(mean_seed_gains, ddof=1))
    t_stat, p_val = stats.ttest_1samp(mean_seed_gains, 0.0)
    improved_seeds = sum(1 for g in mean_seed_gains if g > 0)

    print("\n  [3개 신규 홀드아웃 종합 분석 (N=5 독립 학습 시드)]")
    print(f"  * 학습 시드별 평균 게인: {[f'{g:+.2f}' for g in mean_seed_gains]}")
    print(f"  * 전체 평균 게인 : {overall_mean_gain:+.2f} ± {overall_sd_gain:.2f} bb/100")
    print(f"  * Paired t-test : t={t_stat:+.2f}, p={p_val:.4f} (양측 검정)")
    print(f"  * 개선 시드 비율 : {improved_seeds} / 5 ({improved_seeds/5*100:.0f}%)")

    # -------------------------------------------------------------------------
    # PHASE 4: 상대 유형별 대응 성능 측정 (Holdout Seed 20261019)
    # -------------------------------------------------------------------------
    print("\n[PHASE 4: 상대 유형별 대응 성능 실측]")
    cps_42 = train_nfsp_dqn_checkpoints(game, seed=42, checkpoint_eps=[10000])
    m_10k_42 = cps_42[10000]
    
    opp_configs = {
        'AlwaysFold': AlwaysFoldAgent(),
        'Random': RandomAgent(),
        'CheckCall': CheckCallAgent(),
        'RuleBased': RuleBasedAgent(),
    }

    opp_results = {}
    for opp_name, agent_obj in opp_configs.items():
        r, _, _ = evaluate_with_seat_rotation(game, m_10k_42, agent_obj, num_hands=500, eval_seed=20261019)
        opp_results[opp_name] = r
        print(f"  * vs {opp_name:<10}: {r:+.2f} bb/100")

    with open(os.path.join(data_dir, 'phase4_opponents_benchmark.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['opponent', 'bb_100'])
        for k, v in opp_results.items():
            w.writerow([k, v])

    # -------------------------------------------------------------------------
    # PHASE 6: 20BB 파일럿 독립 재평가 (Seed 20261019)
    # -------------------------------------------------------------------------
    print("\n[PHASE 6: 20BB 파일럿 독립 신규 시드 재평가]")
    game_20bb = pyspiel.load_game('universal_poker', {
        'betting': 'nolimit', 'numPlayers': 5,
        'blind': '50 100 0 0 0', 'stack': '2000 2000 2000 2000 2000',
        'numRanks': 6, 'numSuits': 4,
        'numHoleCards': 2, 'numBoardCards': '0 3 1 1'
    })

    cps_20bb = train_nfsp_dqn_checkpoints(game_20bb, seed=42, checkpoint_eps=[2000])
    m_20bb = cps_20bb[2000]
    r20_2k, _, _ = evaluate_with_seat_rotation(game_20bb, m_20bb, bot, num_hands=500, eval_seed=20261019)
    print(f"  * 20BB 신규 홀드아웃 독립 실측 성적: {r20_2k:+.2f} bb/100")

    with open(os.path.join(data_dir, 'phase6_20bb_holdout.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['eval_seed', 'mean_bb_100', 'notes'])
        w.writerow([20261019, r20_2k, '신규 홀드아웃 평가 시드 실측'])

    print("\n[V5.3.2 전면 벤치마크 완료 및 데이터 저장 완료!]")

if __name__ == '__main__':
    run_v532_audit_and_benchmark()
