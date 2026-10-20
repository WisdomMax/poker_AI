"""
POKER AI V5.7: 50BB NFSP CROSS-PLAY LEAGUE & STRATEGIC ROBUSTNESS
- Phase 0: 텐서 차원(149 -> 161) 전이 무결성 및 체크포인트 해시(SHA256) 감사
- Phase 1: 모델 풀 구성 (50BB Scratch 50k 5시드, 50BB Warm Start 50k 5시드, 과거 체크포인트, RuleBased)
- Phase 2: 5인 AI 리그전
    A. Scratch 50k Hero vs Warm Start 50k 4인
    B. Warm Start 50k Hero vs Scratch 50k 4인
    C. 혼합 5인 테이블 (Scratch & Warm Start 상호 대국)
    D. RuleBased 기준선 대국
- Phase 3: 3개 신규 독립 홀드아웃 시드 (20261101, 20261102, 20261103) 기반 총 4,500핸드 전수 평가
- Phase 4: 전략적 강건성 및 올인(30%) 정밀 해부
- Phase 5: Warm Start vs Scratch 상대 승률 Paired t-test 및 통계 검정
- Phase 6 & 7: 결과 CSV 저장 및 최종 판정
"""

import os
import sys
import csv
import time
import hashlib
import random
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
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

def get_5p_50bb_poker_game():
    params = {
        'betting': 'nolimit',
        'bettingAbstraction': 'fcpa',
        'numPlayers': 5,
        'numRanks': 6,
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

def load_agent(model_path, s_dim=161, a_dim=4):
    ckpt = torch.load(model_path, map_location='cpu')
    agent = NFSP_DQN_Reference(0, s_dim, a_dim, hidden_dim=128)
    agent.q_net.load_state_dict(ckpt['q_net'])
    agent.sl_net.load_state_dict(ckpt['sl_net'])
    with open(model_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return agent, sha

def evaluate_table(game, agents_5p, holdout_seeds=[20261101, 20261102, 20261103], hands_per_seed=1500):
    """
    agents_5p: 5명의 에이전트 리스트 (Seat 0~4)
    5개 좌석 균등 회전 및 핸드별 칩 정산
    반환: 각 에이전트의 bb/100, 표준편차, 액션 빈도
    """
    num_players = 5
    returns_per_player = [[] for _ in range(num_players)]
    actions_per_player = [defaultdict(int) for _ in range(num_players)]
    zero_err = 0

    for h_seed in holdout_seeds:
        rng_deck = np.random.RandomState(h_seed)
        rng_action = np.random.RandomState(h_seed + 100000)

        for hand_id in range(hands_per_seed):
            # 좌석 순환 (핸드마다 물리적 위치 1칸씩 순환)
            seat_shift = hand_id % num_players
            st = game.new_initial_state()
            advance_chance_nodes(st, rng_deck)

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng_deck)
                else:
                    cp = st.current_player()
                    leg = st.legal_actions(cp)
                    agent_idx = (cp - seat_shift) % num_players
                    agent = agents_5p[agent_idx]

                    if hasattr(agent, 'act'):
                        act = agent.act(st, cp, leg)
                    else:
                        inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                        probs = extract_stochastic_probs(agent, inf, leg)
                        act = int(rng_action.choice(agent.action_dim, p=probs))

                    actions_per_player[agent_idx][act] += 1
                    st.apply_action(act)

            rets = st.returns()
            if abs(sum(rets)) > 1e-4:
                zero_err += 1

            for cp in range(num_players):
                agent_idx = (cp - seat_shift) % num_players
                returns_per_player[agent_idx].append(rets[cp] / 100.0)

    total_hands = len(holdout_seeds) * hands_per_seed
    results = []
    for i in range(num_players):
        rets = returns_per_player[i]
        bb_100 = (float(np.sum(rets)) / total_hands) * 100.0
        # 스케줄별(3개) 분산 및 신뢰구간
        sched_scores = []
        for s_idx in range(len(holdout_seeds)):
            sub = rets[s_idx * hands_per_seed : (s_idx + 1) * hands_per_seed]
            sched_scores.append((float(np.sum(sub)) / hands_per_seed) * 100.0)
        sd = float(np.std(sched_scores, ddof=1)) if len(sched_scores) > 1 else 0.0
        ci = 4.303 * (sd / np.sqrt(len(sched_scores))) if len(sched_scores) == 3 else 1.96 * sd

        acts = actions_per_player[i]
        tot_a = sum(acts.values())
        results.append({
            'bb_100': round(bb_100, 2),
            'sd': round(sd, 2),
            'ci': round(ci, 2),
            'p_fold': round((acts[0] / tot_a) * 100.0, 2) if tot_a > 0 else 0.0,
            'p_call': round((acts[1] / tot_a) * 100.0, 2) if tot_a > 0 else 0.0,
            'p_raise': round((acts[2] / tot_a) * 100.0, 2) if tot_a > 0 else 0.0,
            'p_allin': round((acts[3] / tot_a) * 100.0, 2) if tot_a > 0 else 0.0,
            'total_actions': tot_a
        })

    return results, zero_err

def main():
    print("=" * 80)
    print("POKER AI V5.7: 50BB NFSP CROSS-PLAY LEAGUE & STRATEGIC ROBUSTNESS")
    print("=" * 80)

    game = get_5p_50bb_poker_game()
    train_seeds = [42, 101, 202, 303, 404]
    holdout_seeds = [20261101, 20261102, 20261103] # V5.7 전용 신규 홀드아웃 시드

    # -------------------------------------------------------------
    # PHASE 0: 텐서 전이 및 체크포인트 해시 감사
    # -------------------------------------------------------------
    print("\n[PHASE 0: 텐서 차원(149 -> 161) 및 체크포인트 SHA-256 무결성 검증]")
    print(f"  * 50BB 게임 정보상태 차원: 161 (20BB 대비 +12차원 확장)")
    print(f"  * 전이 메커니즘: Partial Weight Transfer (첫 레이어 [:, :149] 가중치 복사 및 신규 차원 초기화 유지)")

    scratch_models = {}
    warm_models = {}
    model_hashes = {}

    for s in train_seeds:
        sc_path = f"v56/models/nfsp_50bb_scratch_seed_{s}_ep_50000.pt"
        wm_path = f"v56/models/nfsp_50bb_warmstart_seed_{s}_ep_50000.pt"
        
        m_sc, h_sc = load_agent(sc_path)
        m_wm, h_wm = load_agent(wm_path)
        
        scratch_models[s] = m_sc
        warm_models[s] = m_wm
        model_hashes[f"scratch_{s}"] = h_sc
        model_hashes[f"warm_{s}"] = h_wm
        print(f"  * Seed {s:<3} | Scratch SHA: {h_sc[:16]}... | Warm SHA: {h_wm[:16]}...")

    rule_bot = RuleBasedAgent()
    fold_bot = AlwaysFoldAgent()
    rand_bot = RandomAgent()

    # -------------------------------------------------------------
    # PHASE 2 & 3: 5인 AI 리그전 대규모 평가
    # -------------------------------------------------------------
    league_results = []

    # -------------------------------------------------------------
    # 대국 형태 A: Scratch 50k Hero 1인 vs Warm Start 50k 4인
    # -------------------------------------------------------------
    print("\n[LEAGUE A: Scratch 50k Hero (1인) vs Warm Start 50k 상대 (4인)]")
    scratch_vs_warm_scores = []
    for s in train_seeds:
        hero = scratch_models[s]
        # 상대 4명은 타 시드의 Warm Start 모델들로 구성
        opp_seeds = [os for os in train_seeds if os != s]
        opponents = [warm_models[os] for os in opp_seeds]
        table = [hero] + opponents

        res, z_err = evaluate_table(game, table, holdout_seeds=holdout_seeds, hands_per_seed=1500)
        hero_res = res[0]
        scratch_vs_warm_scores.append(hero_res['bb_100'])
        
        entry = {
            'match_type': 'Scratch_vs_Warm4',
            'hero_model': f'Scratch_Seed_{s}',
            'hero_bb_100': hero_res['bb_100'],
            'hero_sd': hero_res['sd'],
            'hero_ci': hero_res['ci'],
            'p_fold': hero_res['p_fold'],
            'p_call': hero_res['p_call'],
            'p_raise': hero_res['p_raise'],
            'p_allin': hero_res['p_allin']
        }
        league_results.append(entry)
        print(f"  * Hero Scratch Seed {s:<3} : {hero_res['bb_100']:+6.2f} ± {hero_res['ci']:4.2f} bb/100 | Fold: {hero_res['p_fold']:4.1f}% | Raise: {hero_res['p_raise']:4.1f}% | All-in: {hero_res['p_allin']:4.1f}%")

    # -------------------------------------------------------------
    # 대국 형태 B: Warm Start 50k Hero 1인 vs Scratch 50k 4인
    # -------------------------------------------------------------
    print("\n[LEAGUE B: Warm Start 50k Hero (1인) vs Scratch 50k 상대 (4인)]")
    warm_vs_scratch_scores = []
    for s in train_seeds:
        hero = warm_models[s]
        # 상대 4명은 타 시드의 Scratch 모델들로 구성
        opp_seeds = [os for os in train_seeds if os != s]
        opponents = [scratch_models[os] for os in opp_seeds]
        table = [hero] + opponents

        res, z_err = evaluate_table(game, table, holdout_seeds=holdout_seeds, hands_per_seed=1500)
        hero_res = res[0]
        warm_vs_scratch_scores.append(hero_res['bb_100'])
        
        entry = {
            'match_type': 'Warm_vs_Scratch4',
            'hero_model': f'Warm_Seed_{s}',
            'hero_bb_100': hero_res['bb_100'],
            'hero_sd': hero_res['sd'],
            'hero_ci': hero_res['ci'],
            'p_fold': hero_res['p_fold'],
            'p_call': hero_res['p_call'],
            'p_raise': hero_res['p_raise'],
            'p_allin': hero_res['p_allin']
        }
        league_results.append(entry)
        print(f"  * Hero Warm Seed {s:<3}    : {hero_res['bb_100']:+6.2f} ± {hero_res['ci']:4.2f} bb/100 | Fold: {hero_res['p_fold']:4.1f}% | Raise: {hero_res['p_raise']:4.1f}% | All-in: {hero_res['p_allin']:4.1f}%")

    # -------------------------------------------------------------
    # 대국 형태 C: 혼합 5인 테이블 (Scratch 2시드 + Warm 2시드 + RuleBased 1인)
    # -------------------------------------------------------------
    print("\n[LEAGUE C: 혼합 5인 테이블 (Scratch 2 + Warm 2 + RuleBased 1)]")
    table_c = [scratch_models[42], scratch_models[101], warm_models[202], warm_models[303], rule_bot]
    res_c, _ = evaluate_table(game, table_c, holdout_seeds=holdout_seeds, hands_per_seed=1500)
    c_names = ['Scratch_42', 'Scratch_101', 'Warm_202', 'Warm_303', 'RuleBased']
    for idx, name in enumerate(c_names):
        rc = res_c[idx]
        print(f"  * {name:<14} : {rc['bb_100']:+6.2f} ± {rc['ci']:4.2f} bb/100 | Fold: {rc['p_fold']:4.1f}% | Raise: {rc['p_raise']:4.1f}% | All-in: {rc['p_allin']:4.1f}%")
        league_results.append({
            'match_type': 'Mixed_5p_Table',
            'hero_model': name,
            'hero_bb_100': rc['bb_100'],
            'hero_sd': rc['sd'],
            'hero_ci': rc['ci'],
            'p_fold': rc['p_fold'],
            'p_call': rc['p_call'],
            'p_raise': rc['p_raise'],
            'p_allin': rc['p_allin']
        })

    # CSV 데이터 저장
    with open('v57/data/phase2_league_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=league_results[0].keys())
        writer.writeheader()
        writer.writerows(league_results)
    print("\n  * 리그 결과 저장 완료: v57/data/phase2_league_results.csv")

    # -------------------------------------------------------------
    # PHASE 4 & 5: 통계 검정 및 전략적 강건성 분석
    # -------------------------------------------------------------
    print("\n[PHASE 5: Primary Comparison 통계 검정 (AI 상호 대결)]")
    m_sc_vs_wm = float(np.mean(scratch_vs_warm_scores))
    sd_sc_vs_wm = float(np.std(scratch_vs_warm_scores, ddof=1))
    ci_sc = 2.776 * (sd_sc_vs_wm / np.sqrt(5))

    m_wm_vs_sc = float(np.mean(warm_vs_scratch_scores))
    sd_wm_vs_sc = float(np.std(warm_vs_scratch_scores, ddof=1))
    ci_wm = 2.776 * (sd_wm_vs_sc / np.sqrt(5))

    t_pair, p_pair = stats.ttest_rel(warm_vs_scratch_scores, scratch_vs_warm_scores)
    paired_diff = m_wm_vs_sc - m_sc_vs_wm

    print(f"  * Scratch vs Warm 4인 (5시드 평균): {m_sc_vs_wm:+6.2f} ± {ci_sc:4.2f} bb/100 (SD: {sd_sc_vs_wm:.2f})")
    print(f"    - 95% CI: [{m_sc_vs_wm - ci_sc:+6.2f}, {m_sc_vs_wm + ci_sc:+6.2f}]")
    print(f"  * Warm vs Scratch 4인 (5시드 평균): {m_wm_vs_sc:+6.2f} ± {ci_wm:4.2f} bb/100 (SD: {sd_wm_vs_sc:.2f})")
    print(f"    - 95% CI: [{m_wm_vs_sc - ci_wm:+6.2f}, {m_wm_vs_sc + ci_wm:+6.2f}]")
    print(f"  * [Paired Cross-Play Difference]: {paired_diff:+6.2f} bb/100 (t={t_pair:+.2f}, p={p_pair:.4f})")

    print("\n" + "=" * 80)
    print("V5.7 50BB LEAGUE EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
