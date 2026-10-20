"""
POKER AI V4.9.5 - PHASE 6: Subgame Benefit Validation & Fullgame Distortion Audit
- 희귀 상태별 EV vs 실제 Full-Game EV 비교
- 정상 게임(Full-Game)에서 Subgame 상황이 실제 발생하는 자연 빈도(Natural Frequency) 실측
- Subgame 학습 모델의 포지션별/상대별 성능 및 정상 게임에서의 행동 재현율 측정
"""

import os
import sys
import math
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator, TARGET_HOLE_CARDS
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return

SUBGAME_SITUATIONS = [
    'SitA_Facing_OpenRaise',
    'SitB_Facing_3Bet_AllIn',
    'SitB_Facing_3Bet_Pot',
    'SitC_Facing_4Bet',
    'SitD_Flop_Facing_Bet',
    'SitE_Turn_Facing_Raise',
    'SitF_River_Facing_AllIn'
]

def audit_natural_frequency(game: pyspiel.Game, n_hands: int = 5000) -> Dict[str, float]:
    """자연스러운 풀-게임에서 3-Bet 및 희귀 서브게임 상태가 발생하는 자연 빈도 측정"""
    print(f"\n[자연 빈도 측정] 5인 홀덤 {n_hands}판 시뮬레이션 중...")
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]
    rng_eval = np.random.RandomState(424242)
    rng_opp = random.Random(525252)

    counts = defaultdict(int)
    total_decision_points = 0
    hero_decision_points = 0

    for h in range(n_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        table_opps = rng_opp.sample(opponents, 4)

        st = game.new_initial_state()
        history_actions = []
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_eval)
            else:
                cur = st.current_player()
                legal = st.legal_actions(cur)
                total_decision_points += 1
                if cur == hero_seat:
                    hero_decision_points += 1
                    # 현재 라운드 및 이전 행동 분석
                    cur_round = st.information_state_string(cur)
                    # 프리플롭 3-Bet 직면 여부 간이 감지
                    if len(history_actions) >= 2 and any(a in [2, 3, 4] for _, a in history_actions[-2:]):
                        counts['facing_raise_or_3bet'] += 1

                if cur == hero_seat:
                    act = int(rng_eval.choice(legal))
                else:
                    ag_id = s2a[cur]
                    opp = table_opps[(ag_id - 1) % len(table_opps)]
                    act = opp.act(st, cur, legal)
                history_actions.append((cur, act))
                st.apply_action(act)

    raise_freq_pct = (counts['facing_raise_or_3bet'] / max(1, hero_decision_points)) * 100.0
    print(f"  총 플레이어 결정점: {total_decision_points}회 (영웅: {hero_decision_points}회)")
    print(f"  자연 레이즈/3-Bet 직면 빈도: {counts['facing_raise_or_3bet']}회 ({raise_freq_pct:.2f}% of hero turns)")
    return {
        'total_hands': n_hands,
        'hero_decision_points': hero_decision_points,
        'facing_raise_count': counts['facing_raise_or_3bet'],
        'natural_raise_freq_pct': raise_freq_pct
    }

def run_subgame_validation(models_to_compare: List[Tuple[str, str]]):
    print("=" * 70)
    print("PHASE 6: SUBGAME BENEFIT VALIDATION & FULLGAME DISTORTION AUDIT")
    print("=" * 70)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    gen = TargetedSubgameGenerator(game)

    # 1. 자연 빈도 감사
    nat_freq = audit_natural_frequency(game, n_hands=3000)

    # 2. 모델별 희귀 상태 EV vs Full-Game EV 비교
    results = []
    for mname, ckpt_path in models_to_compare:
        if not os.path.exists(ckpt_path):
            continue
        model = HoldemActorCritic5A().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # A. 희귀 서브게임 상태별 EV 측정 (AA vs 3-Bet All-in 등)
        sub_evs = []
        for _ in range(150):
            st_sub, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', 'AA', hero_seat=2)
            while not st_sub.is_terminal():
                if st_sub.is_chance_node():
                    advance_chance_nodes(st_sub)
                else:
                    cur_p = st_sub.current_player()
                    lgs = st_sub.legal_actions(cur_p)
                    if cur_p == 2:
                        vec = st_sub.information_state_tensor(2)
                        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in lgs: m[0, a] = True
                        with torch.no_grad():
                            lg, _ = model(x, m)
                            pr = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
                        prs = [pr[a] for a in lgs]
                        prs = np.array(prs)/sum(prs) if sum(prs)>0 else np.ones(len(lgs))/len(lgs)
                        act = int(np.random.choice(lgs, p=prs))
                    else:
                        act = AggressiveBot().act(st_sub, cur_p, lgs)
                    st_sub.apply_action(act)
            sub_evs.append(st_sub.returns()[2] / 100.0)

        mean_subgame_ev = float(np.mean(sub_evs))

        # B. 풀-게임 포지션별 EV 측정 (Button, Cutoff, UTG, SB, BB)
        pos_evs = defaultdict(list)
        wrapper = HoldemSeatRotationWrapper(game)
        opps = [RandomBot(), TightBot(), AggressiveBot(), PremiumHunterBot()]

        for h in range(400):
            a2s = wrapper.get_agent_to_seat(h)
            hero_seat = a2s[0]
            pos_name = ['BTN', 'SB', 'BB', 'UTG', 'CO'][hero_seat]
            st = game.new_initial_state()
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st)
                else:
                    cur = st.current_player()
                    lgs = st.legal_actions(cur)
                    if cur == hero_seat:
                        vec = st.information_state_tensor(cur)
                        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in lgs: m[0, a] = True
                        with torch.no_grad():
                            lg, _ = model(x, m)
                            pr = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
                        prs = [pr[a] for a in lgs]
                        prs = np.array(prs)/sum(prs) if sum(prs)>0 else np.ones(len(lgs))/len(lgs)
                        act = int(np.random.choice(lgs, p=prs))
                    else:
                        act = opps[(cur - 1) % len(opps)].act(st, cur, lgs)
                    st.apply_action(act)
            ret_bb = st.returns()[hero_seat] / 100.0
            pos_evs[pos_name].append(ret_bb)

        full_bb100 = float(np.mean([np.mean(v) for v in pos_evs.values()]) * 100.0)
        print(f"\n[{mname}]")
        print(f"  SitB (AA 3-Bet) EV: {mean_subgame_ev:+6.2f} BB/hand ({mean_subgame_ev*100:+7.1f} bb/100 subset)")
        print(f"  Full-Game bb/100:   {full_bb100:+7.2f} bb/100")
        for p in ['BTN', 'CO', 'UTG', 'SB', 'BB']:
            pe = float(np.mean(pos_evs[p]) * 100.0)
            print(f"    포지션 {p:<3}: {pe:+7.2f} bb/100")

        results.append({
            'model': mname,
            'subgame_sitb_ev_bb': mean_subgame_ev,
            'fullgame_bb100': full_bb100,
            'btn_bb100': float(np.mean(pos_evs['BTN']) * 100.0),
            'co_bb100': float(np.mean(pos_evs['CO']) * 100.0),
            'utg_bb100': float(np.mean(pos_evs['UTG']) * 100.0),
            'sb_bb100': float(np.mean(pos_evs['SB']) * 100.0),
            'bb_bb100': float(np.mean(pos_evs['BB']) * 100.0),
        })

    csv_out = "v495/data/phase6_subgame_validation.csv"
    if results:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[저장 완료] Phase 6 검증 데이터: {csv_out}")

    return results

if __name__ == "__main__":
    compare_list = [
        ('Cond1_FullgamePure', 'v495/models/Cond1_E0_B0_S0_seed_42.pt'),
        ('Cond2_FullgameEps',  'v495/models/Cond2_E1_B0_S0_seed_42.pt'),
        ('Cond5_SubgamePure',  'v495/models/Cond5_E0_B0_S1_seed_42.pt'),
        ('Cond6_SubgameEps',   'v495/models/Cond6_E1_B0_S1_seed_42.pt'),
    ]
    run_subgame_validation(compare_list)
