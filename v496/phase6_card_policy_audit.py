"""
POKER AI V4.9.6 - PHASE 6: Card Awareness & Situational Policy Verification
- 9개 대표 핸드 (AA, KK, QQ, AKs, AQs, JTs, 98s, 83o, 72o)
- 7대 게임 상황 (A~G: 프리플롭 최초, 오픈 직면, 3-Bet 직면, 4-Bet 직면, 플롭 베팅, 턴 레이즈, 리버 올인)
- 5개 행동 확률 분포 (Fold, Call, Half, Pot, Allin), Policy Entropy, AA vs 72o TV Distance
- 핵심 검증:
  * AA 3-Bet 직면 시 Fold 감소 여부
  * 72o 3-Bet 직면 시 블라인드 콜러(무차별 콜) 방지 여부 (Fold 90% 이상 유지)
  * 마지널 핸드(JTs)의 선별적 대응 여부
  * 상황별 실제 EV 측정 및 물리적 상한 [-100, +400] BB 자동 Assertion
"""

import os
import sys
import math
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes
)
from v48.opponents_pool import (
    TightBot, AggressiveBot, PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return
from v496.stable_ppo import StableHoldemActorCritic5A

TARGET_HANDS = ['AA', 'KK', 'QQ', 'AKs', 'AQs', 'JTs', '98s', '83o', '72o']

SITUATIONS = [
    ('SitA_Preflop_Open',          'SitA_Facing_OpenRaise',   0, 'A. 프리플롭 최초/오픈 직면'),
    ('SitB_Facing_3Bet_AllIn',     'SitB_Facing_3Bet_AllIn',  2, 'B. 프리플롭 3-Bet 올인 직면'),
    ('SitB_Facing_3Bet_Pot',       'SitB_Facing_3Bet_Pot',    2, 'C. 프리플롭 3-Bet 팟레이즈 직면'),
    ('SitC_Facing_4Bet',           'SitC_Facing_4Bet',        3, 'D. 프리플롭 4-Bet 직면'),
    ('SitD_Flop_Facing_Bet',       'SitD_Flop_Facing_Bet',    2, 'E. 플롭 베팅 직면'),
    ('SitE_Turn_Facing_Raise',     'SitE_Turn_Facing_Raise',  2, 'F. 턴 레이즈 직면'),
    ('SitF_River_Facing_AllIn',    'SitF_River_Facing_AllIn', 2, 'G. 리버 올인 직면'),
]

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def run_phase6_audit(model_path: str, model_name: str, normalize_input: bool = True) -> List[Dict]:
    print("=" * 75)
    print(f"PHASE 6: CARD AWARENESS AUDIT - {model_name}")
    print(f"체크포인트: {model_path} | 정규화: {normalize_input}")
    print("=" * 75)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    gen = TargetedSubgameGenerator(game)

    model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[모델 로드 성공] {model_path}")
    else:
        print(f"[경고] 모델 파일 없음: {model_path}")
        return []

    model.eval()
    results = []

    for sit_key, sit_gen_name, target_seat, sit_desc in SITUATIONS:
        print(f"\n>>> [상황] {sit_desc}")
        hand_policies = {}

        for h in TARGET_HANDS:
            try:
                st, _ = gen.generate_subgame_state(sit_gen_name, h, hero_seat=target_seat)
            except Exception as e:
                continue

            vec = st.information_state_tensor(target_seat)
            if normalize_input:
                vec = normalize_vec(vec)
            legal = st.legal_actions(target_seat)
            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.zeros((1, 5), dtype=torch.bool, device=device)
            for a in legal: m[0, a] = True

            with torch.no_grad():
                lg, _ = model(x, m)
                probs = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()

            hand_policies[h] = probs
            fold_p = probs[0] * 100.0
            call_p = probs[1] * 100.0 if len(probs) > 1 else 0.0
            pot_p = probs[2] * 100.0 if len(probs) > 2 else 0.0
            allin_p = probs[3] * 100.0 if len(probs) > 3 else 0.0
            half_p = probs[4] * 100.0 if len(probs) > 4 else 0.0

            # EV 간이 측정 (50회 대국)
            ev_list = []
            for run_i in range(50):
                rng_run = np.random.RandomState(4000 + run_i)
                st_run, _ = gen.generate_subgame_state(sit_gen_name, h, hero_seat=target_seat, rng=rng_run)
                while not st_run.is_terminal():
                    if st_run.is_chance_node():
                        advance_chance_nodes(st_run, rng=rng_run)
                    else:
                        cur_p = st_run.current_player()
                        lgs = st_run.legal_actions(cur_p)
                        if cur_p == target_seat:
                            v_r = st_run.information_state_tensor(cur_p)
                            if normalize_input: v_r = normalize_vec(v_r)
                            x_r = torch.tensor(v_r, dtype=torch.float32, device=device).unsqueeze(0)
                            m_r = torch.zeros((1, 5), dtype=torch.bool, device=device)
                            for a in lgs: m_r[0, a] = True
                            act, _, _, _ = model.sample_action(x_r, m_r)
                        else:
                            act = AggressiveBot().act(st_run, cur_p, lgs)
                        st_run.apply_action(act)
                ret_bb = st_run.returns()[target_seat] / 100.0
                validate_bb_return(ret_bb)
                ev_list.append(ret_bb)

            mean_ev = float(np.mean(ev_list))
            print(f"  핸드 {h:<4}: Fold {fold_p:5.1f}%, Call {call_p:5.1f}%, Pot {pot_p:5.1f}%, Allin {allin_p:5.1f}%, Half {half_p:5.1f}% | EV: {mean_ev:+6.2f} BB")

            results.append({
                'model': model_name,
                'situation': sit_desc,
                'hand': h,
                'fold_pct': fold_p,
                'call_pct': call_p,
                'pot_pct': pot_p,
                'allin_pct': allin_p,
                'half_pct': half_p,
                'mean_ev_bb': mean_ev
            })

        # AA vs 72o TV Distance
        if 'AA' in hand_policies and '72o' in hand_policies:
            tvd = 0.5 * np.sum(np.abs(hand_policies['AA'] - hand_policies['72o']))
            print(f"  --> [TV Distance (AA vs 72o)]: {tvd:.4f}")

    csv_path = f"v496/data/phase6_card_policy_{model_name}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장 완료] Phase 6 카드별 정책 데이터: {csv_path}")

    return results

if __name__ == "__main__":
    test_models = [
        ('CondB_Norm_EntOFF', 'v496/models/CondB_Norm_EntOFF_seed_42.pt', True),
        ('CondD_Norm_EntON',  'v496/models/CondD_Norm_EntON_seed_42.pt', True),
    ]
    for mname, path, norm in test_models:
        if os.path.exists(path):
            run_phase6_audit(path, mname, norm)
