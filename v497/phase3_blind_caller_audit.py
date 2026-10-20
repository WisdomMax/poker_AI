"""
POKER AI V4.9.7 - PHASE 3: Blind Caller Defect & Card Sensitivity Audit
- 9개 대표 핸드 (AA, KK, QQ, AKs, AQs, JTs, 98s, 83o, 72o)
- 8대 게임 상황 (A~H: Preflop Unopened, Facing Open, 3-Bet All-in, 3-Bet Pot, 4-Bet, Flop Bet, Turn Raise, River All-in)
- 동일 공개 상태에서 홀카드만 변경하여 정책 민감도 및 행동별 EV 정밀 측정
- 질문 해명: '정규화가 카드 강도 판단을 개선했는가, 아니면 단순히 Fold 편향을 Call 편향으로 이동시켰는가?'
"""

import os
import sys
import math
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import get_5p_holdem_5a_game, advance_chance_nodes
from v48.opponents_pool import TightBot, AggressiveBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v494.ev_boundary_validator import validate_bb_return
from v496.stable_ppo import StableHoldemActorCritic5A

TARGET_HANDS = ['AA', 'KK', 'QQ', 'AKs', 'AQs', 'JTs', '98s', '83o', '72o']

SITUATIONS_8 = [
    ('Sit0_Preflop_Unopened',      'SitA_Facing_OpenRaise',   0, 'A. Preflop Unopened (최초 오픈)'),
    ('SitA_Facing_OpenRaise',      'SitA_Facing_OpenRaise',   3, 'B. Facing Open Raise'),
    ('SitB_Facing_3Bet_AllIn',     'SitB_Facing_3Bet_AllIn',  2, 'C. Facing 3-Bet All-in'),
    ('SitB_Facing_3Bet_Pot',       'SitB_Facing_3Bet_Pot',    2, 'D. Facing 3-Bet Pot'),
    ('SitC_Facing_4Bet',           'SitC_Facing_4Bet',        3, 'E. Facing 4-Bet'),
    ('SitD_Flop_Facing_Bet',       'SitD_Flop_Facing_Bet',    2, 'F. Flop Facing Bet'),
    ('SitE_Turn_Facing_Raise',     'SitE_Turn_Facing_Raise',  2, 'G. Turn Facing Raise'),
    ('SitF_River_Facing_AllIn',    'SitF_River_Facing_AllIn', 2, 'H. River Facing All-in'),
]

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def run_phase3_audit():
    print("=" * 75)
    print("PHASE 3: BLIND CALLER DEFECT & CARD SENSITIVITY AUDIT")
    print("=" * 75)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    gen = TargetedSubgameGenerator(game)

    # CondD 모델(정규화 + EntON, 5개 시드)과 CondA 모델(Raw + EntOFF, 5개 시드) 비교
    models_to_test = [
        ('CondA_Raw_EntOFF_seed_42',  'v496/models/CondA_Raw_EntOFF_seed_42.pt',  False),
        ('CondD_Norm_EntON_seed_42',   'v496/models/CondD_Norm_EntON_seed_42.pt',   True),
        ('CondD_Norm_EntON_seed_101',  'v496/models/CondD_Norm_EntON_seed_101.pt',  True),
        ('CondD_Norm_EntON_seed_202',  'v496/models/CondD_Norm_EntON_seed_202.pt',  True),
    ]

    all_records = []

    for mname, mpath, norm in models_to_test:
        if not os.path.exists(mpath):
            continue
        model = StableHoldemActorCritic5A(input_dim=265, hidden_dim=256, action_dim=5).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device))
        model.eval()

        print(f"\n>>> [모델 진단] {mname} (Norm: {norm})")

        for sit_key, sit_gen, target_seat, sit_desc in SITUATIONS_8:
            hand_probs = {}
            for h in TARGET_HANDS:
                try:
                    st, _ = gen.generate_subgame_state(sit_gen, h, hero_seat=target_seat)
                except:
                    continue

                vec = st.information_state_tensor(target_seat)
                if norm: vec = normalize_vec(vec)
                legal = st.legal_actions(target_seat)
                x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                for a in legal: m[0, a] = True

                with torch.no_grad():
                    lg, _ = model(x, m)
                    probs = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
                hand_probs[h] = probs

                # 실제 EV 측정 (50회 대국)
                evs = []
                for run_i in range(50):
                    rng_run = np.random.RandomState(8000 + run_i)
                    st_run, _ = gen.generate_subgame_state(sit_gen, h, hero_seat=target_seat, rng=rng_run)
                    while not st_run.is_terminal():
                        if st_run.is_chance_node():
                            advance_chance_nodes(st_run, rng=rng_run)
                        else:
                            cur_p = st_run.current_player()
                            lgs = st_run.legal_actions(cur_p)
                            if cur_p == target_seat:
                                v_r = st_run.information_state_tensor(cur_p)
                                if norm: v_r = normalize_vec(v_r)
                                x_r = torch.tensor(v_r, dtype=torch.float32, device=device).unsqueeze(0)
                                m_r = torch.zeros((1, 5), dtype=torch.bool, device=device)
                                for a in lgs: m_r[0, a] = True
                                act, _, _, _ = model.sample_action(x_r, m_r)
                            else:
                                act = AggressiveBot().act(st_run, cur_p, lgs)
                            st_run.apply_action(act)
                    evs.append(st_run.returns()[target_seat] / 100.0)

                mean_ev = float(np.mean(evs))

                all_records.append({
                    'model': mname,
                    'situation': sit_desc,
                    'hand': h,
                    'fold_pct': float(probs[0]*100),
                    'call_pct': float(probs[1]*100) if len(probs)>1 else 0.0,
                    'pot_pct': float(probs[2]*100) if len(probs)>2 else 0.0,
                    'allin_pct': float(probs[3]*100) if len(probs)>3 else 0.0,
                    'half_pct': float(probs[4]*100) if len(probs)>4 else 0.0,
                    'mean_ev_bb': mean_ev
                })

            # AA vs 72o 비교 출력
            if 'AA' in hand_probs and '72o' in hand_probs:
                p_aa = hand_probs['AA']
                p_72 = hand_probs['72o']
                tvd = 0.5 * np.sum(np.abs(p_aa - p_72))
                print(f"  {sit_desc:<30} | AA: Fold {p_aa[0]*100:4.1f}%, Call {p_aa[1]*100:4.1f}% | 72o: Fold {p_72[0]*100:4.1f}%, Call {p_72[1]*100:4.1f}% | TVD: {tvd:.4f}")

    csv_out = "v497/data/phase3_blind_caller_audit.csv"
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[저장 완료] Phase 3 블라인드 콜러 감사 데이터: {csv_out}")

if __name__ == "__main__":
    run_phase3_audit()
