"""
POKER AI V4.9.5 - PHASE 4: AA 3-Bet Policy Repair & Expected Value Test
- AA, KK, QQ, JTs, 72o에 대한 Tight / Aggressive / Selective 상대별 3-Bet 직면 테스트
- Fold, Call, Half-Pot, Pot, All-in 행동 확률 정밀 측정
- 단순 Fold 감소 여부가 아닌, 상대 스타일별 실제 선택 행동 EV vs Fold EV vs Call EV 비교
- 물리적 상한 [-100, +400] BB 자동 검증
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
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v48.opponents_pool import (
    TightBot, AggressiveBot, PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v493.targeted_subgame_generator import TargetedSubgameGenerator
from v494.ev_boundary_validator import validate_bb_return, validate_chips_return, validate_mean_ev

TEST_SCENARIOS = [
    {'hand': 'AA',  'opp_name': 'Tight',      'opp_bot': TightBot(),           'desc': 'AA vs Tight 3-Bet'},
    {'hand': 'AA',  'opp_name': 'Aggressive', 'opp_bot': AggressiveBot(),      'desc': 'AA vs Aggressive 3-Bet'},
    {'hand': 'AA',  'opp_name': 'Selective',  'opp_bot': SelectiveCallerBot(), 'desc': 'AA vs Selective 3-Bet'},
    {'hand': 'KK',  'opp_name': 'Tight',      'opp_bot': TightBot(),           'desc': 'KK vs Tight 3-Bet'},
    {'hand': 'QQ',  'opp_name': 'Aggressive', 'opp_bot': AggressiveBot(),      'desc': 'QQ vs Aggressive 3-Bet'},
    {'hand': 'JTs', 'opp_name': 'Tight',      'opp_bot': TightBot(),           'desc': 'JTs vs Tight 3-Bet'},
    {'hand': 'JTs', 'opp_name': 'Aggressive', 'opp_bot': AggressiveBot(),      'desc': 'JTs vs Aggressive 3-Bet'},
    {'hand': '72o', 'opp_name': 'Aggressive', 'opp_bot': AggressiveBot(),      'desc': '72o vs Aggressive 3-Bet'},
]

def run_phase4_test(models_to_test: List[Tuple[str, str]]) -> List[Dict]:
    print("=" * 70)
    print("PHASE 4: AA 3-BET POLICY REPAIR & EV TEST")
    print("=" * 70)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    gen = TargetedSubgameGenerator(game)

    results = []
    N_EVAL_RUNS = 200

    for model_name, ckpt_path in models_to_test:
        if not os.path.exists(ckpt_path):
            print(f"[경고] 모델 파일 없음: {ckpt_path}")
            continue

        model = HoldemActorCritic5A().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"\n>>> [모델 평가] {model_name} ({ckpt_path})")

        for sc in TEST_SCENARIOS:
            hand_str = sc['hand']
            opp_bot = sc['opp_bot']
            opp_name = sc['opp_name']
            desc = sc['desc']

            # 1. 정책 확률 분포 측정
            st_probe, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', hand_str, hero_seat=2)
            vec = st_probe.information_state_tensor(2)
            legal = st_probe.legal_actions(2)
            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.zeros((1, 5), dtype=torch.bool, device=device)
            for a in legal: m[0, a] = True

            with torch.no_grad():
                lg, _ = model(x, m)
                probs = torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()

            fold_prob = probs[0] * 100.0
            call_prob = probs[1] * 100.0 if len(probs) > 1 else 0.0

            # 2. 강제 Fold vs 강제 Call vs 모델 실제 정책 선택 EV 비교 (Paired Rollout)
            ev_fold_list = []
            ev_call_list = []
            ev_model_list = []

            for run_i in range(N_EVAL_RUNS):
                eval_rng = np.random.RandomState(10000 + run_i)

                # A. 강제 Fold 대국
                st_f, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', hand_str, hero_seat=2, rng=eval_rng)
                st_f.apply_action(0) # Fold
                ret_f = st_f.returns()[2] / 100.0
                validate_bb_return(ret_f)
                ev_fold_list.append(ret_f)

                # B. 강제 Call 대국
                eval_rng_c = np.random.RandomState(10000 + run_i)
                st_c, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', hand_str, hero_seat=2, rng=eval_rng_c)
                st_c.apply_action(1) # Call
                while not st_c.is_terminal():
                    if st_c.is_chance_node():
                        advance_chance_nodes(st_c, rng=eval_rng_c)
                    else:
                        cur_p = st_c.current_player()
                        lgs = st_c.legal_actions(cur_p)
                        act = opp_bot.act(st_c, cur_p, lgs)
                        st_c.apply_action(act)
                ret_c = st_c.returns()[2] / 100.0
                validate_bb_return(ret_c)
                ev_call_list.append(ret_c)

                # C. 모델 실제 온-폴리시 선택 대국
                eval_rng_m = np.random.RandomState(10000 + run_i)
                st_m, _ = gen.generate_subgame_state('SitB_Facing_3Bet_AllIn', hand_str, hero_seat=2, rng=eval_rng_m)
                # 모델 결정
                prs = [probs[a] for a in legal]
                p_sum = sum(prs)
                prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                hero_act = int(eval_rng_m.choice(legal, p=prs))
                st_m.apply_action(hero_act)
                while not st_m.is_terminal():
                    if st_m.is_chance_node():
                        advance_chance_nodes(st_m, rng=eval_rng_m)
                    else:
                        cur_p = st_m.current_player()
                        lgs = st_m.legal_actions(cur_p)
                        if cur_p == 2:
                            # 추가 행동 필요 시 모델 선택
                            vec_c = st_m.information_state_tensor(2)
                            x_c = torch.tensor(vec_c, dtype=torch.float32, device=device).unsqueeze(0)
                            m_c = torch.zeros((1, 5), dtype=torch.bool, device=device)
                            for a in lgs: m_c[0, a] = True
                            with torch.no_grad():
                                lg_c, _ = model(x_c, m_c)
                                prs_c = torch.softmax(lg_c, dim=-1).squeeze(0).cpu().numpy()
                            p_sub = [prs_c[a] for a in lgs]
                            p_sub = np.array(p_sub)/sum(p_sub) if sum(p_sub)>0 else np.ones(len(lgs))/len(lgs)
                            act = int(eval_rng_m.choice(lgs, p=p_sub))
                        else:
                            act = opp_bot.act(st_m, cur_p, lgs)
                        st_m.apply_action(act)
                ret_m = st_m.returns()[2] / 100.0
                validate_bb_return(ret_m)
                ev_model_list.append(ret_m)

            mean_ev_fold = float(np.mean(ev_fold_list))
            mean_ev_call = float(np.mean(ev_call_list))
            mean_ev_model = float(np.mean(ev_model_list))

            print(f"  {desc:<28} | Fold: {fold_prob:5.1f}%, Call: {call_prob:5.1f}% | Fold EV: {mean_ev_fold:+6.2f} BB | Call EV: {mean_ev_call:+6.2f} BB | Model EV: {mean_ev_model:+6.2f} BB")

            results.append({
                'model': model_name,
                'scenario': desc,
                'hand': hand_str,
                'opponent': opp_name,
                'fold_prob_pct': fold_prob,
                'call_prob_pct': call_prob,
                'fold_ev_bb': mean_ev_fold,
                'call_ev_bb': mean_ev_call,
                'model_ev_bb': mean_ev_model,
                'ev_gain_vs_fold': mean_ev_model - mean_ev_fold
            })

    # CSV 저장
    csv_out = "v495/data/phase4_aa_3bet_repair_results.csv"
    if results:
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[저장 완료] Phase 4 정밀 EV 결과: {csv_out}")

    return results

if __name__ == "__main__":
    # 대표 모델들 테스트 준비
    sample_models = [
        ('Baseline_PreTrain', 'v49/models/CondC_5A_Transfer_seed_42_step_20000.pt'),
        ('Cond1_E0_B0_S0', 'v495/models/Cond1_E0_B0_S0_seed_42.pt'),
        ('Cond2_E1_B0_S0', 'v495/models/Cond2_E1_B0_S0_seed_42.pt'),
        ('Cond6_E1_B0_S1', 'v495/models/Cond6_E1_B0_S1_seed_42.pt'),
    ]
    run_phase4_test(sample_models)
