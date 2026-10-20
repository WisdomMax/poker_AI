"""
POKER AI V4.9.2 - PHASE 5: 공정한 기준선 확정 및 독립 평가
- 동일한 265차원 5-Action universal_poker 환경에서 평가
- 조건 비교:
  1. Control: 5A 네트워크 + Half-Pot 마스킹 (순수 4-Action 강제)
  2. Treatment_5A: 5A 네트워크 + Half-Pot 허용 (V4.9 20k Transfer)
  3. Baseline_40k: 5A Baseline PPO 40k 모델
  4. Curriculum_40k: 5A Curriculum PPO 40k 모델 (재레이즈 노출 강화)
- 5개 독립 시드 (42, 101, 202, 303, 404), 시드당 1,000판 (총 5,000판)
- 완전 페어드(Paired) 카드 및 좌석 스케줄 사용
- 상세 기록:
  * 상대별 bb/100, 전체 bb/100, 95% 신뢰구간
  * 모델 간 성능 차이(Treatment - Control) 및 95% CI (대응표본)
  * 포지션별 수익 (SB, BB, UTG, CO, BTN)
  * 프리미엄 핸드 EV (AA, KK, AKs) vs 트래시 핸드 EV (72o)
  * Half-Pot 사용률
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
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v48.opponents_pool import (
    RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot
from v492.phase4_true_adaptive_validation import RigorousAdaptiveBot

SEEDS = [42, 101, 202, 303, 404]
EVAL_HANDS_PER_SEED = 1000
BASE_EVAL_SEED = 77777

def get_eval_opponents():
    return [
        RandomBot(),
        AlwaysCallBot(),
        TightBot(),
        AggressiveBot(),
        PremiumHunterBot(),
        SelectiveCallerBot(),
        BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

def run_paired_evaluation(
    models_config: Dict[str, Dict],
    num_hands: int = EVAL_HANDS_PER_SEED,
    base_seed: int = BASE_EVAL_SEED,
    device: torch.device = torch.device("cpu")
):
    print("=" * 80)
    print("POKER AI V4.9.2 - PHASE 5: 공정한 기준선 및 독립 평가 시작")
    print(f"  평가 핸드 수: 시드당 {num_hands:,}판 (총 {num_hands * len(SEEDS):,}판 페어드 대결)")
    print("=" * 80)

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    opponents = get_eval_opponents()

    # 기록 데이터
    detailed_hand_logs = []
    summary_by_model = defaultdict(list)
    position_returns = defaultdict(lambda: defaultdict(list))
    opp_returns = defaultdict(lambda: defaultdict(list))
    premium_returns = defaultdict(list)
    trash_returns = defaultdict(list)
    halfpot_usage = defaultdict(lambda: {'hp_acts': 0, 'total_acts': 0})

    for s_idx, seed_val in enumerate(SEEDS):
        print(f"\n--- [Seed {seed_val}] 평가 진행 중 ---")
        
        # 모델 인스턴스 로드
        loaded_models = {}
        for cond_name, cfg in models_config.items():
            path_tmpl = cfg['path_template']
            path = path_tmpl.format(seed=seed_val)
            mask_halfpot = cfg.get('mask_halfpot', False)
            
            if os.path.exists(path):
                m = HoldemActorCritic5A().to(device)
                m.load_state_dict(torch.load(path, map_location=device))
                m.eval()
                loaded_models[cond_name] = (m, mask_halfpot)
            else:
                print(f"  경고: 모델 경로 없음 -> {path}")

        # 페어드 핸드 루프
        for h in range(num_hands):
            s2a = wrapper.get_seat_to_agent(h)
            a2s = wrapper.get_agent_to_seat(h)
            hero_seat = a2s[0] # Hero = Agent 0
            pos_name = ["SB", "BB", "UTG", "CO", "BTN"][hero_seat]

            # 테이블 상대 고정 (모든 모델에 동일 상대, 동일 좌석)
            rng_opp = random.Random(base_seed + s_idx * 10000 + h)
            table_opps = rng_opp.sample(opponents, 4)

            # 찬스 노드 카드 시드 고정 (모든 모델에 100% 동일한 홀카드 및 보드카드 분배)
            hand_card_seed = base_seed + 50000 + s_idx * 10000 + h

            # 각 모델별로 동일 카드 게임 진행
            for cond_name, (model, mask_halfpot) in loaded_models.items():
                rng_cards = np.random.RandomState(hand_card_seed)
                rng_model_act = np.random.RandomState(hand_card_seed + 1)

                st = game.new_initial_state()
                hero_acts = []
                hero_hole_cards = []

                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng=rng_cards)
                    else:
                        cur_seat = st.current_player()
                        legal = st.legal_actions(cur_seat)
                        ag_id = s2a[cur_seat]

                        if cur_seat == hero_seat:
                            info = st.information_state_string(cur_seat)
                            hc, _ = parse_cards_from_info_str(info)
                            if hc: hero_hole_cards = hc

                            vec = st.information_state_tensor(cur_seat)
                            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                            m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                            for a in legal: m[0, a] = True
                            
                            # Control 조건: Half-Pot (Action 4) 마스킹
                            if mask_halfpot and 4 in legal:
                                m[0, 4] = False
                                # 만약 4만 남았을 경우 방어
                                if not m.any(): m[0, 4] = True

                            with torch.no_grad():
                                logits, _ = model(x, m)
                                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                            valid_acts = [a for a in legal if m[0, a]]
                            prs = [probs[a] for a in valid_acts]
                            p_sum = sum(prs)
                            prs = np.array(prs) / p_sum if p_sum > 0 else np.ones(len(valid_acts))/len(valid_acts)
                            act = int(rng_model_act.choice(valid_acts, p=prs))
                            
                            hero_acts.append(act)
                            halfpot_usage[cond_name]['total_acts'] += 1
                            if act == 4:
                                halfpot_usage[cond_name]['hp_acts'] += 1
                        else:
                            opp = table_opps[(ag_id - 1) % len(table_opps)]
                            act = opp.act(st, cur_seat, legal)

                        st.apply_action(act)

                ret_chips = st.returns()[hero_seat]
                ret_bb = ret_chips / 100.0

                summary_by_model[cond_name].append(ret_bb)
                position_returns[cond_name][pos_name].append(ret_bb)

                # 상대별 수익 기록 (Hero 맞은편 주요 상대 기준)
                main_opp_name = table_opps[0].name
                opp_returns[cond_name][main_opp_name].append(ret_bb)

                # 프리미엄 vs 트래시 핸드 EV
                if len(hero_hole_cards) == 2:
                    ranks = [c[0] for c in hero_hole_cards]
                    is_pair = (ranks[0] == ranks[1])
                    if (is_pair and ranks[0] in ['A', 'K', 'Q']) or ('A' in ranks and 'K' in ranks):
                        premium_returns[cond_name].append(ret_bb)
                    elif ('7' in ranks and '2' in ranks) or ('8' in ranks and '3' in ranks):
                        trash_returns[cond_name].append(ret_bb)

                detailed_hand_logs.append({
                    'seed': seed_val,
                    'hand_idx': h,
                    'condition': cond_name,
                    'position': pos_name,
                    'hole_cards': "".join(hero_hole_cards),
                    'chips_return': ret_chips,
                    'bb_return': ret_bb,
                    'halfpot_used': (4 in hero_acts)
                })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 통계 분석 및 신뢰구간 산출
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("PHASE 5 독립 평가 결과 종합 (총 5,000판)")
    print("=" * 80)

    summary_rows = []

    for cond_name in models_config.keys():
        arr = np.array(summary_by_model[cond_name])
        n = len(arr)
        mean_bb = float(np.mean(arr))
        std_bb = float(np.std(arr, ddof=1))
        ci95 = 1.96 * (std_bb / math.sqrt(n))

        # Half-Pot 사용률
        hp_info = halfpot_usage[cond_name]
        hp_rate = (hp_info['hp_acts'] / max(1, hp_info['total_acts'])) * 100.0

        # 프리미엄 & 트래시 EV
        prem_ev = float(np.mean(premium_returns[cond_name])) if premium_returns[cond_name] else 0.0
        trash_ev = float(np.mean(trash_returns[cond_name])) if trash_returns[cond_name] else 0.0

        print(f"[{cond_name:22s}] bb/100: {mean_bb:+7.2f} ± {ci95:5.2f} (std: {std_bb:6.2f}) | Half-Pot: {hp_rate:4.1f}% | Prem EV: {prem_ev:+6.2f} | Trash EV: {trash_ev:+6.2f}")

        summary_rows.append({
            'condition': cond_name,
            'n_hands': n,
            'mean_bb100': mean_bb,
            'ci95_bb100': ci95,
            'std_bb100': std_bb,
            'halfpot_usage_pct': hp_rate,
            'premium_hand_ev': prem_ev,
            'trash_hand_ev': trash_ev
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모델 간 성능 차이(Treatment - Control 등) 대응표본 t-검정 및 95% CI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[대응표본 차이 검정 (Paired Difference Analysis)]")
    
    pairs_to_test = [
        ('Treatment_5A', 'Control_Masked'),
        ('Curriculum_40k', 'Baseline_40k'),
        ('Curriculum_40k', 'Treatment_5A')
    ]

    pair_diff_records = []
    for c_treat, c_ctrl in pairs_to_test:
        if c_treat in summary_by_model and c_ctrl in summary_by_model:
            t_arr = np.array(summary_by_model[c_treat])
            c_arr = np.array(summary_by_model[c_ctrl])
            diff = t_arr - c_arr
            n_diff = len(diff)
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            sem_diff = std_diff / math.sqrt(n_diff)
            ci95_diff = 1.96 * sem_diff
            t_stat = mean_diff / max(1e-9, sem_diff)
            
            sig = "유의미함 (p < 0.05)" if (mean_diff - ci95_diff > 0 or mean_diff + ci95_diff < 0) else "유의미하지 않음 (0 포함)"
            print(f"  {c_treat} vs {c_ctrl}:")
            print(f"    차이: {mean_diff:+7.2f} ± {ci95_diff:5.2f} bb/100 (t={t_stat:.2f}) -> {sig}")

            pair_diff_records.append({
                'treatment': c_treat,
                'control': c_ctrl,
                'mean_diff_bb100': mean_diff,
                'ci95_diff_bb100': ci95_diff,
                't_stat': t_stat,
                'significant': sig
            })

    # 포지션별 수익 요약
    print("\n[포지션별 수익 (bb/100)]")
    positions = ["SB", "BB", "UTG", "CO", "BTN"]
    print(f"{'Condition':22s} | " + " | ".join([f"{p:7s}" for p in positions]))
    for cond_name in models_config.keys():
        pos_strs = []
        for p in positions:
            p_mean = np.mean(position_returns[cond_name][p]) if position_returns[cond_name][p] else 0.0
            pos_strs.append(f"{p_mean:+7.1f}")
        print(f"{cond_name:22s} | " + " | ".join(pos_strs))

    # CSV 저장
    os.makedirs("v492/data", exist_ok=True)
    summary_csv = "v492/data/phase5_model_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    diff_csv = "v492/data/phase5_paired_differences.csv"
    with open(diff_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_diff_records[0].keys()))
        writer.writeheader()
        writer.writerows(pair_diff_records)

    print(f"\nPhase 5 평가 요약 저장 완료: {summary_csv}")
    print(f"Phase 5 차이 검정 저장 완료: {diff_csv}")

def run_phase5():
    models_config = {
        'Control_Masked': {
            'path_template': "v49/models/CondC_5A_Transfer_seed_{seed}_step_20000.pt",
            'mask_halfpot': True
        },
        'Treatment_5A': {
            'path_template': "v49/models/CondC_5A_Transfer_seed_{seed}_step_20000.pt",
            'mask_halfpot': False
        },
        'Baseline_40k': {
            'path_template': "v492/models/Baseline_seed_{seed}_step_40000.pt",
            'mask_halfpot': False
        },
        'Curriculum_40k': {
            'path_template': "v492/models/Treatment_Curriculum_seed_{seed}_step_40000.pt",
            'mask_halfpot': False
        }
    }
    run_paired_evaluation(models_config)

if __name__ == '__main__':
    run_phase5()
