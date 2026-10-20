"""
POKER AI V4.9.3 - PHASE 7: 실제 5인 홀덤 풀-게임 일반화 독립 평가
- 평가 대상:
  1. ConditionA_Subgame0 (Baseline: 풀-게임 100%)
  2. ConditionB_Subgame10 (Subgame 10%)
  3. ConditionC_Subgame30 (Subgame 30%)
  4. ConditionD_Subgame50 (Subgame 50%)
- Step 40,000 최종 체크포인트, 3개 독립 시드 (42, 101, 202)
- 시드당 1,000판 (총 3,000판) 완전 페어드 독립 대결
- 비학습 독립 평가 시드 (EVAL_BASE = 99999)
- 측정 항목:
  * 전체 bb/100 (공식: 100 * total_chips / (100 * N) = total_chips / N)
  * 95% 신뢰구간 (핸드 단위 및 시드 단위)
  * 상대별 bb/100
  * 포지션별 bb/100 (SB, BB, UTG, CO, BTN)
  * 핸드 등급별 EV: 프리미엄(AA/KK/QQ), 마지널(JTs/98s), 트래시(72o/83o)
  * 손익 분산 (std) 및 Half-Pot 사용률
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

SEEDS = [42, 101, 202]
EVAL_HANDS_PER_SEED = 1000
BASE_EVAL_SEED = 99999

def run_generalization_eval():
    print("=" * 80)
    print("POKER AI V4.9.3 - PHASE 7: 실제 5인 홀덤 풀-게임 일반화 독립 평가")
    print(f"  평가 핸드 수: 시드당 {EVAL_HANDS_PER_SEED:,}판 (총 {EVAL_HANDS_PER_SEED * len(SEEDS):,}판 페어드 대결)")
    print("=" * 80)

    device = torch.device("cpu")
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)

    opponents = [
        RandomBot(), AlwaysCallBot(), TightBot(), AggressiveBot(),
        PremiumHunterBot(), SelectiveCallerBot(), BluffCatcherBot(),
        RigorousAdaptiveBot()
    ]

    conditions = [
        'ConditionA_Subgame0',
        'ConditionB_Subgame10',
        'ConditionC_Subgame30',
        'ConditionD_Subgame50'
    ]

    hand_returns_by_cond = defaultdict(list)
    seed_returns_by_cond = defaultdict(lambda: defaultdict(list))
    pos_returns_by_cond = defaultdict(lambda: defaultdict(list))
    opp_returns_by_cond = defaultdict(lambda: defaultdict(list))

    tier_returns = {
        'premium': defaultdict(list), # AA, KK, QQ
        'marginal': defaultdict(list), # JTs, 98s, AQs
        'trash': defaultdict(list)     # 72o, 83o
    }
    halfpot_usage = defaultdict(lambda: {'hp': 0, 'total': 0})

    for s_idx, seed_val in enumerate(SEEDS):
        print(f"\n--- [Seed {seed_val}] 평가 진행 중 ---")
        loaded_models = {}
        for cond_name in conditions:
            path = f"v493/models/{cond_name}_seed_{seed_val}_step_40000.pt"
            if os.path.exists(path):
                m = HoldemActorCritic5A().to(device)
                m.load_state_dict(torch.load(path, map_location=device))
                m.eval()
                loaded_models[cond_name] = m
            else:
                print(f"  경고: 모델 없음 -> {path}")

        for h in range(EVAL_HANDS_PER_SEED):
            s2a = wrapper.get_seat_to_agent(h)
            a2s = wrapper.get_agent_to_seat(h)
            hero_seat = a2s[0]
            pos_name = ["SB", "BB", "UTG", "CO", "BTN"][hero_seat]

            rng_opp = random.Random(BASE_EVAL_SEED + s_idx * 10000 + h)
            table_opps = rng_opp.sample(opponents, 4)
            card_seed = BASE_EVAL_SEED + 70000 + s_idx * 10000 + h

            for cond_name, model in loaded_models.items():
                rng_cards = np.random.RandomState(card_seed)
                rng_act = np.random.RandomState(card_seed + 1)
                st = game.new_initial_state()
                hero_hole_cards = []

                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng=rng_cards)
                    else:
                        cur = st.current_player()
                        legal = st.legal_actions(cur)
                        if cur == hero_seat:
                            info = st.information_state_string(cur)
                            hc, _ = parse_cards_from_info_str(info)
                            if hc: hero_hole_cards = hc

                            vec = st.information_state_tensor(cur)
                            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                            m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                            for a in legal: m[0, a] = True

                            with torch.no_grad():
                                logits, _ = model(x, m)
                                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                            prs = [probs[a] for a in legal]
                            p_sum = sum(prs)
                            prs = np.array(prs)/p_sum if p_sum > 0 else np.ones(len(legal))/len(legal)
                            act = int(rng_act.choice(legal, p=prs))

                            halfpot_usage[cond_name]['total'] += 1
                            if act == 4: halfpot_usage[cond_name]['hp'] += 1
                        else:
                            ag_id = s2a[cur]
                            opp = table_opps[(ag_id - 1) % len(table_opps)]
                            act = opp.act(st, cur, legal)

                        st.apply_action(act)

                # Return 계산 (chips -> bb: BB = 100)
                ret_chips = st.returns()[hero_seat]
                ret_bb = ret_chips / 100.0 # bb/hand

                hand_returns_by_cond[cond_name].append(ret_bb)
                seed_returns_by_cond[cond_name][seed_val].append(ret_bb)
                pos_returns_by_cond[cond_name][pos_name].append(ret_bb)
                opp_returns_by_cond[cond_name][table_opps[0].name].append(ret_bb)

                # 핸드 등급 분류
                if len(hero_hole_cards) == 2:
                    ranks = [c[0] for c in hero_hole_cards]
                    is_pair = (ranks[0] == ranks[1])
                    is_suited = (hero_hole_cards[0][1] == hero_hole_cards[1][1])
                    if is_pair and ranks[0] in ['A', 'K', 'Q']:
                        tier_returns['premium'][cond_name].append(ret_bb)
                    elif (is_suited and 'J' in ranks and 'T' in ranks) or (is_suited and '9' in ranks and '8' in ranks):
                        tier_returns['marginal'][cond_name].append(ret_bb)
                    elif ('7' in ranks and '2' in ranks) or ('8' in ranks and '3' in ranks):
                        tier_returns['trash'][cond_name].append(ret_bb)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 통계 집계 및 출력
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("PHASE 7 풀-게임 독립 평가 결과 종합 (총 3,000판)")
    print("=" * 80)

    summary_records = []

    for cond_name in conditions:
        arr_bb_hand = np.array(hand_returns_by_cond[cond_name])
        n = len(arr_bb_hand)
        # 정확한 bb/100 공식
        mean_bb100 = float(np.mean(arr_bb_hand) * 100.0)
        std_bb_hand = float(np.std(arr_bb_hand, ddof=1))
        se_bb100 = (std_bb_hand / math.sqrt(n)) * 100.0
        ci95_bb100 = 1.96 * se_bb100

        # 시드별 bb/100
        seed_bb100s = [np.mean(seed_returns_by_cond[cond_name][s]) * 100.0 for s in SEEDS]

        # Half-Pot 사용률
        hp_info = halfpot_usage[cond_name]
        hp_rate = (hp_info['hp'] / max(1, hp_info['total'])) * 100.0

        # 핸드 등급별 EV (bb/100 환산)
        prem_ev = float(np.mean(tier_returns['premium'][cond_name]) * 100.0) if tier_returns['premium'][cond_name] else 0.0
        marg_ev = float(np.mean(tier_returns['marginal'][cond_name]) * 100.0) if tier_returns['marginal'][cond_name] else 0.0
        trash_ev = float(np.mean(tier_returns['trash'][cond_name]) * 100.0) if tier_returns['trash'][cond_name] else 0.0

        print(f"[{cond_name:20s}] bb/100: {mean_bb100:+7.2f} ± {ci95_bb100:6.2f} (std: {std_bb_hand:5.2f} bb/h) | HalfPot: {hp_rate:4.1f}% | Prem: {prem_ev:+7.1f} | Marg: {marg_ev:+6.1f} | Trash: {trash_ev:+6.1f}")
        print(f"  시드별 bb/100: {[round(x, 1) for x in seed_bb100s]}")

        summary_records.append({
            'condition': cond_name,
            'n_hands': n,
            'mean_bb100': mean_bb100,
            'ci95_bb100': ci95_bb100,
            'std_bb_hand': std_bb_hand,
            'seed_42_bb100': seed_bb100s[0],
            'seed_101_bb100': seed_bb100s[1],
            'seed_202_bb100': seed_bb100s[2],
            'halfpot_usage_pct': hp_rate,
            'premium_ev_bb100': prem_ev,
            'marginal_ev_bb100': marg_ev,
            'trash_ev_bb100': trash_ev
        })

    # 포지션별 수익 출력
    print("\n[포지션별 수익 (bb/100)]")
    positions = ["SB", "BB", "UTG", "CO", "BTN"]
    print(f"{'Condition':20s} | " + " | ".join([f"{p:7s}" for p in positions]))
    for cond_name in conditions:
        pos_strs = []
        for p in positions:
            p_val = np.mean(pos_returns_by_cond[cond_name][p]) * 100.0 if pos_returns_by_cond[cond_name][p] else 0.0
            pos_strs.append(f"{p_val:+7.1f}")
        print(f"{cond_name:20s} | " + " | ".join(pos_strs))

    # CSV 저장
    os.makedirs("v493/data", exist_ok=True)
    out_csv = "v493/data/phase7_fullgame_generalization_summary.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)

    print(f"\nPhase 7 일반화 요약 데이터 저장 완료: {out_csv}")

if __name__ == '__main__':
    run_generalization_eval()
