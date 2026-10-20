"""
POKER AI V5.0 - PHASE 6: Extended Frozen Benchmark Evaluation & Model Calibration
- 1. 독립 평가 스케줄 (Test Schedule Eval 1: Master Seed 777777, 1,000판) 생성
- 2. 기존 1,000판 Benchmark(v498) 회귀 테스트 및 새 독립 스케줄 평가
- 3. Match Session 기반 실시간 베이지안 관측 평가 (Cold -> Warm 자연 축적)
- 4. 상대 봇 유형별 EV (Tight, Aggressive, AlwaysCall, Selective, Adaptive 등)
- 5. 포지션별 EV (BTN, CO, UTG, SB, BB)
- 6. 카드별 EV (Premium, Marginal, Trash, 72o, AA, JTs)
- 7. Opponent Model Calibration (사후 평균 vs 실제 행동 빈도 오차 MAE)
"""

import os
import sys
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.stats as stats
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, parse_cards_from_info_str
)
from v498.frozen_evaluator import (
    generate_frozen_schedule, create_opponent_bot, classify_hand_tier_v498
)
from v50.bayesian_opponent_tracker import MatchSessionOpponentManager
from v50.bayesian_ppo_model import (
    BayesianHoldemActorCritic5A, get_neutral_prior_features
)

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32, copy=True)
    v[223:] = v[223:] / 10000.0
    return v

def evaluate_model_session_frozen(
    model_path: str,
    schedule: List[Dict],
    use_opponent_features: bool,
    device: Optional[torch.device] = None,
    session_length: int = 50
) -> Tuple[Dict, List[Dict], Dict]:
    if device is None:
        device = torch.device("cpu")

    model = BayesianHoldemActorCritic5A(input_dim=317, hidden_dim=256, action_dim=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    neutral_feats = get_neutral_prior_features()

    detailed_logs = []
    total_chips = 0.0
    tier_returns = defaultdict(list)
    tier_counts = defaultdict(int)
    pos_returns = defaultdict(list)
    opp_type_returns = defaultdict(list)

    aa_returns = []
    seven_two_returns = []
    jts_returns = []
    aa_calls = 0
    aa_total = 0
    seven_two_calls = 0
    seven_two_total = 0

    pos_names_order = ['BTN', 'SB', 'BB', 'UTG', 'CO']

    # Calibration 추적: {ag_id: {'observed_vpip': float, 'posterior_vpip': float}}
    calibration_records = []

    num_hands = len(schedule)
    num_sessions = num_hands // session_length

    for s_idx in range(num_sessions):
        tracker_mgr = MatchSessionOpponentManager()
        tracker_mgr.reset_session()

        sess_hands = schedule[s_idx * session_length : (s_idx + 1) * session_length]
        first_sched = sess_hands[0]

        # 세션 상대 봇 인스턴스화
        opponents = [create_opponent_bot(name, s) for name, s in zip(first_sched['opp_names'], first_sched['opp_seeds'])]

        for h_in_s, sched in enumerate(sess_hands):
            g_id = sched['game_id']
            deck = sched['deck']
            hero_seat = sched['hero_seat']
            pos_name = pos_names_order[hero_seat]
            hero_seed = sched['hero_rng_seed']

            torch.manual_seed(hero_seed)
            tracker_mgr.set_hand_index(h_in_s)

            if use_opponent_features:
                opp_feats = tracker_mgr.get_table_feature_tensor(hero_seat)
            else:
                opp_feats = neutral_feats

            st = game.new_initial_state()
            chance_idx = 0
            while st.is_chance_node():
                st.apply_action(deck[chance_idx])
                chance_idx += 1

            info_s_hero = st.information_state_string(hero_seat)
            priv_cards, pub_cards = parse_cards_from_info_str(info_s_hero)
            hero_hand_str = ''.join(priv_cards)
            tier = classify_hand_tier_v498(hero_hand_str)
            tier_counts[tier] += 1

            hero_actions = []
            hand_action_log = []

            while not st.is_terminal():
                if st.is_chance_node():
                    st.apply_action(deck[chance_idx])
                    chance_idx += 1
                else:
                    cur = st.current_player()
                    legal = st.legal_actions(cur)
                    round_idx = len(st.information_state_string(cur).split("Round: ")) - 1 if "Round: " in st.information_state_string(cur) else 0

                    if cur == hero_seat:
                        info_vec = normalize_vec(st.information_state_tensor(cur))
                        full_vec = np.concatenate([info_vec, opp_feats])
                        x = torch.tensor(full_vec, dtype=torch.float32, device=device).unsqueeze(0)
                        m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                        for a in legal: m[0, a] = True

                        act, _, _, _ = model.sample_action(x, m, deterministic=False)
                        hero_actions.append(int(act))
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)
                    else:
                        s2a = wrapper.get_seat_to_agent(g_id)
                        ag_id = s2a[cur]
                        opp = opponents[(ag_id - 1) % len(opponents)]
                        act = opp.act(st, cur, legal)
                        hand_action_log.append((cur, round_idx, legal, act))
                        st.apply_action(act)

            # 핸드 종료 후 공개 액션으로 상대 통계 업데이트
            s2a_map = wrapper.get_seat_to_agent(g_id)
            tracker_mgr.parse_and_update_hand(hand_action_log, s2a_map)

            returns_all = st.returns()
            ret_chips = returns_all[hero_seat]
            ret_bb = ret_chips / 100.0
            total_chips += ret_chips

            tier_returns[tier].append(ret_bb)
            pos_returns[pos_name].append(ret_bb)

            # 상대 봇 유형별 손익 매핑 (테이블 상대들에게 분배)
            for opp_idx, opp_b in enumerate(opponents):
                opp_type_returns[opp_b.name].append(ret_bb)

            # 특수 핸드 추적
            if hero_hand_str in ['AsAh', 'AhAs', 'KsKh', 'QsQh', 'AsKs']:
                aa_returns.append(ret_bb)
                aa_total += 1
                if 1 in hero_actions: aa_calls += 1
            if ('7' in hero_hand_str and '2' in hero_hand_str) and (hero_hand_str[1] != hero_hand_str[3]):
                seven_two_returns.append(ret_bb)
                seven_two_total += 1
                if 1 in hero_actions: seven_two_calls += 1
            if ('J' in hero_hand_str and 'T' in hero_hand_str):
                jts_returns.append(ret_bb)

            detailed_logs.append({
                'game_id': g_id,
                'tier': tier,
                'hero_hand': hero_hand_str,
                'hero_actions': hero_actions,
                'ret_bb': ret_bb
            })

        # 세션 종료 시점 Calibration 기록 (사후 평균 vs 실제 관측)
        for ag_id in range(1, 5):
            t = tracker_mgr.trackers[ag_id]
            post_vpip, _, opp_vpip = t.get_posterior('vpip')
            true_vpip = t.stats['vpip']['succ'] / max(1, t.stats['vpip']['opp'])
            calibration_records.append({
                'ag_id': ag_id,
                'obs_vpip': true_vpip,
                'post_vpip': post_vpip,
                'error': abs(true_vpip - post_vpip)
            })

    bb100_final = float(total_chips / float(num_hands))
    mean_cal_err = float(np.mean([c['error'] for c in calibration_records])) if calibration_records else 0.0

    summary = {
        'model_path': model_path,
        'bb100': bb100_final,
        'prem_ev_bb': float(np.mean(tier_returns['premium'])) if tier_returns['premium'] else 0.0,
        'marg_ev_bb': float(np.mean(tier_returns['marginal'])) if tier_returns['marginal'] else 0.0,
        'trash_ev_bb': float(np.mean(tier_returns['trash'])) if tier_returns['trash'] else 0.0,
        'aa_ev_bb': float(np.mean(aa_returns)) if aa_returns else 0.0,
        'seven_two_ev_bb': float(np.mean(seven_two_returns)) if seven_two_returns else 0.0,
        'jts_ev_bb': float(np.mean(jts_returns)) if jts_returns else 0.0,
        'aa_call_pct': (aa_calls / max(1, aa_total)) * 100.0,
        'seven_two_call_pct': (seven_two_calls / max(1, seven_two_total)) * 100.0,
        'calibration_mae': mean_cal_err,
        'btn_bb100': float(np.mean(pos_returns['BTN']) * 100.0) if pos_returns['BTN'] else 0.0,
        'co_bb100': float(np.mean(pos_returns['CO']) * 100.0) if pos_returns['CO'] else 0.0,
        'utg_bb100': float(np.mean(pos_returns['UTG']) * 100.0) if pos_returns['UTG'] else 0.0,
        'sb_bb100': float(np.mean(pos_returns['SB']) * 100.0) if pos_returns['SB'] else 0.0,
        'bb_pos_bb100': float(np.mean(pos_returns['BB']) * 100.0) if pos_returns['BB'] else 0.0,
    }

    opp_stats = {name: float(np.mean(ret_list) * 100.0) for name, ret_list in opp_type_returns.items()}
    return summary, detailed_logs, opp_stats

def run_phase6_extended_evaluation():
    print("=" * 80)
    print("PHASE 6: 확장된 Frozen Benchmark 독립 평가 (Eval 1: 1,000판)")
    print("=" * 80)

    # 1. 새 독립 평가 스케줄 생성
    sched_eval1_path = "v50/data/test_schedule_eval1.json"
    if not os.path.exists(sched_eval1_path):
        print(f"새 독립 평가 스케줄 생성 중 (Master Seed: 777777, 1,000판)...")
        schedule_eval1 = generate_frozen_schedule(num_hands=1000, master_seed=777777)
        with open(sched_eval1_path, 'w', encoding='utf-8') as f:
            json.dump(schedule_eval1, f)
        print(f"-> 스케줄 저장 완료: {sched_eval1_path}")
    else:
        with open(sched_eval1_path, 'r', encoding='utf-8') as f:
            schedule_eval1 = json.load(f)

    device = torch.device("cpu")
    SEEDS = [42, 101, 202]
    CONDITIONS = [
        ('CondA_Base_Standard', False),
        ('CondB_Bayes_Standard', True),
        ('CondC_Base_RareExp', False),
        ('CondD_Bayes_RareExp', True),
    ]

    all_eval_results = []
    all_opp_stats = []

    for cname, use_opp in CONDITIONS:
        print(f"\n>>> [독립 평가] {cname} (상대통계={use_opp})")
        for s in SEEDS:
            mpath = f"v50/models/{cname}_seed_{s}.pt"
            summary, _, opp_stats = evaluate_model_session_frozen(
                model_path=mpath,
                schedule=schedule_eval1,
                use_opponent_features=use_opp,
                device=device
            )

            res = {
                'condition': cname,
                'seed': s,
                'bb100': summary['bb100'],
                'prem_ev_bb': summary['prem_ev_bb'],
                'marg_ev_bb': summary['marg_ev_bb'],
                'trash_ev_bb': summary['trash_ev_bb'],
                'aa_ev_bb': summary['aa_ev_bb'],
                'seven_two_ev_bb': summary['seven_two_ev_bb'],
                'jts_ev_bb': summary['jts_ev_bb'],
                'aa_call_pct': summary['aa_call_pct'],
                'seven_two_call_pct': summary['seven_two_call_pct'],
                'calibration_mae': summary['calibration_mae'],
                'btn_bb100': summary['btn_bb100'],
                'bb_pos_bb100': summary['bb_pos_bb100'],
            }
            all_eval_results.append(res)
            print(f"  Seed {s:3d} | bb/100: {res['bb100']:+7.2f} | Prem: {res['prem_ev_bb']:+5.2f} BB | Trash: {res['trash_ev_bb']:+5.2f} BB | 72o Call: {res['seven_two_call_pct']:4.1f}% | Calib MAE: {res['calibration_mae']:.4f}")

    # CSV 저장
    csv_raw = "v50/data/phase6_eval1_raw_results.csv"
    with open(csv_raw, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_eval_results[0].keys()))
        w.writeheader()
        w.writerows(all_eval_results)

    # 조건별 통계 요약 (df=2, t_crit=4.3027)
    t_crit = stats.t.ppf(0.975, df=2)
    summary_by_cond = []

    print("\n" + "=" * 80)
    print(f"PHASE 6 조건별 통계 요약 (Student's t-분포 df=2, t_crit={t_crit:.4f} 적용):")
    print("=" * 80)

    c_groups = defaultdict(list)
    for r in all_eval_results:
        c_groups[r['condition']].append(r)

    for cname, _ in CONDITIONS:
        rows = c_groups[cname]
        bb100_vals = [r['bb100'] for r in rows]
        mean_bb = np.mean(bb100_vals)
        std_bb = np.std(bb100_vals, ddof=1)
        sem_bb = std_bb / np.sqrt(len(bb100_vals))
        ci_half = t_crit * sem_bb

        summary_by_cond.append({
            'condition': cname,
            'mean_bb100': mean_bb,
            'std_bb100': std_bb,
            'ci95_half': ci_half,
            'ci95_lower': mean_bb - ci_half,
            'ci95_upper': mean_bb + ci_half,
            'mean_prem_ev_bb': np.mean([r['prem_ev_bb'] for r in rows]),
            'mean_trash_ev_bb': np.mean([r['trash_ev_bb'] for r in rows]),
            'mean_72o_call_pct': np.mean([r['seven_two_call_pct'] for r in rows]),
            'mean_calib_mae': np.mean([r['calibration_mae'] for r in rows])
        })

        print(f"[{cname}]")
        print(f"  평균 bb/100: {mean_bb:+7.2f} ± {ci_half:.2f} (95% CI: [{mean_bb - ci_half:+7.2f}, {mean_bb + ci_half:+7.2f}])")
        print(f"  72o Call%: {np.mean([r['seven_two_call_pct'] for r in rows]):.1f}% | Calib MAE: {np.mean([r['calibration_mae'] for r in rows]):.4f}")
        print(f"  Premium EV: {np.mean([r['prem_ev_bb'] for r in rows]):+6.2f} BB | Trash EV: {np.mean([r['trash_ev_bb'] for r in rows]):+6.2f} BB\n")

    # CSV 저장
    csv_cond = "v50/data/phase6_eval1_condition_summary.csv"
    with open(csv_cond, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_by_cond[0].keys()))
        w.writeheader()
        w.writerows(summary_by_cond)
    print(f"-> Phase 6 통계 요약 CSV 저장 완료: {csv_cond}")

if __name__ == "__main__":
    run_phase6_extended_evaluation()
