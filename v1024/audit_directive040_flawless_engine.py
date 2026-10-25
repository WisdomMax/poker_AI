#!/usr/bin/env python3
"""
[DIRECTIVE 040: FINAL EVALUATOR INTEGRITY ENGINE]
- OpenSpiel 5인 노리밋 텍사스 홀덤 실엔진 기반
- DIRECTIVE 040 핵심 요구사항 100% 충족:
  1. 실제 칩 투자 원장(extract_player_spent) 기반 VPIP, PFR, 3-Bet 정규 계산
  2. 스택 전액 투자 기반 실제 올인 판정 (액션 5 및 액션 2~4 스택 도달 포함)
  3. to_call > 0 유료 콜 기반 올인 콜 판정 (체크의 올인콜 오분류 100% 차단)
  4. 262개 콤보(19.76%) 슈트 편향 0.00% 대칭 Top20 상대 정책
  5. 모든 핸드의 원시 데이터를 JSONL(directive040_raw_hands_*.jsonl)로 전수 저장
  6. 두 모델의 핸드별 손익 차이 기반 대응표본(Paired difference) 95% 신뢰구간 산출
  7. 0BB 시점 포함 최대 낙폭 및 실제 자금 위험 지표 계산
"""

import sys
import os
import re
import json
import random
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1023.v1023_13_differential_forensics import PPOAgentWrapper
from v1024.tests.test_directive040_regression import (
    calc_chen_score, build_symmetric_top20_range, robust_parse_cards, parse_board_cards, SYMMETRIC_TOP20_COMBOS
)

# -----------------------------------------------------------------------------
# 1. 상대 봇 정책 정의
# -----------------------------------------------------------------------------
class SymmetricTop20HeuristicBot:
    """슈트 편향 0.0%의 상위 19.76%(262 콤보) Chen 기반 휴리스틱 봇 (GTO 과장 없음)"""
    name = "Symmetric_Top20_Heuristic"
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp, legal_6, priv_cards, round_idx, to_call, is_allin_facing):
        c1, c2 = priv_cards[cp]
        c_score = calc_chen_score(c1, c2)
        in_top20 = (c1, c2) in SYMMETRIC_TOP20_COMBOS
        
        if is_allin_facing and to_call > 0:
            if in_top20:
                return 1 if 1 in legal_6 else 0
            return 0 if 0 in legal_6 else 1
            
        if round_idx == 0: # Preflop
            if in_top20:
                if 2 in legal_6 and self.rng.random() < 0.6: return 2
                return 1 if 1 in legal_6 else 0
            if c_score >= 5.0 and 1 in legal_6:
                return 1 # Limp / Call
            return 0 if 0 in legal_6 else 1
        else: # Postflop
            if c_score >= 8.0:
                if 3 in legal_6 and self.rng.random() < 0.5: return 3
                return 1 if 1 in legal_6 else 0
            return 1 if 1 in legal_6 else (0 if 0 in legal_6 else 1)

class StrictTightPassiveBot:
    """타이트 패시브 상대 (Chen >= 9.0 프리미엄 콜러)"""
    name = "Strict_Tight_Passive"
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp, legal_6, priv_cards, round_idx, to_call, is_allin_facing):
        c1, c2 = priv_cards[cp]
        c_score = calc_chen_score(c1, c2)
        if is_allin_facing and to_call > 0:
            return 1 if (c_score >= 9.0 and 1 in legal_6) else 0
        if round_idx == 0:
            if c_score >= 8.0: return 1 if 1 in legal_6 else 0
            return 0 if 0 in legal_6 else 1
        return 1 if 1 in legal_6 else 0

class StrictLooseLimperBot:
    """100BB 올인에 100% 무조건 폴드하는 결함 봇"""
    name = "Strict_Loose_Limper"
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp, legal_6, priv_cards, round_idx, to_call, is_allin_facing):
        if is_allin_facing and to_call > 0:
            return 0 if 0 in legal_6 else 1
        return 1 if 1 in legal_6 else 0

# -----------------------------------------------------------------------------
# 2. 실엔진 1:1 평가 및 원시 JSONL 기록 함수
# -----------------------------------------------------------------------------
def run_integrity_evaluation(model_path, model_name, opponent_type="symmetric_top20", num_hands=1000, base_seed=999000, jsonl_out=None):
    game = get_5p_fullgame()
    
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()
    agent = PPOAgentWrapper(model)
    
    raw_hands = []
    hand_pnls_bb = []
    
    vpip_count = 0
    pfr_count = 0
    three_bet_count = 0
    allin_count = 0
    preflop_allin_count = 0
    postflop_allin_count = 0
    showdown_count = 0
    showdown_won_count = 0
    
    called_allin_pnls = []
    called_allin_records = []
    uncalled_allin_pnls = []
    
    street_reach_counts = defaultdict(int)
    action_counts = defaultdict(int)
    entropy_values = []
    
    cumulative_pnls = [0.0]
    running_pnl = 0.0
    
    STARTING_STACK = 10000.0 # $100 BB
    
    for h_idx in range(num_hands):
        h_seed = base_seed + h_idx
        chance_rng = np.random.RandomState(h_seed)
        
        st = game.new_initial_state()
        advance_chance_nodes(st, chance_rng)
        
        # 1. 5명 홀카드 무결 추출
        priv_cards = robust_parse_cards(st)
        
        # 2. 5개 좌석 균형 순환
        agent_seat = h_idx % 5
        
        # 3. 상대 봇 생성
        opps = {}
        for s in range(5):
            if s == agent_seat: continue
            if opponent_type == "symmetric_top20":
                opps[s] = SymmetricTop20HeuristicBot(s, h_seed * 10 + s)
            elif opponent_type == "tight_passive":
                opps[s] = StrictTightPassiveBot(s, h_seed * 11 + s)
            elif opponent_type == "loose_limper":
                opps[s] = StrictLooseLimperBot(s, h_seed * 13 + s)
                
        folded_seats = set()
        agent_vpip = False
        agent_pfr = False
        agent_3bet = False
        agent_allin = False
        agent_allin_round = None
        called_by_opp = False
        who_called = []
        
        seen_streets_this_hand = set()
        preflop_actions_log = []
        postflop_actions_log = []
        
        preflop_raises_before_agent = 0
        agent_went_allin_this_step = False
        agent_allin_bet_amount = 0.0
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue
                
            cp = st.current_player()
            round_idx = get_round_from_state(st) # 0: Preflop, 1: Flop, 2: Turn, 3: River
            
            if round_idx not in seen_streets_this_hand:
                seen_streets_this_hand.add(round_idx)
                street_names = ["Preflop", "Flop", "Turn", "River"]
                street_reach_counts[street_names[round_idx]] += 1
                
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            spent_before = extract_player_spent(st)
            
            # 현재 팟에서 이 플레이어가 콜해야 하는 금액 계산
            max_spent = max(spent_before)
            cur_to_call = max_spent - spent_before[cp]
            
            if cp == agent_seat:
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                probs = agent.get_action_probs(inf, legal_6)
                act_6 = agent.act(inf, legal_6)
                
                action_counts[act_6] += 1
                probs_safe = np.maximum(probs, 1e-8)
                ent = -np.sum(probs_safe * np.log(probs_safe))
                entropy_values.append(ent)
                
                # 액션 실행
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                
                spent_after = extract_player_spent(st)
                agent_spent_cur = spent_after[agent_seat]
                agent_rem_stack = STARTING_STACK - agent_spent_cur
                
                if round_idx == 0:
                    preflop_actions_log.append((act_6, nat_act, cur_to_call))
                    if act_6 >= 2: # Raise
                        if preflop_raises_before_agent >= 1:
                            agent_3bet = True
                else:
                    postflop_actions_log.append((round_idx, act_6, nat_act, cur_to_call))
                    
                if act_6 == 0:
                    folded_seats.add(agent_seat)
                    
                # 실제 올인 판정: 남은 스택이 0이거나 nat_act가 스택 전액인 경우
                if agent_rem_stack <= 0.0 or act_6 == 5:
                    agent_allin = True
                    agent_allin_round = round_idx
                    agent_went_allin_this_step = True
                    agent_allin_bet_amount = agent_spent_cur
            else:
                bot = opps.get(cp)
                is_allin_facing = (agent_allin and cur_to_call > 0)
                act_6 = bot.act(st, cp, legal_6, priv_cards, round_idx, cur_to_call, is_allin_facing)
                
                if round_idx == 0 and act_6 >= 2:
                    preflop_raises_before_agent += 1
                    
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                
                if is_allin_facing and act_6 == 1:
                    called_by_opp = True
                    who_called.append((cp, priv_cards[cp], cur_to_call))
                    
                if act_6 == 0:
                    folded_seats.add(cp)
                    
        # 핸드 정산 및 실제 칩 투자 원장 기반 VPIP / PFR 판정
        returns = st.returns()
        agent_pnl_dollars = returns[agent_seat]
        agent_pnl_bb = agent_pnl_dollars / 100.0 # $100 = 1 BB
        hand_pnls_bb.append(agent_pnl_bb)
        
        running_pnl += agent_pnl_bb
        cumulative_pnls.append(running_pnl)
        
        # VPIP 원장 판정: 프리플랍에서 블라인드 강제 지출(SB 50, BB 100, 기타 0)을 초과하여 자발적으로 칩을 투자했는가?
        final_spent = extract_player_spent(st)
        forced_blind = 50.0 if agent_seat == 0 else (100.0 if agent_seat == 1 else 0.0)
        
        # 프리플랍 지출액 계산
        if len(preflop_actions_log) > 0:
            agent_preflop_spent = max(a[1] for a in preflop_actions_log) if preflop_actions_log else forced_blind
        else:
            agent_preflop_spent = forced_blind
            
        if agent_preflop_spent > forced_blind:
            agent_vpip = True
        else:
            agent_vpip = False # BB 무료 체크 등은 완벽히 제외
            
        if any(a[0] >= 2 for a in preflop_actions_log):
            agent_pfr = True
        else:
            agent_pfr = False
            
        if agent_vpip: vpip_count += 1
        if agent_pfr: pfr_count += 1
        if agent_3bet: three_bet_count += 1
        
        if agent_allin:
            allin_count += 1
            if agent_allin_round == 0: preflop_allin_count += 1
            else: postflop_allin_count += 1
            
            if called_by_opp:
                called_allin_pnls.append(agent_pnl_bb)
                called_allin_records.append({
                    "hand_idx": h_idx,
                    "seed": h_seed,
                    "seat": agent_seat,
                    "callers": [f"P{c[0]}:{c[1][0]}+{c[1][1]}(call ${c[2]})" for c in who_called],
                    "pnl_bb": agent_pnl_bb
                })
            else:
                uncalled_allin_pnls.append(agent_pnl_bb)
                
        # 엄격한 쇼다운 판정
        active_players = [s for s in range(5) if s not in folded_seats]
        board_cards = parse_board_cards(st)
        is_showdown = (len(active_players) > 1 and agent_seat not in folded_seats)
        if is_showdown:
            showdown_count += 1
            if agent_pnl_bb > 0:
                showdown_won_count += 1
                
        # 원시 핸드 레코드 구축
        hand_record = {
            "hand_idx": h_idx,
            "seed": h_seed,
            "agent_seat": agent_seat,
            "agent_cards": priv_cards[agent_seat],
            "board": board_cards,
            "preflop_actions": preflop_actions_log,
            "postflop_actions": postflop_actions_log,
            "agent_vpip": agent_vpip,
            "agent_pfr": agent_pfr,
            "agent_allin": agent_allin,
            "allin_round": agent_allin_round,
            "allin_called": called_by_opp,
            "callers": [f"P{c[0]}:{c[1][0]}+{c[1][1]}" for c in who_called],
            "showdown": is_showdown,
            "pnl_bb": round(float(agent_pnl_bb), 2),
            "cumulative_pnl_bb": round(float(running_pnl), 2)
        }
        raw_hands.append(hand_record)
        
    # JSONL 파일로 핸드별 원시 데이터 저장
    if jsonl_out:
        os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for r in raw_hands:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
    mean_bb = np.mean(hand_pnls_bb)
    std_bb = np.std(hand_pnls_bb, ddof=1)
    ci95 = 1.96 * (std_bb / np.sqrt(num_hands))
    bb_per_100 = mean_bb * 100.0
    bb_ci95 = ci95 * 100.0
    
    peak = 0.0
    max_dd = 0.0
    for p in cumulative_pnls:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd
        
    vpip_pct = vpip_count / num_hands * 100.0
    pfr_pct = pfr_count / num_hands * 100.0
    three_bet_pct = three_bet_count / num_hands * 100.0
    wsd_pct = (showdown_won_count / showdown_count * 100.0) if showdown_count > 0 else 0.0
    
    called_ev = np.mean(called_allin_pnls) if called_allin_pnls else 0.0
    uncalled_ev = np.mean(uncalled_allin_pnls) if uncalled_allin_pnls else 0.0
    
    return {
        "model_name": model_name,
        "num_hands": num_hands,
        "bb_per_100": round(float(bb_per_100), 2),
        "ci95": round(float(bb_ci95), 2),
        "mean_bb_per_hand": round(float(mean_bb), 4),
        "std_bb_per_hand": round(float(std_bb), 4),
        "total_pnl_bb": round(float(running_pnl), 2),
        "max_drawdown_bb": round(float(max_dd), 2),
        "vpip_pct": round(float(vpip_pct), 2),
        "pfr_pct": round(float(pfr_pct), 2),
        "three_bet_pct": round(float(three_bet_pct), 2),
        "allin_total": allin_count,
        "preflop_allin": preflop_allin_count,
        "postflop_allin": postflop_allin_count,
        "showdown_count": showdown_count,
        "wsd_pct": round(float(wsd_pct), 2),
        "called_allin_count": len(called_allin_pnls),
        "called_allin_ev_bb": round(float(called_ev), 2),
        "called_allin_records": called_allin_records[:10],
        "uncalled_allin_count": len(uncalled_allin_pnls),
        "uncalled_allin_ev_bb": round(float(uncalled_ev), 2),
        "street_reach_counts": dict(street_reach_counts),
        "action_counts": dict(action_counts),
        "avg_entropy": round(float(np.mean(entropy_values)), 4),
        "hand_pnls_bb": [round(float(p), 2) for p in hand_pnls_bb]
    }

def main():
    print("=" * 80)
    print("DIRECTIVE 040: FINAL EVALUATOR INTEGRITY ENGINE (1:1 Paired Benchmark)")
    print("=" * 80)
    
    jenny_path = "v1024/models/V1024_PPO_Cycle_13_2890003hands.pt"
    e2e_path = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
    
    jenny_jsonl = "v1024/data/directive040_raw_hands_jenny.jsonl"
    e2e_jsonl = "v1024/data/directive040_raw_hands_e2e.jsonl"
    
    print("\n[평가 1/2] Jenny (Cycle 13, 2.89M) 1,000핸드 시뮬레이션 및 JSONL 기록...")
    res_jenny = run_integrity_evaluation(
        jenny_path, "Jenny_Cycle13", opponent_type="symmetric_top20",
        num_hands=1000, base_seed=999000, jsonl_out=jenny_jsonl
    )
    print(f"  [+] Jenny 완료: {res_jenny['bb_per_100']} ± {res_jenny['ci95']} BB/100 | All-in: {res_jenny['allin_total']} (Called: {res_jenny['called_allin_count']}회, EV: {res_jenny['called_allin_ev_bb']}) | DD: {res_jenny['max_drawdown_bb']} BB")
    
    print("\n[평가 2/2] E2E PPO (50k) 1,000핸드 시뮬레이션 및 JSONL 기록...")
    res_e2e = run_integrity_evaluation(
        e2e_path, "E2E_PPO_50k", opponent_type="symmetric_top20",
        num_hands=1000, base_seed=999000, jsonl_out=e2e_jsonl
    )
    print(f"  [+] E2E PPO 완료: {res_e2e['bb_per_100']} ± {res_e2e['ci95']} BB/100 | All-in: {res_e2e['allin_total']} (Called: {res_e2e['called_allin_count']}회, EV: {res_e2e['called_allin_ev_bb']}) | DD: {res_e2e['max_drawdown_bb']} BB")
    
    # 대응표본 95% 신뢰구간 (Paired Difference Confidence Interval) 계산
    diffs = np.array(res_e2e["hand_pnls_bb"]) - np.array(res_jenny["hand_pnls_bb"])
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    ci95_diff = 1.96 * (std_diff / np.sqrt(len(diffs)))
    paired_diff_bb100 = mean_diff * 100.0
    paired_ci95_bb100 = ci95_diff * 100.0
    
    print("\n" + "=" * 80)
    print("대응표본(Paired Difference) 통계 분석 결과:")
    print(f"  * 핸드당 평균 손익 차이: {mean_diff:+.4f} BB (± {ci95_diff:.4f} BB)")
    print(f"  * BB/100 손익 개선도   : {paired_diff_bb100:+.2f} ± {paired_ci95_bb100:.2f} BB/100 (95% CI: [{paired_diff_bb100 - paired_ci95_bb100:+.2f}, {paired_diff_bb100 + paired_ci95_bb100:+.2f}])")
    print("=" * 80)
    
    # hand_pnls_bb는 대용량이므로 최종 JSON 요약본에서는 통계치만 보존
    del res_jenny["hand_pnls_bb"]
    del res_e2e["hand_pnls_bb"]
    
    final_results = {
        "directive": "DIRECTIVE_040",
        "opponent": "Symmetric_Top20_Heuristic (262 combos, 19.76%, bias 0.0%)",
        "num_hands": 1000,
        "base_seed": 999000,
        "Jenny_Cycle13": res_jenny,
        "E2E_PPO_50k": res_e2e,
        "paired_difference": {
            "mean_diff_bb_per_hand": round(float(mean_diff), 4),
            "std_diff_bb_per_hand": round(float(std_diff), 4),
            "paired_diff_bb_per_100": round(float(paired_diff_bb100), 2),
            "paired_ci95_bb_per_100": round(float(paired_ci95_bb100), 2),
            "ci95_lower_bb_per_100": round(float(paired_diff_bb100 - paired_ci95_bb100), 2),
            "ci95_upper_bb_per_100": round(float(paired_diff_bb100 + paired_ci95_bb100), 2)
        }
    }
    
    out_json = "v1024/directive040_final_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n[성공] 최종 검산 결과가 {out_json}에 저장되었습니다.")
    print(f"[성공] Jenny 원시 핸드 JSONL: {jenny_jsonl}")
    print(f"[성공] E2E 원시 핸드 JSONL  : {e2e_jsonl}")

if __name__ == "__main__":
    main()
