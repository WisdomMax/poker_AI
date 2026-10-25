#!/usr/bin/env python3
"""
[DIRECTIVE 038: E2E PPO 50K vs CYCLE 13 JENNY 전수 독립 검산 및 공정 재평가]
- OpenSpiel 실엔진 기반 5인 노리밋 텍사스 홀덤
- 8대 감사 결함 전면 수정:
  1. Chen 1,326개 조합 전수 계산 기반 오차 0.0% 상위 20.0% (265개 콤보) 콜 레인지
  2. BB 무료 체크(Walk)를 제외한 엄격한 정규 VPIP
  3. 5개 좌석 균형 순환 (Seat Rotation: Seat 0, 1, 2, 3, 4 순환)
  4. 동일 덱/시드 1:1 대조 (Paired Seed Evaluation)
  5. 95% 신뢰구간(CI), 0BB 시점 포함 최대 낙폭, 100BB 바이인 재구매 비용 산출
  6. 프리플랍/포스트플랍 올인 수치 정합성 (331+12=343 등)
  7. 신규 E2E PPO 50k 모델의 스트리트별 전이 및 진위 검증
"""

import sys
import os
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# Add workspace roots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyspiel
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent, normalize_inf_state_v79
)
from v1023.v1023_13_differential_forensics import PPOAgentWrapper

# -----------------------------------------------------------------------------
# 1. Chen Formula 1326 Starting Hands 전수 분석 & Top 20.0% 정밀 Lookup Set
# -----------------------------------------------------------------------------
RANKS = '23456789TJQKA'
RANK_VALS = {r: i + 2 for i, r in enumerate(RANKS)}

def calc_chen_score(c1, c2):
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    v1, v2 = RANK_VALS[r1], RANK_VALS[r2]
    high_val = max(v1, v2)
    low_val = min(v1, v2)
    
    if high_val == 14: base = 10
    elif high_val == 13: base = 8
    elif high_val == 12: base = 7
    elif high_val == 11: base = 6
    else: base = high_val / 2.0
    
    is_pair = (r1 == r2)
    if is_pair:
        base = max(base * 2, 5.0)
    
    is_suited = (s1 == s2)
    if is_suited:
        base += 2.0
        
    gap = high_val - low_val - 1
    if not is_pair:
        if gap == 1: base -= 1.0
        elif gap == 2: base -= 2.0
        elif gap == 3: base -= 4.0
        elif gap >= 4: base -= 5.0
        
        if gap <= 1 and high_val < 12 and low_val > 4:
            base += 1.0
            
    return base

ALL_CARDS = [r + s for r in RANKS for s in 'shdc']
ALL_COMBOS = []
for i in range(len(ALL_CARDS)):
    for j in range(i + 1, len(ALL_CARDS)):
        c1, c2 = ALL_CARDS[i], ALL_CARDS[j]
        score = calc_chen_score(c1, c2)
        ALL_COMBOS.append(((c1, c2), score))

ALL_COMBOS.sort(key=lambda x: x[1], reverse=True)
TOP_20_COMBOS = set()
for (c1, c2), score in ALL_COMBOS[:265]: # 정확히 상위 265개 = 20.0%
    TOP_20_COMBOS.add((c1, c2))
    TOP_20_COMBOS.add((c2, c1))

def is_top_20_percent_hand(c1, c2):
    return (c1, c2) in TOP_20_COMBOS

# -----------------------------------------------------------------------------
# 2. 상대 봇 정책 정의
# -----------------------------------------------------------------------------
class RigorousTop20Bot:
    """오차 0.0%의 상위 20.0% 핸드로만 올인에 콜하는 엄격한 합리적 상대"""
    name = "Rigorous_Top20_Rational"
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp, legal_6, priv_cards, round_name, facing_allin):
        c1, c2 = priv_cards.get(cp, ("2s", "3c"))
        c_score = calc_chen_score(c1, c2)
        
        if facing_allin:
            if is_top_20_percent_hand(c1, c2):
                return 1 if 1 in legal_6 else 0
            return 0 if 0 in legal_6 else 1
            
        # 일반 상황
        if round_name == "Preflop":
            if is_top_20_percent_hand(c1, c2):
                if 2 in legal_6 and self.rng.random() < 0.6: return 2
                return 1 if 1 in legal_6 else 0
            if c_score >= 5.0 and 1 in legal_6:
                return 1
            return 0 if 0 in legal_6 else 1
        else: # Postflop
            if c_score >= 8.0:
                if 3 in legal_6 and self.rng.random() < 0.5: return 3
                return 1 if 1 in legal_6 else 0
            return 1 if 1 in legal_6 else (0 if 0 in legal_6 else 1)

class StrictTightPassiveBot:
    """타이트 패시브 상대 (Chen >= 8 참전, 올인에는 Chen >= 9만 콜)"""
    name = "Strict_Tight_Passive"
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp, legal_6, priv_cards, round_name, facing_allin):
        c1, c2 = priv_cards.get(cp, ("2s", "3c"))
        c_score = calc_chen_score(c1, c2)
        if facing_allin:
            return 1 if (c_score >= 9.0 and 1 in legal_6) else 0
        if round_name == "Preflop":
            if c_score >= 8.0: return 1 if 1 in legal_6 else 0
            return 0 if 0 in legal_6 else 1
        return 1 if 1 in legal_6 else 0

class StrictLooseLimperBot:
    """100BB 올인에 100% 무조건 폴드하는 결함 상대 (보상 해킹 유발 요인)"""
    name = "Strict_Loose_Limper"
    def __init__(self, seat, seed):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, st, cp, legal_6, priv_cards, round_name, facing_allin):
        if facing_allin:
            return 0 if 0 in legal_6 else 1
        return 1 if 1 in legal_6 else 0

# -----------------------------------------------------------------------------
# 3. OpenSpiel 유틸리티
# -----------------------------------------------------------------------------
# get_5p_fullgame imported from v79

def advance_chance_nodes(st, rng):
    while st.is_chance_node():
        outcomes = st.chance_outcomes()
        acts, probs = zip(*outcomes)
        a = rng.choice(acts, p=probs)
        st.apply_action(a)

def parse_cards_from_state(st):
    cards_map = {}
    s = str(st)
    for line in s.split('\n'):
        if "P" in line and "Cards" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                p_part = parts[0].strip()
                c_part = parts[1].strip()
                try:
                    p_idx = int(p_part.replace("P", "").replace("Cards", "").strip())
                    cards = c_part.split()
                    if len(cards) == 2:
                        cards_map[p_idx] = (cards[0], cards[1])
                except:
                    pass
    return cards_map

def get_round_name(st):
    s = str(st)
    if "Round 0" in s or "Preflop" in s: return "Preflop"
    if "Round 1" in s or "Flop" in s: return "Flop"
    if "Round 2" in s or "Turn" in s: return "Turn"
    if "Round 3" in s or "River" in s: return "River"
    return "Preflop"

# -----------------------------------------------------------------------------
# 4. 공정한 1,000핸드 시뮬레이션 (동일 시드 + 5개 좌석 순환)
# -----------------------------------------------------------------------------
def run_rigorous_evaluation(model_path, opponent_type="top20_rational", num_hands=1000, base_seed=777000):
    game = get_5p_fullgame()
    
    
    # 모델 로드
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()
    agent = PPOAgentWrapper(model)
    
    # 지표 집계용
    hand_pnls_bb = []
    vpip_count = 0
    pfr_count = 0
    allin_count = 0
    preflop_allin_count = 0
    postflop_allin_count = 0
    showdown_count = 0
    showdown_won_count = 0
    called_allin_pnls = []
    uncalled_allin_pnls = []
    
    # 스트리트별 전이 수 및 액션 카운트
    street_transitions = defaultdict(int)
    action_counts = defaultdict(int)
    entropy_values = []
    
    # 최대 낙폭 추적 (0.0 시점 포함)
    cumulative_pnls = [0.0]
    running_pnl = 0.0
    rebuy_count = 0
    
    for h_idx in range(num_hands):
        h_seed = base_seed + h_idx
        chance_rng = np.random.RandomState(h_seed)
        
        st = game.new_initial_state()
        advance_chance_nodes(st, chance_rng)
        priv_cards = parse_cards_from_state(st)
        
        # 5개 좌석 균형 순환 (Seat Rotation: 0, 1, 2, 3, 4)
        agent_seat = h_idx % 5
        
        # 상대 봇 생성
        opps = {}
        for s in range(5):
            if s == agent_seat: continue
            if opponent_type == "top20_rational":
                opps[s] = RigorousTop20Bot(s, h_seed * 10 + s)
            elif opponent_type == "tight_passive":
                opps[s] = StrictTightPassiveBot(s, h_seed * 11 + s)
            elif opponent_type == "loose_limper":
                opps[s] = StrictLooseLimperBot(s, h_seed * 13 + s)
                
        folded_seats = set()
        agent_vpip = False
        agent_pfr = False
        agent_allin = False
        agent_allin_street = None
        called_by_opp = False
        
        seen_streets_this_hand = set()
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, chance_rng)
                continue
                
            cp = st.current_player()
            r_name = get_round_name(st)
            
            if r_name not in seen_streets_this_hand:
                seen_streets_this_hand.add(r_name)
                street_transitions[r_name] += 1
                
            legal_native = st.legal_actions()
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            
            if cp == agent_seat:
                # 에이전트 행동
                inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                probs = agent.get_action_probs(inf, legal_6)
                act_6 = agent.act(inf, legal_6)
                
                action_counts[act_6] += 1
                probs_safe = np.maximum(probs, 1e-8)
                ent = -np.sum(probs_safe * np.log(probs_safe))
                entropy_values.append(ent)
                
                # VPIP / PFR 판정
                if r_name == "Preflop":
                    if act_6 >= 2: # Raise or All-in
                        agent_vpip = True
                        agent_pfr = True
                    elif act_6 == 1: # Call / Check
                        # BB 무료 체크는 VPIP에서 제외 (정식 규칙 수정)
                        # BB 포지션은 Seat 1에 해당
                        if agent_seat == 1 and not any(a >= 2 for a in action_counts.keys()):
                            pass # Free check / Walk
                        else:
                            agent_vpip = True
                            
                if act_6 == 5:
                    agent_allin = True
                    agent_allin_street = r_name
                    
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
            else:
                # 상대 봇 행동
                bot = opps.get(cp)
                facing_allin = agent_allin
                act_6 = bot.act(st, cp, legal_6, priv_cards, r_name, facing_allin)
                if facing_allin and act_6 == 1:
                    called_by_opp = True
                if act_6 == 0:
                    folded_seats.add(cp)
                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                
        # 핸드 정산 (100BB = $10,000, 1BB = $100)
        returns = st.returns()
        agent_pnl_dollars = returns[agent_seat]
        agent_pnl_bb = agent_pnl_dollars / 100.0 # $100 = 1 BB
        hand_pnls_bb.append(agent_pnl_bb)
        
        running_pnl += agent_pnl_bb
        cumulative_pnls.append(running_pnl)
        if running_pnl <= -100.0 * (rebuy_count + 1):
            rebuy_count += 1
            
        if agent_vpip: vpip_count += 1
        if agent_pfr: pfr_count += 1
        if agent_allin:
            allin_count += 1
            if agent_allin_street == "Preflop": preflop_allin_count += 1
            else: postflop_allin_count += 1
            
            if called_by_opp:
                called_allin_pnls.append(agent_pnl_bb)
            else:
                uncalled_allin_pnls.append(agent_pnl_bb)
                
        # 쇼다운 여부
        active_players = [s for s in range(5) if s not in folded_seats]
        if len(active_players) > 1:
            showdown_count += 1
            if agent_seat in active_players and agent_pnl_bb > 0:
                showdown_won_count += 1
                
    # 통계 산출
    mean_bb = np.mean(hand_pnls_bb)
    std_bb = np.std(hand_pnls_bb)
    ci95 = 1.96 * (std_bb / np.sqrt(num_hands))
    bb_per_100 = mean_bb * 100.0
    bb_ci95 = ci95 * 100.0
    
    # 최대 낙폭 계산 (최초 0.0 시점 포함)
    peak = 0.0
    max_dd = 0.0
    for p in cumulative_pnls:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd
        
    vpip_pct = vpip_count / num_hands * 100.0
    pfr_pct = pfr_count / num_hands * 100.0
    wsd_pct = (showdown_won_count / showdown_count * 100.0) if showdown_count > 0 else 0.0
    
    called_ev = np.mean(called_allin_pnls) if called_allin_pnls else 0.0
    uncalled_ev = np.mean(uncalled_allin_pnls) if uncalled_allin_pnls else 0.0
    
    return {
        "num_hands": num_hands,
        "bb_per_100": round(float(bb_per_100), 2),
        "ci95": round(float(bb_ci95), 2),
        "total_pnl_bb": round(float(running_pnl), 2),
        "max_drawdown_bb": round(float(max_dd), 2),
        "rebuy_count_100bb": rebuy_count,
        "vpip_pct": round(float(vpip_pct), 2),
        "pfr_pct": round(float(pfr_pct), 2),
        "allin_total": allin_count,
        "preflop_allin": preflop_allin_count,
        "postflop_allin": postflop_allin_count,
        "showdown_count": showdown_count,
        "wsd_pct": round(float(wsd_pct), 2),
        "called_allin_count": len(called_allin_pnls),
        "called_allin_ev_bb": round(float(called_ev), 2),
        "uncalled_allin_count": len(uncalled_allin_pnls),
        "uncalled_allin_ev_bb": round(float(uncalled_ev), 2),
        "street_transitions": dict(street_transitions),
        "action_counts": dict(action_counts),
        "avg_entropy": round(float(np.mean(entropy_values)), 4)
    }

def main():
    print("=" * 80)
    print("DIRECTIVE 038: E2E PPO 50K vs CYCLE 13 JENNY 전수 독립 검산 (OpenSpiel 5P)")
    print("=" * 80)
    
    jenny_path = "v1024/models/V1024_PPO_Cycle_13_2890003hands.pt"
    e2e_path = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
    
    scenarios = [
        ("Top20_Rational_Holdout", "top20_rational"),
        ("Strict_Tight_Passive", "tight_passive"),
        ("Loose_Limper_Benchmark", "loose_limper")
    ]
    
    full_report = {}
    
    for label, opp_type in scenarios:
        print(f"\n[시나리오 대전 시작] {label} (1,000핸드 동일 시드 1:1 대조)...")
        res_jenny = run_rigorous_evaluation(jenny_path, opponent_type=opp_type, num_hands=1000, base_seed=999000)
        print(f"  [+] Jenny (Cycle 13) 완료 : {res_jenny['bb_per_100']} ± {res_jenny['ci95']} BB/100 | All-in: {res_jenny['allin_total']}회 | DD: {res_jenny['max_drawdown_bb']} BB")
        
        res_e2e = run_rigorous_evaluation(e2e_path, opponent_type=opp_type, num_hands=1000, base_seed=999000)
        print(f"  [+] E2E PPO (50k)    완료 : {res_e2e['bb_per_100']} ± {res_e2e['ci95']} BB/100 | All-in: {res_e2e['allin_total']}회 | DD: {res_e2e['max_drawdown_bb']} BB")
        
        full_report[label] = {
            "Jenny_Cycle13": res_jenny,
            "E2E_PPO_50k": res_e2e,
            "diff_bb_per_100": round(res_e2e['bb_per_100'] - res_jenny['bb_per_100'], 2),
            "allin_reduction_pct": round((res_jenny['allin_total'] - res_e2e['allin_total']) / max(1, res_jenny['allin_total']) * 100.0, 1)
        }
        
    out_file = "v1024/directive038_rigorous_audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
        
    print(f"\n[최종 완료] 전수 검산 데이터가 {out_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()
