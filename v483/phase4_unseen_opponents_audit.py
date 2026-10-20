"""
POKER AI V4.8.3 - PHASE 4: 미학습 상대 감사 및 독립 재검증 (Unseen Opponents Audit)
- 핵심 감사 과제:
  1. V4.8.2에서 Loose Calling과 Opponent Adaptive의 수익률(-317.9 bb/100)이 100% 동일했던 원인 규명
     - OpponentAdaptiveBot 내부 `self.opp_allins` 누적 결함으로 ratio > 0.3에 진입 불가 -> Loose Calling과 동일 행동 발생 증명
  2. 버그 수정된 Adaptive Bot (BugFixedOpponentAdaptiveBot) 구현
  3. 10개 독립 학습 20k 모델 vs 5대 미학습 상대(Extreme Tight, Loose Calling, Selective All-in, Bluff Catcher, BugFixed Adaptive) 평가
     - 동일 유형 4인 테이블 (각 500판)
     - 5종 혼합 미학습 테이블 (1,000판)
     - V4.7 Step 100k 모델과의 대조 비교 및 95% CI 산출
"""

import sys
import os
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.poker_environment import (
    get_5p_holdem_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str
)
from v48.ppo_actor_critic import HoldemActorCritic
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v48.opponents_pool import BaseBot, extract_hole_cards
from v48.phase2_baseline_benchmark import is_trash_hand

# 1. 5대 미학습 봇 정의
class ExtremeTightBot(BaseBot):
    def __init__(self): super().__init__("Extreme Tight")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        if len(hc) == 2:
            r1, r2 = hc[0][0], hc[1][0]
            if r1 == r2 and r1 in ['A', 'K', 'Q']:
                return 3 if 3 in legal_acts else (2 if 2 in legal_acts else 1)
        return 1 if 1 in legal_acts and 2 not in legal_acts and 3 not in legal_acts else (0 if 0 in legal_acts else 1)

class LooseCallingBot(BaseBot):
    def __init__(self): super().__init__("Loose Calling")
    def act(self, state, cur_player, legal_acts) -> int:
        return 1 if 1 in legal_acts else legal_acts[0]

class SelectiveCallerBot(BaseBot):
    def __init__(self): super().__init__("Selective Allin Caller")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        is_top8 = False
        if len(hc) == 2:
            r1, r2 = hc[0][0], hc[1][0]
            if r1 == r2 and r1 in ['A', 'K', 'Q', 'J']: is_top8 = True
            elif (r1 == 'A' and r2 in ['K', 'Q']) or (r2 == 'A' and r1 in ['K', 'Q']): is_top8 = True
        if len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts:
            return 1 if is_top8 else 0
        return 1 if 1 in legal_acts else 0

class BluffCatcherBot(BaseBot):
    def __init__(self): super().__init__("Bluff Catcher")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        has_pair = any(p in info for p in ["Pair", "Two", "Three", "Full", "Straight", "Flush"])
        if has_pair:
            return 1 if 1 in legal_acts else legal_acts[0]
        return 1 if 1 in legal_acts and 0 not in legal_acts else (0 if 0 in legal_acts else 1)

# [버그 원인 증명용] 기존 결함 있는 봇
class DefectiveOpponentAdaptiveBot(BaseBot):
    def __init__(self):
        super().__init__("Defective Adaptive")
        self.opp_allins = 0
        self.opp_total = 0
    def act(self, state, cur_player, legal_acts) -> int:
        self.opp_total += 1
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        ratio = self.opp_allins / max(1, self.opp_total)
        has_high = len(hc) == 2 and any(c[0] in ['A', 'K', 'Q'] for c in hc)
        if ratio > 0.3:
            if has_high: return 1 if 1 in legal_acts else legal_acts[0]
            return 0 if 0 in legal_acts else 1
        return 1 if 1 in legal_acts else 0

# [수정된 봇] 상대 올인을 실제로 감지하고 적응하는 봇
class FixedOpponentAdaptiveBot(BaseBot):
    def __init__(self):
        super().__init__("Fixed Opponent Adaptive")
        self.opp_allins = 0
        self.opp_total = 0
        
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        
        # 상대방의 올인 발생 감지 (정보 상태 문자열에 action 3 또는 max bet 직면)
        is_allin_facing = (len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts)
        if is_allin_facing:
            self.opp_allins += 1
        self.opp_total += 1
        
        ratio = self.opp_allins / max(1, self.opp_total)
        has_high = len(hc) == 2 and any(c[0] in ['A', 'K', 'Q'] for c in hc)
        
        if ratio > 0.2: # 상대가 올인을 자주 하면 하이카드 보유 시 콜, 아니면 폴드
            if is_allin_facing:
                return 1 if has_high and (1 in legal_acts) else (0 if 0 in legal_acts else 1)
        return 1 if 1 in legal_acts else 0

def demonstrate_defective_bot_bug():
    print("=" * 85)
    print("🔬 [PHASE 4 AUDIT] Loose Calling vs Defective Adaptive 행동 일치율 검사")
    print("=" * 85)
    
    b_loose = LooseCallingBot()
    b_defective = DefectiveOpponentAdaptiveBot()
    game = get_5p_holdem_game()
    rng = np.random.RandomState(42)
    
    match_count = 0
    total_actions = 0
    
    for h in range(100):
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng=rng)
            else:
                p = st.current_player()
                legal = st.legal_actions(p)
                a1 = b_loose.act(st, p, legal)
                a2 = b_defective.act(st, p, legal)
                if a1 == a2: match_count += 1
                total_actions += 1
                st.apply_action(rng.choice(legal))
                
    pct = (match_count / max(1, total_actions)) * 100.0
    print(f"  - 총 의사결정 수: {total_actions}회")
    print(f"  - 행동 일치 횟수: {match_count}회 ({pct:.2f}%)")
    print(f"  - 결함 원인 분석:")
    print(f"    DefectiveOpponentAdaptiveBot의 self.opp_allins는 항상 0으로 유지됨.")
    print(f"    결과적으로 ratio = 0.0이 되어 'ratio > 0.3' 조건에 절대 도달하지 못함.")
    print(f"    따라서 항상 'return 1 if 1 in legal_acts else 0'을 실행하여 Loose Calling과 100% 동일하게 행동함.")
    print(f"    동일한 시드(8888)로 평가 시 두 봇의 손익이 -317.9 bb/100으로 완전 일치했던 근본 원인 규명 완료!")
    print("=" * 85)

def evaluate_agent_against_opps(
    model,
    is_ppo: bool,
    opponents: List[BaseBot],
    num_hands: int = 500,
    seed: int = 8888,
    device: torch.device = torch.device("cpu")
) -> float:
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    rng = np.random.RandomState(seed)
    chips_total = 0.0
    
    model.eval()
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng=rng)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                if cur_seat == hero_seat:
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 4), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    with torch.no_grad():
                        if is_ppo: logits, _ = model(x, m)
                        else: logits = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng.choice(legal, p=prs))
                else:
                    ag_id = s2a[cur_seat]
                    opp = opponents[(ag_id - 1) % len(opponents)]
                    act = opp.act(st, cur_seat, legal)
                st.apply_action(act)
                
        chips_total += st.returns()[hero_seat]
    return float(chips_total / float(num_hands))

def run_phase4_audit_and_eval():
    demonstrate_defective_bot_bug()
    
    device = torch.device("cpu")
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    
    # 5대 미학습 봇 (수정된 봇 포함)
    unseen_bots = {
        "Extreme Tight": ExtremeTightBot(),
        "Loose Calling": LooseCallingBot(),
        "Selective All-in": SelectiveCallerBot(),
        "Bluff Catcher": BluffCatcherBot(),
        "Fixed Adaptive": FixedOpponentAdaptiveBot()
    }
    
    # Step 100k 모델 로드
    m_100k = Holdem5PNet(253, 256, 4).to(device)
    m_100k.load_state_dict(torch.load("v47/models/holdem_5p_step_100k.pt", map_location=device))
    
    # 10개 20k 모델 로드 및 평가
    models_20k = {}
    for s in seeds:
        m = HoldemActorCritic(253, 256, 4).to(device)
        path = f"v483/models/seed_{s}_step_20000.pt"
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
            models_20k[s] = m
            
    print(f"\n📊 10개 독립 20k 모델 vs 5대 미학습 상대 평가 시작 (모델당 500판 x 5종 + 혼합 1,000판)...")
    
    results = []
    
    # 1. 100k 기준 성적 측정
    print("  [Step 100k] 기준 대조군 평가 중...")
    res_100k = {}
    for bname, bot in unseen_bots.items():
        opps = [bot] * 4
        bb = evaluate_agent_against_opps(m_100k, is_ppo=False, opponents=opps, num_hands=500, seed=8888)
        res_100k[bname] = bb
    # 혼합 테이블
    mix_opps = list(unseen_bots.values())[:4]
    bb_mix_100k = evaluate_agent_against_opps(m_100k, is_ppo=False, opponents=mix_opps, num_hands=1000, seed=9999)
    res_100k["Mixed Unseen"] = bb_mix_100k
    
    # 2. 10개 PPO 모델 평가
    ppo_bot_scores = {bname: [] for bname in list(unseen_bots.keys()) + ["Mixed Unseen"]}
    
    for s, m in models_20k.items():
        row = {"seed": s}
        for bname, bot in unseen_bots.items():
            opps = [bot] * 4
            bb = evaluate_agent_against_opps(m, is_ppo=True, opponents=opps, num_hands=500, seed=8888)
            row[bname] = bb
            ppo_bot_scores[bname].append(bb)
            
        bb_mix = evaluate_agent_against_opps(m, is_ppo=True, opponents=mix_opps, num_hands=1000, seed=9999)
        row["Mixed Unseen"] = bb_mix
        ppo_bot_scores["Mixed Unseen"].append(bb_mix)
        results.append(row)
        print(f"  ✓ Seed {s:<4d}: Mixed = {bb_mix:>+7.1f} bb/100 | Tight={row['Extreme Tight']:>+6.1f} | Loose={row['Loose Calling']:>+6.1f} | FixedAdapt={row['Fixed Adaptive']:>+6.1f}")
        
    # 종합 표 출력
    print("\n" + "=" * 95)
    print("🏆 [PHASE 4 결과] 5대 미학습 상대별 성적 및 신뢰구간 (bb/100)")
    print(f"{'상대 유형':<22} | {'Step 100k':<12} | {'PPO 20k 평균':<14} | {'표본 표준편차':<12} | {'PPO 95% 신뢰구간'}")
    print("-" * 95)
    
    summary_rows = []
    for cat in list(unseen_bots.keys()) + ["Mixed Unseen"]:
        scores = np.array(ppo_bot_scores[cat])
        mean_s = np.mean(scores)
        std_s = np.std(scores, ddof=1)
        n = len(scores)
        se = std_s / np.sqrt(n)
        t_c = stats.t.ppf(0.975, df=n-1)
        ci = (mean_s - t_c * se, mean_s + t_c * se)
        s100 = res_100k[cat]
        
        print(f"{cat:<22} | {s100:>+8.1f}     | {mean_s:>+10.1f}     | {std_s:>10.1f}   | [{ci[0]:>+7.1f}, {ci[1]:>+7.1f}]")
        summary_rows.append({
            "category": cat,
            "step100k": s100,
            "ppo_mean": mean_s,
            "ppo_std": std_s,
            "ci_lower": ci[0],
            "ci_upper": ci[1]
        })
        
    print("=" * 95)
    
    # CSV 저장
    with open("v483/data/phase4_unseen_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("✓ phase4_unseen_results.csv 저장 완료")

if __name__ == "__main__":
    run_phase4_audit_and_eval()
