"""
POKER AI V4.8.1 - PHASE 5: 핸드별 기대수익(EV) 측정 표본 감사 및 대규모 정밀 재측정
- 감사 목적:
  1. V4.8 보고서의 수치(AA +13,643, 72o -5,023)의 실제 표본 수(N=30)의 통계적 취약성 정밀 감사
  2. 표본 수를 각 핸드당 300회(총 2,400판)로 10배 확대
  3. 동일 포지션(UTG) 통제 조건 하에서 PPO League vs Step 100k 모델의 실제 승패, 평균 손익, 표준편차, 95% 신뢰구간 정밀 산출
  4. 결과를 v481/data/hand_ev_audit_results.csv로 저장
"""

import sys
import os
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.poker_environment import (
    get_5p_holdem_game, advance_chance_nodes, parse_cards_from_info_str
)
from v48.ppo_actor_critic import HoldemActorCritic
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v48.opponents_pool import TightBot, AggressiveBot, AlwaysCallBot, RandomBot

def measure_rigorous_hand_ev(
    model,
    is_ppo: bool,
    target_hand_name: str,
    target_hand_cards: List[str],
    num_samples: int = 300,
    device: torch.device = torch.device("cpu")
) -> Dict:
    game = get_5p_holdem_game()
    opponents = [TightBot(), AggressiveBot(), AlwaysCallBot(), RandomBot()]
    
    returns_list = []
    actions_taken = []
    wins = 0
    showdowns = 0
    
    c1, c2 = target_hand_cards[0], target_hand_cards[1]
    r1_target, r2_target = c1[0], c2[0]
    is_suited_target = (c1[1] == c2[1])
    is_pair_target = (r1_target == r2_target)
    
    seed = 1000
    while len(returns_list) < num_samples:
        seed += 1
        random.seed(seed)
        np.random.seed(seed)
        
        st = game.new_initial_state()
        advance_chance_nodes(st) # 홀카드 배분
        
        # UTG (Player 2)
        cur_p = st.current_player()
        if cur_p == 2:
            s_str = st.information_state_string(cur_p)
            hc, _ = parse_cards_from_info_str(s_str)
            if len(hc) == 2:
                r1, r2 = hc[0][0], hc[1][0]
                s1, s2 = hc[0][1], hc[1][1]
                match = False
                if is_pair_target and r1 == r1_target and r2 == r2_target:
                    match = True
                elif not is_pair_target and is_suited_target and s1 == s2:
                    if (r1 == r1_target and r2 == r2_target) or (r1 == r2_target and r2 == r1_target):
                        match = True
                elif not is_pair_target and not is_suited_target and s1 != s2:
                    if (r1 == r1_target and r2 == r2_target) or (r1 == r2_target and r2 == r1_target):
                        match = True
                        
                if match:
                    # 첫 행동 샘플링
                    legal = st.legal_actions(cur_p)
                    vec = st.information_state_tensor(cur_p)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
                    for a in legal: mask[0, a] = True
                    with torch.no_grad():
                        if is_ppo: logits, _ = model(x, mask)
                        else: logits = model(x, mask)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs)>0 else np.ones(len(legal))/len(legal)
                    act = int(np.random.choice(legal, p=prs))
                    actions_taken.append(act)
                    st.apply_action(act)
                    
                    # 핸드 끝까지 진행
                    while not st.is_terminal():
                        if st.is_chance_node(): advance_chance_nodes(st)
                        else:
                            p = st.current_player()
                            l = st.legal_actions(p)
                            if p == cur_p:
                                v_p = st.information_state_tensor(p)
                                xp = torch.tensor(v_p, dtype=torch.float32, device=device).unsqueeze(0)
                                mp = torch.zeros((1, 4), dtype=torch.bool, device=device)
                                for a in l: mp[0, a] = True
                                with torch.no_grad():
                                    if is_ppo: lp, _ = model(xp, mp)
                                    else: lp = model(xp, mp)
                                    pp = torch.softmax(lp, dim=-1).squeeze(0).cpu().numpy()
                                prsp = [pp[a] for a in l]
                                prsp = np.array(prsp)/sum(prsp) if sum(prsp)>0 else np.ones(len(l))/len(l)
                                act_p = int(np.random.choice(l, p=prsp))
                            else:
                                act_p = opponents[(p - 1) % len(opponents)].act(st, p, l)
                            st.apply_action(act_p)
                            
                    rets = st.returns()
                    hero_ret = rets[cur_p]
                    returns_list.append(hero_ret)
                    if hero_ret == max(rets) and hero_ret > 0: wins += 1
                    _, board = parse_cards_from_info_str(st.information_state_string(cur_p))
                    if len(board) >= 5 or sum(r != 0 for r in rets) >= 3: showdowns += 1

    rets_arr = np.array(returns_list)
    mean_val = np.mean(rets_arr)
    std_val = np.std(rets_arr, ddof=1)
    ci95 = 1.96 * (std_val / np.sqrt(len(rets_arr)))
    
    act_counts = np.bincount(actions_taken, minlength=4)
    allin_pct = (act_counts[3] / len(actions_taken)) * 100
    fold_pct = (act_counts[0] / len(actions_taken)) * 100
    
    return {
        "hand": target_hand_name,
        "n_samples": len(rets_arr),
        "mean_ev": mean_val,
        "std_ev": std_val,
        "ci95": ci95,
        "ci_lower": mean_val - ci95,
        "ci_upper": mean_val + ci95,
        "win_rate": (wins / len(rets_arr)) * 100,
        "allin_pct": allin_pct,
        "fold_pct": fold_pct
    }

def run_phase5_audit():
    print("=" * 80)
    print("🔍 [V4.8.1 PHASE 5] 핸드별 EV 측정 표본 전수 감사 및 정밀 재측정 (N=300)")
    print("=" * 80)
    
    print("⚠️ [V4.8 표본 수 감사 결과]")
    print("  - V4.8 phase7_deep_validation.py 확인 결과: num_trials = 30 으로 실행됨.")
    print("  - 표본 N=30은 표준오차(SE = std / sqrt(30))가 수천 칩에 달하여 신뢰구간이 지나치게 넓은 한계가 확인됨.")
    print("  - 본 V4.8.1에서는 표본 수를 N=300으로 10배 확대하여 통계적 신뢰성을 확보함.\n")
    
    device = torch.device("cpu")
    ppo_model = HoldemActorCritic(253, 256, 4).to(device)
    ppo_model.load_state_dict(torch.load("v48/models/Exp_D_PPO_League_step_10k.pt", map_location=device))
    ppo_model.eval()
    
    old_model = Holdem5PNet(253, 256, 4).to(device)
    old_model.load_state_dict(torch.load("v47/models/holdem_5p_step_100k.pt", map_location=device))
    old_model.eval()
    
    target_hands = [
        ("AA",  ["Ah", "As"]),
        ("KK",  ["Kh", "Ks"]),
        ("AKs", ["Ah", "Kh"]),
        ("72o", ["7h", "2s"])
    ]
    
    csv_rows = []
    print(f"{'핸드':<6} | {'모델':<14} | {'표본':<5} | {'평균 EV (Chips)':<18} | {'95% CI':<24} | {'승률':<6} | {'All-in%'}")
    print("-" * 88)
    
    for name, cards in target_hands:
        # PPO League
        res_ppo = measure_rigorous_hand_ev(ppo_model, is_ppo=True, target_hand_name=name, target_hand_cards=cards, num_samples=300)
        # Step 100k
        res_old = measure_rigorous_hand_ev(old_model, is_ppo=False, target_hand_name=name, target_hand_cards=cards, num_samples=300)
        
        for m_lbl, r in [("PPO League", res_ppo), ("Step 100k", res_old)]:
            print(f"{name:<6} | {m_lbl:<14} | {r['n_samples']:<5} | {r['mean_ev']:+10.1f} chips    | [{r['ci_lower']:+8.1f}, {r['ci_upper']:+8.1f}]  | {r['win_rate']:>4.1f}% | {r['allin_pct']:>5.1f}%")
            csv_rows.append({
                "hand": name,
                "model": m_lbl,
                "n_samples": r["n_samples"],
                "mean_ev": round(r["mean_ev"], 1),
                "std_ev": round(r["std_ev"], 1),
                "ci95": round(r["ci95"], 1),
                "ci_lower": round(r["ci_lower"], 1),
                "ci_upper": round(r["ci_upper"], 1),
                "win_rate": round(r["win_rate"], 1),
                "allin_pct": round(r["allin_pct"], 1),
                "fold_pct": round(r["fold_pct"], 1)
            })
        print("-" * 88)
        
    csv_path = "v481/data/hand_ev_audit_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"\n💾 정밀 핸드 EV 감사 결과 CSV 저장 완료: {csv_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_phase5_audit()
