"""
POKER AI V4.9 - PHASE 6: 9대 상대별 독립 평가 (Independent Evaluation against 9 Opponents)
- 9대 상대 봇:
  [학습 참여 봇]
  1. Random
  2. Always Call
  3. Tight
  4. Aggressive
  5. Premium Hunter
  6. Selective Caller
  7. Bluff Catcher
  [평가 전용 미학습 봇 (Unseen)]
  8. Rule-Based
  9. Fixed Adaptive
- 측정 항목:
  - 4인 동일 테이블 (각 300판) 및 9종 혼합 테이블 (1,000판)
  - bb/100, 95% 신뢰구간, 상대별 기대수익, 포지션별 기대수익(SB, BB, UTG, CO, BTN),
    올인 빈도, Half-Pot 사용률, 핸드 강도별 기대수익, 손익 분산(std), 최대 손실(Max Drawdown)
- 3대 조건(Cond A vs Cond B vs Cond C) 전수 비교
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
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, POSITION_NAMES
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v49.model_adapter_4a_to_5a import FourActionToFiveActionAdapter
from v48.ppo_actor_critic import HoldemActorCritic
from v48.opponents_pool import (
    BaseBot, RandomBot, AlwaysCallBot, TightBot, AggressiveBot,
    RuleBasedBot, PremiumHunterBot
)
from v483.phase4_unseen_opponents_audit import (
    SelectiveCallerBot, BluffCatcherBot, FixedOpponentAdaptiveBot
)
from v48.phase2_baseline_benchmark import is_trash_hand

SEEDS = [42, 101, 202, 303, 404]

def get_9_opponents_dict() -> Dict[str, BaseBot]:
    return {
        "Random": RandomBot(),
        "Always Call": AlwaysCallBot(),
        "Tight": TightBot(),
        "Aggressive": AggressiveBot(),
        "Premium Hunter": PremiumHunterBot(),
        "Selective Caller": SelectiveCallerBot(),
        "Bluff Catcher": BluffCatcherBot(),
        "Rule-Based (Unseen)": RuleBasedBot(),
        "Fixed Adaptive (Unseen)": FixedOpponentAdaptiveBot()
    }

def evaluate_table_detailed(
    model: torch.nn.Module,
    opponents: List[BaseBot],
    num_hands: int = 500,
    seed: int = 99999,
    device: torch.device = torch.device("cpu")
) -> Dict:
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    rng = np.random.RandomState(seed)
    
    chips_list = []
    wins = 0
    allin_acts = 0
    halfpot_acts = 0
    total_acts = 0
    
    pos_chips = {p: 0.0 for p in range(5)}
    pos_hands = {p: 0 for p in range(5)}
    
    prem_chips = 0.0
    prem_count = 0
    trash_chips = 0.0
    trash_count = 0
    
    model.eval()
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        hero_seat = a2s[0]
        pos_hands[hero_seat] += 1
        
        st = game.new_initial_state()
        hero_hole = []
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                cur_str = st.information_state_string(cur_seat)
                
                if cur_seat == hero_seat and not hero_hole:
                    hero_hole, _ = parse_cards_from_info_str(cur_str)
                    
                if cur_seat == hero_seat:
                    total_acts += 1
                    vec = st.information_state_tensor(cur_seat)
                    x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.zeros((1, 5), dtype=torch.bool, device=device)
                    for a in legal: m[0, a] = True
                    
                    with torch.no_grad():
                        logits, _ = model(x, m)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prs = [probs[a] for a in legal]
                    prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                    act = int(rng.choice(legal, p=prs))
                    if act == 3: allin_acts += 1
                    elif act == 4: halfpot_acts += 1
                else:
                    ag_id = s2a[cur_seat]
                    opp = opponents[(ag_id - 1) % len(opponents)]
                    act = opp.act(st, cur_seat, legal)
                st.apply_action(act)
                
        ret = st.returns()[hero_seat]
        chips_list.append(ret)
        pos_chips[hero_seat] += ret
        if ret == max(st.returns()) and ret > 0: wins += 1
        
        if len(hero_hole) == 2:
            r1, r2 = hero_hole[0][0], hero_hole[1][0]
            if (r1 == r2 and r1 in ['A', 'K', 'Q']) or (r1 == 'A' and r2 == 'K') or (r2 == 'A' and r1 == 'K'):
                prem_chips += ret
                prem_count += 1
            elif is_trash_hand(hero_hole):
                trash_chips += ret
                trash_count += 1
                
    bb_arr = np.array(chips_list)
    mean_bb = float(np.mean(bb_arr)) # chip mean (per hand = 100칩 기준 bb/100)
    std_bb = float(np.std(bb_arr, ddof=1))
    
    # 95% CI (게임 레벨 변동성)
    n = len(bb_arr)
    se = std_bb / np.sqrt(n)
    t_c = stats.t.ppf(0.975, df=n-1)
    ci_game = (mean_bb - t_c * se, mean_bb + t_c * se)
    
    # 최대 손실 (Max Single Loss)
    max_loss = float(np.min(bb_arr))
    
    pos_ev = {POSITION_NAMES[p]: (pos_chips[p] / max(1, pos_hands[p])) for p in range(5)}
    
    return {
        "bb100": mean_bb,
        "std_bb": std_bb,
        "ci_game": ci_game,
        "win_rate": (wins / float(num_hands)) * 100.0,
        "allin_pct": (allin_acts / max(1, total_acts)) * 100.0,
        "halfpot_pct": (halfpot_acts / max(1, total_acts)) * 100.0,
        "max_loss": max_loss,
        "prem_ev": (prem_chips / max(1, prem_count)),
        "trash_ev": (trash_chips / max(1, trash_count)),
        "pos_ev": pos_ev
    }

def run_phase6_evaluations():
    print("=" * 95)
    print("📊 [POKER AI V4.9] PHASE 6: 9대 상대별 독립 정밀 평가 (동일 4인 & 혼합 테이블)")
    print("=" * 95)
    
    device = torch.device("cpu")
    opponents_dict = get_9_opponents_dict()
    
    # 대표 모델 로드
    # Cond A (4A Adapter Seed 42 20k)
    m4 = HoldemActorCritic(253, 256, 4).to(device)
    m4.load_state_dict(torch.load("v483/models/seed_42_step_20000.pt", map_location=device))
    mod_a = FourActionToFiveActionAdapter(m4).to(device)
    
    # Cond B (5A Scratch Seed 42 20k)
    mod_b = HoldemActorCritic5A(265, 256, 5).to(device)
    mod_b.load_state_dict(torch.load("v49/models/CondB_5A_Scratch_seed_42_step_20000.pt", map_location=device))
    
    # Cond C (5A Transfer Seed 42 20k)
    mod_c = HoldemActorCritic5A(265, 256, 5).to(device)
    mod_c.load_state_dict(torch.load("v49/models/CondC_5A_Transfer_seed_42_step_20000.pt", map_location=device))
    
    eval_models = {
        "CondA (4A Adapter)": mod_a,
        "CondB (5A Scratch)": mod_b,
        "CondC (5A Transfer)": mod_c
    }
    
    results = []
    
    print("\n[1] 9대 상대별 4인 동일 테이블 평가 (각 300판):")
    print(f"{'상대 유형':<24} | {'Cond A (4A)':<16} | {'Cond B (Scratch)':<18} | {'Cond C (Transfer)':<18} | {'우위 조건'}")
    print("-" * 95)
    
    for oname, bot in opponents_dict.items():
        opp_list = [bot] * 4
        res_a = evaluate_table_detailed(mod_a, opp_list, num_hands=300, seed=1111)
        res_b = evaluate_table_detailed(mod_b, opp_list, num_hands=300, seed=1111)
        res_c = evaluate_table_detailed(mod_c, opp_list, num_hands=300, seed=1111)
        
        best = "Cond C" if res_c["bb100"] >= max(res_a["bb100"], res_b["bb100"]) else ("Cond B" if res_b["bb100"] >= res_a["bb100"] else "Cond A")
        print(f"{oname:<24} | {res_a['bb100']:>+8.1f} bb/100  | {res_b['bb100']:>+10.1f} bb/100   | {res_c['bb100']:>+10.1f} bb/100   | 🏆 {best}")
        results.append({
            "opp_type": oname,
            "bb100_a": res_a["bb100"], "bb100_b": res_b["bb100"], "bb100_c": res_c["bb100"],
            "halfpot_c": res_c["halfpot_pct"], "allin_c": res_c["allin_pct"]
        })
        
    # [2] 9종 혼합 테이블 평가 (1,000판)
    print("\n" + "=" * 95)
    print("[2] 9종 혼합 테이블 종합 평가 (1,000판 대규모 독립 테스트):")
    mixed_opps = list(opponents_dict.values())
    mix_res_a = evaluate_table_detailed(mod_a, mixed_opps, num_hands=1000, seed=22222)
    mix_res_b = evaluate_table_detailed(mod_b, mixed_opps, num_hands=1000, seed=22222)
    mix_res_c = evaluate_table_detailed(mod_c, mixed_opps, num_hands=1000, seed=22222)
    
    print(f"{'지표':<25} | {'Cond A (4A Adapter)':<20} | {'Cond B (5A Scratch)':<20} | {'Cond C (5A Transfer)'}")
    print("-" * 95)
    print(f"{'평균 수익률 (bb/100)':<25} | {mix_res_a['bb100']:>+12.1f} bb/100   | {mix_res_b['bb100']:>+12.1f} bb/100   | {mix_res_c['bb100']:>+12.1f} bb/100")
    print(f"{'게임 95% 신뢰구간':<25} | [{mix_res_a['ci_game'][0]:>+6.1f}, {mix_res_a['ci_game'][1]:>+6.1f}]       | [{mix_res_b['ci_game'][0]:>+6.1f}, {mix_res_b['ci_game'][1]:>+6.1f}]       | [{mix_res_c['ci_game'][0]:>+6.1f}, {mix_res_c['ci_game'][1]:>+6.1f}]")
    print(f"{'승률 (Win Rate)':<25} | {mix_res_a['win_rate']:>12.1f} %       | {mix_res_b['win_rate']:>12.1f} %       | {mix_res_c['win_rate']:>12.1f} %")
    print(f"{'All-in 비율':<25} | {mix_res_a['allin_pct']:>12.1f} %       | {mix_res_b['allin_pct']:>12.1f} %       | {mix_res_c['allin_pct']:>12.1f} %")
    print(f"{'Half-Pot 사용률':<25} | {mix_res_a['halfpot_pct']:>12.1f} %       | {mix_res_b['halfpot_pct']:>12.1f} %       | {mix_res_c['halfpot_pct']:>12.1f} %")
    print(f"{'손익 분산 (표준편차)':<25} | {mix_res_a['std_bb']:>12.1f} chips     | {mix_res_b['std_bb']:>12.1f} chips     | {mix_res_c['std_bb']:>12.1f} chips")
    print(f"{'프리미엄 핸드 EV':<25} | {mix_res_a['prem_ev']:>+12.1f} chips   | {mix_res_b['prem_ev']:>+12.1f} chips   | {mix_res_c['prem_ev']:>+12.1f} chips")
    print(f"{'트래시 핸드 EV':<25} | {mix_res_a['trash_ev']:>+12.1f} chips   | {mix_res_b['trash_ev']:>+12.1f} chips   | {mix_res_c['trash_ev']:>+12.1f} chips")
    
    print("\n[3] 포지션별 기대수익 (bb/100):")
    for pos in ["SB (Small Blind)", "BB (Big Blind)", "UTG (Under the Gun)", "CO (Cutoff)", "BTN (Button)"]:
        print(f"  - {pos:<22}: Cond A = {mix_res_a['pos_ev'][pos]:>+7.1f} | Cond B = {mix_res_b['pos_ev'][pos]:>+7.1f} | Cond C = {mix_res_c['pos_ev'][pos]:>+7.1f}")
        
    print("=" * 95)
    
    # CSV 저장
    csv_path = "v49/data/phase6_opponents_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ 상대별 평가 결과 저장: {csv_path}")

if __name__ == "__main__":
    run_phase6_evaluations()
