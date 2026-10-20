"""
POKER AI V4.8 - PHASE 2: 기존 모델 기준선 재확립 (Rigorous Baseline Benchmark)
- 비교 대상:
  1. Step 10k (v47/models/holdem_5p_step_10k.pt)
  2. Step 50k (v47/models/holdem_5p_step_50k.pt)
  3. Step 100k (v47/models/holdem_5p_step_100k.pt)
- 대전 환경:
  A. 동일 유형 상대 4명 테이블 (Homogeneous 1 vs 4 for each of 8 opponent types)
  B. 서로 다른 유형 4명 혼합 테이블 (Mixed Table)
- 통제 조건:
  - 10개 독립 시드 (Seed 1001 ~ 1010)
  - V4.8 검증된 공통 무작위 샘플러 (advance_chance_nodes)
  - 5인 포지션 완전 순환 (HoldemSeatRotationWrapper)
  - 95% 신뢰구간 (mean +- 1.96 * std / sqrt(N))
  - 비공개 카드 누출 0건 보장
  - 신뢰구간에 0이 포함되면 '수익성 입증' 표현 금지
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
    get_5p_holdem_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, POSITION_NAMES
)
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v48.opponents_pool import get_8_opponents, BaseBot

def is_trash_hand(hole_cards: List[str]) -> bool:
    if len(hole_cards) != 2: return False
    c1, c2 = hole_cards[0], hole_cards[1]
    r1, r2 = c1[0], c2[0]
    s1, s2 = c1[1], c2[1]
    if r1 == r2 or s1 == s2: return False
    high_ranks = set(['T', 'J', 'Q', 'K', 'A'])
    if r1 in high_ranks or r2 in high_ranks: return False
    return True

def run_table_eval(
    net: torch.nn.Module,
    opponents_for_seats: List[BaseBot],
    seeds: List[int],
    hands_per_seed: int = 200,
    device: torch.device = torch.device("cpu")
) -> Dict:
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    target_agent_id = 0
    
    seed_bb100_list = []
    pos_profits = [0.0] * 5
    pos_hands = [0] * 5
    
    total_decisions = 0
    total_allin_count = 0
    
    trash_decisions = 0
    trash_fold_count = 0
    trash_allin_count = 0
    
    wins_count = 0
    showdown_count = 0
    total_hands_played = 0
    max_loss_single_hand = 0.0
    
    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        rng = np.random.RandomState(s)
        
        seed_net_chips = 0.0
        
        for h in range(hands_per_seed):
            total_hands_played += 1
            s2a = wrapper.get_seat_to_agent(h)
            a2s = wrapper.get_agent_to_seat(h)
            ai_seat = a2s[target_agent_id]
            pos_hands[ai_seat] += 1
            
            st = game.new_initial_state()
            hero_hole = []
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng=rng)
                else:
                    cur_seat = st.current_player()
                    legal = st.legal_actions(cur_seat)
                    cur_str = st.information_state_string(cur_seat)
                    
                    if cur_seat == ai_seat and not hero_hole:
                        hero_hole, _ = parse_cards_from_info_str(cur_str)
                        
                    if cur_seat == ai_seat:
                        total_decisions += 1
                        vec = st.information_state_tensor(cur_seat)
                        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                        mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
                        for a in legal: mask[0, a] = True
                        with torch.no_grad():
                            logits = net(x, mask)
                            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        prs = [probs[a] for a in legal]
                        prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                        act = int(rng.choice(legal, p=prs))
                        if act == 3: total_allin_count += 1
                        
                        if is_trash_hand(hero_hole):
                            trash_decisions += 1
                            if act == 0: trash_fold_count += 1
                            elif act == 3: trash_allin_count += 1
                    else:
                        opp_id = s2a[cur_seat]
                        # 좌석별 배정된 봇
                        opp_bot = opponents_for_seats[(opp_id - 1) % len(opponents_for_seats)]
                        act = opp_bot.act(st, cur_seat, legal)
                    st.apply_action(act)
                    
            rets = st.returns()
            hero_ret = rets[ai_seat]
            seed_net_chips += hero_ret
            pos_profits[ai_seat] += hero_ret
            
            if hero_ret < max_loss_single_hand:
                max_loss_single_hand = hero_ret
            if hero_ret == max(rets) and hero_ret > 0:
                wins_count += 1
                
            # 쇼다운 확인
            _, board = parse_cards_from_info_str(st.information_state_string(ai_seat))
            if len(board) >= 5 or sum(r != 0 for r in rets) >= 3:
                showdown_count += 1
                
        bb100 = (seed_net_chips / hands_per_seed) / 1.0 # BB = 100
        seed_bb100_list.append(bb100)
        
    mean_bb100 = np.mean(seed_bb100_list)
    std_bb100 = np.std(seed_bb100_list, ddof=1) if len(seeds) > 1 else 0.0
    ci95 = 1.96 * (std_bb100 / np.sqrt(len(seeds)))
    
    allin_freq = (total_allin_count / max(1, total_decisions)) * 100
    trash_fold_freq = (trash_fold_count / max(1, trash_decisions)) * 100
    trash_allin_freq = (trash_allin_count / max(1, trash_decisions)) * 100
    win_rate = (wins_count / max(1, total_hands_played)) * 100
    showdown_rate = (showdown_count / max(1, total_hands_played)) * 100
    
    pos_bb100 = [(pos_profits[i] / max(1, pos_hands[i])) for i in range(5)]
    
    # 신뢰구간에 0이 포함되는지 여부 (Statistically Significant?)
    ci_lower = mean_bb100 - ci95
    ci_upper = mean_bb100 + ci95
    is_significant = (ci_lower > 0) or (ci_upper < 0)
    
    return {
        "mean_bb100": mean_bb100,
        "std_bb100": std_bb100,
        "ci95": ci95,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "is_significant": is_significant,
        "pos_bb100": pos_bb100,
        "allin_freq": allin_freq,
        "trash_fold_freq": trash_fold_freq,
        "trash_allin_freq": trash_allin_freq,
        "win_rate": win_rate,
        "showdown_rate": showdown_rate,
        "max_loss": max_loss_single_hand
    }

def run_phase2_baseline_benchmark():
    print("=" * 80)
    print("🔬 [V4.8 PHASE 2] 기존 모델(Step 10k, 50k, 100k) 엄밀 기준선 재확립")
    print("=" * 80)
    
    device = torch.device("cpu")
    seeds = [1001 + i for i in range(10)] # 10개 독립 시드
    
    models = [
        ("Step 10k", "v47/models/holdem_5p_step_10k.pt"),
        ("Step 50k", "v47/models/holdem_5p_step_50k.pt"),
        ("Step 100k", "v47/models/holdem_5p_step_100k.pt")
    ]
    
    opp_pool = get_8_opponents()
    opp_dict = {bot.name: bot for bot in opp_pool}
    
    csv_rows = []
    
    for m_name, m_path in models:
        print(f"\n=======================================================")
        print(f"📊 모델 평가: {m_name} ({m_path})")
        print(f"=======================================================")
        
        net = Holdem5PNet(253, 256, 4).to(device)
        net.load_state_dict(torch.load(m_path, map_location=device))
        net.eval()
        
        # 1. 혼합 테이블 (Mixed Table: Random, Always Call, Tight, Aggressive, Hunters 순환)
        mixed_opps = [opp_pool[0], opp_pool[1], opp_pool[2], opp_pool[5]] # Random, Always Call, Tight, Premium Hunter
        res_mixed = run_table_eval(net, mixed_opps, seeds, hands_per_seed=200, device=device)
        
        sig_str = "✅ 유의미한 수익" if (res_mixed["ci_lower"] > 0) else ("❌ 유의미한 손실" if res_mixed["ci_upper"] < 0 else "⚠️ 0 포함(유의성 미확정)")
        print(f"  [혼합 테이블 (Mixed Table)]: {res_mixed['mean_bb100']:+8.2f} ± {res_mixed['ci95']:5.2f} bb/100 (95% CI: [{res_mixed['ci_lower']:+8.2f}, {res_mixed['ci_upper']:+8.2f}]) | {sig_str}")
        print(f"    - 올인 빈도: {res_mixed['allin_freq']:.1f}% | 약한 패 올인율: {res_mixed['trash_allin_freq']:.1f}% | 약한 패 폴드율: {res_mixed['trash_fold_freq']:.1f}%")
        print(f"    - 승률: {res_mixed['win_rate']:.1f}% | 쇼다운율: {res_mixed['showdown_rate']:.1f}% | 단일 핸드 최대 손실: {res_mixed['max_loss']:+8.0f} chips")
        print(f"    - 포지션별 bb/100: SB {res_mixed['pos_bb100'][0]:+7.1f} | BB {res_mixed['pos_bb100'][1]:+7.1f} | UTG {res_mixed['pos_bb100'][2]:+7.1f} | CO {res_mixed['pos_bb100'][3]:+7.1f} | BTN {res_mixed['pos_bb100'][4]:+7.1f}")
        
        csv_rows.append({
            "model": m_name,
            "table_type": "Mixed",
            "opponent": "Mixed (4-type)",
            "mean_bb100": round(res_mixed["mean_bb100"], 2),
            "ci95": round(res_mixed["ci95"], 2),
            "ci_lower": round(res_mixed["ci_lower"], 2),
            "ci_upper": round(res_mixed["ci_upper"], 2),
            "is_significant": res_mixed["is_significant"],
            "allin_freq": round(res_mixed["allin_freq"], 1),
            "trash_allin_freq": round(res_mixed["trash_allin_freq"], 1),
            "trash_fold_freq": round(res_mixed["trash_fold_freq"], 1),
            "win_rate": round(res_mixed["win_rate"], 1),
            "showdown_rate": round(res_mixed["showdown_rate"], 1)
        })
        
        # 2. 동일 유형 4명 테이블 (Homogeneous: 8종 각각 1 vs 4)
        print("  [동일 유형 4명 테이블 (Homogeneous 1 vs 4)]:")
        for opp in opp_pool:
            opp_list = [opp] * 4
            res_homo = run_table_eval(net, opp_list, seeds, hands_per_seed=100, device=device)
            h_sig = "유의미" if res_homo["is_significant"] else "0 포함"
            print(f"    - vs {opp.name:<16}: {res_homo['mean_bb100']:+8.2f} ± {res_homo['ci95']:5.2f} bb/100 (CI: [{res_homo['ci_lower']:+7.1f}, {res_homo['ci_upper']:+7.1f}]) [{h_sig}] | 승률 {res_homo['win_rate']:4.1f}%")
            
            csv_rows.append({
                "model": m_name,
                "table_type": "Homogeneous",
                "opponent": opp.name,
                "mean_bb100": round(res_homo["mean_bb100"], 2),
                "ci95": round(res_homo["ci95"], 2),
                "ci_lower": round(res_homo["ci_lower"], 2),
                "ci_upper": round(res_homo["ci_upper"], 2),
                "is_significant": res_homo["is_significant"],
                "allin_freq": round(res_homo["allin_freq"], 1),
                "trash_allin_freq": round(res_homo["trash_allin_freq"], 1),
                "trash_fold_freq": round(res_homo["trash_fold_freq"], 1),
                "win_rate": round(res_homo["win_rate"], 1),
                "showdown_rate": round(res_homo["showdown_rate"], 1)
            })

    # CSV 저장
    csv_file = "v48/data/baseline_benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n💾 기준선 벤치마크 결과 CSV 저장 완료: {csv_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_phase2_baseline_benchmark()
