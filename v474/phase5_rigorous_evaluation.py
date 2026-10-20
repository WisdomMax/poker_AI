"""
POKER AI V4.7.4 - PHASE 5: 정상 무작위 환경 정밀 재평가 (Rigorous Benchmark Re-evaluation)
- 평가 대상 모델:
  1. Step 10k (v47/models/holdem_5p_step_10k.pt)
  2. Step 50k (v47/models/holdem_5p_step_50k.pt)
  3. Step 100k (v47/models/holdem_5p_step_100k.pt)
- 대전 상대 (6종):
  1. RandomHoldemBot
  2. AlwaysCallHoldemBot
  3. TightHoldemBot
  4. AggressiveHoldemBot
  5. RuleBasedHoldemBot
  6. PremiumHunterBot (Anti-All-in Trap Bot)
- 통제 조건:
  - 10개 독립 시드 (Seed 1001 ~ 1010)
  - 공통 무작위 카드 샘플러 (advance_through_chance_nodes)
  - 5개 포지션(SB, BB, UTG, CO, BTN) 완전 순환 (HoldemSeatRotationWrapper)
  - 95% 신뢰구간 (CI = mean +- 1.96 * std / sqrt(N))
  - 약한 카드(Trash Hand: 2-7, 3-8 offsuit 등) 수신 시 올인 및 폴드 빈도 추적
  - 결함 환경(V4.7.1/V4.7.2) vs 정상 환경(V4.7.4) 수치 비교
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
from v47.phase1_holdem_rules_audit import get_5p_holdem_game
from v47.phase2_position_fairness import HoldemSeatRotationWrapper, POSITION_NAMES
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v47.phase5_benchmark_eval import (
    RandomHoldemBot, AlwaysCallHoldemBot, TightHoldemBot, AggressiveHoldemBot, RuleBasedHoldemBot
)
from v472.phase3_anti_allin_bots import PremiumHunterBot
from common_poker_utils import advance_through_chance_nodes

def is_trash_hand(hole_cards_str: str) -> bool:
    """
    대표적인 약한 카드 (27o, 28o, 37o, 38o, 48o, 29o 등 노페어 노수티드 10이하) 판별
    """
    if len(hole_cards_str) < 4: return False
    c1, c2 = hole_cards_str[:2], hole_cards_str[2:4]
    r1, r2 = c1[0], c2[0]
    s1, s2 = c1[1], c2[1]
    
    # 페어나 수티드는 제외
    if r1 == r2 or s1 == s2: return False
    
    # T, J, Q, K, A가 포함되어 있으면 제외
    high_ranks = set(['T', 'J', 'Q', 'K', 'A'])
    if r1 in high_ranks or r2 in high_ranks: return False
    
    # 대표적인 언더카드 조합 (예: 2와 7, 3과 8 등)
    return True

def evaluate_model_on_opponents(
    model_name: str,
    model_path: str,
    seeds: List[int],
    hands_per_seed: int = 500
) -> Dict:
    device = torch.device("cpu")
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    
    net = Holdem5PNet(253, 256, 4).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    # 6종 상대 봇 풀
    opp_pool = [
        RandomHoldemBot(),
        AlwaysCallHoldemBot(),
        TightHoldemBot(),
        AggressiveHoldemBot(),
        RuleBasedHoldemBot(),
        PremiumHunterBot()
    ]
    
    seed_bb100_results = []
    seed_net_chips_results = []
    
    opp_profit_accum = {bot.name: 0.0 for bot in opp_pool}
    opp_hands_accum = {bot.name: 0 for bot in opp_pool}
    
    trash_decisions_total = 0
    trash_allin_count = 0
    trash_fold_count = 0
    
    target_agent_id = 0
    action_names = ["Fold", "Call", "Bet", "All-in"]
    
    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        rng = np.random.RandomState(s)
        
        seed_chips = 0.0
        
        for h in range(hands_per_seed):
            s2a = wrapper.get_seat_to_agent(h)
            a2s = wrapper.get_agent_to_seat(h)
            ai_seat = a2s[target_agent_id]
            
            st = game.new_initial_state()
            hero_hole_str = ""
            
            while not st.is_terminal():
                if st.is_chance_node():
                    advance_through_chance_nodes(st, rng=rng)
                else:
                    cur_seat = st.current_player()
                    legal = st.legal_actions(cur_seat)
                    cur_str = st.information_state_string(cur_seat)
                    
                    if cur_seat == ai_seat and not hero_hole_str:
                        if "Private: " in cur_str:
                            hero_hole_str = cur_str.split("Private: ")[1].split("]")[0]
                            
                    if cur_seat == ai_seat:
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
                        
                        # 약한 카드 행동 양상 추적
                        if is_trash_hand(hero_hole_str):
                            trash_decisions_total += 1
                            if act == 3: trash_allin_count += 1
                            elif act == 0: trash_fold_count += 1
                    else:
                        opp_id = s2a[cur_seat]
                        # 6종 상대 풀 전체가 매 핸드 골고루 참여하도록 핸드 인덱스 h와 결합
                        opp_idx = (h + opp_id - 1) % len(opp_pool)
                        opp_bot = opp_pool[opp_idx]
                        act = opp_bot.act(st, cur_seat, legal)
                        
                    st.apply_action(act)
                    
            returns = st.returns()
            hero_return = returns[ai_seat]
            seed_chips += hero_return
            
            # 상대 봇별 손익 귀속 (상대의 손실 = AI의 수익)
            for s_idx in range(5):
                if s_idx != ai_seat:
                    opp_id = s2a[s_idx]
                    bot_idx = (h + opp_id - 1) % len(opp_pool)
                    bot_name = opp_pool[bot_idx].name
                    opp_profit_accum[bot_name] -= returns[s_idx]
                    opp_hands_accum[bot_name] += 1
                    
        bb100 = (seed_chips / hands_per_seed) / 1.0 # BB = 100 chips
        seed_bb100_results.append(bb100)
        seed_net_chips_results.append(seed_chips)
        
    mean_bb100 = np.mean(seed_bb100_results)
    std_bb100 = np.std(seed_bb100_results, ddof=1)
    ci95 = 1.96 * (std_bb100 / np.sqrt(len(seeds)))
    
    trash_allin_pct = (trash_allin_count / trash_decisions_total * 100) if trash_decisions_total > 0 else 0.0
    trash_fold_pct = (trash_fold_count / trash_decisions_total * 100) if trash_decisions_total > 0 else 0.0
    
    return {
        "model_name": model_name,
        "mean_bb100": mean_bb100,
        "std_bb100": std_bb100,
        "ci95": ci95,
        "mean_chips_per_hand": np.mean(seed_net_chips_results) / hands_per_seed,
        "seed_bb100_list": seed_bb100_results,
        "opp_profits": {k: opp_profit_accum[k] / max(1, opp_hands_accum[k]) for k in opp_profit_accum},
        "trash_allin_pct": trash_allin_pct,
        "trash_fold_pct": trash_fold_pct,
        "trash_decisions": trash_decisions_total
    }

def run_phase5_rigorous_evaluation():
    print("=" * 80)
    print("🚀 [V4.7.4 PHASE 5] 정상 무작위 환경 엄밀 벤치마크 평가 (10개 시드, 6종 상대)")
    print("=" * 80)
    
    seeds = [1001 + i for i in range(10)]
    models = [
        ("Step 10k", "v47/models/holdem_5p_step_10k.pt", -24.45), # (이름, 경로, 기존 결함보고치)
        ("Step 50k", "v47/models/holdem_5p_step_50k.pt", +499.68),
        ("Step 100k", "v47/models/holdem_5p_step_100k.pt", +1037.04)
    ]
    
    summary_results = []
    
    for m_name, m_path, old_reported_bb100 in models:
        print(f"\nEvaluating {m_name} across 10 independent seeds ({len(seeds)*500:,} hands)...")
        res = evaluate_model_on_opponents(m_name, m_path, seeds, hands_per_seed=500)
        res["old_reported_bb100"] = old_reported_bb100
        summary_results.append(res)
        
        print(f"  👉 {m_name} 정상 환경 성적: {res['mean_bb100']:+8.2f} ± {res['ci95']:.2f} bb/100 (95% CI)")
        print(f"     [시드별 분포]: {[round(x, 1) for x in res['seed_bb100_list']]}")
        print(f"     [약한 카드(Trash Hand) 행동]: All-in {res['trash_allin_pct']:.1f}%, Fold {res['trash_fold_pct']:.1f}% (총 {res['trash_decisions']:,}회)")
        print(f"     [기존 결함 보고치]: {old_reported_bb100:+8.2f} bb/100 -> 변동폭: {res['mean_bb100'] - old_reported_bb100:+8.2f} bb/100")

    # 결과 종합 비교표 출력
    print("\n" + "=" * 80)
    print("📊 [V4.7.4 정상 환경 vs 기존 결함 환경 최종 비교표]")
    print("=" * 80)
    print(f"{'모델명':<10} | {'기존 결함 보고치':<16} | {'신규 정상 측정치 (95% CI)':<26} | {'약한 패 올인율':<12} | {'약한 패 폴드율'}")
    print("-" * 80)
    for r in summary_results:
        print(f"{r['model_name']:<10} | {r['old_reported_bb100']:+8.2f} bb/100     | {r['mean_bb100']:+8.2f} ± {r['ci95']:5.2f} bb/100     | {r['trash_allin_pct']:>6.1f}%      | {r['trash_fold_pct']:>6.1f}%")

    print("\n[상대 봇별 세부 성적 (Chips/Hand 기여도)]")
    for r in summary_results:
        print(f"\n  🎯 [{r['model_name']}]")
        for opp_name, prof in r["opp_profits"].items():
            print(f"    - vs {opp_name:<16}: {prof:+8.2f} chips/hand")

    # CSV 저장
    csv_file = "v474/data/rigorous_eval_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model_name", "old_reported_bb100", "mean_bb100", "std_bb100", "ci95", "trash_allin_pct", "trash_fold_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_results:
            writer.writerow({
                "model_name": r["model_name"],
                "old_reported_bb100": r["old_reported_bb100"],
                "mean_bb100": round(r["mean_bb100"], 2),
                "std_bb100": round(r["std_bb100"], 2),
                "ci95": round(r["ci95"], 2),
                "trash_allin_pct": round(r["trash_allin_pct"], 1),
                "trash_fold_pct": round(r["trash_fold_pct"], 1)
            })
    print(f"\n💾 정상 벤치마크 평가 결과 CSV 저장 완료: {csv_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_phase5_rigorous_evaluation()
