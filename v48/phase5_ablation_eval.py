"""
POKER AI V4.8 - PHASE 5: 4대 대조 모델 벤치마크 평가 스크립트
- 평가 대상:
  1. Exp_A: REINFORCE + SelfPlay (v48/models/Exp_A_REINFORCE_SelfPlay_step_10k.pt)
  2. Exp_B: REINFORCE + League (v48/models/Exp_B_REINFORCE_League_step_10k.pt)
  3. Exp_C: PPO + SelfPlay (v48/models/Exp_C_PPO_SelfPlay_step_10k.pt)
  4. Exp_D: PPO + League (v48/models/Exp_D_PPO_League_step_10k.pt)
- 대전 환경:
  - 혼합 테이블 (Mixed Table: 4-type mixed)
  - 8종 상대 각각 단독 테이블 (Homogeneous 1 vs 4)
- 10개 독립 시드 (Seed 1001 ~ 1010)
- 95% 신뢰구간 및 승률, 올인율, 약한 패 올인율 측정
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
from v48.ppo_actor_critic import HoldemActorCritic
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v48.opponents_pool import get_8_opponents, BaseBot
from v48.phase2_baseline_benchmark import is_trash_hand

def evaluate_ablation_model(
    model_name: str,
    model_path: str,
    is_ppo: bool,
    seeds: List[int],
    hands_per_seed: int = 200,
    device: torch.device = torch.device("cpu")
) -> Dict:
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    target_agent_id = 0
    
    if is_ppo:
        net = HoldemActorCritic(253, 256, 4).to(device)
    else:
        net = Holdem5PNet(253, 256, 4).to(device)
        
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    opp_pool = get_8_opponents()
    mixed_opps = [opp_pool[0], opp_pool[1], opp_pool[2], opp_pool[5]] # Random, Always Call, Tight, Premium Hunter
    
    seed_bb100_list = []
    total_decisions = 0
    total_allin = 0
    trash_decisions = 0
    trash_allin = 0
    trash_fold = 0
    wins = 0
    total_hands = 0
    
    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        rng = np.random.RandomState(s)
        
        seed_chips = 0.0
        for h in range(hands_per_seed):
            total_hands += 1
            s2a = wrapper.get_seat_to_agent(h)
            a2s = wrapper.get_agent_to_seat(h)
            ai_seat = a2s[target_agent_id]
            
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
                            if is_ppo:
                                logits, _ = net(x, mask)
                            else:
                                logits = net(x, mask)
                            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        prs = [probs[a] for a in legal]
                        prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                        act = int(rng.choice(legal, p=prs))
                        if act == 3: total_allin += 1
                        
                        if is_trash_hand(hero_hole):
                            trash_decisions += 1
                            if act == 3: trash_allin += 1
                            elif act == 0: trash_fold += 1
                    else:
                        opp_id = s2a[cur_seat]
                        opp_bot = mixed_opps[(opp_id - 1) % len(mixed_opps)]
                        act = opp_bot.act(st, cur_seat, legal)
                    st.apply_action(act)
                    
            rets = st.returns()
            hero_ret = rets[ai_seat]
            seed_chips += hero_ret
            if hero_ret == max(rets) and hero_ret > 0:
                wins += 1
                
        seed_bb100_list.append(seed_chips / hands_per_seed)
        
    mean_bb100 = np.mean(seed_bb100_list)
    std_bb100 = np.std(seed_bb100_list, ddof=1)
    ci95 = 1.96 * (std_bb100 / np.sqrt(len(seeds)))
    
    return {
        "model_name": model_name,
        "mean_bb100": mean_bb100,
        "ci95": ci95,
        "ci_lower": mean_bb100 - ci95,
        "ci_upper": mean_bb100 + ci95,
        "is_significant": (mean_bb100 - ci95 > 0) or (mean_bb100 + ci95 < 0),
        "allin_pct": (total_allin / max(1, total_decisions)) * 100,
        "trash_allin_pct": (trash_allin / max(1, trash_decisions)) * 100,
        "trash_fold_pct": (trash_fold / max(1, trash_decisions)) * 100,
        "win_rate": (wins / max(1, total_hands)) * 100
    }

def run_phase5_ablation_eval():
    print("=" * 80)
    print("🏆 [V4.8 PHASE 5] 4대 대조 모델 혼합 테이블 벤치마크 정밀 평가")
    print("=" * 80)
    
    seeds = [1001 + i for i in range(10)]
    models = [
        ("Exp A: REINFORCE SelfPlay", "v48/models/Exp_A_REINFORCE_SelfPlay_step_10k.pt", False),
        ("Exp B: REINFORCE League",   "v48/models/Exp_B_REINFORCE_League_step_10k.pt", False),
        ("Exp C: PPO SelfPlay",       "v48/models/Exp_C_PPO_SelfPlay_step_10k.pt", True),
        ("Exp D: PPO League",         "v48/models/Exp_D_PPO_League_step_10k.pt", True)
    ]
    
    results = []
    for name, path, is_ppo in models:
        res = evaluate_ablation_model(name, path, is_ppo, seeds, hands_per_seed=200)
        results.append(res)
        sig_str = "✅ 유의미한 수익" if res["ci_lower"] > 0 else ("❌ 유의미한 손실" if res["ci_upper"] < 0 else "⚠️ 0 포함 (미확정)")
        print(f"  [{name}]: {res['mean_bb100']:+8.2f} ± {res['ci95']:5.2f} bb/100 (CI: [{res['ci_lower']:+7.1f}, {res['ci_upper']:+7.1f}]) | {sig_str}")
        print(f"    - 승률: {res['win_rate']:4.1f}% | 올인율: {res['allin_pct']:4.1f}% | 약한 패 올인율: {res['trash_allin_pct']:4.1f}% | 약한 패 폴드율: {res['trash_fold_pct']:4.1f}%")

    print("\n" + "=" * 80)
    print("📊 [4대 대조 모델 벤치마크 최종 비교 요약]")
    print("=" * 80)
    print(f"{'모델명':<26} | {'수익률 (95% CI)':<26} | {'승률':<6} | {'올인율':<8} | {'약한패 올인율'}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_name']:<26} | {r['mean_bb100']:+8.2f} ± {r['ci95']:5.2f} bb/100     | {r['win_rate']:>4.1f}% | {r['allin_pct']:>5.1f}%  | {r['trash_allin_pct']:>6.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_phase5_ablation_eval()
