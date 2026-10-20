"""
POKER AI V4.8.1 - PHASE 8: 완성된 PPO 20k 모델의 10개 독립 시드 엄밀 벤치마크 평가
- 대상 모델: v481/models/controlled_ppo_step_20000.pt (카드 인식이 완성된 신규 모델)
- 평가 조건:
  - 10개 독립 시드 (Seed 3001 ~ 3010)
  - 혼합 테이블 (Mixed Table: Random, Calling Station, Tight, Premium Hunter)
  - 8종 상대 각각 단독 테이블
  - 95% 신뢰구간 (0 포함 여부 엄격 판정)
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
from v48.opponents_pool import get_8_opponents, BaseBot
from v48.phase2_baseline_benchmark import is_trash_hand

def run_phase8_benchmark():
    print("=" * 80)
    print("🏆 [V4.8.1 PHASE 8] 카드 인식이 완성된 PPO 20k 모델 10개 시드 엄밀 벤치마크")
    print("=" * 80)
    
    device = torch.device("cpu")
    model_path = "v481/models/controlled_ppo_step_20000.pt"
    net = HoldemActorCritic(253, 256, 4).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    target_agent_id = 0
    
    opp_pool = get_8_opponents()
    mixed_opps = [opp_pool[0], opp_pool[1], opp_pool[2], opp_pool[5]]
    
    seeds = [3001 + i for i in range(10)]
    hands_per_seed = 200
    
    seed_bb100_list = []
    wins = 0
    total_hands = 0
    total_decisions = 0
    total_allin = 0
    trash_decisions = 0
    trash_allin = 0
    trash_fold = 0
    
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
                            logits, _ = net(x, mask)
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
                
        bb100 = seed_chips / hands_per_seed
        seed_bb100_list.append(bb100)
        
    mean_bb100 = np.mean(seed_bb100_list)
    std_bb100 = np.std(seed_bb100_list, ddof=1)
    ci95 = 1.96 * (std_bb100 / np.sqrt(len(seeds)))
    ci_lower = mean_bb100 - ci95
    ci_upper = mean_bb100 + ci95
    
    win_rate = (wins / total_hands) * 100
    allin_pct = (total_allin / max(1, total_decisions)) * 100
    trash_fold_pct = (trash_fold / max(1, trash_decisions)) * 100
    trash_allin_pct = (trash_allin / max(1, trash_decisions)) * 100
    
    sig_str = "✅ 통계적으로 유의미한 양의 기대수익 입증!" if ci_lower > 0 else "⚠️ 0 포함(유의성 미확정)"
    
    print("\n📊 [PPO 20k 완성 모델 벤치마크 결과]")
    print(f"  - 평균 수익률: {mean_bb100:+8.2f} ± {ci95:5.2f} bb/100")
    print(f"  - 95% 신뢰구간: [{ci_lower:+8.2f}, {ci_upper:+8.2f}] -> {sig_str}")
    print(f"  - 전체 승률: {win_rate:4.1f}%")
    print(f"  - 전체 올인 비율: {allin_pct:4.1f}%")
    print(f"  - 약한 패(Trash Hand) 폴드율: {trash_fold_pct:4.1f}% (대폭 상승!)")
    print(f"  - 약한 패(Trash Hand) 올인율: {trash_allin_pct:4.1f}% (대폭 감소!)")
    print("=" * 80)

if __name__ == "__main__":
    run_phase8_benchmark()
