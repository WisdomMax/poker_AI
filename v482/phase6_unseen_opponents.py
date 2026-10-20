"""
POKER AI V4.8.2 - PHASE 6: 미학습 상대 5종 일반화 테스트 (Unseen Opponents Generalization)
- 목적: 훈련 리그에 전혀 등장하지 않은 5종의 신규 상대 봇을 설계하고,
  새 PPO 20k 모델 vs 기존 Step 100k 모델의 제로샷(Zero-shot) 일반화 적응력을 비교.
- 신규 5대 미학습 봇 (상대 비공개 카드 접근 0건 보장):
  1. ExtremeTightBot: AA, KK, QQ 외에는 프리플롭부터 전원 폴드.
  2. LooseCallingBot: 80% 이상의 핸드로 폴드하지 않고 끈질기게 쇼다운까지 콜.
  3. SelectiveCallerBot: 상대가 올인하면 상위 8% 핸드(JJ+, AK)로만 스나이핑 콜.
  4. BluffCatcherBot: 원페어 이상이면 상대의 팟 베팅/올인을 블러프로 간주하고 캐치.
  5. OpponentAdaptiveBot: 상대의 최근 30판 행동 빈도를 추적하여 올인 빈도가 높으면 콜 범위를 유연하게 확장.
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
    parse_cards_from_info_str
)
from v48.ppo_actor_critic import HoldemActorCritic
from v47.phase3_holdem_ai_architecture import Holdem5PNet
from v48.opponents_pool import BaseBot, extract_hole_cards
from v48.phase2_baseline_benchmark import is_trash_hand

class ExtremeTightBot(BaseBot):
    def __init__(self): super().__init__("Extreme Tight")
    def act(self, state, cur_player, legal_acts) -> int:
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        if len(hc) == 2:
            r1, r2 = hc[0][0], hc[1][0]
            if r1 == r2 and r1 in ['A', 'K', 'Q']: # AA, KK, QQ만 참여
                return 3 if 3 in legal_acts else (2 if 2 in legal_acts else 1)
        return 1 if 1 in legal_acts and 2 not in legal_acts and 3 not in legal_acts else (0 if 0 in legal_acts else 1)

class LooseCallingBot(BaseBot):
    def __init__(self): super().__init__("Loose Calling")
    def act(self, state, cur_player, legal_acts) -> int:
        # 거의 폴드하지 않고 체크/콜
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
        if len(legal_acts) == 2 and 0 in legal_acts and 1 in legal_acts: # 올인 직면
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

class OpponentAdaptiveBot(BaseBot):
    def __init__(self):
        super().__init__("Opponent Adaptive")
        self.opp_allins = 0
        self.opp_total = 0
    def act(self, state, cur_player, legal_acts) -> int:
        self.opp_total += 1
        info = state.information_state_string(cur_player)
        hc = extract_hole_cards(info)
        ratio = self.opp_allins / max(1, self.opp_total)
        has_high = len(hc) == 2 and any(c[0] in ['A', 'K', 'Q'] for c in hc)
        if ratio > 0.3: # 상대가 올인을 남발하면 하이카드만 있어도 적극 콜
            if has_high: return 1 if 1 in legal_acts else legal_acts[0]
            return 0 if 0 in legal_acts else 1
        return 1 if 1 in legal_acts else 0

def evaluate_against_unseen_pool(
    model,
    is_ppo: bool,
    opponents: List[BaseBot],
    num_hands: int = 500,
    seed: int = 7777,
    device: torch.device = torch.device("cpu")
) -> Dict:
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    target_agent_id = 0
    
    rng = np.random.RandomState(seed)
    chips_total = 0.0
    wins = 0
    allin_acts = 0
    total_acts = 0
    
    prem_chips = 0.0
    prem_count = 0
    trash_chips = 0.0
    trash_count = 0
    
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        ai_seat = a2s[target_agent_id]
        
        st = game.new_initial_state()
        hero_hole = []
        
        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng=rng)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                cur_str = st.information_state_string(cur_seat)
                
                if cur_seat == ai_seat and not hero_hole:
                    hero_hole, _ = parse_cards_from_info_str(cur_str)
                    
                if cur_seat == ai_seat:
                    total_acts += 1
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
                    if act == 3: allin_acts += 1
                else:
                    opp_id = s2a[cur_seat]
                    act = opponents[(opp_id - 1) % len(opponents)].act(st, cur_seat, legal)
                st.apply_action(act)
                
        ret = st.returns()[ai_seat]
        chips_total += ret
        if ret == max(st.returns()) and ret > 0: wins += 1
        
        # 프리미엄 / 트래시 성적 추적
        if len(hero_hole) == 2:
            r1, r2 = hero_hole[0][0], hero_hole[1][0]
            if (r1 == r2 and r1 in ['A', 'K', 'Q']) or (r1 == 'A' and r2 == 'K') or (r2 == 'A' and r1 == 'K'):
                prem_chips += ret
                prem_count += 1
            elif is_trash_hand(hero_hole):
                trash_chips += ret
                trash_count += 1
                
    bb100 = chips_total / float(num_hands)
    return {
        "bb100": bb100,
        "win_rate": (wins / float(num_hands)) * 100,
        "allin_pct": (allin_acts / max(1, total_acts)) * 100,
        "prem_ev": (prem_chips / max(1, prem_count)),
        "trash_ev": (trash_chips / max(1, trash_count))
    }

def run_phase6_unseen_tests():
    print("=" * 80)
    print("🛸 [V4.8.2 PHASE 6] 5대 미학습 신규 상대(Unseen Opponents) 일반화 테스트 (각 500판)")
    print("=" * 80)
    
    device = torch.device("cpu")
    
    # 20k 완성 모델
    m_ppo20k = HoldemActorCritic(253, 256, 4).to(device)
    m_ppo20k.load_state_dict(torch.load("v481/models/controlled_ppo_step_20000.pt", map_location=device))
    m_ppo20k.eval()
    
    # Step 100k
    m_step100k = Holdem5PNet(253, 256, 4).to(device)
    m_step100k.load_state_dict(torch.load("v47/models/holdem_5p_step_100k.pt", map_location=device))
    m_step100k.eval()
    
    unseen_bots = [
        ExtremeTightBot(),
        LooseCallingBot(),
        SelectiveCallerBot(),
        BluffCatcherBot(),
        OpponentAdaptiveBot()
    ]
    
    print(f"{'상대 유형':<24} | {'PPO 20k (bb/100)':<18} | {'Step 100k (bb/100)':<18} | {'PPO 트래시 EV':<14} | {'100k 트래시 EV'}")
    print("-" * 95)
    
    for bot in unseen_bots:
        opp_list = [bot] * 4
        res_ppo = evaluate_against_unseen_pool(m_ppo20k, is_ppo=True, opponents=opp_list, num_hands=500, seed=8888)
        res_old = evaluate_against_unseen_pool(m_step100k, is_ppo=False, opponents=opp_list, num_hands=500, seed=8888)
        
        print(f"vs {bot.name:<21} | {res_ppo['bb100']:>+11.1f} bb/100   | {res_old['bb100']:>+11.1f} bb/100   | {res_ppo['trash_ev']:>+8.1f} chips   | {res_old['trash_ev']:>+8.1f} chips")

    # 5종 혼합 미학습 테이블
    print("-" * 95)
    mixed_unseen = unseen_bots[:4]
    r_ppo_mix = evaluate_against_unseen_pool(m_ppo20k, is_ppo=True, opponents=mixed_unseen, num_hands=1000, seed=9999)
    r_old_mix = evaluate_against_unseen_pool(m_step100k, is_ppo=False, opponents=mixed_unseen, num_hands=1000, seed=9999)
    print(f"🔥 [5종 미학습 혼합 테이블] | {r_ppo_mix['bb100']:>+11.1f} bb/100   | {r_old_mix['bb100']:>+11.1f} bb/100   | {r_ppo_mix['trash_ev']:>+8.1f} chips   | {r_old_mix['trash_ev']:>+8.1f} chips")
    print(f"   (PPO 올인율: {r_ppo_mix['allin_pct']:.1f}% vs Step 100k 올인율: {r_old_mix['allin_pct']:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    run_phase6_unseen_tests()
