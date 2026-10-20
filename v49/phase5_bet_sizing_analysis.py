"""
POKER AI V4.9 - PHASE 5: 베팅 크기 학습 분석 (Bet Sizing Intelligence Analysis)
- 분석 핸드 9종: AA, KK, QQ, AKs, AQs, JTs, 98s, 72o, 83o
- 핵심 분석 과제:
  1. 9개 핸드의 5대 행동 확률 분포 (Fold%, Call%, Half-Pot%, Pot%, All-in%)
  2. AA의 올인 의존도 탈피 및 Half-Pot / Pot 분산 여부
  3. JTs의 과도한 올인(44%) 감소 및 하프팟/콜로의 이동 여부
  4. Half-Pot의 실제 사용률 및 조건부 EV 검증 (vs Tight vs Loose 상대)
  5. 베팅 크기 다변화가 전체 수익률 및 분산에 미친 영향
"""

import sys
import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v49.poker_environment_5a import (
    get_5p_holdem_5a_game, HoldemSeatRotationWrapper, advance_chance_nodes,
    parse_cards_from_info_str, ACTION_NAMES_5A
)
from v49.ppo_actor_critic_5a import HoldemActorCritic5A
from v49.hand_tensors_5a import get_standard_hand_tensors_5a, HAND_SEEDS_5A
from v49.model_adapter_4a_to_5a import FourActionToFiveActionAdapter
from v48.ppo_actor_critic import HoldemActorCritic
from v48.opponents_pool import TightBot, AlwaysCallBot, RandomBot
from v483.phase4_unseen_opponents_audit import SelectiveCallerBot, BluffCatcherBot

HAND_ORDER_9 = ['AA', 'KK', 'QQ', 'AKs', 'AQs', 'JTs', '98s', '83o', '72o']
SEEDS = [42, 101, 202, 303, 404]

def analyze_hand_policies_by_condition():
    device = torch.device("cpu")
    hand_tensors = get_standard_hand_tensors_5a(device=device)
    
    # 3대 조건 모델 로드
    cond_models = {"CondA_4A": [], "CondB_5A_Scratch": [], "CondC_5A_Transfer": []}
    
    for s in SEEDS:
        # Cond A
        p_a = f"v483/models/seed_{s}_step_20000.pt"
        if os.path.exists(p_a):
            m4 = HoldemActorCritic(253, 256, 4).to(device)
            m4.load_state_dict(torch.load(p_a, map_location=device))
            adapter = FourActionToFiveActionAdapter(m4).to(device)
            adapter.eval()
            cond_models["CondA_4A"].append(adapter)
            
        # Cond B
        p_b = f"v49/models/CondB_5A_Scratch_seed_{s}_step_20000.pt"
        if os.path.exists(p_b):
            mb = HoldemActorCritic5A(265, 256, 5).to(device)
            mb.load_state_dict(torch.load(p_b, map_location=device))
            mb.eval()
            cond_models["CondB_5A_Scratch"].append(mb)
            
        # Cond C
        p_c = f"v49/models/CondC_5A_Transfer_seed_{s}_step_20000.pt"
        if os.path.exists(p_c):
            mc = HoldemActorCritic5A(265, 256, 5).to(device)
            mc.load_state_dict(torch.load(p_c, map_location=device))
            mc.eval()
            cond_models["CondC_5A_Transfer"].append(mc)
            
    print("=" * 105)
    print("🃏 [PHASE 5] 9개 핵심 핸드 프리플롭 정책 비교 (Cond A vs Cond B vs Cond C)")
    print("=" * 105)
    print(f"{'핸드':<6} | {'조건':<18} | {'Fold (0)':<12} | {'Call (1)':<12} | {'Pot (2)':<12} | {'All-in (3)':<14} | {'Half-Pot (4)'}")
    print("-" * 105)
    
    results = {}
    for h in HAND_ORDER_9:
        x, m, _ = hand_tensors[h]
        x, m = x.to(device), m.to(device)
        results[h] = {}
        for cname, mlist in cond_models.items():
            if not mlist: continue
            probs_list = []
            for mod in mlist:
                with torch.no_grad():
                    logits, _ = mod(x, m)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    probs_list.append(probs)
            mean_p = np.mean(probs_list, axis=0) * 100.0
            std_p = np.std(probs_list, axis=0) * 100.0
            results[h][cname] = (mean_p, std_p)
            print(f"{h:<6} | {cname:<18} | {mean_p[0]:>5.1f}% (±{std_p[0]:>4.1f}) | {mean_p[1]:>5.1f}% (±{std_p[1]:>4.1f}) | {mean_p[2]:>5.1f}% (±{std_p[2]:>4.1f}) | {mean_p[3]:>6.1f}% (±{std_p[3]:>4.1f}) | {mean_p[4]:>6.1f}% (±{std_p[4]:>4.1f})")
        print("-" * 105)
        
    return cond_models, hand_tensors

def simulate_hand_bet_ev(
    model: torch.nn.Module,
    hand_name: str,
    action_to_force: int,
    opponents: List,
    num_sims: int = 250,
    seed: int = 7777
) -> float:
    game = get_5p_holdem_5a_game()
    wrapper = HoldemSeatRotationWrapper(game)
    h_seed = HAND_SEEDS_5A[hand_name]
    
    chips_total = 0.0
    rng = np.random.RandomState(seed)
    
    for sim in range(num_sims):
        s2a = wrapper.get_seat_to_agent(sim)
        a2s = wrapper.get_agent_to_seat(sim)
        hero_seat = a2s[0]
        
        # 영구 재현을 위해 지정된 핸드가 들어오도록 카드 분배
        rng_hole = np.random.RandomState(h_seed)
        st = game.new_initial_state()
        advance_chance_nodes(st, rng=rng_hole)
        hero_seat = st.current_player() # UTG (포지션 2)
        
        rng_board = np.random.RandomState(seed + sim * 79)
        first_turn = True
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng=rng_board)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                if cur_seat == hero_seat:
                    if first_turn:
                        act = action_to_force if action_to_force in legal else legal[0]
                        first_turn = False
                    else:
                        vec = st.information_state_tensor(cur_seat)
                        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                        m = torch.zeros((1, 5), dtype=torch.bool)
                        for a in legal: m[0, a] = True
                        with torch.no_grad():
                            logits, _ = model(x, m)
                            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                        prs = [probs[a] for a in legal]
                        prs = np.array(prs) / sum(prs) if sum(prs) > 0 else np.ones(len(legal))/len(legal)
                        act = int(rng.choice(legal, p=prs))
                else:
                    ag_id = s2a[cur_seat]
                    opp = opponents[(ag_id - 1) % len(opponents)]
                    act = opp.act(st, cur_seat, legal)
                st.apply_action(act)
                
        chips_total += st.returns()[hero_seat]
    return float(chips_total / float(num_sims))

def run_phase5_analysis():
    cond_models, hand_tensors = analyze_hand_policies_by_condition()
    
    if not cond_models["CondC_5A_Transfer"]:
        print("Models not ready yet.")
        return
        
    rep_model = cond_models["CondC_5A_Transfer"][0]
    
    print("\n" + "=" * 90)
    print("🔬 [PHASE 5 EV 심층 비교] AA 및 JTs의 베팅 크기별 조건부 기대수익 (chips)")
    print("=" * 90)
    print(f"{'핸드':<6} | {'상대 환경':<18} | {'Half-Pot EV':<15} | {'Pot EV':<15} | {'All-in EV':<15} | {'Call EV'}")
    print("-" * 90)
    
    opp_envs = {
        "vs Tight": [TightBot()] * 4,
        "vs Loose Calling": [AlwaysCallBot()] * 4,
        "vs Selective Caller": [SelectiveCallerBot()] * 4
    }
    
    for h in ['AA', 'JTs']:
        for oname, opps in opp_envs.items():
            ev_half = simulate_hand_bet_ev(rep_model, h, 4, opps, num_sims=200, seed=1234)
            ev_pot = simulate_hand_bet_ev(rep_model, h, 2, opps, num_sims=200, seed=1234)
            ev_allin = simulate_hand_bet_ev(rep_model, h, 3, opps, num_sims=200, seed=1234)
            ev_call = simulate_hand_bet_ev(rep_model, h, 1, opps, num_sims=200, seed=1234)
            print(f"{h:<6} | {oname:<18} | {ev_half:>+10.1f} chips | {ev_pot:>+10.1f} chips | {ev_allin:>+10.1f} chips | {ev_call:>+10.1f} chips")
        print("-" * 90)
        
    print("=" * 90)

if __name__ == "__main__":
    run_phase5_analysis()
