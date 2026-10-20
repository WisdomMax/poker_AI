"""
POKER AI V4.8.1 - PHASE 1: 카드 입력 텐서 무결성 및 인코딩 구조 전수 분석
- 목적:
  1. OpenSpiel universal_poker 253차원 information_state_tensor의 내부 레이아웃 규명
  2. 비공개 홀카드(Private Cards) 인덱스 및 원-핫 인코딩 위치 확인
  3. 8종 핸드(AA, KK, QQ, AKs, AQs, JTs, 72o, 83o)의 입력 텐서 차이(L1/L2 Norm) 측정
  4. PPO League 및 기존 체크포인트 신경망에 주입했을 때 중간 Feature, Logits, Softmax 확률의 변화 측정
"""

import sys
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v48.poker_environment import get_5p_holdem_game, advance_chance_nodes, parse_cards_from_info_str
from v48.ppo_actor_critic import HoldemActorCritic
from v47.phase3_holdem_ai_architecture import Holdem5PNet

def analyze_tensor_layout():
    print("=" * 80)
    print("🔬 [V4.8.1 PHASE 1] 253차원 Information State Tensor 구조 정밀 감사")
    print("=" * 80)
    
    game = get_5p_holdem_game()
    st = game.new_initial_state()
    advance_chance_nodes(st) # 홀카드 배분 완료
    
    cur_p = st.current_player()
    info_str = st.information_state_string(cur_p)
    info_vec = np.array(st.information_state_tensor(cur_p))
    
    print(f"  - 게임 상태 문자열: {info_str}")
    print(f"  - 텐서 전체 차원: {len(info_vec)}")
    print(f"  - 0이 아닌 원소 개수: {np.count_nonzero(info_vec)}")
    print(f"  - 0이 아닌 인덱스들: {np.where(info_vec != 0)[0].tolist()}")
    
    # 카드가 바뀔 때 어떤 인덱스가 변하는지 2개 상태 비교
    st2 = game.new_initial_state()
    advance_chance_nodes(st2)
    info_str2 = st2.information_state_string(cur_p)
    info_vec2 = np.array(st2.information_state_tensor(cur_p))
    
    diff_indices = np.where(info_vec != info_vec2)[0]
    print(f"\n  [두 상태 간 변동 인덱스 비교]")
    print(f"    - 상태 1: {info_str}")
    print(f"    - 상태 2: {info_str2}")
    print(f"    - 변동된 인덱스 목록 ({len(diff_indices)}개): {diff_indices.tolist()}")
    
    return game

def inspect_8_hands_sensitivity():
    print("\n" + "=" * 80)
    print("🃏 [8종 핸드별 입력 텐서, Logits, Softmax 정책 민감도 전수 측정]")
    print("=" * 80)
    
    device = torch.device("cpu")
    game = get_5p_holdem_game()
    
    # 모델 로드
    ppo_league_path = "v48/models/Exp_D_PPO_League_step_10k.pt"
    old_100k_path = "v47/models/holdem_5p_step_100k.pt"
    
    ppo_model = HoldemActorCritic(253, 256, 4).to(device)
    ppo_model.load_state_dict(torch.load(ppo_league_path, map_location=device))
    ppo_model.eval()
    
    step100k_model = Holdem5PNet(253, 256, 4).to(device)
    step100k_model.load_state_dict(torch.load(old_100k_path, map_location=device))
    step100k_model.eval()
    
    target_hands = ["AA", "KK", "QQ", "AKs", "AQs", "JTs", "72o", "83o"]
    hand_tensors = {}
    
    # 8종 핸드 상태 수집 (UTG 프리플롭 첫 턴, 액션 이력 동일: 라운드 0 시작 시점)
    # OpenSpiel에서 특정 핸드가 나올 때까지 시뮬레이션
    for th in target_hands:
        for seed in range(1, 100000):
            np.random.seed(seed)
            st = game.new_initial_state()
            advance_chance_nodes(st)
            cur_p = st.current_player()
            if cur_p == 2: # UTG
                s_str = st.information_state_string(cur_p)
                hc, _ = parse_cards_from_info_str(s_str)
                if len(hc) == 2:
                    r1, r2 = hc[0][0], hc[1][0]
                    s1, s2 = hc[0][1], hc[1][1]
                    m = False
                    if th == "AA" and r1 == 'A' and r2 == 'A': m = True
                    elif th == "KK" and r1 == 'K' and r2 == 'K': m = True
                    elif th == "QQ" and r1 == 'Q' and r2 == 'Q': m = True
                    elif th == "AKs" and ((r1=='A' and r2=='K') or (r1=='K' and r2=='A')) and s1 == s2: m = True
                    elif th == "AQs" and ((r1=='A' and r2=='Q') or (r1=='Q' and r2=='A')) and s1 == s2: m = True
                    elif th == "JTs" and ((r1=='J' and r2=='T') or (r1=='T' and r2=='J')) and s1 == s2: m = True
                    elif th == "72o" and ((r1=='7' and r2=='2') or (r1=='2' and r2=='7')) and s1 != s2: m = True
                    elif th == "83o" and ((r1=='8' and r2=='3') or (r1=='3' and r2=='8')) and s1 != s2: m = True
                    
                    if m:
                        vec = st.information_state_tensor(cur_p)
                        legal = st.legal_actions(cur_p)
                        hand_tensors[th] = (vec, legal, s_str)
                        break
                        
    print(f"  - 8종 핸드 텐서 수집 완료: {list(hand_tensors.keys())}")
    
    # AA 입력 텐서와 다른 핸드 텐서의 거리 비교
    aa_vec = np.array(hand_tensors["AA"][0])
    print("\n[AA 기준 입력 텐서 L1 거리 및 L2 거리]")
    for th in target_hands:
        v = np.array(hand_tensors[th][0])
        l1 = np.sum(np.abs(v - aa_vec))
        l2 = np.linalg.norm(v - aa_vec)
        print(f"  - AA vs {th:<4}: L1 거리 = {l1:.2f}, L2 거리 = {l2:.4f}")

    # PPO League 모델과 Step 100k 모델의 출력 비교
    print("\n[PPO League vs Step 100k: 8종 핸드별 출력 로짓 및 확률]")
    print(f"{'핸드':<6} | {'모델':<14} | {'Fold Logit':<10} | {'Call Logit':<10} | {'Pot Logit':<10} | {'All-in Logit':<12} | {'Softmax [F, C, P, A]'}")
    print("-" * 95)
    
    for th in target_hands:
        vec, legal, s_str = hand_tensors[th]
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
        for a in legal: mask[0, a] = True
        
        # PPO
        with torch.no_grad():
            ppo_logits, ppo_val = ppo_model(x, mask)
            ppo_probs = torch.softmax(ppo_logits, dim=-1).squeeze(0).cpu().numpy()
            ppo_l = ppo_logits.squeeze(0).cpu().numpy()
            
        # Step 100k
        with torch.no_grad():
            old_logits = step100k_model(x, mask)
            old_probs = torch.softmax(old_logits, dim=-1).squeeze(0).cpu().numpy()
            old_l = old_logits.squeeze(0).cpu().numpy()
            
        print(f"{th:<6} | {'PPO League':<14} | {ppo_l[0]:+8.3f}   | {ppo_l[1]:+8.3f}   | {ppo_l[2]:+8.3f}   | {ppo_l[3]:+10.3f}   | [{ppo_probs[0]:.3f}, {ppo_probs[1]:.3f}, {ppo_probs[2]:.3f}, {ppo_probs[3]:.3f}]")
        print(f"{'':<6} | {'Step 100k':<14} | {old_l[0]:+8.3f}   | {old_l[1]:+8.3f}   | {old_l[2]:+8.3f}   | {old_l[3]:+10.3f}   | [{old_probs[0]:.3f}, {old_probs[1]:.3f}, {old_probs[2]:.3f}, {old_probs[3]:.3f}]")
        print("-" * 95)

if __name__ == "__main__":
    analyze_tensor_layout()
    inspect_8_hands_sensitivity()
