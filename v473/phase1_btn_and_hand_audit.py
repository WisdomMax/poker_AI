"""
POKER AI V4.7.3 - PHASE 1 & 4: BTN 수익 집중 원인 규명, 경계값 검사, 원시 핸드 로그 및 극단적 40판 복원
- 감사 대상:
  1. V4.7.2에서 보고된 BTN 수익 편중(+17,488,450 chips)의 근본 원인 규명
  2. 물리 좌석(Physical Seat) vs 논리 에이전트(Agent ID) 매핑 및 보상 귀속 정밀 검증
  3. 모든 판의 게임당 순수익이 -10,000 <= hero_return <= 40,000 경계 조건을 만족하는지
  4. 2,500판 원시 핸드 로그 생성 (v473/data/detailed_hand_logs.csv)
  5. 최대 승리 20판 + 최대 패배 20판 (총 40판) 텍스트 액션 로그 복원 (v473/logs/extreme_40_hands.txt)
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
from v47.phase5_benchmark_eval import HoldemBot, RandomHoldemBot, AlwaysCallHoldemBot, TightHoldemBot, RuleBasedHoldemBot

def run_phase1_and_4_audit(num_hands: int = 2500, seed: int = 1001):
    print("=" * 80)
    print("🔍 [V4.7.3 PHASE 1 & 4] 시뮬레이터 찬스 노드 결함 규명 및 원시 핸드 전수 감사")
    print("=" * 80)
    
    device = torch.device("cpu")
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    
    net = Holdem5PNet(253, 256, 4).to(device)
    model_path = "v47/models/holdem_5p_step_100k.pt"
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    opponents = [
        RandomHoldemBot(),       # Opp 0
        AlwaysCallHoldemBot(),   # Opp 1
        TightHoldemBot(),        # Opp 2
        RuleBasedHoldemBot()     # Opp 3
    ]
    opp_names = ["Random", "Always Call", "Tight", "Rule-Based"]
    target_agent_id = 0
    action_names = ["Fold", "Call", "Bet", "All-in"]
    
    # -------------------------------------------------------------
    # 1. 버그 재현 모드: st.chance_outcomes()[0][0] (V4.7.2 방식)
    # -------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    bug_seat_wins = {i: 0 for i in range(5)}
    bug_profit_by_seat = {i: 0.0 for i in range(5)}
    bug_hero_profit_by_seat = {i: 0.0 for i in range(5)}
    bug_hero_hands_by_seat = {i: 0 for i in range(5)}
    
    bug_boundary_violations = []
    
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        ai_seat = a2s[target_agent_id]
        bug_hero_hands_by_seat[ai_seat] += 1
        
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.is_chance_node():
                # V4.7.2 버그: 항상 첫 번째 찬스 아웃컴 선택
                st.apply_action(st.chance_outcomes()[0][0])
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
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
                    act = int(np.random.choice(legal, p=prs))
                else:
                    opp_id = s2a[cur_seat]
                    opp_bot = opponents[(opp_id - 1) % len(opponents)]
                    act = opp_bot.act(st, cur_seat, legal)
                st.apply_action(act)
                
        returns = st.returns()
        hero_return = returns[ai_seat]
        
        # 경계값 검사
        if not (-10000.0 <= hero_return <= 40000.0):
            bug_boundary_violations.append((h, hero_return))
            
        # 승자 판정
        max_ret = max(returns)
        if max_ret > 0:
            winner_seat = returns.index(max_ret)
            bug_seat_wins[winner_seat] += 1
            bug_profit_by_seat[winner_seat] += max_ret
            
        bug_hero_profit_by_seat[ai_seat] += hero_return

    print("📊 [버그 재현 결과: chance_outcomes()[0][0] 사용]")
    print(f"  - 총 핸드 수: {num_hands:,}")
    print(f"  - 경계 조건 (-10,000 <= return <= 40,000) 위반: {len(bug_boundary_violations)}건")
    print("  - 물리 좌석별 승리 횟수 및 총 승리 칩:")
    for s in range(5):
        print(f"    Seat {s} ({POSITION_NAMES[s]}): 승리 {bug_seat_wins[s]:,}/{num_hands:,}회 "
              f"({bug_seat_wins[s]/num_hands*100:5.1f}%), 총 승리 칩: {bug_profit_by_seat[s]:+13,f}")
    print("  - AI 에이전트가 앉았던 좌석별 순수익 (Hero Net Chips):")
    for s in range(5):
        print(f"    AI on Seat {s} ({POSITION_NAMES[s]}): {bug_hero_profit_by_seat[s]:+13,f} chips "
              f"({bug_hero_hands_by_seat[s]}판, 평균: {bug_hero_profit_by_seat[s]/bug_hero_hands_by_seat[s]:+8.1f} chips/hand)")

    # -------------------------------------------------------------
    # 2. 정상 무작위 샘플링 모드: np.random.choice(acts, p=prs)
    # -------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    correct_seat_wins = {i: 0 for i in range(5)}
    correct_hero_profit_by_seat = {i: 0.0 for i in range(5)}
    correct_hero_hands_by_seat = {i: 0 for i in range(5)}
    correct_hero_wins_by_seat = {i: 0 for i in range(5)}
    
    detailed_hand_records = []
    hand_action_histories = []
    
    for h in range(num_hands):
        s2a = wrapper.get_seat_to_agent(h)
        a2s = wrapper.get_agent_to_seat(h)
        ai_seat = a2s[target_agent_id]
        correct_hero_hands_by_seat[ai_seat] += 1
        
        st = game.new_initial_state()
        history = []
        hero_actions = []
        hero_hole_cards = ""
        board_cards = ""
        
        while not st.is_terminal():
            if st.is_chance_node():
                # 정상적인 무작위 샘플링
                acts, prs = zip(*st.chance_outcomes())
                act = int(np.random.choice(acts, p=prs))
                st.apply_action(act)
            else:
                cur_seat = st.current_player()
                legal = st.legal_actions(cur_seat)
                cur_str = st.information_state_string(cur_seat)
                
                # 홀카드/보드카드 파싱
                if cur_seat == ai_seat and not hero_hole_cards:
                    if "Private: " in cur_str:
                        hero_hole_cards = cur_str.split("Private: ")[1].split("]")[0]
                        
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
                    act = int(np.random.choice(legal, p=prs))
                    hero_actions.append(action_names[act])
                else:
                    opp_id = s2a[cur_seat]
                    opp_bot = opponents[(opp_id - 1) % len(opponents)]
                    act = opp_bot.act(st, cur_seat, legal)
                    
                history.append((cur_seat, act, cur_str))
                st.apply_action(act)
                
        returns = st.returns()
        hero_return = returns[ai_seat]
        correct_hero_profit_by_seat[ai_seat] += hero_return
        
        # 보드카드 추출
        last_str = st.information_state_string(ai_seat)
        if "Public: " in last_str:
            board_cards = last_str.split("Public: ")[1].split("]")[0]
            
        max_ret = max(returns)
        winner_seat = returns.index(max_ret) if max_ret > 0 else -1
        if winner_seat >= 0:
            correct_seat_wins[winner_seat] += 1
            if winner_seat == ai_seat:
                correct_hero_wins_by_seat[ai_seat] += 1
                
        is_showdown = (sum(r != 0 for r in returns) > 2) or (hero_return != 0 and len(board_cards) > 0)
        
        # CSV 레코드
        detailed_hand_records.append({
            "hand_id": h,
            "ai_seat": ai_seat,
            "ai_position": POSITION_NAMES[ai_seat],
            "hero_hole_cards": hero_hole_cards,
            "board_cards": board_cards,
            "hero_actions": ";".join(hero_actions),
            "hero_return": hero_return,
            "hero_bb100": (hero_return / 100.0) * 100.0, # 1판당 bb/100 기여도
            "winner_seat": winner_seat,
            "winner_position": POSITION_NAMES[winner_seat] if winner_seat >= 0 else "Split/Fold",
            "pot_size": sum(abs(r) for r in returns) / 2.0,
            "is_showdown": is_showdown,
            "returns_all": ";".join(map(str, returns))
        })
        
        hand_action_histories.append({
            "hand_id": h,
            "ai_seat": ai_seat,
            "hero_return": hero_return,
            "hero_hole_cards": hero_hole_cards,
            "board_cards": board_cards,
            "returns": returns,
            "history": history
        })

    print("\n📊 [정상 무작위 샘플링 결과: chance_outcomes() 확률 샘플링]")
    print(f"  - 총 핸드 수: {num_hands:,}")
    print("  - 물리 좌석별 승리 횟수:")
    for s in range(5):
        print(f"    Seat {s} ({POSITION_NAMES[s]}): 승리 {correct_seat_wins[s]:,}/{num_hands:,}회 "
              f"({correct_seat_wins[s]/num_hands*100:5.1f}%)")
    print("  - AI 에이전트가 앉았던 좌석별 순수익 (Hero Net Chips):")
    total_correct_profit = sum(correct_hero_profit_by_seat.values())
    for s in range(5):
        print(f"    AI on Seat {s} ({POSITION_NAMES[s]}): {correct_hero_profit_by_seat[s]:+13,f} chips "
              f"({correct_hero_hands_by_seat[s]}판, 승률: {correct_hero_wins_by_seat[s]/correct_hero_hands_by_seat[s]*100:4.1f}%, "
              f"평균: {correct_hero_profit_by_seat[s]/correct_hero_hands_by_seat[s]:+8.1f} chips/hand)")
    print(f"  - 전체 순수익 합계: {total_correct_profit:+13,f} chips")
    print(f"  - 전체 평균 수익률: {(total_correct_profit / num_hands):+8.2f} chips/hand = {(total_correct_profit / num_hands / 1.0):+8.2f} bb/100 (BB=100)")

    # -------------------------------------------------------------
    # 3. CSV 파일 저장: v473/data/detailed_hand_logs.csv
    # -------------------------------------------------------------
    csv_file = "v473/data/detailed_hand_logs.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(detailed_hand_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_hand_records)
    print(f"\n💾 원시 핸드 로그 {len(detailed_hand_records):,}개 저장 완료: {csv_file}")

    # -------------------------------------------------------------
    # 4. 극단적 승패 40판 복원 (최대 승리 20판, 최대 패배 20판)
    # -------------------------------------------------------------
    sorted_by_profit = sorted(hand_action_histories, key=lambda x: x["hero_return"], reverse=True)
    top_20_wins = sorted_by_profit[:20]
    top_20_losses = sorted_by_profit[-20:][::-1] # 가장 큰 손실부터
    
    txt_file = "v473/logs/extreme_40_hands.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("🃏 [V4.7.3 PHASE 4] 극단적 승패 40판 상세 텍스트 복원 감사 로그\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("🏆 [TOP 20 최대 승리 핸드 (Max Wins)]\n")
        f.write("=" * 80 + "\n")
        for rank, h_data in enumerate(top_20_wins, 1):
            f.write(f"\n[WIN #{rank:02d}] Hand #{h_data['hand_id']} | Hero Profit: {h_data['hero_return']:+10,f} chips\n")
            f.write(f"  Position: Seat {h_data['ai_seat']} ({POSITION_NAMES[h_data['ai_seat']]})\n")
            f.write(f"  Hole Cards: {h_data['hero_hole_cards']} | Board Cards: {h_data['board_cards']}\n")
            f.write(f"  Final Returns: {h_data['returns']}\n")
            f.write("  Action Sequence:\n")
            for cur_seat, act, cur_str in h_data["history"]:
                actor = "Hero(AI)" if cur_seat == h_data["ai_seat"] else f"Opp(Seat {cur_seat})"
                f.write(f"    - {actor}: {action_names[act]} (Action {act})\n")
                
        f.write("\n" + "=" * 80 + "\n")
        f.write("💀 [TOP 20 최대 패배 핸드 (Max Losses)]\n")
        f.write("=" * 80 + "\n")
        for rank, h_data in enumerate(top_20_losses, 1):
            f.write(f"\n[LOSS #{rank:02d}] Hand #{h_data['hand_id']} | Hero Loss: {h_data['hero_return']:+10,f} chips\n")
            f.write(f"  Position: Seat {h_data['ai_seat']} ({POSITION_NAMES[h_data['ai_seat']]})\n")
            f.write(f"  Hole Cards: {h_data['hero_hole_cards']} | Board Cards: {h_data['board_cards']}\n")
            f.write(f"  Final Returns: {h_data['returns']}\n")
            f.write("  Action Sequence:\n")
            for cur_seat, act, cur_str in h_data["history"]:
                actor = "Hero(AI)" if cur_seat == h_data["ai_seat"] else f"Opp(Seat {cur_seat})"
                f.write(f"    - {actor}: {action_names[act]} (Action {act})\n")

    print(f"📄 극단적 승패 40판 상세 텍스트 복원 완료: {txt_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_phase1_and_4_audit()
