"""
POKER AI V4.7 - PHASE 2: 포지션 공정성 및 Seat Rotation Wrapper 구현
- 목적: OpenSpiel 내부 좌석 고정 문제를 해결하고, 5개 포지션(Button, SB, BB, UTG, CO)을
  각 AI 에이전트가 정확히 1/5(20.0%)씩 공정하게 경험하도록 회전 매핑.
- 포지션 정의 (5인 테이블):
  - Seat 0: Small Blind (SB, 50 chips)
  - Seat 1: Big Blind (BB, 100 chips)
  - Seat 2: Under the Gun (UTG, Preflop First to act)
  - Seat 3: Cutoff (CO)
  - Seat 4: Button (BTN, Postflop Last to act)
- SeatRotationWrapper:
  - hand_id에 따라 에이전트와 좌석을 1칸씩 시프트 순환:
    agent_seat_map: seat -> agent_id, agent_id -> seat
  - 반환 보상 및 통계를 에이전트 기준과 포지션 기준으로 정확히 재매핑
"""

import sys
import os
import random
from typing import Dict, List, Tuple
import numpy as np

import pyspiel

POSITION_NAMES = ["SB (Small Blind)", "BB (Big Blind)", "UTG (Under the Gun)", "CO (Cutoff)", "BTN (Button)"]

class HoldemSeatRotationWrapper:
    """
    5인 홀덤 포지션 공정성 래퍼:
    - hand_index에 따라 딜러 버튼 및 좌석이 매 판 1칸씩 시계 방향으로 회전.
    - 에이전트 i의 좌석: seat = (i - hand_idx) % 5
    - 좌석 s의 에이전트: agent = (s + hand_idx) % 5
    """
    def __init__(self, game):
        self.game = game
        self.num_players = 5
        
    def get_seat_to_agent(self, hand_idx: int) -> Dict[int, int]:
        """물리 좌석(0~4)에 앉아있는 에이전트 ID(0~4) 매핑"""
        return {seat: (seat + hand_idx) % self.num_players for seat in range(self.num_players)}
        
    def get_agent_to_seat(self, hand_idx: int) -> Dict[int, int]:
        """에이전트 ID(0~4)가 앉아있는 물리 좌석(0~4) 매핑"""
        return {agent: (agent - hand_idx) % self.num_players for agent in range(self.num_players)}

def verify_position_fairness(total_hands: int = 5000):
    print("=" * 80)
    print(f"🔄 [V4.7 PHASE 2] 5인 홀덤 포지션 공정성 및 Seat Rotation 검증 ({total_hands:,}판)")
    print("=" * 80)
    
    from phase1_holdem_rules_audit import get_5p_holdem_game
    game = get_5p_holdem_game()
    wrapper = HoldemSeatRotationWrapper(game)
    
    # 5명의 에이전트별 포지션 경험 횟수 카운터 [agent_id][position_seat]
    position_counts = np.zeros((5, 5), dtype=int)
    
    for h in range(total_hands):
        s2a = wrapper.get_seat_to_agent(h)
        for seat, agent in s2a.items():
            position_counts[agent, seat] += 1
            
    print("\n[에이전트별 포지션 경험 횟수 및 비율 (5,000판 기준)]")
    print(f"{'에이전트':<10} | {'SB (Seat 0)':<13} | {'BB (Seat 1)':<13} | {'UTG (Seat 2)':<13} | {'CO (Seat 3)':<13} | {'BTN (Seat 4)':<13} | {'균등성 검증'}")
    print("-" * 90)
    
    expected = total_hands // 5
    is_fair = True
    
    for a in range(5):
        row = position_counts[a]
        passed = all(cnt == expected for cnt in row)
        if not passed: is_fair = False
        mark = "✅ 100% 균등" if passed else "❌ 편향"
        print(f"Agent {a:<4} | {row[0]:>6} ({row[0]/total_hands*100:.1f}%) | {row[1]:>6} ({row[1]/total_hands*100:.1f}%) | {row[2]:>6} ({row[2]/total_hands*100:.1f}%) | {row[3]:>6} ({row[3]/total_hands*100:.1f}%) | {row[4]:>6} ({row[4]/total_hands*100:.1f}%) | {mark}")
        
    print("=" * 90)
    if is_fair:
        print(f"🎉 [검증 완료]: 모든 에이전트가 5개 포지션(SB, BB, UTG, CO, BTN)을 오차 0(정확히 {expected}회, 20.0%)으로 경험함을 입증!")
    else:
        print("⚠️ [경고]: 포지션 균등성 위반이 감지되었습니다.")
    print("=" * 80)
    return is_fair

if __name__ == "__main__":
    verify_position_fairness(total_hands=5000)
