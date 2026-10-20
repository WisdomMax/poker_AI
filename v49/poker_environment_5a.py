"""
POKER AI V4.9 - 5-ACTION (fchpa) 5인 텍사스 홀덤 환경 모듈
- OpenSpiel universal_poker의 5-Action fchpa 추상화 적용:
  Action 0: Fold
  Action 1: Call / Check
  Action 2: Pot Bet / Raise
  Action 3: All-in
  Action 4: Half-Pot Bet / Raise
- 5인 플레이어, 52장 덱, 4개 라운드(Preflop, Flop, Turn, River)
- 블라인드: SB 50, BB 100, 시작 스택: 10,000 chips
- 공통 확률 샘플러 및 좌석 회전 래퍼 내장
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import pyspiel

ACTION_NAMES_5A = ["Fold", "Call/Check", "Pot Bet", "All-in", "Half-Pot"]
POSITION_NAMES = ["SB (Small Blind)", "BB (Big Blind)", "UTG (Under the Gun)", "CO (Cutoff)", "BTN (Button)"]

def get_5p_holdem_5a_game() -> pyspiel.Game:
    """5인 Texas Hold'em fchpa No-Limit OpenSpiel 게임 생성"""
    params = {
        "betting": "nolimit",
        "bettingAbstraction": "fchpa",
        "numPlayers": 5,
        "numRanks": 13,
        "numSuits": 4,
        "numHoleCards": 2,
        "numRounds": 4,
        "numBoardCards": "0 3 1 1",
        "blind": "50 100 0 0 0",
        "stack": "10000 10000 10000 10000 10000",
        "firstPlayer": "3 1 1 1"
    }
    return pyspiel.load_game("universal_poker", params)

class HoldemSeatRotationWrapper:
    """
    5인 테이블 좌석 회전 래퍼:
    - hand_index에 따라 매 판 딜러 버튼 및 좌석이 1칸씩 순환.
    - agent_to_seat: agent_id -> physical_seat = (agent_id - hand_idx) % 5
    - seat_to_agent: physical_seat -> agent_id = (physical_seat + hand_idx) % 5
    """
    def __init__(self, game):
        self.game = game
        self.num_players = 5
        
    def get_seat_to_agent(self, hand_idx: int) -> Dict[int, int]:
        return {seat: (seat + hand_idx) % self.num_players for seat in range(self.num_players)}
        
    def get_agent_to_seat(self, hand_idx: int) -> Dict[int, int]:
        return {agent: (agent - hand_idx) % self.num_players for agent in range(self.num_players)}

def sample_chance_action(state: pyspiel.State, rng: Optional[np.random.RandomState] = None) -> int:
    """확률 가중치 기반 무작위 찬스 액션 선택 (결정론적 [0][0] 금지)"""
    if not state.is_chance_node():
        raise ValueError(f"State is not a chance node! Current player: {state.current_player()}")
    outcomes = state.chance_outcomes()
    if not outcomes:
        raise RuntimeError("chance_outcomes() is empty!")
    actions, probs = zip(*outcomes)
    p_sum = sum(probs)
    norm_probs = [p / p_sum for p in probs]
    if rng is not None:
        idx = rng.choice(len(actions), p=norm_probs)
    else:
        idx = np.random.choice(len(actions), p=norm_probs)
    return actions[idx]

def advance_chance_nodes(state: pyspiel.State, rng: Optional[np.random.RandomState] = None) -> List[int]:
    """모든 찬스 노드를 통과할 때까지 무작위 카드를 배분"""
    actions_taken = []
    while state.is_chance_node():
        a = sample_chance_action(state, rng=rng)
        state.apply_action(a)
        actions_taken.append(a)
    return actions_taken

def parse_cards_from_info_str(info_str: str) -> Tuple[List[str], List[str]]:
    """information_state_string에서 private cards와 public cards 파싱"""
    private_cards = []
    public_cards = []
    if "Private: " in info_str:
        priv_part = info_str.split("Private: ")[1].split("]")[0].strip()
        if priv_part:
            private_cards = [priv_part[i:i+2] for i in range(0, len(priv_part), 2)]
    if "Public: " in info_str:
        pub_part = info_str.split("Public: ")[1].split("]")[0].strip()
        if pub_part:
            public_cards = [pub_part[i:i+2] for i in range(0, len(pub_part), 2)]
    return private_cards, public_cards
