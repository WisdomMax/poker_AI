"""
POKER AI V4.8 - 공통 홀덤 환경 및 평가 유틸리티
- V4.7.4 검증 완료된 공통 Chance Sampler를 필수로 사용.
- 결정론적 카드 선택([0][0]) 원천 차단.
- 5인 텍사스 홀덤 규칙, 포지션 회전, 보상 정산 표준화.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import pyspiel

POSITION_NAMES = ["SB (Small Blind)", "BB (Big Blind)", "UTG (Under the Gun)", "CO (Cutoff)", "BTN (Button)"]
ACTION_NAMES = ["Fold", "Call/Check", "Pot Bet/Raise", "All-in"]

def get_5p_holdem_game():
    params = {
        "betting": "nolimit",
        "bettingAbstraction": "fcpa",
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
    """
    확률 가중치 기반 무작위 찬스 액션 선택 (결정론적 [0][0] 금지).
    """
    if not state.is_chance_node():
        raise ValueError(f"State is not a chance node! Current player: {state.current_player()}")
    outcomes = state.chance_outcomes()
    if not outcomes:
        raise RuntimeError("chance_outcomes() is empty!")
    actions, probs = zip(*outcomes)
    p_sum = sum(probs)
    if p_sum <= 0:
        raise ValueError(f"Invalid probability sum: {p_sum}")
    norm_p = np.array(probs, dtype=np.float64) / p_sum
    if rng is not None:
        return int(rng.choice(actions, p=norm_p))
    return int(np.random.choice(actions, p=norm_p))

def advance_chance_nodes(state: pyspiel.State, rng: Optional[np.random.RandomState] = None) -> List[int]:
    actions = []
    while state.is_chance_node() and not state.is_terminal():
        act = sample_chance_action(state, rng=rng)
        state.apply_action(act)
        actions.append(act)
    return actions

def parse_cards_from_info_str(info_str: str) -> Tuple[List[str], List[str]]:
    private = ""
    public = ""
    if "Private: " in info_str:
        private = info_str.split("Private: ")[1].split("]")[0]
    if "Public: " in info_str:
        public = info_str.split("Public: ")[1].split("]")[0]
    hole_cards = [private[i:i+2] for i in range(0, len(private), 2)]
    board_cards = [public[i:i+2] for i in range(0, len(public), 2)]
    return hole_cards, board_cards
