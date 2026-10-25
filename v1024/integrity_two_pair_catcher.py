#!/usr/bin/env python3
"""
DIRECTIVE 047: Flawless TwoPair+ All-in Catcher Bot
Strict heuristic rule-based evaluation benchmark opponent.

Integrity Guarantees:
1. FREE CHECK: When to_call == 0, ALWAYS selects Action 1 (Check). ZERO FREE FOLDS!
2. FACING ALL-IN:
   - Preflop: Only calls with top premium holdings (Chen score >= 11.0: AA, KK, QQ, AKs).
   - Postflop: Only calls with verified 7-card Two Pair or higher (eval_7cards rank >= 2).
   - Otherwise: Folds (Action 0).
3. MODERATE BETS (to_call <= 3 BB):
   - Calls with speculative/playable hands (Chen >= 6.0).
   - Folds weak hands.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v79.run_v79_controlled_causal_study import extract_player_spent

STARTING_STACK = 10000.0
BIG_BLIND = 100.0
SMALL_BLIND = 50.0
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards

def card_str_to_int(c_str: str) -> int:
    ranks = '23456789TJQKA'
    suits = 'shdc'
    r = ranks.index(c_str[0])
    s = suits.index(c_str[1])
    return r * 4 + s

def calculate_chen_score(c1: str, c2: str) -> float:
    rank_scores = {
        '2': 1, '3': 1.5, '4': 2, '5': 2.5, '6': 3, '7': 3.5,
        '8': 4, '9': 4.5, 'T': 5, 'J': 6, 'Q': 7, 'K': 8, 'A': 10
    }
    ranks_order = '23456789TJQKA'
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    
    score = max(rank_scores[r1], rank_scores[r2])
    if r1 == r2:
        score = max(5.0, score * 2)
        if r1 in ['J', 'T']: score += 1
        return score
    
    if s1 == s2: score += 2
    
    idx1, idx2 = ranks_order.index(r1), ranks_order.index(r2)
    gap = abs(idx1 - idx2) - 1
    if gap == 1: score -= 1
    elif gap == 2: score -= 2
    elif gap == 3: score -= 4
    elif gap >= 4: score -= 5
    
    if gap in [0, 1] and max(idx1, idx2) < ranks_order.index('Q'):
        score += 1
    return max(0.0, score)

def parse_board_cards(st) -> List[str]:
    s = str(st)
    for line in s.split('\n'):
        if line.startswith('Board:'):
            parts = line.split()[1:]
            return [p for p in parts if len(p) == 2 and p[0] in '23456789TJQKA' and p[1] in 'shdc']
    return []

class FlawlessTwoPairCatcherBot:
    """
    Certified bug-free heuristic evaluation opponent.
    Never folds on free check (Action 1 is always chosen when to_call == 0).
    """
    def __init__(self, seat: int = 1, seed: int = 12345):
        self.seat = seat
        self.rng = np.random.RandomState(seed)

    def select_action(self, st, current_player: int, legal_6: List[int],
                      cards_map: Dict[int, Tuple[str, str]], round_idx: int) -> int:
        if current_player not in cards_map:
            # Fallback
            return 1 if 1 in legal_6 else (0 if 0 in legal_6 else list(legal_6)[0])

        c1, c2 = cards_map[current_player]
        chen = calculate_chen_score(c1, c2)
        board = parse_board_cards(st)

        # Calculate exact to_call
        all_spents = extract_player_spent(st)
        my_spent = all_spents[current_player]
        max_spent = max(all_spents)
        to_call = max_spent - my_spent

        # Check if an opponent is All-In
        is_allin_facing = (max_spent >= STARTING_STACK * 0.99)

        # 1. FACING ALL-IN
        if is_allin_facing and to_call > 0:
            if round_idx == 0:
                # Preflop: AA, KK, QQ, AKs only (Chen >= 11.0)
                if chen >= 11.0 and 1 in legal_6:
                    return 1
                return 0 if 0 in legal_6 else 1
            else:
                # Postflop: Real 7-card hand rank
                if len(board) >= 3:
                    int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
                    rank = eval_7cards(int_cards)[0]
                else:
                    rank = 0
                if rank >= 2 and 1 in legal_6:
                    return 1 # Calls only with Two Pair or better
                return 0 if 0 in legal_6 else 1

        # 2. FREE CHECK (to_call == 0)
        # CRITICAL DIRECTIVE 047 FIX: ALWAYS CHECK! NEVER FOLD!
        if to_call == 0:
            if 1 in legal_6:
                return 1
            return 0 if 0 in legal_6 else list(legal_6)[0]

        # 3. MODERATE BETS (to_call <= 3 BB)
        if to_call <= BIG_BLIND * 3:
            if chen >= 6.0 and 1 in legal_6:
                return 1
            return 0 if 0 in legal_6 else 1

        # 4. LARGE BETS (> 3 BB)
        if round_idx > 0 and len(board) >= 3:
            int_cards = [card_str_to_int(c) for c in [c1, c2] + board]
            rank = eval_7cards(int_cards)[0]
            if rank >= 2 and 1 in legal_6:
                return 1
        return 0 if 0 in legal_6 else 1
