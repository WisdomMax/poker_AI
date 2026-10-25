"""
Human-Friendly Policy Layer for Poker AI Human Arena (DIRECTIVE 030)
====================================================================
Designed to prevent irrational preflop all-in spamming while:
1. NEVER modifying original PPO neural network checkpoints.
2. NEVER peeking at opponent hole cards or future deck cards (Strictly zero cheating).
3. Allowing legitimate all-ins for premium hands (AA, KK, QQ, AK) and short-stack push/fold.
4. Downsizing irrational trash-hand all-ins (Chen < 8) into rational standard poker sizings:
   - Premium (Chen >= 11): 100% All-in preserved.
   - Strong (Chen 8~10): All-in allowed facing heavy bets; otherwise downsized to 100% or 200% Pot.
   - Weak/Trash (Chen < 8): Downsized to Fold (0), Check/Call (1), or Standard 50% Pot Raise (2).
5. Providing full side-by-side mode switching:
   - 'original': Research-grade raw PPO argmax policy
   - 'human_friendly': Context & hand-strength aware rational poker policy
"""

import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

def calculate_chen_score(card1: str, card2: str) -> float:
    """Bill Chen Formula for Texas Hold'em Preflop Hand Strength"""
    if not card1 or not card2 or card1 == "??" or card2 == "??":
        return 0.0
    ranks = '23456789TJQKA'
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    v1, v2 = ranks.index(r1), ranks.index(r2)
    if v1 < v2:
        v1, v2 = v2, v1
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    
    score_map = {
        'A': 10.0, 'K': 8.0, 'Q': 7.0, 'J': 6.0, 'T': 5.0,
        '9': 4.5, '8': 4.0, '7': 3.5, '6': 3.0, '5': 2.5, '4': 2.0, '3': 1.5, '2': 1.0
    }
    score = score_map[r1]
    is_pair = (r1 == r2)
    if is_pair:
        score = max(5.0, score * 2)
    
    if s1 == s2:
        score += 2.0
        
    gap = v1 - v2 - 1
    if not is_pair:
        if gap == 0:
            pass
        elif gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        else:
            score -= 5.0
            
    if not is_pair and gap <= 1 and v1 < ranks.index('Q'):
        score += 1.0
        
    return round(score, 1)

class HumanFriendlyPolicyAdapter:
    """
    Rational Policy Adapter that intercepts raw model action logits
    and applies GTO-grounded risk management without peeking hidden info.
    """
    
    @staticmethod
    def filter_action(
        raw_act: int,
        model_probs: List[float],
        private_cards: List[str],
        street: int, # 0=Preflop, 1=Flop, 2=Turn, 3=River
        pot: float,
        facing_bet: float,
        legal_actions: List[int],
        mode: str = "human_friendly"
    ) -> Tuple[int, str]:
        """
        Filters or preserves action based on mode.
        Returns: (final_action_id, reason_log)
        """
        if mode != "human_friendly":
            return raw_act, "Original PPO"
            
        # If raw action is not All-in (5), keep it as is
        if raw_act != 5:
            return raw_act, "Natural Model Choice"
            
        # If action IS All-in, apply rational poker risk governance
        # 1. Preflop All-in Governance
        if street == 0:
            c1, c2 = private_cards[0], private_cards[1]
            c_score = calculate_chen_score(c1, c2)
            
            # Premium Hands: AA, KK, QQ, AKs, JJ (Chen >= 11) -> Legit All-in!
            if c_score >= 11.0:
                return 5, f"Legit Premium All-in (Chen={c_score})"
                
            # Strong Hands: TT, 99, AQs, AQo, AJs (Chen 8.0 ~ 10.5)
            # Facing a large bet (3-bet or 4-bet, facing_bet >= 500), All-in is standard push/fold
            if c_score >= 8.0:
                if facing_bet >= 500:
                    return 5, f"Legit Shove vs 3Bet+ (Chen={c_score})"
                # Otherwise, downsize open-shove to standard 100% or 200% Pot Raise
                if 4 in legal_actions:
                    return 4, f"Downsized Strong Hand to 200% Pot (Chen={c_score})"
                elif 3 in legal_actions:
                    return 3, f"Downsized Strong Hand to 100% Pot (Chen={c_score})"
                elif 2 in legal_actions:
                    return 2, f"Downsized Strong Hand to 50% Pot (Chen={c_score})"
                return 1, "Downsized to Call"
                
            # Weak / Trash Hands: Chen < 8.0 (72o, 83o, Q4o, T6o, etc.)
            # Irrational 100BB open-shove with trash should NEVER happen in human play
            # Check secondary model preferences among legal actions
            sorted_acts = sorted(
                [a for a in legal_actions if a != 5],
                key=lambda a: model_probs[a] if a < len(model_probs) else 0.0,
                reverse=True
            )
            
            # If the model had high fold probability or facing bet is large, fold trash
            if facing_bet > 100 and 0 in legal_actions:
                return 0, f"Filtered Trash All-in -> Fold (Chen={c_score})"
                
            # In an unraised pot, if 50% or 100% pot raise was considered, use standard bluff sizing
            # Otherwise Call or Fold
            for alt in sorted_acts:
                if alt in [2, 3]: # Standard 50% or 100% raise
                    return alt, f"Filtered Trash All-in -> Sized Bluff (Chen={c_score})"
                if alt == 1:
                    return 1, f"Filtered Trash All-in -> Check/Call (Chen={c_score})"
                if alt == 0:
                    return 0, f"Filtered Trash All-in -> Fold (Chen={c_score})"
                    
            return 1 if 1 in legal_actions else 0, "Filtered Trash All-in -> Fallback"

        # 2. Postflop All-in Governance
        # In postflop, if pot is relatively small (< 2000 chips) and someone open-shoves 10,000 with low equity
        # Keep all-in if facing a bet >= 3000 chips (already pot-committed)
        if facing_bet >= 3000:
            return 5, "Legit Pot-Committed Postflop All-in"
            
        # If open-shoving 10,000 chips into a 500-chip pot (overbet 20x),
        # downsize to 100% or 200% pot raise to allow authentic poker play
        if pot < 3000 and 4 in legal_actions:
            return 4, "Postflop Overbet Downsized to 200% Pot"
        elif pot < 3000 and 3 in legal_actions:
            return 3, "Postflop Overbet Downsized to 100% Pot"
            
        return 5, "Legit Postflop All-in"
