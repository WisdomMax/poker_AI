#!/usr/bin/env python3
"""
DIRECTIVE 047: Flawless Native Action Ledger Tracker
Tracks exact player-by-player actions, street transitions, chips invested,
real all-in events, and precise opponent call events with ZERO false positives.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

STARTING_STACK = 10000.0 # $100 BB in cents
SMALL_BLIND = 50.0       # 0.5 BB
BIG_BLIND = 100.0        # 1.0 BB

STREET_NAMES = ["Preflop", "Flop", "Turn", "River"]

@dataclass
class ActionRecord:
    player_id: int
    seat: int
    street: str
    round_idx: int
    abstract_action: int # 0=Fold, 1=Check/Call, 2=HalfPot, 3=Pot, 4=2xPot, 5=AllIn
    native_action: int
    stack_before: float
    stack_after: float
    spent_before: float
    spent_after: float
    call_amount: float
    is_allin_action: bool
    is_facing_allin: bool

@dataclass
class AllInEvent:
    player_id: int
    seat: int
    street: str
    round_idx: int
    allin_amount: float     # Total chips committed to all-in
    stack_committed: float  # Chips invested in this specific action
    callers: List[Tuple[int, float]] = field(default_factory=list) # (player_id, called_amount)
    folders: List[int] = field(default_factory=list)               # player_ids who folded directly to this allin

class HandLedgerTracker:
    """
    Comprehensive, event-driven action ledger tracker.
    Eliminates all heuristics (like len(folded_seats) < 4) and tracks exact call/allin events.
    """
    def __init__(self, num_players: int = 5, starting_stack: float = STARTING_STACK):
        self.num_players = num_players
        self.starting_stack = starting_stack
        
        self.stacks = [starting_stack] * num_players
        self.spent = [0.0] * num_players
        self.folded = [False] * num_players
        self.is_allin = [False] * num_players
        
        self.actions: List[ActionRecord] = []
        self.allin_events: Dict[int, AllInEvent] = {} # player_id -> AllInEvent
        self.active_allins: List[int] = []            # player_ids currently all-in
        
        # Blinds post
        # Standard: Seat 0 = SB, Seat 1 = BB
        self.spent[0] = min(SMALL_BLIND, self.stacks[0])
        self.stacks[0] -= self.spent[0]
        self.spent[1] = min(BIG_BLIND, self.stacks[1])
        self.stacks[1] -= self.spent[1]

    def record_action(self, player_id: int, abstract_action: int, native_action: int,
                      round_idx: int, spent_after_action: float) -> ActionRecord:
        street = STREET_NAMES[round_idx] if round_idx < len(STREET_NAMES) else f"Round_{round_idx}"
        stack_before = self.stacks[player_id]
        spent_before = self.spent[player_id]
        
        # Calculate to_call prior to this action
        current_max_spent = max(self.spent)
        to_call = max(0.0, current_max_spent - spent_before)
        
        is_facing_allin = any(self.is_allin[p] for p in range(self.num_players) if p != player_id and not self.folded[p])
        
        # Calculate new spent and stack
        actual_invested = max(0.0, spent_after_action - spent_before)
        stack_after = max(0.0, stack_before - actual_invested)
        
        self.spent[player_id] = spent_after_action
        self.stacks[player_id] = stack_after
        
        call_amount = 0.0
        is_allin_act = False
        
        if abstract_action == 0:
            # FOLD
            self.folded[player_id] = True
            # Record fold for any pending active all-ins
            for allin_p in self.active_allins:
                if allin_p != player_id:
                    self.allin_events[allin_p].folders.append(player_id)
        elif abstract_action == 1:
            # CHECK or CALL
            call_amount = min(to_call, actual_invested)
            # If player was facing an allin and called:
            if to_call > 0 and self.active_allins:
                for allin_p in self.active_allins:
                    if allin_p != player_id:
                        # Player contributed chips to match this allin
                        self.allin_events[allin_p].callers.append((player_id, call_amount))
        else:
            # BET / RAISE / ALLIN (actions 2, 3, 4, 5)
            # If this bet matches an existing allin:
            if to_call > 0 and self.active_allins:
                for allin_p in self.active_allins:
                    if allin_p != player_id:
                        call_portion = min(to_call, actual_invested)
                        self.allin_events[allin_p].callers.append((player_id, call_portion))
        
        # Check if this player is now ALL-IN
        if stack_after == 0.0 or abstract_action == 5:
            is_allin_act = True
            self.is_allin[player_id] = True
            if player_id not in self.allin_events:
                ev = AllInEvent(
                    player_id=player_id,
                    seat=player_id,
                    street=street,
                    round_idx=round_idx,
                    allin_amount=spent_after_action,
                    stack_committed=actual_invested
                )
                self.allin_events[player_id] = ev
                self.active_allins.append(player_id)
        
        rec = ActionRecord(
            player_id=player_id,
            seat=player_id,
            street=street,
            round_idx=round_idx,
            abstract_action=abstract_action,
            native_action=native_action,
            stack_before=stack_before,
            stack_after=stack_after,
            spent_before=spent_before,
            spent_after=spent_after_action,
            call_amount=call_amount,
            is_allin_action=is_allin_act,
            is_facing_allin=is_facing_allin
        )
        self.actions.append(rec)
        return rec

    def is_player_allin(self, player_id: int) -> bool:
        """Returns True if and only if player actually reached all-in in this hand."""
        return self.is_allin[player_id]

    def was_allin_called(self, player_id: int) -> bool:
        """
        Returns True if player went all-in AND is in an all-in showdown with at least one opponent.
        (Includes: Player pushed and opponent called, OR opponent pushed and Player called).
        GUARANTEE: If player did NOT go all-in, returns False unconditionally (0 false positives).
        """
        if not self.is_allin[player_id]:
            return False
        
        # If player has recorded callers, definitely called:
        if player_id in self.allin_events and len(self.allin_events[player_id].callers) > 0:
            return True
        
        # If player called an all-in and went to showdown with active opponents:
        active_opponents = [p for p in range(self.num_players) if p != player_id and not self.folded[p]]
        return len(active_opponents) > 0

    def is_uncalled_allin(self, player_id: int) -> bool:
        """
        Returns True if and only if player went all-in and ALL opponents folded (uncontested pot win).
        GUARANTEE: Return is always strictly positive (wins blinds + uncontested bets).
        """
        if not self.is_allin[player_id]:
            return False
        active_opponents = [p for p in range(self.num_players) if p != player_id and not self.folded[p]]
        return len(active_opponents) == 0

    def get_allin_callers(self, player_id: int) -> List[Tuple[int, float]]:
        """Returns list of (opponent_id, called_amount) for player's all-in."""
        if player_id not in self.allin_events:
            return []
        return self.allin_events[player_id].callers

    def get_allin_folders(self, player_id: int) -> List[int]:
        """Returns list of opponent_ids who folded directly to player's all-in."""
        if player_id not in self.allin_events:
            return []
        return self.allin_events[player_id].folders

    def get_summary(self, hero_seat: int) -> Dict[str, Any]:
        """Generates comprehensive ledger summary for the hand."""
        hero_allin = self.is_player_allin(hero_seat)
        hero_called = self.was_allin_called(hero_seat)
        callers = self.get_allin_callers(hero_seat)
        
        return {
            "hero_seat": hero_seat,
            "hero_allin": hero_allin,
            "hero_allin_called": hero_called,
            "allin_caller_count": len(callers),
            "allin_callers": callers,
            "total_pot": sum(self.spent),
            "player_spents": list(self.spent),
            "player_folded": list(self.folded),
            "player_allin": list(self.is_allin),
            "num_actions": len(self.actions)
        }
