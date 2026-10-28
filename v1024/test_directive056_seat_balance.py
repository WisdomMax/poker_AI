#!/usr/bin/env python3
"""
v1024/test_directive056_seat_balance.py
DIRECTIVE 056 OPERATIONAL GATE:
Unit test verifying Perfectly Balanced Symmetrical Latin Square Seat Rotation.

Specification:
In 1,000 table hands:
- Every Agent (A, B, C, D, E) MUST be assigned to every seat (0, 1, 2, 3, 4) EXACTLY 160 times.
- Every Agent MUST sit out EXACTLY 200 times.
- Frozen opponent MUST appear in every seat EXACTLY 200 times.
- Zero position bias, zero learner bias (100.000% mathematical symmetry).
"""

import numpy as np

def get_perfect_latin_square_seat_assignment(hand_idx: int) -> dict:
    """
    Computes perfectly symmetrical Latin square seat assignments for hand_idx (1-based).
    Cycle period: 25 hands.
    """
    h0 = hand_idx - 1
    c25 = h0 % 25
    r = c25 % 5       # Frozen seat 0..4
    k = c25 // 5      # Rotation block 0..4
    
    frozen_seat = r
    sit_out_agent = (r + k) % 5
    
    avail_seats = [s for s in range(5) if s != frozen_seat]
    
    seat_to_agent = {}
    for s_idx, seat in enumerate(avail_seats):
        agent = (r + k + s_idx + 1) % 5
        seat_to_agent[seat] = agent
        
    return {
        "frozen_seat": frozen_seat,
        "sit_out_agent": sit_out_agent,
        "seat_to_agent": seat_to_agent # seat -> agent_idx (0..4)
    }

def test_seat_rotation_balance():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 056 GATE] Testing Perfect Symmetrical Latin Square Seat Balance")
    print("========================================================================")
    
    TOTAL_HANDS = 1000
    seat_matrix = np.zeros((5, 5), dtype=int)
    sit_out_counts = np.zeros(5, dtype=int)
    frozen_seat_counts = np.zeros(5, dtype=int)
    
    for h in range(1, TOTAL_HANDS + 1):
        assign = get_perfect_latin_square_seat_assignment(h)
        frozen_seat = assign["frozen_seat"]
        sit_out = assign["sit_out_agent"]
        
        frozen_seat_counts[frozen_seat] += 1
        sit_out_counts[sit_out] += 1
        
        for seat, agent in assign["seat_to_agent"].items():
            seat_matrix[agent, seat] += 1
            
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    
    print("\n[Seat Allocation Matrix across 1,000 Hands]")
    print("Agent       | Seat 0 | Seat 1 | Seat 2 | Seat 3 | Seat 4 | Sit-Out | Total")
    print("-------------------------------------------------------------------------")
    for a_idx, a_name in enumerate(agents):
        row = seat_matrix[a_idx]
        sit_out = sit_out_counts[a_idx]
        tot = sum(row) + sit_out
        print(f"{a_name:11s} | {row[0]:6d} | {row[1]:6d} | {row[2]:6d} | {row[3]:6d} | {row[4]:6d} | {sit_out:7d} | {tot:5d}")
        
    print("-------------------------------------------------------------------------")
    print(f"Frozen Opp  | {frozen_seat_counts[0]:6d} | {frozen_seat_counts[1]:6d} | {frozen_seat_counts[2]:6d} | {frozen_seat_counts[3]:6d} | {frozen_seat_counts[4]:6d} |       - |  1000")
    
    # Strict Assertions
    assert np.all(seat_matrix == 160), f"[FAIL] Seat allocation is not exactly 160 for all!\n{seat_matrix}"
    assert np.all(sit_out_counts == 200), f"[FAIL] Sit-out count is not exactly 200 for all!\n{sit_out_counts}"
    assert np.all(frozen_seat_counts == 200), f"[FAIL] Frozen seat count is not exactly 200 for all!\n{frozen_seat_counts}"
    
    print("\n  -> [PASS] Every agent sits in every seat EXACTLY 160 times!")
    print("  -> [PASS] Every agent rests EXACTLY 200 times!")
    print("  -> [PASS] Frozen opponent occupies every seat EXACTLY 200 times!")
    print("========================================================================")
    print(">>> [GATE PASS] Perfectly Balanced Symmetrical Latin Square Verified 100%!")
    print("========================================================================\n")
    return True

if __name__ == "__main__":
    test_seat_rotation_balance()
