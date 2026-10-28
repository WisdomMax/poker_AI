#!/usr/bin/env python3
"""
v1024/test_directive056_pfsp_sign.py
Unit test verifying CorrectedAlphaStarPFSPSampler directly imported from directive056_production_engine.py.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from v1024.directive056_production_engine import CorrectedAlphaStarPFSPSampler

def test_pfsp_sign():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 056 GATE] Testing CorrectedAlphaStarPFSPSampler Direct Import")
    print("========================================================================")
    
    pool = ["Agent_A_2000h", "Agent_B_2000h", "Agent_C_2000h"]
    sampler = CorrectedAlphaStarPFSPSampler(pool, temperature=10.0, exploration_floor=0.1)
    
    # Simulate matches:
    # Weak: opponent lost -15.0 BB/hand
    # Neutral: opponent made 0.0 BB/hand
    # Strong: opponent won +15.0 BB/hand
    for _ in range(10):
        sampler.record_match("Agent_A_2000h", opponent_net_bb=-15.0)
        sampler.record_match("Agent_B_2000h", opponent_net_bb=0.0)
        sampler.record_match("Agent_C_2000h", opponent_net_bb=+15.0)
        
    probs = sampler.get_probabilities()
    p_weak = probs["Agent_A_2000h"]
    p_neutral = probs["Agent_B_2000h"]
    p_strong = probs["Agent_C_2000h"]
    
    print(f"Probabilities after 10 matches:")
    print(f"  Weak Opponent   (opp_avg = -15.0 BB): {p_weak:.5f} ({p_weak*100:.2f}%)")
    print(f"  Neutral Opponent (opp_avg =   0.0 BB): {p_neutral:.5f} ({p_neutral*100:.2f}%)")
    print(f"  Strong Opponent  (opp_avg = +15.0 BB): {p_strong:.5f} ({p_strong*100:.2f}%)")
    
    assert p_strong > p_neutral > p_weak, "[FAIL] Monotonicity violated!"
    assert p_strong > 5.0 * p_weak, "[FAIL] Hard opponent prioritization ratio insufficient!"
    assert p_weak >= 0.1 / len(pool) - 1e-6, "[FAIL] Exploration floor violated!"
    
    print("  -> [PASS] P(Strong) > P(Neutral) > P(Weak) strictly confirmed.")
    print(f"  -> [PASS] Strong vs Weak ratio: {p_strong / p_weak:.2f}x.")
    print("========================================================================")
    print(">>> [GATE PASS] Production CorrectedAlphaStarPFSPSampler 100% Sound!")
    print("========================================================================\n")
    return True

if __name__ == "__main__":
    test_pfsp_sign()
