#!/usr/bin/env python3
"""
v1024/test_directive057_league_mechanics.py
Unit tests for DIRECTIVE 057 Dynamic League, Immutability, and Exploration Floor.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v1024.directive057_production_engine import (
    LeagueOpponentPool, compute_sha256, CHECKPOINT_DIR, AGENT_NAMES
)

def test_static_league_immutability():
    print("[*] Testing Static League immutability...")
    base_models = [{
        "id": f"frozen_7k_{a}",
        "path": os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt"),
        "meta": {"agent": a}
    } for a in AGENT_NAMES]
    
    pool = LeagueOpponentPool(base_models, mode="static", sampler_type="pfsp")
    assert len(pool.opponents) == 5, f"Expected 5 base opponents, got {len(pool.opponents)}"
    
    # Attempting to add snapshot in static mode must raise RuntimeError
    try:
        pool.add_snapshot("snap_500h_Agent_A", base_models[0]["path"], {})
        assert False, "Static pool should NOT accept new snapshots!"
    except RuntimeError as e:
        print(f"  -> Successfully caught expected error: {e}")
    print("  [PASS] Static League immutability verified.")

def test_dynamic_league_expansion_and_exploration_floor():
    print("\n[*] Testing Dynamic League expansion and exploration floor...")
    base_models = [{
        "id": f"frozen_7k_{a}",
        "path": os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt"),
        "meta": {"agent": a}
    } for a in AGENT_NAMES]
    
    pool = LeagueOpponentPool(
        base_models,
        mode="dynamic",
        sampler_type="pfsp",
        min_exploration_prob=0.05,
        temperature=10.0
    )
    assert len(pool.opponents) == 5
    
    # Add snapshot
    snap_id = "snap_500h_Agent_A"
    snap_path = base_models[0]["path"]
    sha_before = compute_sha256(snap_path)
    
    pool.add_snapshot(snap_id, snap_path, {"sha256": sha_before})
    assert len(pool.opponents) == 6, f"Expected 6 opponents after snapshot, got {len(pool.opponents)}"
    assert pool.opponents[snap_id]["sha256"] == sha_before
    
    # Record extreme win and extreme loss to test floor
    # Learner wins heavily vs snap_id (opponent lost -500 BB)
    pool.record_result(snap_id, learner_net_bb=+500.0, opponent_net_bb=-500.0)
    # Learner loses heavily vs frozen_7k_Agent_A (opponent won +500 BB)
    pool.record_result("frozen_7k_Agent_A", learner_net_bb=-500.0, opponent_net_bb=+500.0)
    
    probs = pool.get_selection_probabilities()
    n = len(pool.opponents)
    min_allowed = 0.05 / n # min exploration floor
    
    print(f"  Probabilities after matches (Pool size: {n}):")
    for oid, p in probs.items():
        print(f"    {oid:22s} -> {p*100:6.2f}%")
        assert p >= min_allowed - 1e-7, f"Opponent {oid} prob {p} fell below exploration floor {min_allowed}!"
        
    # Verify strong opponent got highest probability
    assert probs["frozen_7k_Agent_A"] > probs[snap_id], "Strong opponent must have higher probability than weak opponent!"
    print(f"  Strong opponent prob: {probs['frozen_7k_Agent_A']*100:.2f}% vs Weak opponent prob: {probs[snap_id]*100:.2f}%")
    print("  [PASS] Dynamic League expansion and exploration floor verified.")

def main():
    print("=== [DIRECTIVE 057] LEAGUE MECHANICS GATE TEST ===")
    test_static_league_immutability()
    test_dynamic_league_expansion_and_exploration_floor()
    print("\n[GATE PASS] All Directive 057 League Mechanics tests passed perfectly!")

if __name__ == "__main__":
    main()
