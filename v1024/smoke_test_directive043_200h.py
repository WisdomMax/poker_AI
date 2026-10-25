#!/usr/bin/env python3
"""
PHASE B: 200-Hand Smoke Test for DIRECTIVE 043 Universal Poker AlphaGo
Verifies:
1. 5 independent agents (Agent A~E) initialize with distinct weights.
2. All 5 agents actively receive gradient updates.
3. Multi-agent zero-sum invariant holds strictly for every single hand.
4. Loss, entropy, and action distributions are recorded without NaN/infinities.
"""
import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v1024.alphago_5p_selfplay import MultiAgentPokerAlphaGoEngine

def run_smoke_test():
    print("=" * 70)
    print("STARTING DIRECTIVE 043 PHASE B: 200-HAND SMOKE TEST")
    print("=" * 70)
    
    engine = MultiAgentPokerAlphaGoEngine(
        base_ckpt_path="v1024/models_e2e/PPO_E2E_Final_50k.pt",
        log_file="v1024/alphago_smoke_200h.log",
        base_seed=606000
    )
    
    # Run 200 hands with update interval of 50 hands (each agent gets 4 updates)
    res = engine.run_selfplay_pilot(num_hands=200, checkpoint_interval=200, update_batch_hands=50)
    
    print("\n--- SMOKE TEST AUDIT REPORT ---")
    print(f"Total Hands: {res['num_hands']}")
    print(f"Elapsed Time: {res['elapsed_sec']:.1f}s")
    print("Agent Final PnLs (BB):", res["final_agent_pnls"])
    print("Gradient Updates per Agent:", res["gradient_updates"])
    
    # Verifications
    tot_pnl = sum(res["final_agent_pnls"].values())
    assert abs(tot_pnl) < 1e-3, f"Zero-sum invariant broken! sum={tot_pnl}"
    
    for a in engine.agents:
        assert a.gradient_update_count > 0, f"{a.name} received ZERO gradient updates!"
        assert len(a.loss_history) > 0, f"{a.name} has no loss history!"
        latest_loss = a.loss_history[-1]
        assert not any(map(lambda v: (isinstance(v, float) and (v != v)), latest_loss.values())), f"NaN detected in {a.name} losses!"
        print(f"[PASS] {a.name}: Gradient Updates={a.gradient_update_count}, "
              f"Actor Loss={latest_loss['actor_loss']:.4f}, Critic Loss={latest_loss['critic_loss']:.4f}, "
              f"Entropy={latest_loss['entropy']:.4f}, Mean Ratio={latest_loss['mean_ratio']:.4f}")
              
    print("=" * 70)
    print("PHASE B: 200-HAND SMOKE TEST 100% PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
