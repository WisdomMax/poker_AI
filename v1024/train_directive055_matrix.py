#!/usr/bin/env python3
"""
v1024/train_directive055_matrix.py
DIRECTIVE 055 TASK 3: Controlled 2x2 Factorial Reinforcement Learning Engine (8,000 Table Hands).

Factorial Matrix:
- Factor A (Opponent Selection): Uniform vs Corrected PFSP (2 levels)
- Factor B (Opponent Pool): Frozen 2k Pool vs Extended Pool (2k + 7k) (2 levels)
- Seeds: 2 Replicated Seeds (801000, 802000)
- Hands per session: 1,000 table hands (Checkpoints at 500h Canary and 1,000h Final)
- Total learning: 4 arms x 2 seeds x 1,000 hands = 8,000 table hands.

Integrates all TASK 1 verified components:
- 5-Agent Symmetrical Learning & Seat Rotation
- Dual Decoupled RNGs (deck_rng & action_rng)
- Active RealProcessCPUController (<= 70% target)
- Exact Categorical KL Divergence, Policy Entropy & NaN Protection
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v1024.directive055_production_engine import run_directive055_arm

EXP_DIR = "v1024/exp_directive055"
TOTAL_HANDS = 1000
SEEDS = [801000, 802000]

ARMS = [
    ("uniform", "frozen2k"),
    ("pfsp", "frozen2k"),
    ("uniform", "extended"),
    ("pfsp", "extended")
]

def run_matrix_production():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 055 TASK 3] Starting 2x2 Factorial Matrix Production")
    print(f"    4 Arms x 2 Seeds = 8 Sessions | 1,000 Hands each (Total: 8,000 Hands)")
    print("========================================================================")
    
    os.makedirs(EXP_DIR, exist_ok=True)
    summary_matrix = {}
    t_global_start = time.time()
    
    session_idx = 0
    for league_type, pool_type in ARMS:
        for seed in SEEDS:
            session_idx += 1
            arm_name = f"arm_{league_type}_{pool_type}_seed_{seed}"
            out_dir = os.path.join(EXP_DIR, arm_name)
            
            print(f"\n[{session_idx}/8] Launching {arm_name}...")
            arm_summary = run_directive055_arm(
                arm_name=arm_name,
                league_type=league_type,
                opponent_pool_type=pool_type,
                total_hands=TOTAL_HANDS,
                seed_base=seed,
                out_dir=out_dir
            )
            summary_matrix[arm_name] = arm_summary
            
            # Canary Health Check at 500h checkpoint
            ckpt_500 = os.path.join(out_dir, "ckpt_hand_500")
            assert os.path.exists(ckpt_500), f"[FAIL] Canary 500h checkpoint missing: {ckpt_500}"
            ckpt_1000 = os.path.join(out_dir, "ckpt_hand_1000")
            assert os.path.exists(ckpt_1000), f"[FAIL] Final 1000h checkpoint missing: {ckpt_1000}"
            
    summary_path = os.path.join(EXP_DIR, "matrix_production_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_hands": 8000,
            "sessions": len(summary_matrix),
            "total_elapsed_seconds": round(time.time() - t_global_start, 2),
            "matrix": summary_matrix
        }, f, indent=2)
        
    print("\n========================================================================")
    print(f">>> [ALL SESSIONS COMPLETED] 8,000 Table Hands Production Finished!")
    print(f"    Total Elapsed Time: {round(time.time() - t_global_start, 2)}s")
    print(f"    Summary saved to: {summary_path}")
    print("========================================================================\n")

if __name__ == "__main__":
    run_matrix_production()
