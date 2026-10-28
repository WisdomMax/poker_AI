#!/usr/bin/env python3
"""
v1024/train_directive056_scaleup.py
DIRECTIVE 056 TASK 4: Controlled Extended League Scale-Up (2,000 Hands x 4 Sessions = 8,000 Table Hands).

Matrix:
- Uniform Extended: Seed 901000 (2,000h) & Seed 902000 (2,000h)
- Corrected PFSP Extended: Seed 901000 (2,000h) & Seed 902000 (2,000h)
- Checkpoints at 500h, 1,000h, 1,500h, 2,000h.
- Total production: 8,000 table hands across 4 sessions.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v1024.directive056_production_engine import run_directive056_arm

EXP056_DIR = "v1024/exp_directive056"
TOTAL_HANDS = 2000
SEEDS = [901000, 902000]
LEAGUES = ["uniform", "pfsp"]

def run_scaleup_production():
    print("\n========================================================================")
    print(">>> [DIRECTIVE 056 TASK 4] 2,000-Hand Extended League Scale-Up")
    print(f"    4 Sessions x 2,000 Hands = 8,000 Table Hands | Seeds: {SEEDS}")
    print("========================================================================")
    
    os.makedirs(EXP056_DIR, exist_ok=True)
    summary_matrix = {}
    t_global = time.time()
    
    sess_id = 0
    for league in LEAGUES:
        for seed in SEEDS:
            sess_id += 1
            arm_name = f"arm_{league}_extended_seed_{seed}"
            out_dir = os.path.join(EXP056_DIR, arm_name)
            
            print(f"\n[{sess_id}/4] Launching {arm_name} (2,000 Hands)...")
            arm_summary = run_directive056_arm(
                arm_name=arm_name,
                league_type=league,
                opponent_pool_type="extended",
                total_hands=TOTAL_HANDS,
                seed_base=seed,
                out_dir=out_dir
            )
            summary_matrix[arm_name] = arm_summary
            
            # Verify checkpoints
            for h in [500, 1000, 1500, 2000]:
                ckpt = os.path.join(out_dir, f"ckpt_hand_{h}")
                assert os.path.exists(ckpt), f"[FAIL] Missing checkpoint {ckpt}"
                
    summary_path = os.path.join(EXP056_DIR, "scaleup_production_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_hands": 8000,
            "sessions": len(summary_matrix),
            "total_elapsed_seconds": round(time.time() - t_global, 2),
            "matrix": summary_matrix
        }, f, indent=2)
        
    print("\n========================================================================")
    print(f">>> [ALL 4 SESSIONS COMPLETED] 8,000 Hands Scale-Up Finished in {round(time.time() - t_global, 2)}s!")
    print(f"    Summary saved to: {summary_path}")
    print("========================================================================\n")

if __name__ == "__main__":
    run_scaleup_production()
