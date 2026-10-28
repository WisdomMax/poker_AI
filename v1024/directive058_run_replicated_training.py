#!/usr/bin/env python3
"""
v1024/directive058_run_replicated_training.py
TASK 6: Replicated Production Training Suite (16,000 Table Hands)
- 8 Replicated Production Sessions across Seed 1101000 & Seed 1102000
- 2,000 table hands per session
- CPU <= 70% cumulative regulation
- 500h checkpoints (500, 1000, 1500, 2000) with bitwise pause/resume meta_state.pt
- Generates directive058_training_replications.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Any

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from v1024.directive058_production_engine import (
    Directive058ProductionSession,
    compute_sha256,
    AGENT_NAMES
)

REPLICATED_DIR = "v1024/exp_directive058_replicated"
OUT_REPLICATIONS_JSON = "directive058_training_replications.json"

REPLICATED_ARMS = [
    {
        "arm_name": "arm_uniform_static_seed_1101000",
        "league_mode": "static",
        "sampler_type": "uniform",
        "seed_base": 1101000
    },
    {
        "arm_name": "arm_uniform_static_seed_1102000",
        "league_mode": "static",
        "sampler_type": "uniform",
        "seed_base": 1102000
    },
    {
        "arm_name": "arm_pfsp_static_seed_1101000",
        "league_mode": "static",
        "sampler_type": "pfsp",
        "seed_base": 1101000
    },
    {
        "arm_name": "arm_pfsp_static_seed_1102000",
        "league_mode": "static",
        "sampler_type": "pfsp",
        "seed_base": 1102000
    },
    {
        "arm_name": "arm_uniform_dynamic_seed_1101000",
        "league_mode": "dynamic",
        "sampler_type": "uniform",
        "seed_base": 1101000
    },
    {
        "arm_name": "arm_uniform_dynamic_seed_1102000",
        "league_mode": "dynamic",
        "sampler_type": "uniform",
        "seed_base": 1102000
    },
    {
        "arm_name": "arm_pfsp_dynamic_seed_1101000",
        "league_mode": "dynamic",
        "sampler_type": "pfsp",
        "seed_base": 1101000
    },
    {
        "arm_name": "arm_pfsp_dynamic_seed_1102000",
        "league_mode": "dynamic",
        "sampler_type": "pfsp",
        "seed_base": 1102000
    }
]

def run_all_replications():
    print("=================================================================")
    print(">>> [TASK 6] DIRECTIVE 058 REPLICATED PRODUCTION SUITE (16,000h)")
    print("=================================================================")
    
    os.makedirs(REPLICATED_DIR, exist_ok=True)
    suite_summary = {}
    total_start = time.time()
    
    for idx, arm_cfg in enumerate(REPLICATED_ARMS):
        arm_name = arm_cfg["arm_name"]
        arm_out = os.path.join(REPLICATED_DIR, arm_name)
        print(f"\n[{idx+1}/8] Starting Replicated Session: {arm_name} (2,000h)...")
        
        session = Directive058ProductionSession(
            arm_name=arm_name,
            league_mode=arm_cfg["league_mode"],
            sampler_type=arm_cfg["sampler_type"],
            total_hands=2000,
            seed_base=arm_cfg["seed_base"],
            out_dir=arm_out,
            checkpoint_interval=500,
            target_cpu_ratio=0.70
        )
        
        summary = session.run()
        
        # Verify 2,000h checkpoints
        ckpt_2k = os.path.join(arm_out, "ckpt_hand_2000")
        ckpt_shas = {}
        for agent in AGENT_NAMES:
            p_path = os.path.join(ckpt_2k, f"{agent}.pt")
            ckpt_shas[agent] = compute_sha256(p_path) if os.path.exists(p_path) else "MISSING"
            
        summary["checkpoints_2000h_sha256"] = ckpt_shas
        suite_summary[arm_name] = summary
        print(f"  -> Finished {arm_name} in {summary['elapsed_seconds']}s | CPU: {summary['final_cpu_ratio_pct']}% | Models verified.")
        
    tot_elapsed = time.time() - total_start
    print(f"\n=================================================================")
    print(f"[+] All 8 Replicated Sessions Completed in {round(tot_elapsed, 2)}s!")
    print(f"=================================================================")
    
    # Save master replication JSON
    out_data = {
        "metadata": {
            "title": "DIRECTIVE 058 Replicated Production Training Master Suite",
            "total_sessions": len(REPLICATED_ARMS),
            "hands_per_session": 2000,
            "total_hands": len(REPLICATED_ARMS) * 2000,
            "seeds": [1101000, 1102000],
            "total_elapsed_seconds": round(tot_elapsed, 2)
        },
        "sessions": suite_summary
    }
    
    with open(OUT_REPLICATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"[+] Master JSON saved to {OUT_REPLICATIONS_JSON}")

if __name__ == "__main__":
    run_all_replications()