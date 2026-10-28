#!/usr/bin/env python3
"""
v1024/train_directive057_scaleup.py
Executes DIRECTIVE 057: 16,000-Hand 2x2 Factorial League Training
- Arm 1: Uniform Static (Seeds 1001000, 1002000)
- Arm 2: PFSP Static (Seeds 1001000, 1002000)
- Arm 3: Uniform Dynamic (Seeds 1001000, 1002000)
- Arm 4: PFSP Dynamic (Seeds 1001000, 1002000)
Total: 8 sessions x 2,000 hands = 16,000 table hands.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v1024.directive057_production_engine import Directive057ProductionSession

EXP_BASE_DIR = "v1024/exp_directive057"
SEEDS = [1001000, 1002000]
TOTAL_HANDS_PER_ARM = 2000

CONFIGS = [
    # Arm 1: Uniform Static
    {"arm_name": "arm_uniform_static", "mode": "static", "sampler": "uniform"},
    # Arm 2: PFSP Static
    {"arm_name": "arm_pfsp_static", "mode": "static", "sampler": "pfsp"},
    # Arm 3: Uniform Dynamic
    {"arm_name": "arm_uniform_dynamic", "mode": "dynamic", "sampler": "uniform"},
    # Arm 4: PFSP Dynamic
    {"arm_name": "arm_pfsp_dynamic", "mode": "dynamic", "sampler": "pfsp"}
]

def main():
    print("=================================================================")
    print(">>> [DIRECTIVE 057] 16,000-HAND 2x2 FACTORIAL LEAGUE TRAINING")
    print(f"    Configurations: {len(CONFIGS)} arms x {len(SEEDS)} seeds")
    print(f"    Total Table Hands: {len(CONFIGS) * len(SEEDS) * TOTAL_HANDS_PER_ARM}")
    print("=================================================================")
    
    os.makedirs(EXP_BASE_DIR, exist_ok=True)
    all_summaries = {}
    total_start = time.time()
    
    for cfg in CONFIGS:
        for seed in SEEDS:
            session_name = f"{cfg['arm_name']}_seed_{seed}"
            session_out = os.path.join(EXP_BASE_DIR, session_name)
            
            print(f"\n---------------------------------------------------------")
            print(f"[*] Starting Session: {session_name}")
            print(f"    Mode: {cfg['mode']} | Sampler: {cfg['sampler']} | Seed: {seed}")
            print(f"---------------------------------------------------------")
            
            session = Directive057ProductionSession(
                arm_name=session_name,
                league_mode=cfg["mode"],
                sampler_type=cfg["sampler"],
                total_hands=TOTAL_HANDS_PER_ARM,
                seed_base=seed,
                out_dir=session_out
            )
            
            summary = session.run()
            all_summaries[session_name] = summary
            
    total_elapsed = round(time.time() - total_start, 2)
    master_summary = {
        "directive": "DIRECTIVE_057",
        "total_sessions": len(all_summaries),
        "total_table_hands": len(all_summaries) * TOTAL_HANDS_PER_ARM,
        "total_elapsed_seconds": total_elapsed,
        "sessions": all_summaries
    }
    
    summary_path = os.path.join(EXP_BASE_DIR, "production_master_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
        
    print("\n=================================================================")
    print(f">>> [ALL SESSIONS COMPLETE] 16,000 Table Hands Finished in {total_elapsed}s!")
    print(f"    Master summary saved to: {summary_path}")
    print("=================================================================")

if __name__ == "__main__":
    main()
