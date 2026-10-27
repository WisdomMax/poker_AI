#!/usr/bin/env python3
"""
v1024/run_directive050_phaseA.py
Runs DIRECTIVE 050 Phase A (Task 1, 2, 3), generates audit JSON, and produces the Phase A report.
"""

import json
import os
import sys
import time

sys.path.insert(0, ".")

from v1024.audit_directive050_tasks import run_task1_branching_analysis, run_task2_250pair_unit_statistics
from v1024.train_directive050_integrated_pilot import run_smoke_pilot

def main():
    print("================================================================================")
    print("DIRECTIVE 050: PHASE A INTEGRATED SMOKE & CRITICAL AUDIT")
    print("================================================================================")
    t0 = time.time()
    
    # Task 1
    t1 = run_task1_branching_analysis()
    
    # Task 2
    t2 = run_task2_250pair_unit_statistics()
    
    # Task 3
    t3 = run_smoke_pilot(num_hands=150, seed_base=91000)
    
    elapsed = time.time() - t0
    
    results = {
        "directive": "050-PhaseA",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(elapsed, 2),
        "task1_branching_analysis": t1,
        "task2_250pair_unit_statistics": t2,
        "task3_smoke_pilot": t3
    }
    
    json_path = "v1024/directive050_smoke_audit_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nAudit results successfully saved to {json_path}")
    
    # Generate Phase A report
    import subprocess
    subprocess.run(["python3", "v1024/generate_directive050_phaseA_report.py"], check=True)
    print("[Phase A Complete]")

if __name__ == "__main__":
    main()
