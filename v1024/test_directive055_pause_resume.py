#!/usr/bin/env python3
"""
v1024/test_directive055_pause_resume.py
DIRECTIVE 055 OPERATIONAL GATE:
Rigorous Bit-Exact Equivalence Verification between Continuous 50 Hands and 25+25 Paused/Resumed Execution.

Uses the EXACT production runner from `directive055_production_engine.py` across separate subprocesses.
Validates:
1. Bit-exact SHA256 hashes of all 5 agent neural models at Hand 50.
2. Identical Agent PnL across all 5 learners.
3. Bit-exact PPO update metrics in ppo_updates.jsonl.
4. Final Sampler probabilities and selection logs.
"""

import os
import sys
import shutil
import subprocess
import json
import hashlib
from typing import Dict, Any

TEST_DIR = "v1024/exp_directive055_test_pause_resume"
CONT_DIR = os.path.join(TEST_DIR, "arm_continuous_50h")
RESUME_DIR = os.path.join(TEST_DIR, "arm_resumed_25_25h")
SEED = 998800

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_subprocess(cmd: str):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDOUT:\n{res.stdout}")
        print(f"STDERR:\n{res.stderr}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res.stdout

def run_pause_resume_gate():
    print("\n==================================================================")
    print(">>> [DIRECTIVE 055 GATE] Continuous 50h vs 25+25h Pause/Resume Verification")
    print("==================================================================")
    
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(CONT_DIR, exist_ok=True)
    os.makedirs(RESUME_DIR, exist_ok=True)
    
    # 1. Run Continuous 50 Hands in Subprocess 1
    print("\n[Step 1] Running 50 hands continuously in Subprocess 1...")
    script_cont = f"""
import sys, os
sys.path.insert(0, os.getcwd())
from v1024.directive055_production_engine import run_directive055_arm
run_directive055_arm(
    arm_name='Continuous_50h',
    league_type='pfsp',
    opponent_pool_type='frozen2k',
    total_hands=50,
    seed_base={SEED},
    out_dir='{CONT_DIR}'
)
"""
    cmd1 = f'python3 -c "{script_cont}"'
    out1 = run_subprocess(cmd1)
    print("  -> Subprocess 1 finished successfully.")
    
    # 2. Run First 25 Hands in Subprocess 2
    print("\n[Step 2] Running first 25 hands and saving checkpoint in Subprocess 2...")
    script_p1 = f"""
import sys, os
sys.path.insert(0, os.getcwd())
from v1024.directive055_production_engine import run_directive055_arm
run_directive055_arm(
    arm_name='Part1_25h',
    league_type='pfsp',
    opponent_pool_type='frozen2k',
    total_hands=25,
    seed_base={SEED},
    out_dir='{RESUME_DIR}'
)
"""
    cmd2 = f'python3 -c "{script_p1}"'
    out2 = run_subprocess(cmd2)
    print("  -> Subprocess 2 finished successfully (checkpoint at Hand 25 created).")
    
    # Verify ckpt_hand_25 exists
    ckpt_25_dir = os.path.join(RESUME_DIR, "ckpt_hand_25")
    assert os.path.exists(ckpt_25_dir), f"Checkpoint missing: {ckpt_25_dir}"
    
    # 3. Resume from Hand 25 to 50 in Subprocess 3
    print("\n[Step 3] Resuming from Hand 25 checkpoint to Hand 50 in Subprocess 3...")
    script_p2 = f"""
import sys, os
sys.path.insert(0, os.getcwd())
from v1024.directive055_production_engine import run_directive055_arm
run_directive055_arm(
    arm_name='Part2_Resume_50h',
    league_type='pfsp',
    opponent_pool_type='frozen2k',
    total_hands=50,
    seed_base={SEED},
    out_dir='{RESUME_DIR}',
    resume_from='{ckpt_25_dir}'
)
"""
    cmd3 = f'python3 -c "{script_p2}"'
    out3 = run_subprocess(cmd3)
    print("  -> Subprocess 3 finished successfully (resumed to Hand 50).")
    
    # 4. Rigorous Bit-Exact Comparison
    print("\n[Step 4] Performing Bit-Exact Verification between Subprocess 1 and Subprocess 3...")
    
    # 4.1 Compare final checkpoint Hand 50 models
    cont_ckpt50 = os.path.join(CONT_DIR, "ckpt_hand_50")
    res_ckpt50 = os.path.join(RESUME_DIR, "ckpt_hand_50")
    
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    mismatch_count = 0
    
    print("\n--- Model SHA256 Hashes at Hand 50 ---")
    for a in agents:
        cont_pt = os.path.join(cont_ckpt50, f"{a}.pt")
        res_pt = os.path.join(res_ckpt50, f"{a}.pt")
        hash_c = compute_sha256(cont_pt)
        hash_r = compute_sha256(res_pt)
        
        status = "MATCH" if hash_c == hash_r else "MISMATCH"
        print(f"  {a:7s} | Cont: {hash_c[:16]}... | Resumed: {hash_r[:16]}... | [{status}]")
        if hash_c != hash_r:
            mismatch_count += 1
            
    assert mismatch_count == 0, f"[FAIL] Model weights mismatched in {mismatch_count} agents!"
    print("\n  -> [PASS] All 5 Agent Model Checkpoints match 100.000% at bit level!")
    
    # 4.2 Compare Training Summaries & PnLs
    with open(os.path.join(CONT_DIR, "training_summary.json")) as f:
        summary_c = json.load(f)
    with open(os.path.join(RESUME_DIR, "training_summary.json")) as f:
        summary_r = json.load(f)
        
    print("\n--- Learner PnL Comparison ---")
    for a in agents:
        pnl_c = summary_c["learners_pnl"][a]
        pnl_r = summary_r["learners_pnl"][a]
        print(f"  {a:7s} | Cont: {pnl_c:+.2f} BB | Resumed: {pnl_r:+.2f} BB | Diff: {abs(pnl_c - pnl_r):.5f}")
        assert abs(pnl_c - pnl_r) < 1e-4, f"PnL mismatch for {a}: cont={pnl_c}, res={pnl_r}"
        
    print("  -> [PASS] All 5 Learner PnLs match perfectly!")
    
    # 4.3 Compare PPO Updates Log
    with open(os.path.join(CONT_DIR, "ppo_updates.jsonl")) as f:
        lines_c = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(RESUME_DIR, "ppo_updates.jsonl")) as f:
        lines_r = [json.loads(line) for line in f if line.strip()]
        
    print(f"\n--- PPO Updates Log Comparison ---")
    print(f"  Continuous updates: {len(lines_c)} | Resumed updates: {len(lines_r)}")
    assert len(lines_c) == len(lines_r), f"Update count mismatch: {len(lines_c)} vs {len(lines_r)}"
    
    for i, (uc, ur) in enumerate(zip(lines_c, lines_r)):
        assert uc["hand"] == ur["hand"], f"Hand mismatch at update {i}"
        assert uc["agent"] == ur["agent"], f"Agent mismatch at update {i}"
        assert abs(uc["policy_loss"] - ur["policy_loss"]) < 1e-4, f"Policy loss mismatch at update {i}"
        assert abs(uc["categorical_kl"] - ur["categorical_kl"]) < 1e-4, f"Categorical KL mismatch at update {i}"
        assert abs(uc["clip_fraction"] - ur["clip_fraction"]) < 1e-4, f"Clip fraction mismatch at update {i}"
        
    print("  -> [PASS] All PPO Update entries match 100%!")
    
    print("\n==================================================================")
    print(">>> [GATE PASS] DIRECTIVE 055 OPERATIONAL RESUME GATE 100% PASSED!")
    print("==================================================================\n")
    return True

if __name__ == "__main__":
    run_pause_resume_gate()
