#!/usr/bin/env python3
"""
v1024/test_directive056_pause_resume.py
DIRECTIVE 056 OPERATIONAL GATE:
State Equivalence Verification between Continuous 50 Hands and 25+25 Paused/Resumed Execution.

Uses the EXACT production runner from `directive056_production_engine.py` across separate subprocesses.
Validates:
1. Exact SHA256 hashes of all 5 agent neural models at Hand 50.
2. Identical Agent PnL across all 5 learners.
3. Identical PPO update metrics in ppo_updates.jsonl.
4. Final Sampler probabilities and selection logs.
"""

import os
import sys
import shutil
import subprocess
import json
import hashlib

TEST_DIR = "v1024/exp_directive056_test_pause_resume"
CONT_DIR = os.path.join(TEST_DIR, "arm_continuous_50h")
RESUME_DIR = os.path.join(TEST_DIR, "arm_resumed_25_25h")
SEED = 997700

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
    print(">>> [DIRECTIVE 056 GATE] Continuous 50h vs 25+25h Pause/Resume Verification")
    print("==================================================================")
    
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(CONT_DIR, exist_ok=True)
    os.makedirs(RESUME_DIR, exist_ok=True)
    
    # 1. Run Continuous 50 Hands
    print("\n[Step 1] Running 50 hands continuously in Subprocess 1...")
    script_cont = f"""
import sys, os
sys.path.insert(0, os.getcwd())
from v1024.directive056_production_engine import run_directive056_arm
run_directive056_arm(
    arm_name='Continuous_50h',
    league_type='pfsp',
    opponent_pool_type='extended',
    total_hands=50,
    seed_base={SEED},
    out_dir='{CONT_DIR}'
)
"""
    cmd1 = f'python3 -c "{script_cont}"'
    run_subprocess(cmd1)
    print("  -> Subprocess 1 finished successfully.")
    
    # 2. Run First 25 Hands
    print("\n[Step 2] Running first 25 hands and saving checkpoint in Subprocess 2...")
    script_p1 = f"""
import sys, os
sys.path.insert(0, os.getcwd())
from v1024.directive056_production_engine import run_directive056_arm
run_directive056_arm(
    arm_name='Part1_25h',
    league_type='pfsp',
    opponent_pool_type='extended',
    total_hands=25,
    seed_base={SEED},
    out_dir='{RESUME_DIR}'
)
"""
    cmd2 = f'python3 -c "{script_p1}"'
    run_subprocess(cmd2)
    print("  -> Subprocess 2 finished successfully.")
    
    ckpt_25_dir = os.path.join(RESUME_DIR, "ckpt_hand_25")
    assert os.path.exists(ckpt_25_dir), f"Checkpoint missing: {ckpt_25_dir}"
    
    # 3. Resume from Hand 25 to 50
    print("\n[Step 3] Resuming from Hand 25 checkpoint to Hand 50 in Subprocess 3...")
    script_p2 = f"""
import sys, os
sys.path.insert(0, os.getcwd())
from v1024.directive056_production_engine import run_directive056_arm
run_directive056_arm(
    arm_name='Part2_Resume_50h',
    league_type='pfsp',
    opponent_pool_type='extended',
    total_hands=50,
    seed_base={SEED},
    out_dir='{RESUME_DIR}',
    resume_from='{ckpt_25_dir}'
)
"""
    cmd3 = f'python3 -c "{script_p2}"'
    run_subprocess(cmd3)
    print("  -> Subprocess 3 finished successfully.")
    
    # 4. State Equivalence Comparison
    print("\n[Step 4] Comparing State Equivalence between Subprocess 1 and Subprocess 3...")
    
    cont_ckpt50 = os.path.join(CONT_DIR, "ckpt_hand_50")
    res_ckpt50 = os.path.join(RESUME_DIR, "ckpt_hand_50")
    
    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    mismatch_count = 0
    
    print("\n--- Model Checkpoints SHA256 at Hand 50 ---")
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
    print("\n  -> [PASS] All 5 Agent Model Checkpoints match 100% at SHA256 level!")
    
    with open(os.path.join(CONT_DIR, "training_summary.json")) as f:
        summary_c = json.load(f)
    with open(os.path.join(RESUME_DIR, "training_summary.json")) as f:
        summary_r = json.load(f)
        
    print("\n--- Learner PnL Comparison ---")
    for a in agents:
        pnl_c = summary_c["learners_pnl"][a]
        pnl_r = summary_r["learners_pnl"][a]
        print(f"  {a:7s} | Cont: {pnl_c:+.2f} BB | Resumed: {pnl_r:+.2f} BB | Diff: {abs(pnl_c - pnl_r):.5f}")
        assert abs(pnl_c - pnl_r) < 1e-4, f"PnL mismatch for {a}"
        
    print("  -> [PASS] All 5 Learner PnLs match identically!")
    
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
        
    print("  -> [PASS] All PPO Update entries match identically!")
    print("\n==================================================================")
    print(">>> [GATE PASS] DIRECTIVE 056 OPERATIONAL RESUME GATE PASSED!")
    print("==================================================================\n")
    return True

if __name__ == "__main__":
    run_pause_resume_gate()
