#!/usr/bin/env python3
"""
Generate complete SHA256 manifest for all 160 models and checkpoints in exp_directive057.
"""
import os
import hashlib
import json
import torch

BASE_DIR = "v1024/exp_directive057"
manifest = {
    "generated_at": "2026-10-29T01:43:00+09:00",
    "base_dir": BASE_DIR,
    "total_sessions": 8,
    "sessions": {}
}

total_pt_files = 0
total_bytes = 0

for session in sorted(os.listdir(BASE_DIR)):
    s_path = os.path.join(BASE_DIR, session)
    if not os.path.isdir(s_path):
        continue
    manifest["sessions"][session] = {
        "checkpoints": {},
        "logs": {}
    }
    
    # Check log files
    for log_name in ["cpu_metrics.jsonl", "league_selection_history.jsonl", "ppo_updates.jsonl", "training_summary.json"]:
        lp = os.path.join(s_path, log_name)
        if os.path.exists(lp):
            with open(lp, "rb") as f:
                data = f.read()
            manifest["sessions"][session]["logs"][log_name] = {
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()
            }
            
    # Check checkpoint directories
    for ckpt_dir in sorted(os.listdir(s_path)):
        cp = os.path.join(s_path, ckpt_dir)
        if not os.path.isdir(cp):
            continue
        models = {}
        for fname in sorted(os.listdir(cp)):
            if fname.endswith(".pt"):
                fp = os.path.join(cp, fname)
                size = os.path.getsize(fp)
                with open(fp, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
                models[fname] = {
                    "size_bytes": size,
                    "sha256": sha
                }
                total_pt_files += 1
                total_bytes += size
        if models:
            manifest["sessions"][session]["checkpoints"][ckpt_dir] = models

manifest["summary"] = {
    "total_pt_models": total_pt_files,
    "total_bytes": total_bytes,
    "total_gigabytes": round(total_bytes / (1024**3), 3)
}

out_path = os.path.join(BASE_DIR, "checkpoint_manifest_sha256.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"[+] Successfully generated manifest with {total_pt_files} models ({round(total_bytes/(1024**3), 3)} GB) -> {out_path}")
