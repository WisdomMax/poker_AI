#!/usr/bin/env python3
"""
DIRECTIVE 044: Checkpoint Integrity & Weight Distance Audit
- Computes SHA256 of all 25 checkpoints
- Computes parameter step distances: ||theta_{t} - theta_{t-1}||
- Computes inter-agent parameter distances (Agent A vs B vs C vs D vs E)
- Verifies model parameter divergence and independence
"""
import glob
import hashlib
import json
import os
import torch
import numpy as np

CHECKPOINT_DIR = "v1024/models_alphago"
BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_param_vector(state_dict):
    tensors = []
    for k in sorted(state_dict.keys()):
        # exclude optimizer states if any, take model weights
        t = state_dict[k]
        if isinstance(t, torch.Tensor):
            tensors.append(t.detach().cpu().flatten())
    return torch.cat(tensors).numpy()

def audit_checkpoints():
    print("=== DIRECTIVE 044: CHECKPOINT INTEGRITY & WEIGHT DISTANCE AUDIT ===")
    
    # 1. Base 50k
    assert os.path.exists(BASE_50K_PATH), f"Missing base 50k: {BASE_50K_PATH}"
    base_sha = compute_sha256(BASE_50K_PATH)
    base_state = torch.load(BASE_50K_PATH, map_location="cpu", weights_only=False)
    base_vec = get_param_vector(base_state["model_state_dict"] if "model_state_dict" in base_state else base_state)
    print(f"Base 50k SHA256: {base_sha} | Total params: {len(base_vec):,}")

    agents = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    milestones = ["200h", "500h", "1000h", "1500h", "2000h"]

    checkpoint_info = {}
    agent_vectors = {a: {} for a in agents}

    for a in agents:
        for m in milestones:
            fname = f"{a}_{m}.pt"
            fpath = os.path.join(CHECKPOINT_DIR, fname)
            assert os.path.exists(fpath), f"Missing checkpoint: {fpath}"
            sha = compute_sha256(fpath)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            checkpoint_info[fname] = {
                "agent": a,
                "milestone": m,
                "path": fpath,
                "sha256": sha,
                "size_mb": round(size_mb, 2)
            }
            ckpt_data = torch.load(fpath, map_location="cpu", weights_only=False)
            state = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
            vec = get_param_vector(state)
            agent_vectors[a][m] = vec

    print(f"Verified all 25 milestone checkpoints! (5 agents x 5 milestones)")

    # 2. Step distances across training: ||theta_{t} - theta_{t-1}||
    print("\n--- Parameter Evolution (Distance from Base 50k & Step Evolution) ---")
    step_evolution = {}
    for a in agents:
        step_evolution[a] = {}
        prev_vec = base_vec
        dist_from_base = np.linalg.norm(agent_vectors[a]["2000h"] - base_vec)
        cos_sim_base = np.dot(agent_vectors[a]["2000h"], base_vec) / (np.linalg.norm(agent_vectors[a]["2000h"]) * np.linalg.norm(base_vec))
        
        step_diffs = []
        for m in milestones:
            curr_vec = agent_vectors[a][m]
            step_dist = float(np.linalg.norm(curr_vec - prev_vec))
            step_diffs.append({"milestone": m, "delta_from_prev": round(step_dist, 4)})
            prev_vec = curr_vec
            
        step_evolution[a] = {
            "total_dist_from_50k": round(float(dist_from_base), 4),
            "final_cos_sim_with_50k": round(float(cos_sim_base), 5),
            "step_deltas": step_diffs
        }
        print(f"{a:8s} | Total delta from 50k: {dist_from_base:6.4f} | Cos Sim: {cos_sim_base:7.5f} | Step deltas: {[s['delta_from_prev'] for s in step_diffs]}")

    # 3. Inter-agent distances at 2000h (Proving independence)
    print("\n--- Inter-Agent Independence at 2000h (Pairwise Euclidean Distance & Cosine Sim) ---")
    inter_agent = {}
    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            if i < j:
                v1 = agent_vectors[a1]["2000h"]
                v2 = agent_vectors[a2]["2000h"]
                dist = float(np.linalg.norm(v1 - v2))
                sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                pair_key = f"{a1}_vs_{a2}"
                inter_agent[pair_key] = {
                    "euclidean_distance": round(dist, 4),
                    "cosine_similarity": round(sim, 5)
                }
                print(f"{pair_key:22s} | Distance: {dist:6.4f} | Cosine Sim: {sim:7.5f}")

    results = {
        "base_50k_sha256": base_sha,
        "total_checkpoints": len(checkpoint_info),
        "checkpoint_catalog": checkpoint_info,
        "step_evolution": step_evolution,
        "inter_agent_at_2000h": inter_agent
    }

    out_json = "v1024/directive044_checkpoint_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit results successfully saved to {out_json}")

if __name__ == "__main__":
    audit_checkpoints()
