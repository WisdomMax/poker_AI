#!/usr/bin/env python3
"""
v1024/evaluate_directive052r_matrix.py
Evaluates Flat Reward vs Corrected GAE across Uniform and PFSP on 200 independent holdout seeds.
50 Seeds x 5 Heroes x 5 Seats = 1,250 hands per arm x 4 arms = 5,000 hands total.
"""

import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer

BIG_BLIND = 100.0
AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
CHECKPOINT_DIR = "v1024/models_alphago"

def load_strict_model(model_path: str) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[FAIL_FAST] Required model path does not exist: {model_path}")
    m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m.eval()
    return m

def evaluate_052r_matrix():
    print("==================================================================")
    print("STARTING DIRECTIVE 052-R INDEPENDENT MATRIX EVALUATION")
    print("Flat vs Corrected GAE (50 Seeds x 5 Heroes x 5 Seats = 1,250h / arm)")
    print("==================================================================")
    
    game = get_5p_fullgame()
    frozen_pool = [f"{a}_2000h" for a in AGENT_NAMES]
    frozen_models = {c: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{c}.pt")) for c in frozen_pool}
    
    base_dir = "v1024/exp_directive052r"
    model_groups = {
        "Baseline_7000h": {a: load_strict_model(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES},
        "Uniform_Flat": {a: load_strict_model(os.path.join(base_dir, f"arm_uniform_flat_seed_401000/{a}.pt")) for a in AGENT_NAMES},
        "Uniform_GAE": {a: load_strict_model(os.path.join(base_dir, f"arm_uniform_gae_seed_401000/{a}.pt")) for a in AGENT_NAMES},
        "PFSP_Flat": {a: load_strict_model(os.path.join(base_dir, f"arm_pfsp_flat_seed_401000/{a}.pt")) for a in AGENT_NAMES},
        "PFSP_GAE": {a: load_strict_model(os.path.join(base_dir, f"arm_pfsp_gae_seed_401000/{a}.pt")) for a in AGENT_NAMES}
    }
    
    NUM_SEEDS = 50
    SEED_BASE = 666000
    results = {}
    raw_logs = []
    seed_means_dict = {}
    
    t0 = time.time()
    for grp_name, hero_models in model_groups.items():
        hands_pnl = []
        seed_pnl = defaultdict(list)
        allin_count = 0
        total_decisions = 0
        
        for s_idx in range(NUM_SEEDS):
            seed = SEED_BASE + s_idx
            for hero_name in AGENT_NAMES:
                hero_m = hero_models[hero_name]
                for seat in range(5):
                    rng = np.random.RandomState(seed)
                    st = game.new_initial_state()
                    advance_chance_nodes(st, rng)
                    
                    table_opps = [frozen_pool[(seat + i + 1) % len(frozen_pool)] for i in range(4)]
                    table = []
                    opp_iter = 0
                    for pos in range(5):
                        if pos == seat:
                            table.append(("HERO", hero_name))
                        else:
                            table.append(("OPP", table_opps[opp_iter]))
                            opp_iter += 1
                            
                    while not st.is_terminal():
                        if st.is_chance_node(): advance_chance_nodes(st, rng); continue
                        cp = st.current_player()
                        role, ag_id = table[cp]
                        leg = True6ActionAdapter.get_legal_6actions(st, cp)
                        norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                        m = hero_m if role == "HERO" else frozen_models[ag_id]
                        with torch.no_grad():
                            lgt, _ = m(torch.from_numpy(norm_inf).unsqueeze(0))
                            msk = torch.full_like(lgt, -1e9)
                            for a in leg: msk[0, a] = 0.0
                            act = int(torch.argmax(lgt + msk, dim=-1).item())
                        if role == "HERO":
                            total_decisions += 1
                            if act == 5: allin_count += 1
                        st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
                        
                    ret = st.returns()[seat] / BIG_BLIND
                    hands_pnl.append(ret)
                    seed_pnl[seed].append(ret)
                    raw_logs.append({"group": grp_name, "seed": seed, "hero": hero_name, "seat": seat, "pnl": float(ret)})
                    
        pnl_arr = np.array(hands_pnl)
        seed_means = np.array([np.mean(seed_pnl[s]) * 100.0 for s in range(SEED_BASE, SEED_BASE + NUM_SEEDS)])
        seed_means_dict[grp_name] = seed_means
        sem = seed_means.std() / np.sqrt(NUM_SEEDS)
        ci95 = [float(seed_means.mean() - 1.96 * sem), float(seed_means.mean() + 1.96 * sem)]
        
        results[grp_name] = {
            "mean_bb100": float(pnl_arr.mean() * 100.0),
            "seed_ci95": ci95,
            "allin_rate_pct": float((allin_count / max(1, total_decisions)) * 100.0)
        }
        print(f"  -> {grp_name}: {results[grp_name]['mean_bb100']:+.2f} BB/100 (95% CI: [{ci95[0]:.2f}, {ci95[1]:.2f}]), All-in: {results[grp_name]['allin_rate_pct']:.2f}%")

    # Comparisons
    diff_u_gae_vs_flat = seed_means_dict["Uniform_GAE"] - seed_means_dict["Uniform_Flat"]
    diff_p_gae_vs_flat = seed_means_dict["PFSP_GAE"] - seed_means_dict["PFSP_Flat"]
    diff_pfsp_vs_uni_gae = seed_means_dict["PFSP_GAE"] - seed_means_dict["Uniform_GAE"]
    
    comparisons = {
        "Uniform_GAE_minus_Flat": {
            "mean_diff": float(diff_u_gae_vs_flat.mean()),
            "ci95": [float(diff_u_gae_vs_flat.mean() - 1.96 * diff_u_gae_vs_flat.std() / np.sqrt(NUM_SEEDS)),
                     float(diff_u_gae_vs_flat.mean() + 1.96 * diff_u_gae_vs_flat.std() / np.sqrt(NUM_SEEDS))]
        },
        "PFSP_GAE_minus_Flat": {
            "mean_diff": float(diff_p_gae_vs_flat.mean()),
            "ci95": [float(diff_p_gae_vs_flat.mean() - 1.96 * diff_p_gae_vs_flat.std() / np.sqrt(NUM_SEEDS)),
                     float(diff_p_gae_vs_flat.mean() + 1.96 * diff_p_gae_vs_flat.std() / np.sqrt(NUM_SEEDS))]
        },
        "PFSP_vs_Uniform_under_GAE": {
            "mean_diff": float(diff_pfsp_vs_uni_gae.mean()),
            "ci95": [float(diff_pfsp_vs_uni_gae.mean() - 1.96 * diff_pfsp_vs_uni_gae.std() / np.sqrt(NUM_SEEDS)),
                     float(diff_pfsp_vs_uni_gae.mean() + 1.96 * diff_pfsp_vs_uni_gae.std() / np.sqrt(NUM_SEEDS))]
        }
    }
    
    print("\n>>> [TASK 5 KEY VERDICTS]")
    print(f"  Uniform GAE vs Flat: {comparisons['Uniform_GAE_minus_Flat']['mean_diff']:+.2f} BB/100 (95% CI: {comparisons['Uniform_GAE_minus_Flat']['ci95']})")
    print(f"  PFSP GAE vs Flat: {comparisons['PFSP_GAE_minus_Flat']['mean_diff']:+.2f} BB/100 (95% CI: {comparisons['PFSP_GAE_minus_Flat']['ci95']})")
    print(f"  PFSP vs Uniform (under Corrected GAE): {comparisons['PFSP_vs_Uniform_under_GAE']['mean_diff']:+.2f} BB/100 (95% CI: {comparisons['PFSP_vs_Uniform_under_GAE']['ci95']})")

    out_file = "v1024/directive052r_task5_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "comparisons": comparisons}, f, indent=2)
    print(f"\n[DONE] Saved Task 5 Benchmark to {out_file}")

if __name__ == "__main__":
    evaluate_052r_matrix()
