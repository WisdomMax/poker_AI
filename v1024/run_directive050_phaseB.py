#!/usr/bin/env python3
"""
v1024/run_directive050_phaseB.py
Executes DIRECTIVE 050 Phase B:
1. Full Main Self-Play Pilot (500 hands)
2. PFSP Mixed Historical League Pilot (500 hands)
3. Independent Holdout Evaluation (200 hands on fixed evaluation pool)
4. Saves full checkpoints and outputs directive050_pilot_results.json
"""

import os
import sys
import json
import time
import hashlib
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import scipy.stats as stats

sys.path.insert(0, ".")

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, normalize_inf_state_v79
)
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.audit_directive049c_final_gate import DynamicAlphaStarPFSPSampler
from v1024.train_directive050_integrated_pilot import IndependentPPOAgent, compute_sha256

CHECKPOINT_DIR = "v1024/models_alphago"
BIG_BLIND = 100.0
AGENT_NAMES = ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]

def train_full_main_pilot(num_hands: int = 500, seed_base: int = 100000) -> Dict[str, Any]:
    print(f"\n>>> [Phase B1] Training Full Main Self-Play ({num_hands} hands)...")
    game = get_5p_fullgame()
    learners = {a: IndependentPPOAgent(a, os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
    
    decisions = defaultdict(int)
    pnl = defaultdict(float)
    
    for h_i in range(1, num_hands + 1):
        seed = seed_base + h_i
        rng = np.random.RandomState(seed)
        
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
                continue
            cp = st.current_player()
            agent_name = AGENT_NAMES[cp]
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
            
            act, _ = learners[agent_name].select_action(norm_inf, legal_6)
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            decisions[agent_name] += 1
            
        returns = [st.returns()[p] / BIG_BLIND for p in range(5)]
        for p in range(5):
            a_name = AGENT_NAMES[p]
            learners[a_name].record_hand_reward(returns[p])
            pnl[a_name] += returns[p]
            
        if h_i % 25 == 0:
            for a_name in AGENT_NAMES:
                learners[a_name].ppo_update()
                
    # Save Full Main checkpoints
    os.makedirs(os.path.join(CHECKPOINT_DIR, "pilot_full_main"), exist_ok=True)
    for a in AGENT_NAMES:
        torch.save(learners[a].model.state_dict(), os.path.join(CHECKPOINT_DIR, "pilot_full_main", f"{a}_pilot.pt"))
        
    return {
        "num_hands": num_hands,
        "decisions": dict(decisions),
        "pnl_bb": dict(pnl),
        "grad_steps": {a: learners[a].gradient_steps for a in AGENT_NAMES}
    }

def train_pfsp_league_pilot(num_hands: int = 500, seed_base: int = 200000) -> Dict[str, Any]:
    print(f"\n>>> [Phase B2] Training PFSP Mixed Historical League ({num_hands} hands)...")
    game = get_5p_fullgame()
    learners = {a: IndependentPPOAgent(a, os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt")) for a in AGENT_NAMES}
    
    frozen_pool = [f"{a}_2000h" for a in AGENT_NAMES]
    frozen_models = {}
    for c in frozen_pool:
        m = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{c}.pt"), map_location="cpu", weights_only=True)
        m.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        m.eval()
        frozen_models[c] = m
        
    sampler = DynamicAlphaStarPFSPSampler(frozen_pool, temperature=10.0, exploration_floor=0.1)
    
    participation = defaultdict(int)
    decisions = defaultdict(int)
    pnl = defaultdict(float)
    
    for h_i in range(1, num_hands + 1):
        seed = seed_base + h_i
        rng = np.random.RandomState(seed)
        
        # 3 Main Learners rotated + 2 PFSP Frozen opponents
        perm_learners = [AGENT_NAMES[(h_i + i) % 5] for i in range(3)]
        opp1, _ = sampler.sample_opponent(rng, hand_id=h_i)
        opp2, _ = sampler.sample_opponent(rng, hand_id=h_i)
        table = perm_learners + [opp1, opp2]
        
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)
        
        while not st.is_terminal():
            if st.is_chance_node():
                advance_chance_nodes(st, rng)
                continue
            cp = st.current_player()
            ag_id = table[cp]
            legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
            norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
            
            if cp < 3:
                act, _ = learners[ag_id].select_action(norm_inf, legal_6)
                decisions[ag_id] += 1
            else:
                with torch.no_grad():
                    logits, _ = frozen_models[ag_id](torch.from_numpy(norm_inf).unsqueeze(0))
                    mask = torch.full_like(logits, -1e9)
                    for a in legal_6: mask[0, a] = 0.0
                    act = int(torch.argmax(logits + mask, dim=-1).item())
                    
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            
        returns = [st.returns()[p] / BIG_BLIND for p in range(5)]
        for p in range(3):
            a_name = table[p]
            learners[a_name].record_hand_reward(returns[p])
            pnl[a_name] += returns[p]
            participation[a_name] += 1
            
        hero_net1 = returns[0] - returns[3]
        hero_net2 = returns[0] - returns[4]
        sampler.record_match(opp1, hero_net1)
        sampler.record_match(opp2, hero_net2)
        
        if h_i % 25 == 0:
            for a_name in AGENT_NAMES:
                learners[a_name].ppo_update()
                
    # Save PFSP League checkpoints
    os.makedirs(os.path.join(CHECKPOINT_DIR, "pilot_pfsp_league"), exist_ok=True)
    for a in AGENT_NAMES:
        torch.save(learners[a].model.state_dict(), os.path.join(CHECKPOINT_DIR, "pilot_pfsp_league", f"{a}_pilot.pt"))
        
    return {
        "num_hands": num_hands,
        "participation": dict(participation),
        "decisions": dict(decisions),
        "pnl_bb": dict(pnl),
        "grad_steps": {a: learners[a].gradient_steps for a in AGENT_NAMES}
    }

def run_holdout_benchmark(eval_hands: int = 200, eval_seed: int = 300000) -> Dict[str, Any]:
    print(f"\n>>> [Phase B3] Running Independent Holdout Benchmark ({eval_hands} hands)...")
    game = get_5p_fullgame()
    
    # Evaluate 3 groups: Baseline 7000h vs Full Main Pilot vs PFSP League Pilot
    models_7k = {}
    models_main = {}
    models_pfsp = {}
    
    for a in AGENT_NAMES:
        m7 = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        m7.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"{a}_7000h.pt"), map_location="cpu", weights_only=True))
        m7.eval()
        models_7k[a] = m7
        
        mm = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        mm.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "pilot_full_main", f"{a}_pilot.pt"), map_location="cpu", weights_only=True))
        mm.eval()
        models_main[a] = mm
        
        mp = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256)
        mp.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "pilot_pfsp_league", f"{a}_pilot.pt"), map_location="cpu", weights_only=True))
        mp.eval()
        models_pfsp[a] = mp
        
    # Evaluate in 5-player round-robin
    pnl_7k_total = 0.0
    pnl_main_total = 0.0
    pnl_pfsp_total = 0.0
    
    # 200 hands evaluated
    for h in range(1, eval_hands + 1):
        rng = np.random.RandomState(eval_seed + h)
        # Symmetrical evaluation
        for grp_name, grp_models in [("7k", models_7k), ("main", models_main), ("pfsp", models_pfsp)]:
            st = game.new_initial_state()
            advance_chance_nodes(st, rng)
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng); continue
                cp = st.current_player()
                a_name = AGENT_NAMES[cp]
                leg = True6ActionAdapter.get_legal_6actions(st, cp)
                norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                with torch.no_grad():
                    logits, _ = grp_models[a_name](torch.from_numpy(norm_inf).unsqueeze(0))
                    mask = torch.full_like(logits, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    act = int(torch.argmax(logits + mask, dim=-1).item())
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            rets = [st.returns()[p] / BIG_BLIND for p in range(5)]
            if grp_name == "7k": pnl_7k_total += sum(rets)
            elif grp_name == "main": pnl_main_total += sum(rets)
            elif grp_name == "pfsp": pnl_pfsp_total += sum(rets)
            
    print("  Holdout Benchmark Complete:")
    print(f"    Baseline 7000h Total PnL: {pnl_7k_total:+.2f} BB")
    print(f"    Full Main Pilot Total PnL: {pnl_main_total:+.2f} BB")
    print(f"    PFSP League Pilot Total PnL: {pnl_pfsp_total:+.2f} BB")
    
    return {
        "eval_hands": eval_hands,
        "baseline_7k_bb": pnl_7k_total,
        "pilot_full_main_bb": pnl_main_total,
        "pilot_pfsp_league_bb": pnl_pfsp_total,
        "scientific_verdict": "PERFORMANCE_NOT_YET_PROVEN"
    }

def main():
    print("================================================================================")
    print("DIRECTIVE 050: PHASE B (1,000-HAND PILOT LEAGUE TRAINING & HOLDOUT BENCHMARK)")
    print("================================================================================")
    t0 = time.time()
    
    res_main = train_full_main_pilot(num_hands=500)
    res_pfsp = train_pfsp_league_pilot(num_hands=500)
    res_eval = run_holdout_benchmark(eval_hands=200)
    
    elapsed = time.time() - t0
    
    results = {
        "directive": "050-PhaseB",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(elapsed, 2),
        "full_main_pilot": res_main,
        "pfsp_league_pilot": res_pfsp,
        "holdout_benchmark": res_eval
    }
    
    out_path = "v1024/directive050_pilot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nPhase B Results saved to {out_path} ({elapsed:.1f}s)")

if __name__ == "__main__":
    main()
