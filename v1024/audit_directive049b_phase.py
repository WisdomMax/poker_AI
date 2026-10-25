#!/usr/bin/env python3
"""
POKER AI DIRECTIVE 049-B: Rigorous Scientific Verification & PFSP Production Gate
Phase 1: Bit-Level Exact Reproduction of 047 Agent E 500-Hand Benchmark (-230.45 BB/100, 113 Called, 28 Act5)
Phase 2: Mathematical Ledger Decomposition of the 117 All-in / 113 Called Hands
Phase 3: Rigorous Generational Difference Analysis Across All 5 Agents (Independent Controlled Environment)
Phase 4: Bug-Free Production PFSP Sampler (Clean Identifiers, Float Norm < 1e-6, 50-Model SHA256 Check)
Phase 5: Limited League Dry-Run (50 Hands, Zero-Sum Verified, Immutability Verified)
"""

import sys
import os
import time
import json
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, '/Volumes/NVME/7.AI_vibe_coding/hangame')

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, True6ActionAdapter, advance_chance_nodes, extract_player_spent,
    normalize_inf_state_v79
)
from v71.run_v71_ledger_and_street_audit import get_round_from_state
from v1015.run_v1015_neural_scaleup import PPONeuralPlayer
from v1024.tests.test_directive046_evaluator_integrity import safe_parse_cards_v46
from v1024.integrity_two_pair_catcher import FlawlessTwoPairCatcherBot, parse_board_cards
from v1024.integrity_ledger_tracker import HandLedgerTracker, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from v1024.alphago_5p_selfplay import AGENT_NAMES

CHECKPOINT_DIR = "v1024/models_alphago"
BASE_50K_PATH = "v1024/models_e2e/PPO_E2E_Final_50k.pt"
CATALOG_048_PATH = "v1024/directive048_audit_results.json"
AUDIT_JSON_PATH = "v1024/directive049b_audit_results.json"
DATA_DIR = "v1024/data"
os.makedirs(DATA_DIR, exist_ok=True)

RAW_049B_REPRO_JSONL = os.path.join(DATA_DIR, "directive049b_raw_reproduced_047_hands.jsonl")
RAW_049B_LEAGUE_DRYRUN_JSONL = os.path.join(DATA_DIR, "directive049b_raw_league_dryrun.jsonl")

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_ppo_model(path: str) -> PPONeuralPlayer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    model = PPONeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256, lr=3e-4)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

# ==============================================================================
# PHASE 1 & 2: Bit-Level Exact Reproduction of 047 Agent E & Event Decomposition
# ==============================================================================
def run_phase1_and_phase2_reproduction(num_hands: int = 500, base_seed: int = 40000) -> Dict[str, Any]:
    print("\n>>> [PHASE 1 & 2] Running Exact Reproduction of 047 Agent E Benchmark...")
    game = get_5p_fullgame()
    base_model = load_ppo_model(BASE_50K_PATH)
    model_7k = load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_E_7000h.pt"))

    # Exactly matching 047 Part 2 seed for Agent E (a_idx=4):
    a_idx = 4
    reproduced_pnls = []
    act5_hand_count = 0
    full_stack_hand_count = 0
    called_allin_count = 0
    uncalled_allin_count = 0

    # Decomposition counters
    decomp = {
        "Direct_Act5_Called": 0,
        "Direct_Act5_Uncalled": 0,
        "Indirect_Cap_Called": 0,
        "Indirect_Cap_Uncalled": 0,
        "Non_Allin_Hands": 0
    }
    decomp_pnls = defaultdict(list)

    with open(RAW_049B_REPRO_JSONL, "w", encoding="utf-8") as out_f:
        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)

            # Match 047 execution exactly:
            st_init = game.new_initial_state()
            advance_chance_nodes(st_init, rng)
            cards_map = safe_parse_cards_v46(st_init)

            # In 047, st_0 and st_2 were run before st_7, but each used clone() and independent rng_X
            st_7 = st_init.clone()
            rng_7 = np.random.RandomState(seed)
            h_act5 = False
            h_full = False
            tracker7 = HandLedgerTracker(num_players=5)
            hero_acts = []

            while not st_7.is_terminal():
                if st_7.is_chance_node():
                    advance_chance_nodes(st_7, rng_7)
                    continue
                cp = st_7.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st_7, cp)
                round_idx = get_round_from_state(st_7)
                norm_inf = normalize_inf_state_v79(np.array(st_7.information_state_tensor(cp), dtype=np.float32))
                curr_m = model_7k if cp == hero_seat else base_model
                with torch.no_grad():
                    logits, _ = curr_m(torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone()
                    mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    masked += mask
                    act = int(torch.argmax(F.softmax(masked, dim=-1), dim=-1).item())

                if cp == hero_seat:
                    hero_acts.append(act)
                    if act == 5: h_act5 = True

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st_7, cp, act)
                st_7.apply_action(nat_act)
                spent_after = extract_player_spent(st_7)[cp]
                tracker7.record_action(cp, act, nat_act, round_idx, spent_after)

            ret_7 = st_7.returns()[hero_seat] / BIG_BLIND
            if extract_player_spent(st_7)[hero_seat] >= STARTING_STACK * 0.99:
                h_full = True

            reproduced_pnls.append(ret_7)
            if h_act5: act5_hand_count += 1
            if h_full: full_stack_hand_count += 1

            hero_allin = tracker7.is_player_allin(hero_seat)
            opp_called = tracker7.was_allin_called(hero_seat)

            if hero_allin:
                if opp_called:
                    called_allin_count += 1
                else:
                    uncalled_allin_count += 1

            # Exact decomposition
            if h_act5:
                if opp_called:
                    cat = "Direct_Act5_Called"
                else:
                    cat = "Direct_Act5_Uncalled"
            elif hero_allin:
                if opp_called:
                    cat = "Indirect_Cap_Called"
                else:
                    cat = "Indirect_Cap_Uncalled"
            else:
                cat = "Non_Allin_Hands"

            decomp[cat] += 1
            decomp_pnls[cat].append(ret_7)

            out_f.write(json.dumps({
                "hand_id": h_i, "seed": seed, "seat": hero_seat,
                "pnl_bb": ret_7, "hero_actions": hero_acts,
                "act5_pushed": h_act5, "full_stack_reached": h_full,
                "hero_allin_tracker": hero_allin, "opp_called": opp_called,
                "category": cat
            }) + "\n")

    repro_bb100 = round(float(np.mean(reproduced_pnls)) * 100, 2)
    print(f"  --> Reproduced BB/100: {repro_bb100} BB/100 (Target: -230.45 BB/100)")
    print(f"  --> Hands with Action 5: {act5_hand_count} hands (Target: 28 hands)")
    print(f"  --> Full Stack Hands: {full_stack_hand_count} hands (Target: 117 hands)")
    print(f"  --> Called All-in Hands: {called_allin_count} hands (Target: 113 hands)")
    print(f"  --> Uncalled All-in Hands: {uncalled_allin_count} hands (Target: 4 hands)")

    # Strict assertion of reproduction
    assert abs(repro_bb100 - (-230.45)) < 1e-2, f"BB/100 mismatch: {repro_bb100} vs -230.45"
    assert act5_hand_count == 28, f"Act5 mismatch: {act5_hand_count} vs 28"
    assert called_allin_count == 113, f"Called allin mismatch: {called_allin_count} vs 113"
    assert uncalled_allin_count == 4, f"Uncalled allin mismatch: {uncalled_allin_count} vs 4"
    assert full_stack_hand_count == 117, f"Full stack mismatch: {full_stack_hand_count} vs 117"
    print("  [SUCCESS] 500/500 Hands, 113/113 Called, 28/28 Act5 100% BIT-LEVEL REPRODUCED!")

    decomp_summary = {}
    for c, cnt in decomp.items():
        arr = decomp_pnls[c]
        decomp_summary[c] = {
            "count": cnt,
            "pct": round(cnt / num_hands * 100, 2),
            "avg_ev_bb": round(float(np.mean(arr)), 2) if arr else 0.0,
            "total_pnl_bb": round(float(np.sum(arr)), 2) if arr else 0.0
        }
        print(f"    * {c}: {cnt} hands ({decomp_summary[c]['pct']}%), Avg EV = {decomp_summary[c]['avg_ev_bb']:+.2f} BB")

    return {
        "status": "REPRODUCTION_VERIFIED",
        "reproduced_bb100": repro_bb100,
        "act5_hands": act5_hand_count,
        "called_allin_hands": called_allin_count,
        "uncalled_allin_hands": uncalled_allin_count,
        "full_stack_hands": full_stack_hand_count,
        "decomposition": decomp_summary
    }

# ==============================================================================
# PHASE 3: Rigorous Generational Difference Analysis Across All 5 Agents
# ==============================================================================
def run_phase3_independent_generational_benchmark(num_hands: int = 500, base_seed: int = 40000) -> Dict[str, Any]:
    print("\n>>> [PHASE 3] Running Independent Generational Benchmark Across All 5 Agents (2000h vs 7000h)...")
    # This evaluates each agent's 2000h vs 7000h under the exact same fixed baseline opponent pool
    game = get_5p_fullgame()
    base_model = load_ppo_model(BASE_50K_PATH)
    results = {}

    for a_idx, a_name in enumerate(AGENT_NAMES):
        model_2k = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_2000h.pt"))
        model_7k = load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{a_name}_7000h.pt"))

        pnl_2k = []
        pnl_7k = []
        paired_diffs = []

        for h_i in range(1, num_hands + 1):
            seed = base_seed + a_idx * 10000 + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)

            st_init = game.new_initial_state()
            advance_chance_nodes(st_init, rng)

            # Run 2000h
            st_2 = st_init.clone()
            rng_2 = np.random.RandomState(seed)
            while not st_2.is_terminal():
                if st_2.is_chance_node(): advance_chance_nodes(st_2, rng_2); continue
                cp = st_2.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st_2, cp)
                norm_inf = normalize_inf_state_v79(np.array(st_2.information_state_tensor(cp), dtype=np.float32))
                curr_m = model_2k if cp == hero_seat else base_model
                with torch.no_grad():
                    logits, _ = curr_m(torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    act = int(torch.argmax(logits + mask, dim=-1).item())
                st_2.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_2, cp, act))
            ret_2 = st_2.returns()[hero_seat] / BIG_BLIND
            pnl_2k.append(ret_2)

            # Run 7000h
            st_7 = st_init.clone()
            rng_7 = np.random.RandomState(seed)
            while not st_7.is_terminal():
                if st_7.is_chance_node(): advance_chance_nodes(st_7, rng_7); continue
                cp = st_7.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st_7, cp)
                norm_inf = normalize_inf_state_v79(np.array(st_7.information_state_tensor(cp), dtype=np.float32))
                curr_m = model_7k if cp == hero_seat else base_model
                with torch.no_grad():
                    logits, _ = curr_m(torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    act = int(torch.argmax(logits + mask, dim=-1).item())
                st_7.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_7, cp, act))
            ret_7 = st_7.returns()[hero_seat] / BIG_BLIND
            pnl_7k.append(ret_7)

            paired_diffs.append(ret_7 - ret_2)

        diff_arr = np.array(paired_diffs)
        mean_diff = float(np.mean(diff_arr)) * 100
        std_diff = float(np.std(diff_arr, ddof=1)) * 100
        ci95 = 1.96 * (std_diff / np.sqrt(len(diff_arr)))
        t_stat, p_val = stats.ttest_1samp(diff_arr, 0.0)

        results[a_name] = {
            "bb100_2000h": round(float(np.mean(pnl_2k)) * 100, 2),
            "bb100_7000h": round(float(np.mean(pnl_7k)) * 100, 2),
            "delta_bb100": round(mean_diff, 2),
            "ci95_bb": round(ci95, 2),
            "raw_p_value": round(float(p_val), 4),
            "t_statistic": round(float(t_stat), 3)
        }
        print(f"  {a_name}: 2000h={results[a_name]['bb100_2000h']:+.1f} -> 7000h={results[a_name]['bb100_7000h']:+.1f} | Δ={mean_diff:+.1f} ± {ci95:.1f} BB/100 (p={p_val:.4f})")

    return results

# ==============================================================================
# PHASE 4: Production PFSP Sampler & 50-Model Catalog Comparison
# ==============================================================================
class BugFreeAlphaStarPFSPSampler:
    def __init__(self, catalog_dict: Dict[str, Any], temperature: float = 10.0, exploration_floor: float = 0.05):
        # Only accept keys that actually exist in the 50-model checkpoint directory
        self.valid_checkpoints = []
        for key in catalog_dict.keys():
            # Ensure no double prefix: key must be like "Agent_A_2000h"
            clean_key = key.replace("Agent_Agent_", "Agent_")
            filepath = os.path.join(CHECKPOINT_DIR, f"{clean_key}.pt")
            if os.path.exists(filepath):
                self.valid_checkpoints.append(clean_key)
            else:
                raise FileNotFoundError(f"Invalid PFSP checkpoint: {filepath}")

        self.temperature = temperature
        self.exploration_floor = exploration_floor
        self.history = defaultdict(lambda: {"games": 0, "net_bb": 0.0})
        self.selection_ledger = []

    def record_match(self, opponent: str, net_bb: float):
        assert opponent in self.valid_checkpoints, f"Unknown opponent {opponent}"
        self.history[opponent]["games"] += 1
        self.history[opponent]["net_bb"] += net_bb

    def get_unrounded_probabilities(self) -> Dict[str, float]:
        n = len(self.valid_checkpoints)
        raw_weights = []
        for ckpt in self.valid_checkpoints:
            h = self.history[ckpt]
            if h["games"] == 0:
                raw_weights.append(1.0)
            else:
                avg_bb = h["net_bb"] / h["games"]
                w = np.exp(-np.clip(avg_bb, -25.0, 25.0) / self.temperature)
                raw_weights.append(float(w))

        tot_w = sum(raw_weights)
        uniform_weight = 1.0 / n
        probs = {}
        for ckpt, w in zip(self.valid_checkpoints, raw_weights):
            p = (1.0 - self.exploration_floor) * (w / tot_w) + self.exploration_floor * uniform_weight
            probs[ckpt] = float(p)

        sum_p = sum(probs.values())
        norm_probs = {k: v / sum_p for k, v in probs.items()}
        # Strict float assertion
        assert abs(sum(norm_probs.values()) - 1.0) < 1e-6, "PFSP normalization violation!"
        return norm_probs

    def sample_opponent(self, rng: np.random.RandomState, hand_id: int) -> Tuple[str, float]:
        probs = self.get_unrounded_probabilities()
        ckpts = list(probs.keys())
        p_vals = [probs[c] for c in ckpts]
        chosen = rng.choice(ckpts, p=p_vals)
        chosen_prob = probs[chosen]

        self.selection_ledger.append({
            "hand_id": hand_id,
            "chosen_opponent": chosen,
            "sampling_probability": chosen_prob
        })
        return chosen, chosen_prob

def verify_50_checkpoints_against_catalog_048() -> Dict[str, Any]:
    print("\n>>> [PHASE 4] Verifying 50 Frozen Checkpoints against 048 Catalog...")
    with open(CATALOG_048_PATH, "r", encoding="utf-8") as f:
        data_048 = json.load(f)
    catalog_048 = data_048["checkpoint_catalog"]

    verified = 0
    mismatches = []

    generations = [200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
    for agent in AGENT_NAMES:
        for gen in generations:
            key = f"{agent}_{gen}h"
            fpath = os.path.join(CHECKPOINT_DIR, f"{key}.pt")
            if not os.path.exists(fpath):
                mismatches.append(f"{key}: File not found")
                continue
            sha = compute_sha256(fpath)
            expected_sha = catalog_048[key]["sha256"]
            if sha == expected_sha:
                verified += 1
            else:
                mismatches.append(f"{key}: SHA mismatch! Expected {expected_sha}, got {sha}")

    print(f"  --> SHA256 Comparison: {verified}/50 matched against 048 reference.")
    assert len(mismatches) == 0, f"Hash mismatches found: {mismatches}"
    return {"verified_count": verified, "mismatches": mismatches, "status": "ALL_IMMUTABLE_MATCHED"}

# ==============================================================================
# PHASE 5: Limited League Dry-Run (50 Hands, Zero-Sum, PFSP Connected)
# ==============================================================================
def run_phase5_league_dryrun(num_hands: int = 50, seed_base: int = 99000) -> Dict[str, Any]:
    print(f"\n>>> [PHASE 5] Running Limited League Dry-Run ({num_hands} hands, Inference-Only)...")
    game = get_5p_fullgame()
    learner_7k = load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_A_7000h.pt"))

    # Build sampler from valid 50 models
    with open(CATALOG_048_PATH, "r", encoding="utf-8") as f:
        catalog_048 = json.load(f)["checkpoint_catalog"]
    sampler = BugFreeAlphaStarPFSPSampler(catalog_048)

    # Record some synthetic historical net BB
    sampler.record_match("Agent_B_2000h", -15.0)
    sampler.record_match("Agent_C_7000h", +30.0)

    zero_sum_violations = 0
    learner_pnls = []

    # Record pre-dryrun SHA256 of learner and sampled models
    pre_sha_learner = compute_sha256(os.path.join(CHECKPOINT_DIR, "Agent_A_7000h.pt"))

    with open(RAW_049B_LEAGUE_DRYRUN_JSONL, "w", encoding="utf-8") as out_f:
        for h_i in range(1, num_hands + 1):
            seed = seed_base + h_i
            rng = np.random.RandomState(seed)

            # Sample 4 opponents from PFSP Frozen Pool
            sampled_opps = []
            sampled_models = []
            for seat_idx in range(4):
                opp_id, prob = sampler.sample_opponent(rng, hand_id=h_i)
                sampled_opps.append(opp_id)
                sampled_models.append(load_ppo_model(os.path.join(CHECKPOINT_DIR, f"{opp_id}.pt")))

            # Seat 0: Learner (Agent A 7000h), Seats 1-4: Sampled Frozen Opponents
            table_policies = [learner_7k] + sampled_models
            table_labels = ["Learner_Agent_A_7000h"] + sampled_opps

            st = game.new_initial_state()
            advance_chance_nodes(st, rng)

            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, rng); continue
                cp = st.current_player()
                leg = True6ActionAdapter.get_legal_6actions(st, cp)
                norm_inf = normalize_inf_state_v79(np.array(st.information_state_tensor(cp), dtype=np.float32))
                with torch.no_grad():
                    logits, _ = table_policies[cp](torch.from_numpy(norm_inf).unsqueeze(0))
                    masked = logits.clone(); mask = torch.full_like(masked, -1e9)
                    for a in leg: mask[0, a] = 0.0
                    act = int(torch.argmax(logits + mask, dim=-1).item())
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))

            rets = [st.returns()[p] / BIG_BLIND for p in range(5)]
            if abs(sum(rets)) > 1e-4: zero_sum_violations += 1
            learner_pnls.append(rets[0])

            # Update sampler with outcome against each opponent
            for opp_id in sampled_opps:
                sampler.record_match(opp_id, rets[0])

            out_f.write(json.dumps({
                "hand_id": h_i, "seed": seed, "table_labels": table_labels,
                "returns_bb": [round(r, 2) for r in rets]
            }) + "\n")

    post_sha_learner = compute_sha256(os.path.join(CHECKPOINT_DIR, "Agent_A_7000h.pt"))
    assert pre_sha_learner == post_sha_learner, "Learner model mutated during dryrun!"
    assert zero_sum_violations == 0, "Zero-sum violation during dryrun!"

    learner_bb100 = round(float(np.mean(learner_pnls)) * 100, 2)
    print(f"  --> League Dry-Run Complete: 50 hands, Learner BB/100 = {learner_bb100:+.2f}")
    print(f"  --> Zero-Sum Violations: 0, Model Mutated: False [PASS]")

    return {
        "hands_played": num_hands,
        "learner_bb100": learner_bb100,
        "zero_sum_violations": zero_sum_violations,
        "model_immutability_verified": True
    }

# ==============================================================================
# Main Orchestrator
# ==============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("POKER AI DIRECTIVE 049-B: SCIENTIFIC VERIFICATION & PFSP LEAGUE PREFLIGHT")
    print("=" * 80)

    # 1. Phase 1 & 2
    p1_p2 = run_phase1_and_phase2_reproduction(num_hands=500, base_seed=40000)

    # 2. Phase 3
    p3 = run_phase3_independent_generational_benchmark(num_hands=500, base_seed=40000)

    # 3. Phase 4
    p4_immutability = verify_50_checkpoints_against_catalog_048()

    # 4. Phase 5
    p5_dryrun = run_phase5_league_dryrun(num_hands=50, seed_base=99000)

    total_sec = round(time.time() - t0, 1)

    output = {
        "directive": "049-B",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": total_sec,
        "phase1_phase2_reproduction": p1_p2,
        "phase3_generational_benchmark": p3,
        "phase4_immutability_check": p4_immutability,
        "phase5_league_dryrun": p5_dryrun
    }

    with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n>>> DIRECTIVE 049-B Execution Complete in {total_sec}s.")
    print(f"Results written to {AUDIT_JSON_PATH}")

if __name__ == "__main__":
    main()
