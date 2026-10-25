#!/usr/bin/env python3
"""
DIRECTIVE 049 Preflight & Rigorous Audit Suite
- Part 1: Corrected Holm-Bonferroni Step-Down Function (with Cumulative Maximum)
- Part 2: 50-Checkpoint Catalog & Immutability Verification
- Part 3: Agent E 6-Event Ledger Multi-Flag Re-audit (reproducing 047 conditions)
- Part 4: Balanced Generational Cross-Play Matrix (Seat-Rotated, 7000h vs 2000h)
- Part 5: AlphaStar-style PFSP (Prioritized Fictitious Self-Play) Opponent Sampler Implementation
- All Raw JSONL datasets preserved for full reproducibility
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
AUDIT_JSON_PATH = "v1024/directive049_audit_results.json"
DATA_DIR = "v1024/data"
os.makedirs(DATA_DIR, exist_ok=True)

RAW_AGENT_E_JSONL = os.path.join(DATA_DIR, "directive049_raw_agent_e_allin_audit.jsonl")
RAW_CROSSPLAY_JSONL = os.path.join(DATA_DIR, "directive049_raw_crossplay_audit.jsonl")
RAW_CATCHER_PAIRED_JSONL = os.path.join(DATA_DIR, "directive049_raw_catcher_paired.jsonl")

# ==============================================================================
# 1. Corrected Holm-Bonferroni Procedure (with Cumulative Maximum Monotonicity)
# ==============================================================================
def holm_bonferroni_corrected(p_values: List[float]) -> List[float]:
    """
    Standard Holm-Bonferroni step-down adjustment:
    Sorted p-values: p_(1) <= p_(2) <= ... <= p_(m)
    adj_p_(k) = min(1.0, max(adj_p_(k-1), (m - k + 1) * p_(k)))
    Ensures monotonicity: adj_p_(k) >= adj_p_(k-1)
    """
    m = len(p_values)
    if m == 0:
        return []
    
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_sorted = [0.0] * m
    
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        k = rank + 1
        adj = min(1.0, (m - k + 1) * p)
        running_max = max(running_max, adj)
        adjusted_sorted[rank] = (orig_idx, running_max)
    
    # Restore original order
    final_adj = [0.0] * m
    for orig_idx, adj_p in adjusted_sorted:
        final_adj[orig_idx] = round(adj_p, 4)
    return final_adj

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
# 2. 50-Checkpoint Full Catalog & Immutability Verification
# ==============================================================================
def build_50_checkpoint_catalog() -> Dict[str, Any]:
    print(">>> [Part 2] Building Full 50-Checkpoint Official Catalog...")
    generations = [200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
    catalog = {}
    verified = 0

    for agent in AGENT_NAMES:
        for gen in generations:
            key = f"{agent}_{gen}h"
            filename = f"{agent}_{gen}h.pt"
            filepath = os.path.join(CHECKPOINT_DIR, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                sha = compute_sha256(filepath)
                catalog[key] = {
                    "filename": filename,
                    "agent": agent,
                    "generation_hands": gen,
                    "sha256": sha,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "optimizer_restored": False,
                    "status": "VERIFIED_ON_DISK"
                }
                verified += 1
            else:
                catalog[key] = {
                    "filename": filename,
                    "status": "MISSING"
                }
    print(f"Catalog complete: {verified}/50 checkpoints verified on disk.")
    return catalog

# ==============================================================================
# 3. Agent E 6-Event Ledger Multi-Flag Re-audit (Reproducing 047 conditions)
# ==============================================================================
def run_agent_e_multi_flag_allin_audit(num_hands: int = 500, seed_base: int = 40000) -> Dict[str, Any]:
    """
    Strict ledger re-audit reproducing the exact seed conditions of 047 (base_seed=40000).
    Separates independent event flags and assigns a mutually exclusive primary category:
    Flags:
      - hero_pushed_direct_allin (Act 5)
      - hero_converted_allin (Act 2,3,4 reaching all-in cap)
      - facing_opp_allin
      - hero_called_opp_allin
      - hero_folded_to_opp_allin
      - uncalled_allin
      - showdown_reached
    """
    print(f"\n>>> [Part 3] Running Agent E Multi-Flag All-in Re-Audit under 047 Seeds ({num_hands} hands)...")
    game = get_5p_fullgame()
    ckpt_e = os.path.join(CHECKPOINT_DIR, "Agent_E_7000h.pt")
    model_e = load_ppo_model(ckpt_e)
    catcher = FlawlessTwoPairCatcherBot(seat=1, seed=seed_base)

    primary_category_counts = Counter()
    primary_category_pnls = defaultdict(list)
    flag_counts = Counter()

    with open(RAW_AGENT_E_JSONL, "w", encoding="utf-8") as out_f:
        for h_i in range(1, num_hands + 1):
            seed = seed_base + h_i
            hero_seat = (h_i - 1) % 5
            rng = np.random.RandomState(seed)
            st = game.new_initial_state()
            advance_chance_nodes(st, rng)
            cards_map = safe_parse_cards_v46(st)

            tracker = HandLedgerTracker(num_players=5)

            # Flags for this hand
            flag_hero_pushed = False
            flag_hero_converted = False
            flag_facing_allin = False
            flag_hero_called_allin = False
            flag_hero_folded_to_allin = False
            hero_action_list = []

            while not st.is_terminal():
                if st.is_chance_node():
                    advance_chance_nodes(st, rng)
                    continue

                cp = st.current_player()
                legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                round_idx = get_round_from_state(st)
                spent_before = extract_player_spent(st)
                to_call = max(spent_before) - spent_before[cp]

                opp_allin_active = any(tracker.is_player_allin(p) for p in range(5) if p != cp)

                if cp == hero_seat:
                    raw_inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)
                    t_state = torch.from_numpy(norm_inf).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = model_e(t_state)
                        masked = logits.clone()
                        mask = torch.full_like(masked, -1e9)
                        for a in legal_6: mask[0, a] = 0.0
                        act_6 = int(torch.argmax(logits + mask, dim=-1).item())

                    hero_action_list.append((round_idx, act_6, to_call))

                    if opp_allin_active and to_call > 0:
                        flag_facing_allin = True
                        if act_6 == 0:
                            flag_hero_folded_to_allin = True
                        elif act_6 in [1, 2, 3, 4, 5]:
                            flag_hero_called_allin = True

                    if act_6 == 5:
                        flag_hero_pushed = True
                else:
                    act_6 = catcher.select_action(st, cp, legal_6, cards_map, round_idx)

                nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                st.apply_action(nat_act)
                spent_after = extract_player_spent(st)[cp]
                tracker.record_action(cp, act_6, nat_act, round_idx, spent_after)

                if cp == hero_seat and act_6 in [2, 3, 4] and tracker.is_player_allin(hero_seat):
                    flag_hero_converted = True

            hero_pnl_bb = st.returns()[hero_seat] / BIG_BLIND
            hero_is_allin = tracker.is_player_allin(hero_seat)
            allin_was_called = tracker.was_allin_called(hero_seat)
            flag_uncalled = hero_is_allin and not allin_was_called

            # Primary Mutually Exclusive Categorization:
            if flag_hero_pushed and allin_was_called:
                cat = "Cat1_HeroPushed_Called"
            elif flag_hero_pushed and flag_uncalled:
                cat = "Cat2_HeroPushed_Uncalled"
            elif flag_facing_allin and flag_hero_called_allin:
                cat = "Cat3_FacingAllin_HeroCalled"
            elif flag_facing_allin and flag_hero_folded_to_allin:
                cat = "Cat4_FacingAllin_HeroFolded"
            elif flag_hero_converted:
                cat = "Cat5_HeroConverted_Allin"
            else:
                cat = "Cat6_Normal_NonAllin_Hand"

            primary_category_counts[cat] += 1
            primary_category_pnls[cat].append(hero_pnl_bb)

            # Record Flags
            if flag_hero_pushed: flag_counts["hero_pushed"] += 1
            if flag_hero_converted: flag_counts["hero_converted"] += 1
            if flag_facing_allin: flag_counts["facing_allin"] += 1
            if flag_hero_called_allin: flag_counts["called_allin"] += 1
            if flag_hero_folded_to_allin: flag_counts["folded_to_allin"] += 1
            if flag_uncalled: flag_counts["uncalled_allin"] += 1

            record = {
                "hand_id": h_i,
                "seed": seed,
                "hero_seat": hero_seat,
                "primary_category": cat,
                "pnl_bb": round(hero_pnl_bb, 2),
                "flags": {
                    "hero_pushed": flag_hero_pushed,
                    "hero_converted": flag_hero_converted,
                    "facing_allin": flag_facing_allin,
                    "called_allin": flag_hero_called_allin,
                    "folded_to_allin": flag_hero_folded_to_allin,
                    "uncalled": flag_uncalled
                },
                "actions": hero_action_list
            }
            out_f.write(json.dumps(record) + "\n")

    summary = {}
    for cat, cnt in sorted(primary_category_counts.items()):
        pnls = primary_category_pnls[cat]
        summary[cat] = {
            "count": cnt,
            "pct": round(cnt / num_hands * 100, 2),
            "avg_ev_bb": round(float(np.mean(pnls)), 2) if pnls else 0.0,
            "total_pnl_bb": round(float(np.sum(pnls)), 2) if pnls else 0.0
        }
        print(f"  {cat}: {cnt} hands ({summary[cat]['pct']}%), Avg EV = {summary[cat]['avg_ev_bb']:+.2f} BB")

    print(f"\n  Independent Flag Totals:")
    for flg, cnt in flag_counts.items():
        print(f"    - {flg}: {cnt} occurrences")

    return {
        "primary_summary": summary,
        "flag_totals": dict(flag_counts),
        "total_hands": num_hands,
        "seed_base": seed_base
    }

# ==============================================================================
# 4. Balanced Generational Cross-Play Matrix (Seat Rotated)
# ==============================================================================
def run_balanced_generational_crossplay(num_rounds: int = 100, seed_base: int = 80000) -> Dict[str, Any]:
    """
    Runs a fully balanced, seat-rotated generational match:
    2 Agents from 7000h (A & B) vs 3 Agents from 2000h (C, D, E)
    Seats rotate every round: each configuration played for 5 seats x 100 hands = 500 hands.
    Tests whether 7000h generation has a statistically robust edge over 2000h generation.
    """
    print(f"\n>>> [Part 4] Running Balanced Generational Cross-Play ({num_rounds * 5} total hands)...")
    game = get_5p_fullgame()
    policies_7k = {
        "A": load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_A_7000h.pt")),
        "B": load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_B_7000h.pt"))
    }
    policies_2k = {
        "C": load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_C_2000h.pt")),
        "D": load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_D_2000h.pt")),
        "E": load_ppo_model(os.path.join(CHECKPOINT_DIR, "Agent_E_2000h.pt"))
    }

    base_assignment = [
        ("Agent_A_7000h", policies_7k["A"]),
        ("Agent_B_7000h", policies_7k["B"]),
        ("Agent_C_2000h", policies_2k["C"]),
        ("Agent_D_2000h", policies_2k["D"]),
        ("Agent_E_2000h", policies_2k["E"]),
    ]

    hand_idx = 0
    pnl_by_policy = defaultdict(list)
    zero_sum_violations = 0

    with open(RAW_CROSSPLAY_JSONL, "w", encoding="utf-8") as out_f:
        for r_i in range(num_rounds):
            for shift in range(5):
                hand_idx += 1
                seed = seed_base + hand_idx
                rng = np.random.RandomState(seed)

                # Rotate seats to ensure absolute seat fairness
                current_assignment = base_assignment[shift:] + base_assignment[:shift]
                current_policies = [item[1] for item in current_assignment]
                current_labels = [item[0] for item in current_assignment]

                st = game.new_initial_state()
                advance_chance_nodes(st, rng)

                while not st.is_terminal():
                    if st.is_chance_node():
                        advance_chance_nodes(st, rng)
                        continue
                    cp = st.current_player()
                    legal_6 = True6ActionAdapter.get_legal_6actions(st, cp)
                    raw_inf = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    norm_inf = normalize_inf_state_v79(raw_inf)
                    with torch.no_grad():
                        logits, _ = current_policies[cp](torch.from_numpy(norm_inf).unsqueeze(0))
                        masked = logits.clone()
                        mask = torch.full_like(masked, -1e9)
                        for a in legal_6: mask[0, a] = 0.0
                        act_6 = int(torch.argmax(logits + mask, dim=-1).item())
                    nat_act = True6ActionAdapter.map_6action_to_engine_amount(st, cp, act_6)
                    st.apply_action(nat_act)

                rets = [st.returns()[p] / BIG_BLIND for p in range(5)]
                if abs(sum(rets)) > 1e-4:
                    zero_sum_violations += 1

                for p in range(5):
                    label = current_labels[p]
                    pnl_by_policy[label].append(rets[p])

                out_f.write(json.dumps({
                    "hand_id": hand_idx,
                    "seed": seed,
                    "shift": shift,
                    "seat_labels": current_labels,
                    "returns_bb": [round(r, 2) for r in rets]
                }) + "\n")

    # Aggregate
    results = {}
    team_7k_pnls = []
    team_2k_pnls = []

    for label, pnls in sorted(pnl_by_policy.items()):
        arr = np.array(pnls)
        bb100 = float(np.mean(arr)) * 100
        std = float(np.std(arr, ddof=1)) * 100
        ci95 = 1.96 * (std / np.sqrt(len(arr)))
        results[label] = {
            "bb_per_100": round(bb100, 2),
            "ci95_bb": round(ci95, 2),
            "total_pnl_bb": round(float(np.sum(arr)), 2),
            "hands_played": len(arr)
        }
        if "7000h" in label:
            team_7k_pnls.extend(pnls)
        else:
            team_2k_pnls.extend(pnls)
        print(f"  {label}: {bb100:+.2f} ± {ci95:.2f} BB/100 (across {len(arr)} hands)")

    team_7k_mean = float(np.mean(team_7k_pnls)) * 100
    team_2k_mean = float(np.mean(team_2k_pnls)) * 100
    t_stat, p_val = stats.ttest_ind(team_7k_pnls, team_2k_pnls)

    crossplay_summary = {
        "policy_results": results,
        "team_7000h_bb100": round(team_7k_mean, 2),
        "team_2000h_bb100": round(team_2k_mean, 2),
        "team_difference_p_value": round(float(p_val), 4),
        "total_hands": hand_idx,
        "zero_sum_violations": zero_sum_violations
    }
    print(f"\n  --> Team 7000h Avg = {team_7k_mean:+.2f} BB/100 vs Team 2000h Avg = {team_2k_mean:+.2f} BB/100 (p={p_val:.4f})")
    print(f"  --> Zero-sum violations = {zero_sum_violations}")
    return crossplay_summary

# ==============================================================================
# 5. AlphaStar-style PFSP (Prioritized Fictitious Self-Play) Opponent Sampler
# ==============================================================================
class PrioritizedFictitiousSelfPlaySampler:
    """
    Standard AlphaStar PFSP Sampler for Texas Hold'em Multi-Agent League:
    P(opponent i) = f(payoff_i) / sum(f(payoff_j))
    Where f(x) prioritizes opponents where the agent has a win rate near 50% or negative EV,
    preventing both exploitation of trivial opponents and forgetting of tough past policies.
    """
    def __init__(self, historical_checkpoints: List[str]):
        self.checkpoints = historical_checkpoints
        # Payoff tracking: (wins, total_games, net_bb)
        self.matchup_history = defaultdict(lambda: {"games": 0, "net_bb": 0.0})

    def record_match(self, opponent_id: str, net_bb: float):
        self.matchup_history[opponent_id]["games"] += 1
        self.matchup_history[opponent_id]["net_bb"] += net_bb

    def get_sampling_distribution(self, weighting_mode: str = "hard_opponents") -> Dict[str, float]:
        """
        Calculates normalized sampling probabilities summing to exactly 1.0.
        Weighting function: f(net_bb) = exp(-clip(avg_bb, -50, 50) / temperature)
        Harder opponents (lower net_bb) receive higher sampling priority.
        """
        if not self.checkpoints:
            return {}

        weights = []
        for ckpt in self.checkpoints:
            hist = self.matchup_history[ckpt]
            if hist["games"] == 0:
                # Prioritize unexplored opponents with default exploration weight
                weights.append(1.0)
            else:
                avg_bb = hist["net_bb"] / hist["games"]
                if weighting_mode == "hard_opponents":
                    # Focus on opponents where hero loses or draws
                    w = np.exp(-np.clip(avg_bb, -20.0, 20.0) / 10.0)
                else:
                    w = 1.0
                weights.append(float(w))

        tot = sum(weights)
        probs = [w / tot for w in weights]
        return {ckpt: round(prob, 4) for ckpt, prob in zip(self.checkpoints, probs)}

# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("DIRECTIVE 049 PREFLIGHT & RIGOROUS AUDIT SUITE")
    print("=" * 80)

    # 1. Unit test of corrected Holm procedure
    test_raw_p = [0.0006, 0.0236, 0.4346, 0.4648, 0.7912]
    corr_p = holm_bonferroni_corrected(test_raw_p)
    print(">>> [Part 1] Holm-Bonferroni Corrected Procedure Test:")
    for raw, adj in zip(test_raw_p, corr_p):
        print(f"  raw p={raw:.4f} --> adj p={adj:.4f}")
    assert all(corr_p[i] <= corr_p[i+1] for i in range(len(corr_p)-1)), "Holm monotonicity violated!"
    print("  [PASS] Holm-Bonferroni monotonicity strictly verified.")

    # 2. 50-checkpoint catalog
    catalog = build_50_checkpoint_catalog()

    # 3. Agent E 6-event re-audit
    agent_e_audit = run_agent_e_multi_flag_allin_audit(num_hands=500, seed_base=40000)

    # 4. Balanced cross-play matrix
    crossplay_results = run_balanced_generational_crossplay(num_rounds=100, seed_base=80000)

    # 5. PFSP Sampler verification
    print("\n>>> [Part 5] Testing AlphaStar PFSP Sampler...")
    sample_pool = [f"Agent_A_{g}h" for g in [1000, 2000, 3000, 4000, 5000, 6000]]
    sampler = PrioritizedFictitiousSelfPlaySampler(sample_pool)
    # Simulate some matches
    sampler.record_match("Agent_A_2000h", -15.0) # Hero lost
    sampler.record_match("Agent_A_4000h", +20.0) # Hero won
    dist = sampler.get_sampling_distribution("hard_opponents")
    sum_dist = sum(dist.values())
    print("  PFSP Sample distribution:", dist)
    print(f"  Sum of probabilities: {sum_dist:.4f} (Strictly Normalized)")
    assert abs(sum_dist - 1.0) < 1e-3, "PFSP probabilities must sum to 1.0!"

    total_sec = round(time.time() - t0, 1)

    output = {
        "directive": "049",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": total_sec,
        "checkpoint_catalog_50_verified": sum(1 for v in catalog.values() if v.get("status") == "VERIFIED_ON_DISK"),
        "checkpoint_catalog": catalog,
        "agent_e_multi_flag_allin_audit": agent_e_audit,
        "balanced_crossplay_results": crossplay_results,
        "pfsp_sampler_test": {
            "distribution": dist,
            "sum_probabilities": round(sum_dist, 4)
        }
    }

    with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n>>> DIRECTIVE 049 Preflight & Audit Completed in {total_sec}s.")
    print(f"Results written to {AUDIT_JSON_PATH}")

if __name__ == "__main__":
    main()
